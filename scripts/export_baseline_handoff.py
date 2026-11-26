"""Export a progressive, verified index of the six B1 checkpoint handoffs."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.project_config import (
    PROJECT_ROOT, file_sha256, load_config, read_jsonl, require, resolve_base_path,
    rows_digest, write_json,
)


def _record(base, path):
    require(path.resolve().is_relative_to(base), f"Handoff file escapes the project folder: {path}")
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty handoff file: {path}")
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _official_rows(base, manifest, name):
    description = manifest["datasets"][name]
    path = base / "01_data" / name
    record = _record(base, path)
    rows = read_jsonl(path)
    require(record["sha256"] == description["file_sha256"] and rows_digest(rows) == description["rows_sha256"],
            f"Official data hash differs: {name}")
    require(len(rows) == description["rows"] and len({row["id"] for row in rows}) == len(rows),
            f"Official data count or IDs differ: {name}")
    require([row["id"] for row in rows] == description["ids"], f"Official ordered IDs differ: {name}")
    return rows, record


def _verify_run(base, folder, config, setting, seed, train, validation, rare_ids, class_names, trainer_sha):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    run = json.loads((folder / "run_config.json").read_text())
    result = json.loads((folder / "results.json").read_text())
    spec = run["specification"]
    identity = ("B1", setting, seed)
    require((spec["condition"], spec["setting"], spec["seed"]) == identity
            and (result["condition"], result["setting"], result["seed"]) == identity,
            f"B1 run identity differs: {folder}")
    require(spec["protocol_id"] == config["protocol_id"] and spec["mode"] == "main"
            and spec["training"] == config["training"], f"B1 frozen training policy differs: {folder}")
    require(rows_digest([spec]) == run["fingerprint"] == result["fingerprint"], f"B1 fingerprint differs: {folder}")
    require(run["config_provenance_sha256"] == rows_digest([config]), f"B1 frozen configuration differs: {folder}")
    require(spec["training_rows_sha256"] == rows_digest(train)
            and spec["validation_rows_sha256"] == rows_digest(validation), f"B1 official subset differs: {folder}")
    require(spec["rare_class_ids"] == rare_ids and spec["class_names"] == class_names,
            f"B1 label or rare-class metadata differs: {folder}")
    require(spec["trainer_source_sha256"] == result["trainer_source_sha256"] == trainer_sha,
            f"B1 trainer source differs: {folder}")
    require(isinstance(result.get("code_commit"), str) and len(result["code_commit"]) == 40,
            f"B1 code commit is unavailable: {folder}")
    require(result["real_count"] == result["training_rows"] == len(train) and result["synthetic_count"] == 0,
            f"B1 training counts differ: {folder}")

    predictions = read_jsonl(folder / "validation_predictions.jsonl")
    ids = [row["id"] for row in validation]
    labels = np.asarray([row["label"] for row in validation])
    require([row["id"] for row in predictions] == ids
            and [row["true_label"] for row in predictions] == labels.tolist(), f"B1 validation prediction IDs or labels differ: {folder}")
    require([row["text"] for row in predictions] == [row["text"] for row in validation],
            f"B1 validation prediction text differs: {folder}")
    require(all((row["condition"], row["setting"], row["seed"], row["split"]) == (*identity, "validation")
                for row in predictions), f"B1 prediction identity differs: {folder}")
    with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as stored:
        probabilities = stored["probabilities"]
        require(stored["ids"].tolist() == ids and np.array_equal(stored["labels"], labels),
                f"B1 probability IDs or labels differ: {folder}")
    require(probabilities.shape == (len(validation), len(class_names)) and probabilities.dtype == np.float32
            and np.isfinite(probabilities).all()
            and probabilities.min() >= 0 and probabilities.max() <= 1
            and np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6),
            f"B1 probabilities are invalid: {folder}")
    predicted = probabilities.argmax(axis=1)
    require([row["predicted_label"] for row in predictions] == predicted.tolist()
            and np.allclose([row["confidence"] for row in predictions], probabilities.max(axis=1), rtol=0, atol=1e-7),
            f"B1 predictions or confidence differ from probabilities: {folder}")
    metrics = {
        "macro_f1": float(f1_score(labels, predicted, labels=list(range(len(class_names))), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "rare_macro_f1": float(f1_score(labels, predicted, labels=rare_ids, average="macro", zero_division=0)),
        "rare_macro_recall": float(recall_score(labels, predicted, labels=rare_ids, average="macro", zero_division=0)),
    }
    require(all(math.isclose(result[key], value, rel_tol=0, abs_tol=1e-12) for key, value in metrics.items()),
            f"B1 recomputed validation metrics differ: {folder}")
    with (folder / "history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    scores = [float(row["macro_f1"]) for row in history]
    require(scores and len(history) == result["epochs_run"]
            and [int(row["epoch"]) for row in history] == list(range(1, len(history) + 1))
            and scores.index(max(scores)) + 1 == result["best_epoch"]
            and math.isclose(max(scores), result["macro_f1"], rel_tol=0, abs_tol=1e-12),
            f"B1 selected checkpoint history differs: {folder}")
    runtime = json.loads((folder / "runtime.json").read_text())
    require(runtime["device"] == result["device"] and runtime["precision"] == result["precision"] == "float32",
            f"B1 runtime metadata differs: {folder}")

    best = folder / "best_model"
    require(best.is_dir() and any((best / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
            and (best / "tokenizer_config.json").is_file()
            and any((best / name).is_file() for name in ("tokenizer.json", "vocab.txt")),
            f"B1 complete model or tokenizer is missing: {folder}")
    model_config = json.loads((best / "config.json").read_text())
    require(model_config["id2label"] == {str(index): name for index, name in enumerate(class_names)},
            f"B1 saved model labels differ: {folder}")
    names = ("run_config.json", "results.json", "runtime.json", "history.csv", "stage_log.csv",
             "per_class_metrics.csv", "validation_predictions.jsonl", "validation_probabilities.npz")
    files = [folder / name for name in names] + sorted(path for path in best.rglob("*") if path.is_file())
    return {"run_id": f"B1_{setting}_seed_{seed}", "condition": "B1", "setting": setting, "seed": seed,
            "status": "verified", "model_directory": best.relative_to(base).as_posix(),
            "code_commit": result["code_commit"], "trainer_source_sha256": trainer_sha,
            "model_revision": config["training"]["model_revision"], "metrics": metrics,
            "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "files": [_record(base, path) for path in files]}


def export_baseline_handoff(config_path=None, base_path=None, report_root=None):
    config_path = Path(config_path) if config_path else PROJECT_ROOT / "configs/experiment.yaml"
    config = load_config(config_path)
    require(config.get("status") == "frozen" and config["training"].get("learning_rate"),
            "Freeze the shared classifier configuration before exporting B1 handoffs")
    base = resolve_base_path(config, base_path)
    manifest_path = base / "01_data/manifest.json"
    pinned_manifest = config.get("classifier_pilot", {}).get("selection", {}).get("data_manifest_sha256")
    require(pinned_manifest and file_sha256(manifest_path) == pinned_manifest,
            "B1 handoff official manifest differs from the frozen pilot release")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["protocol_id"] == config["protocol_id"], "B1 handoff data protocol differs")
    for key in ("dataset", "settings", "seeds"):
        require(manifest["config"][key] == config[key], f"B1 handoff data configuration differs: {key}")
    metadata = {}
    for name in ("label_mapping.json", "rare_classes.json"):
        path = base / "01_data" / name
        require(file_sha256(path) == manifest["metadata_files"][name], f"B1 handoff metadata hash differs: {name}")
        metadata[name] = json.loads(path.read_text())
    mapping = metadata["label_mapping.json"]
    class_names = [mapping[str(index)] for index in range(config["dataset"]["num_classes"])]
    require(len(mapping) == len(class_names), "B1 handoff label mapping has extra classes")
    rare_ids = metadata["rare_classes.json"]["rare_class_ids"]
    require(rare_ids == manifest["rare_class_ids"], "B1 handoff rare-class IDs differ from the official manifest")
    trainer_sha = file_sha256(PROJECT_ROOT / "src/training.py")
    validation, validation_record = _official_rows(base, manifest, "validation.jsonl")
    completed, missing, input_records = [], [], []
    for setting in config["settings"]:
        for seed in config["seeds"]:
            train, training_record = _official_rows(base, manifest, f"{setting}/seed_{seed}.jsonl")
            input_records.append(training_record)
            folder = base / "04_checkpoints/B1" / setting / f"seed_{seed}"
            result_path = folder / "results.json"
            if not result_path.is_file() or json.loads(result_path.read_text()).get("status") != "completed":
                missing.append({"run_id": f"B1_{setting}_seed_{seed}", "condition": "B1", "setting": setting, "seed": seed,
                                "status": "pending", "reason": "not_started" if not folder.exists() else "incomplete",
                                "expected_model_directory": (folder / "best_model").relative_to(base).as_posix()})
                continue
            completed.append(_verify_run(base, folder, config, setting, seed, train, validation, rare_ids, class_names, trainer_sha))
    index = {"schema_version": 1, "protocol_id": config["protocol_id"], "status": "partial" if missing else "complete",
             "expected_runs": len(input_records), "completed_runs": len(completed), "missing_runs": missing,
             "configuration_file": config_path.name, "configuration_file_sha256": file_sha256(config_path),
             "trainer_source_sha256": trainer_sha, "data_manifest": _record(base, manifest_path),
             "metadata_files": [_record(base, base / "01_data" / name) for name in metadata],
             "validation_data": validation_record, "training_data": input_records, "checkpoints": completed,
             "path_base": "CS682_Project", "contents": "Verified B1 artifacts only; paths are relative to the project output folder. No weights are copied."}
    local_path = base / "handoff" / f"{config['protocol_id']}_b1_checkpoints.json"
    if local_path.exists():
        previous = json.loads(local_path.read_text())
        require(previous["configuration_file_sha256"] == index["configuration_file_sha256"],
                "Existing B1 handoff uses another frozen configuration")
        require(all(previous[key] == index[key] for key in
                    ("trainer_source_sha256", "data_manifest", "metadata_files", "validation_data", "training_data")),
                "Existing B1 handoff trainer or official data changed")
        current = {row["run_id"]: row for row in completed}
        require(all(current.get(row["run_id"]) == row for row in previous["checkpoints"]),
                "A previously verified B1 checkpoint changed or disappeared")
    reports = Path(report_root) if report_root else PROJECT_ROOT / "results/metrics" / config["protocol_id"]
    for path in (local_path, reports / "b1_checkpoint_index.json"):
        if not path.is_file() or json.loads(path.read_text()) != index:
            write_json(path, index)
    print(f"B1 handoff: {len(completed)}/{len(input_records)} verified; status={index['status']}")
    print(local_path)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-path", type=Path)
    args = parser.parse_args()
    export_baseline_handoff(args.config, args.base_path)
