"""Reporting-only evaluation of frozen baseline checkpoints on the official test split."""

import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import training
from .baselines import _eda_parameters
from .classifier_pilot import write_csv
from .project_config import (
    PROJECT_ROOT, file_sha256, load_config, read_jsonl, require,
    resolve_base_path, rows_digest, write_json, write_jsonl,
)


def _shared_policy(config, base):
    policy = {key: config[key] for key in (
        "protocol_id", "dataset", "training", "seeds", "settings", "baselines",
    )}
    for name in ("label_mapping", "rare_classes"):
        policy[f"{name}_sha256"] = file_sha256(base / "01_data" / f"{name}.json")
    return rows_digest([policy])


def _released_rows(path, description):
    require(file_sha256(path) == description["file_sha256"], f"Released file checksum differs: {path}")
    rows = read_jsonl(path)
    require(rows_digest(rows) == description["rows_sha256"] and len(rows) == description["rows"],
            f"Released records differ: {path}")
    ids = [row["id"] for row in rows]
    require(len(set(ids)) == len(ids) and ("ids" not in description or ids == description["ids"]),
            f"Released ordered IDs differ: {path}")
    counts = {str(label): count for label, count in sorted(Counter(row["label"] for row in rows).items())}
    require(counts == description["per_class"], f"Released class counts differ: {path}")
    return rows


def _training_release(config, base):
    manifest_path = base / "01_data/manifest.json"
    selection = config.get("classifier_pilot", {}).get("selection", {})
    require(selection.get("data_manifest_sha256") == file_sha256(manifest_path),
            "The training release manifest differs from the frozen pilot release.")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["protocol_id"] == config["protocol_id"]
            and manifest["config"] == {key: config[key] for key in ("dataset", "settings", "seeds")},
            "Released training split configuration differs.")
    for name in ("label_mapping.json", "rare_classes.json"):
        require(file_sha256(base / "01_data" / name) == manifest["metadata_files"][name],
                f"Released metadata checksum differs: {name}")
    num_classes = config["dataset"]["num_classes"]
    rare_ids = training._rare_ids(base, num_classes)
    require(rare_ids == manifest["rare_class_ids"], "Released rare-class IDs differ.")
    validation = _released_rows(base / "01_data/validation.jsonl", manifest["datasets"]["validation.jsonl"])
    training_hashes = {}
    for setting in config["settings"]:
        for seed in config["seeds"]:
            relative = f"{setting}/seed_{seed}.jsonl"
            description = manifest["datasets"][relative]
            original = _released_rows(base / "01_data" / relative, description)
            training_hashes[(setting, seed, "B1")] = rows_digest(original)
            for condition in ("B2", "B3"):
                data_relative = f"01_data/baselines/{condition}/{relative}"
                report_relative = f"01_data/baselines/reports/{condition}/{relative[:-6]}.json"
                report_path = base / report_relative
                report = json.loads(report_path.read_text())
                specification = report["specification"]
                require(report.get("status") == "prepared" and rows_digest([specification]) == report["config_sha256"],
                        f"Invalid baseline preparation report: {report_path}")
                for record in (report, specification):
                    require((record["condition"], record["setting"], record["seed"])
                            == (condition, setting, seed), f"Baseline release identity differs: {report_path}")
                    require(record["source_file_sha256"] == description["file_sha256"]
                            and record["source_rows_sha256"] == description["rows_sha256"],
                            f"Baseline source differs from the released real subset: {report_path}")
                parameters = _eda_parameters(config) if condition == "B3" else {"target": "largest_selected_class_count"}
                require(specification["protocol_id"] == config["protocol_id"]
                        and specification["source_path"] == f"01_data/{relative}"
                        and specification["num_classes"] == num_classes
                        and specification["dataset_source"] == {"id": config["dataset"]["dataset_id"],
                                                                 "revision": config["dataset"]["dataset_revision"]}
                        and specification["method"] == config["baselines"][condition]
                        and specification["parameters"] == parameters,
                        f"Baseline preparation policy differs from the frozen release: {report_path}")
                require(report["data_path"] == data_relative and report["report_path"] == report_relative,
                        f"Baseline release path differs: {report_path}")
                rows = _released_rows(base / data_relative, report)
                require(rows[:len(original)] == original, f"Baseline changed the released real prefix: {data_relative}")
                training_hashes[(setting, seed, condition)] = rows_digest(rows)
    return {"training_rows_sha256": training_hashes, "validation_rows_sha256": rows_digest(validation),
            "rare_class_ids": rare_ids, "class_names": training._class_names(base, num_classes)}


def _main_entries(config, base, plan):
    require(config.get("status") == "frozen", "Freeze the classifier policy before test reporting.")
    rate = config["training"].get("learning_rate")
    require(isinstance(rate, (int, float)) and not isinstance(rate, bool) and math.isfinite(rate) and rate > 0,
            "A frozen positive learning rate is required.")
    require(config["training"].get("precision") == "float32", "Test reporting uses the shared float32 policy.")
    paths = [base / "05_results/metrics" / condition / setting / f"seed_{seed}.json"
             for setting, seed, condition in plan]
    require(all(path.is_file() for path in paths),
            "Complete every baseline validation result before opening the test split.")
    policy_sha256 = _shared_policy(config, base)
    release = _training_release(config, base)
    entries = []
    for (setting, seed, condition), path in zip(plan, paths):
        result = json.loads(path.read_text())
        require(result.get("status") == "completed", f"Incomplete baseline: {path}")
        require((result.get("setting"), result.get("seed"), result.get("condition"))
                == (setting, seed, condition), f"Baseline identity differs: {path}")
        require(result.get("shared_training_policy_sha256") == policy_sha256,
                f"Baseline does not match the frozen training policy: {path}")
        reused = bool(result.get("training_reused"))
        require(not reused or (condition == "B2" and setting == "low_resource"),
                "Only balanced B2 may reuse a B1 checkpoint.")
        source_condition = "B1" if reused else condition
        folder = base / "04_checkpoints" / source_condition / setting / f"seed_{seed}"
        run_path = folder / "run_config.json"
        source_result_path = folder / "results.json"
        require(run_path.is_file() and source_result_path.is_file(), f"Missing completed main run: {folder}")
        run = json.loads(run_path.read_text())
        source_result = json.loads(source_result_path.read_text())
        specification = run["specification"]
        require(rows_digest([specification]) == run["fingerprint"], f"Checkpoint specification checksum differs: {folder}")
        require(source_result.get("status") == "completed"
                and run["fingerprint"] == source_result.get("fingerprint") == result.get("fingerprint"),
                f"Baseline/checkpoint provenance differs: {folder}")
        require((specification.get("mode"), specification.get("condition"), specification.get("setting"),
                 specification.get("seed"), specification.get("protocol_id"))
                == ("main", source_condition, setting, seed, config["protocol_id"]),
                f"Expected the matching main checkpoint, not a pilot checkpoint: {folder}")
        require(specification["training"] == config["training"], f"Checkpoint training policy differs: {folder}")
        require(specification.get("trainer_source_sha256") == file_sha256(Path(training.__file__)),
                f"Checkpoint released trainer source differs: {folder}")
        require(specification.get("training_rows_sha256")
                == release["training_rows_sha256"][(setting, seed, source_condition)],
                f"Checkpoint was not trained on the released subset: {folder}")
        for key in ("validation_rows_sha256", "rare_class_ids", "class_names"):
            require(specification.get(key) == release[key], f"Checkpoint released {key} differs: {folder}")
        best = folder / "best_model"
        required = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"}
        require(required <= {file.name for file in best.glob("*") if file.is_file()},
                f"Checkpoint model or tokenizer is incomplete: {best}")
        if reused:
            real = base / "01_data" / setting / f"seed_{seed}.jsonl"
            oversampled = base / "01_data/baselines/B2" / setting / f"seed_{seed}.jsonl"
            require(real.read_bytes() == oversampled.read_bytes(), "Balanced B2 inputs no longer match B1.")
            require(result.get("reused_from") == folder.relative_to(base).as_posix(), "Invalid B2 source reference.")
        checkpoint_files = sorted(file for file in best.rglob("*") if file.is_file())
        checkpoint_files += [run_path, source_result_path]
        entries.append({
            "condition": condition, "setting": setting, "seed": seed,
            "reused": reused, "folder": folder, "validation_result": result,
            "checkpoint_sha256": {file.relative_to(folder).as_posix(): file_sha256(file)
                                  for file in checkpoint_files},
            "shared_training_policy_sha256": policy_sha256,
        })
    return entries


def _official_test(config, base):
    manifest_path = base / "01_data/manifest.json"
    selection = config.get("classifier_pilot", {}).get("selection", {})
    require(selection.get("data_manifest_sha256") == file_sha256(manifest_path),
            "The test release manifest differs from the frozen pilot release.")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["protocol_id"] == config["protocol_id"], "Test release protocol differs.")
    require(manifest["source"]["dataset_id"] == config["dataset"]["dataset_id"]
            and manifest["source"]["revision"] == config["dataset"]["dataset_revision"],
            "Test dataset revision differs from the frozen release.")
    description = manifest["datasets"]["test.jsonl"]
    path = base / "01_data/test.jsonl"
    require(file_sha256(path) == description["file_sha256"], "Official test file checksum differs.")
    rows = read_jsonl(path)
    require(len(rows) == description["rows"] == manifest["source"]["official_test_rows"],
            "Official test row count differs.")
    require(rows_digest(rows) == description["rows_sha256"], "Official test record checksum differs.")
    ids = [row["id"] for row in rows]
    require(ids == description["ids"] and len(set(ids)) == len(ids)
            and all(identifier.startswith("test:") for identifier in ids), "Official test ordered IDs differ.")
    require(set(row["label"] for row in rows) == set(range(config["dataset"]["num_classes"])),
            "Official test labels differ from the classifier mapping.")
    return rows, description["file_sha256"], file_sha256(manifest_path)


def _paths(base, condition, setting, seed):
    relative = Path(condition) / setting / f"seed_{seed}"
    return {
        "predictions": base / "05_results/predictions" / relative.with_suffix(".jsonl"),
        "probabilities": base / "05_results/probabilities/test" / relative.with_suffix(".npz"),
        "per_class": base / "05_results/per_class_metrics/test" / relative.with_suffix(".csv"),
        "cost": base / "06_cost_logs/inference" / relative.with_suffix(".json"),
        "result": base / "05_results/metrics/test" / relative.with_suffix(".json"),
        "specification": base / "05_results/inference_runs" / relative / "specification.json",
    }


def _evaluate_checkpoint(entry, rows, config, base):
    libraries = training._libraries()
    torch = libraries.torch
    device = training.select_device(config["training"].get("device", "auto"), torch)
    runtime = training._runtime(libraries, device)
    started = time.perf_counter()
    start_time = datetime.now(timezone.utc).isoformat()
    best = entry["folder"] / "best_model"
    model = None
    try:
        tokenizer = libraries.AutoTokenizer.from_pretrained(str(best), local_files_only=True)
        features, tokens = training._tokenize(rows, tokenizer, config["training"]["max_length"])
        loader = libraries.DataLoader(
            features, batch_size=config["training"]["eval_batch_size"], shuffle=False, num_workers=0,
            collate_fn=libraries.DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        )
        model = libraries.AutoModelForSequenceClassification.from_pretrained(
            str(best), local_files_only=True, attn_implementation="eager",
        ).float().to(device)
        require(model.config.num_labels == config["dataset"]["num_classes"], "Checkpoint label count differs.")
        rare_ids = training._rare_ids(base, config["dataset"]["num_classes"])
        metrics, labels, predictions, probabilities, seconds = training._evaluate(
            model, loader, libraries, device, config["dataset"]["num_classes"], rare_ids,
        )
        require(labels.tolist() == [row["label"] for row in rows], "Test prediction order differs.")
        metrics["test_loss"] = metrics.pop("validation_loss")
        precision, recall, f1, support = libraries.precision_recall_fscore_support(
            labels, predictions, labels=list(range(config["dataset"]["num_classes"])), zero_division=0,
        )
        names = training._class_names(base, config["dataset"]["num_classes"])
        per_class = [{"label": label, "intent": names[label], "rare_class": label in rare_ids,
                      "precision": float(precision[label]), "recall": float(recall[label]),
                      "f1": float(f1[label]), "support": int(support[label])}
                     for label in range(config["dataset"]["num_classes"])]
        timing = {"inference_seconds": seconds, "total_seconds": time.perf_counter() - started,
                  "nonpadding_tokens": tokens, "start_time": start_time,
                  "end_time": datetime.now(timezone.utc).isoformat(), "runtime": runtime}
        return metrics, labels, predictions, probabilities, per_class, timing
    finally:
        model = None
        training._release(torch, device)


def _save_probabilities(path, probabilities, labels, ids):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, probabilities=probabilities, labels=labels, ids=np.asarray(ids))
    temporary.replace(path)


def _report_entry(entry, rows, config, base, test_sha256, manifest_sha256, completed):
    condition, setting, seed = entry["condition"], entry["setting"], entry["seed"]
    paths = _paths(base, condition, setting, seed)
    specification = {
        "condition": condition, "setting": setting, "seed": seed, "split": "test",
        "protocol_id": config["protocol_id"], "reporting_only": True,
        "checkpoint_sha256": entry["checkpoint_sha256"], "test_file_sha256": test_sha256,
        "data_manifest_sha256": manifest_sha256,
        "shared_training_policy_sha256": entry["shared_training_policy_sha256"],
        "evaluation_source_sha256": file_sha256(Path(__file__)),
        "training_source_sha256": file_sha256(Path(training.__file__)),
    }
    fingerprint = rows_digest([specification])
    if paths["specification"].exists():
        saved = json.loads(paths["specification"].read_text())
        require(saved == {"fingerprint": fingerprint, "specification": specification},
                f"Existing test reporting inputs differ: {paths['specification']}")
    else:
        require(not paths["result"].exists(), "Existing test result has no reporting specification.")
        write_json(paths["specification"], {"fingerprint": fingerprint, "specification": specification})
    if paths["result"].exists():
        result = json.loads(paths["result"].read_text())
        require(result.get("status") == "completed" and result.get("fingerprint") == fingerprint,
                f"Invalid completed test report: {paths['result']}")
        for name in ("predictions", "probabilities", "per_class", "cost"):
            require(paths[name].is_file() and file_sha256(paths[name]) == result["artifact_sha256"][name],
                    f"Completed test artifact differs: {paths[name]}")
        print(f"Reusing test report {condition}/{setting}/seed_{seed}", flush=True)
        return result

    source_result = entry["validation_result"]
    identity = {"condition": condition, "setting": setting, "seed": seed, "split": "test"}
    if entry["reused"]:
        source = completed[(setting, seed, "B1")]
        source_paths = _paths(base, "B1", setting, seed)
        predictions = [{**row, "condition": "B2", "reused_from_condition": "B1"}
                       for row in read_jsonl(source_paths["predictions"])]
        write_jsonl(paths["predictions"], predictions)
        for name in ("probabilities", "per_class"):
            paths[name].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_paths[name], paths[name])
        metrics = {key: source[key] for key in (
            "macro_f1", "accuracy", "rare_macro_f1", "rare_macro_recall", "test_loss",
        )}
        timing = {"inference_seconds": 0.0, "total_seconds": 0.0, "nonpadding_tokens": 0,
                  "runtime": source["runtime"], "start_time": None, "end_time": None}
        reuse = {"inference_reused": True, "reused_from_condition": "B1",
                 "source_fingerprint": source["fingerprint"], "source_total_seconds": source["total_seconds"],
                 "accounting_note": "Identical balanced B2 inputs reuse B1 test predictions; count inference once."}
    else:
        metrics, labels, predicted, probabilities, per_class, timing = _evaluate_checkpoint(entry, rows, config, base)
        predictions = [{"id": row["id"], "true_label": int(labels[index]),
                        "predicted_label": int(predicted[index]), "confidence": float(probabilities[index].max()),
                        **identity} for index, row in enumerate(rows)]
        write_jsonl(paths["predictions"], predictions)
        _save_probabilities(paths["probabilities"], probabilities, labels, [row["id"] for row in rows])
        write_csv(paths["per_class"], per_class)
        reuse = {"inference_reused": False}
    metrics["rare_class_f1"] = metrics["rare_macro_f1"]
    metrics["rare_class_recall"] = metrics["rare_macro_recall"]
    provenance = {"code_commit": training._code_commit(), "evaluation_source_sha256": specification["evaluation_source_sha256"],
                  "model_revision": config["training"]["model_revision"],
                  "frozen_config_sha256": rows_digest([config])}
    write_json(paths["cost"], {"stage": "test_inference", **identity, **timing, **reuse, **provenance,
                              "timing_note": "Elapsed inference time; accelerator utilization was not measured."})
    result = {"status": "completed", "fingerprint": fingerprint, **identity, **metrics, **timing, **reuse, **provenance,
              "reporting_only": True, "test_used_for_selection": False, "test_rows": len(rows),
              "additional_inference_seconds": timing["total_seconds"],
              "shared_training_policy_sha256": entry["shared_training_policy_sha256"],
              "checkpoint_fingerprint": source_result["fingerprint"],
              "best_epoch": source_result["best_epoch"], "epochs_run": source_result["epochs_run"],
              "real_count": source_result["real_count"], "synthetic_count": source_result["synthetic_count"],
              "training_rows": source_result["training_rows"],
              "additional_draws": source_result.get("additional_draws", source_result["training_rows"]
                                                     - source_result["real_count"] - source_result["synthetic_count"]),
              "artifact_sha256": {name: file_sha256(paths[name]) for name in ("predictions", "probabilities", "per_class", "cost")}}
    write_json(paths["result"], result)
    print(f"Reported test {condition}/{setting}/seed_{seed}: macro-F1={metrics['macro_f1']:.4f}", flush=True)
    return result


def evaluate_baselines(config_path=None, base_path=None, execute=False):
    """Evaluate the completed frozen matrix once; test scores never select settings."""
    config = load_config(config_path)
    base = resolve_base_path(config, base_path)
    plan = [(setting, int(seed), condition) for setting in config["settings"] for seed in config["seeds"]
            for condition in ("B1", "B2", "B3")]
    if not execute:
        for setting, seed, condition in plan:
            print(f"Test reporting: {condition}/{setting}/seed_{seed}")
        print(f"Preview only: {len(plan)} result entries. --run requires frozen policy and all main validation results.")
        return []
    entries = _main_entries(config, base, plan)
    rows, test_sha256, manifest_sha256 = _official_test(config, base)
    completed = {}
    for entry in entries:
        result = _report_entry(entry, rows, config, base, test_sha256, manifest_sha256, completed)
        completed[(entry["setting"], entry["seed"], entry["condition"])] = result
    summary = [{key: result[key] for key in (
        "setting", "condition", "seed", "macro_f1", "accuracy", "rare_class_f1", "rare_class_recall",
        "rare_macro_f1", "rare_macro_recall",
        "test_rows", "real_count", "synthetic_count", "training_rows", "additional_draws", "best_epoch", "epochs_run",
        "inference_reused", "additional_inference_seconds",
    )} for result in completed.values()]
    write_csv(base / "05_results/metrics/baseline_test_results.csv", summary)
    write_csv(PROJECT_ROOT / "results/metrics" / config["protocol_id"] / "baseline_test_results.csv", summary)
    return summary
