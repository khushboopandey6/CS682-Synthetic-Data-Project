"""Execute Module A's baseline matrix with explicit balanced-B2 reuse."""

import json
import shutil
from pathlib import Path

from .classifier_pilot import write_csv
from .project_config import PROJECT_ROOT, file_sha256, load_config, read_jsonl, resolve_base_path, rows_digest, write_json, write_jsonl


def _copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"Existing exported artifact differs: {destination}")
    else:
        shutil.copy2(source, destination)


def run_baselines(config_path=None, base_path=None, execute=False, settings=None, seeds=None):
    config = load_config(config_path)
    base = resolve_base_path(config, base_path)
    settings = settings or list(config["settings"])
    seeds = seeds or config["seeds"]
    if not set(settings) <= set(config["settings"]) or not set(seeds) <= set(config["seeds"]):
        raise ValueError("Requested baseline settings or seeds are outside the shared protocol")
    plan = [(setting, int(seed), condition) for setting in settings for seed in seeds
            for condition in ("B1", "B2", "B3")]
    if not execute:
        for setting, seed, condition in plan:
            suffix = " (reuse B1 when input bytes match)" if setting == "low_resource" and condition == "B2" else ""
            print(f"{condition}/{setting}/seed_{seed}{suffix}")
        print(f"Preview: {len(plan)} result entries; main training requires the frozen pilot configuration.")
        return []
    if config.get("status") != "frozen" or not config["training"].get("learning_rate"):
        raise ValueError("Complete and freeze the shared classifier pilot before main baselines")
    from .training import train_classifier
    policy = {key: config[key] for key in ("protocol_id", "dataset", "training", "seeds", "settings", "baselines")}
    policy["label_mapping_sha256"] = file_sha256(base / "01_data/label_mapping.json")
    policy["rare_classes_sha256"] = file_sha256(base / "01_data/rare_classes.json")
    shared_policy_sha256 = rows_digest([policy])

    for setting, seed, condition in plan:
        real_path = base / "01_data" / setting / f"seed_{seed}.jsonl"
        training_path = real_path if condition == "B1" else base / "01_data" / "baselines" / condition / setting / f"seed_{seed}.jsonl"
        if condition == "B2" and setting == "low_resource" and training_path.read_bytes() == real_path.read_bytes():
            b1_folder = base / "04_checkpoints" / "B1" / setting / f"seed_{seed}"
            b1 = train_classifier(real_path, base / "01_data/validation.jsonl", "B1", setting, seed, config, base_path=base)
            result = {**b1, "condition": "B2", "training_reused": True,
                      "reused_from": b1_folder.relative_to(base).as_posix(),
                      "source_training_seconds": b1["training_seconds"],
                      "source_total_seconds": b1["total_seconds"],
                      "training_seconds": 0.0, "total_seconds": 0.0, "additional_training_seconds": 0.0,
                      "accounting_note": "Identical balanced inputs reuse B1. Source training is counted once in project totals."}
            folder = b1_folder
        else:
            result = train_classifier(training_path, base / "01_data/validation.jsonl", condition,
                                      setting, seed, config, base_path=base)
            result = {**result, "training_reused": False, "additional_training_seconds": result["total_seconds"]}
            folder = base / "04_checkpoints" / condition / setting / f"seed_{seed}"
        metrics_path = base / "05_results" / "metrics" / condition / setting / f"seed_{seed}.json"
        result["shared_training_policy_sha256"] = shared_policy_sha256
        result["config_provenance_sha256"] = rows_digest([config])
        write_json(metrics_path, result)
        prediction_path = base / "05_results" / "validation_predictions" / condition / setting / f"seed_{seed}.jsonl"
        if result["training_reused"]:
            predictions = [{**row, "condition": "B2", "reused_from_condition": "B1"}
                           for row in read_jsonl(folder / "validation_predictions.jsonl")]
            if prediction_path.exists() and read_jsonl(prediction_path) != predictions:
                raise ValueError(f"Existing B2 identity predictions differ: {prediction_path}")
            if not prediction_path.exists():
                write_jsonl(prediction_path, predictions)
        else:
            _copy_verified(folder / "validation_predictions.jsonl", prediction_path)
        if not result["training_reused"]:
            _copy_verified(folder / "stage_log.csv",
                           base / "06_cost_logs" / "training" / condition / setting / f"seed_{seed}.csv")

    rows = []
    for path in sorted((base / "05_results/metrics").glob("B*/*/seed_*.json")):
        result = json.loads(path.read_text())
        if result.get("shared_training_policy_sha256") != shared_policy_sha256:
            raise ValueError(f"Cannot aggregate results from a different shared configuration: {path}")
        rows.append({
            "setting": result["setting"], "condition": result["condition"], "seed": result["seed"],
            "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
            "rare_class_f1": result.get("rare_class_f1", result.get("rare_macro_f1")),
            "rare_class_recall": result.get("rare_class_recall", result.get("rare_macro_recall")),
            "real_count": result["real_count"], "synthetic_count": result["synthetic_count"],
            "training_rows": result["training_rows"], "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"], "optimizer_steps": result["optimizer_steps"],
            "training_reused": result["training_reused"],
            "additional_training_seconds": result["additional_training_seconds"], "device": result["device"],
        })
    if rows:
        write_csv(base / "05_results/metrics/baseline_validation_results.csv", rows)
        write_csv(PROJECT_ROOT / "results/metrics" / config["protocol_id"] / "baseline_validation_results.csv", rows)
    print(f"Exported {len(rows)} baseline result entries, including explicit B2 identities.")
    return rows
