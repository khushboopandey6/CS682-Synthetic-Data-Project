"""Train and report F0/F1/F2 classifiers on matched synthetic data (Module D).

This module only *calls* the shared trainer in ``src/training.py`` and the
reporting helpers in ``src/inference.py``. It never modifies them, so the
trainer checksum recorded by the baseline checkpoints stays valid.

Expected inputs under the shared base path (the Drive ``CS682_Project`` folder):

    01_data/manifest.json                            released split manifest
    01_data/validation.jsonl                         fixed validation split
    01_data/{setting}/seed_{seed}.jsonl              released real subset
    03_filtered_data/matched/{F}/{setting}/seed_{seed}.jsonl
                                                     matched synthetic rows (Module C)

Outputs mirror the baseline runner:

    04_checkpoints/{F}/{setting}/seed_{seed}/        written by the shared trainer
    05_results/metrics/{F}/{setting}/seed_{seed}.json
    05_results/validation_predictions/{F}/{setting}/seed_{seed}.jsonl
    06_cost_logs/training/{F}/{setting}/seed_{seed}.csv
    05_results/metrics/synthetic_validation_results.csv
    and, after ``evaluate_synthetic_test``, the same test artifacts as the baselines.
"""

import json
from collections import Counter
from pathlib import Path

from . import inference, training
from .baseline_runs import _copy_verified
from .classifier_pilot import write_csv
from .project_config import (
    PROJECT_ROOT, file_sha256, load_config, read_jsonl, require,
    resolve_base_path, rows_digest, write_json,
)

SYNTHETIC_CONDITIONS = ("F0", "F1", "F2")
NON_GENERATED_SOURCES = {None, "", "real", "oversampled", "eda"}
REQUIRED_FIELDS = ("id", "text", "label", "source", "setting", "seed")


def matched_path(base, condition, setting, seed):
    return base / "03_filtered_data" / "matched" / condition / setting / f"seed_{seed}.jsonl"


def _plan(config, conditions, settings, seeds):
    conditions = list(conditions or SYNTHETIC_CONDITIONS)
    settings = list(settings or config["settings"])
    seeds = [int(seed) for seed in (seeds or config["seeds"])]
    require(set(conditions) <= set(SYNTHETIC_CONDITIONS), "Synthetic runs accept only F0, F1 and F2.")
    require(set(settings) <= set(config["settings"]), "Requested settings are outside the shared protocol.")
    require(set(seeds) <= {int(seed) for seed in config["seeds"]}, "Requested seeds are outside the shared protocol.")
    return [(setting, seed, condition) for setting in settings for seed in seeds for condition in conditions]


def _released_real(config, base, setting, seed):
    """Load the released real subset and validation split, verified against the frozen manifest."""
    manifest_path = base / "01_data/manifest.json"
    selection = config.get("classifier_pilot", {}).get("selection", {})
    require(selection.get("data_manifest_sha256") == file_sha256(manifest_path),
            "01_data/manifest.json differs from the frozen pilot release; use the official 01_data folder.")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("protocol_id") == config["protocol_id"], "Released data protocol differs.")
    relative = f"{setting}/seed_{seed}.jsonl"
    real = inference._released_rows(base / "01_data" / relative, manifest["datasets"][relative])
    validation = inference._released_rows(base / "01_data/validation.jsonl", manifest["datasets"]["validation.jsonl"])
    return real, validation


def _target_counts(real_rows, ratio):
    return {label: int(round(count * ratio)) for label, count in Counter(row["label"] for row in real_rows).items()}


def load_matched(base, condition, setting, seed, real_rows, validation_rows, num_classes, ratio=1.0):
    """Read one matched synthetic file and enforce the shared data contract (Section 15.4 and 11)."""
    path = matched_path(base, condition, setting, seed)
    require(path.is_file(), f"Missing matched synthetic data from Module C: {path}")
    rows = read_jsonl(path)
    reserved = {row["id"] for row in real_rows} | {row["id"] for row in validation_rows}
    seen = set()
    for row in rows:
        require(all(field in row for field in REQUIRED_FIELDS), f"Synthetic row lacks {REQUIRED_FIELDS}: {path}")
        require(isinstance(row["id"], str) and row["id"].strip(), f"Synthetic IDs must be nonempty strings: {path}")
        require(row["id"] not in seen, f"Repeated synthetic ID {row['id']}: {path}")
        require(row["id"] not in reserved, f"Synthetic ID collides with a real or validation ID: {row['id']}")
        seen.add(row["id"])
        require(row["source"] not in NON_GENERATED_SOURCES, f"Matched rows must be generated, not {row['source']!r}: {path}")
        require((row["setting"], int(row["seed"])) == (setting, seed),
                f"Synthetic row {row['id']} belongs to {row['setting']}/seed_{row['seed']}, not {setting}/seed_{seed}.")
        require(isinstance(row["label"], int) and not isinstance(row["label"], bool) and 0 <= row["label"] < num_classes,
                f"Invalid synthetic label in {row['id']}.")
        require(isinstance(row["text"], str) and row["text"].strip(), f"Blank synthetic text in {row['id']}.")
    counts = Counter(row["label"] for row in rows)
    targets = _target_counts(real_rows, ratio)
    over = {label: count for label, count in counts.items() if count > targets.get(label, 0)}
    require(not over, f"Matched counts exceed the {ratio}x augmentation target for classes {sorted(over)}: {path}")
    return rows, {str(label): counts.get(label, 0) for label in range(num_classes)}, file_sha256(path)


def check_matched_counts(base, setting, seed, real_rows, validation_rows, num_classes, ratio=1.0):
    """All three matched files must exist and agree per class before any F run (Section 11)."""
    per_condition = {}
    for condition in SYNTHETIC_CONDITIONS:
        _, counts, _ = load_matched(base, condition, setting, seed, real_rows, validation_rows, num_classes, ratio)
        per_condition[condition] = counts
    reference = per_condition["F0"]
    for condition, counts in per_condition.items():
        differing = [label for label in reference if counts[label] != reference[label]]
        require(not differing, f"{condition} per-class counts differ from F0 for {setting}/seed_{seed} "
                               f"(classes {differing[:10]}). Matched datasets must have equal counts per class.")
    return reference


def run_synthetic(config_path=None, base_path=None, execute=False, conditions=None, settings=None, seeds=None):
    """Preview, or with ``execute=True`` train, the requested F0/F1/F2 runs with the shared trainer."""
    config = load_config(config_path)
    base = resolve_base_path(config, base_path)
    plan = _plan(config, conditions, settings, seeds)
    if not execute:
        for setting, seed, condition in plan:
            print(f"{condition}/{setting}/seed_{seed}  <-  {matched_path(base, condition, setting, seed)}")
        print(f"Preview: {len(plan)} synthetic runs. Pass execute=True (or --run) to train.")
        return []
    require(config.get("status") == "frozen" and config["training"].get("learning_rate"),
            "Freeze the shared classifier policy before synthetic runs.")
    num_classes = config["dataset"]["num_classes"]
    ratio = float(config.get("augmentation", {}).get("primary_ratio", 1.0))
    policy_sha256 = inference._shared_policy(config, base)
    results = []
    for setting, seed, condition in plan:
        real, validation = _released_real(config, base, setting, seed)
        matched_counts = check_matched_counts(base, setting, seed, real, validation, num_classes, ratio)
        synthetic, _, synthetic_sha256 = load_matched(base, condition, setting, seed, real, validation, num_classes, ratio)
        result = training.train_classifier(real + synthetic, validation, condition, setting, seed, config, base_path=base)
        targets = _target_counts(real, ratio)
        result = {**result, "training_reused": False, "additional_training_seconds": result["total_seconds"],
                  "shared_training_policy_sha256": policy_sha256,
                  "config_provenance_sha256": rows_digest([config]),
                  "matched_data_path": matched_path(base, condition, setting, seed).relative_to(base).as_posix(),
                  "matched_data_sha256": synthetic_sha256,
                  "matched_counts_per_class": matched_counts,
                  "augmentation_target": ratio,
                  "actual_augmentation_ratio": len(synthetic) / len(real),
                  "classes_below_target": sum(1 for label, target in targets.items()
                                              if int(matched_counts[str(label)]) < target)}
        write_json(base / "05_results/metrics" / condition / setting / f"seed_{seed}.json", result)
        folder = base / "04_checkpoints" / condition / setting / f"seed_{seed}"
        _copy_verified(folder / "validation_predictions.jsonl",
                       base / "05_results/validation_predictions" / condition / setting / f"seed_{seed}.jsonl")
        _copy_verified(folder / "stage_log.csv", base / "06_cost_logs/training" / condition / setting / f"seed_{seed}.csv")
        results.append(result)
    summarize_synthetic(config, base)
    return results


def summarize_synthetic(config, base):
    """Collect every completed F validation result into one CSV for Module E."""
    policy_sha256 = inference._shared_policy(config, base)
    rows = []
    for path in sorted((base / "05_results/metrics").glob("F*/*/seed_*.json")):
        result = json.loads(path.read_text())
        require(result.get("shared_training_policy_sha256") == policy_sha256,
                f"Cannot aggregate results from a different shared configuration: {path}")
        rows.append({key: result.get(key) for key in (
            "setting", "condition", "seed", "macro_f1", "accuracy", "rare_macro_f1", "rare_macro_recall",
            "real_count", "synthetic_count", "training_rows", "actual_augmentation_ratio", "classes_below_target",
            "best_epoch", "epochs_run", "optimizer_steps", "additional_training_seconds", "device", "hardware",
        )})
    if rows:
        write_csv(base / "05_results/metrics/synthetic_validation_results.csv", rows)
        write_csv(PROJECT_ROOT / "results/metrics" / config["protocol_id"] / "synthetic_validation_results.csv", rows)
    return rows


def _synthetic_entry(config, base, setting, seed, condition, policy_sha256):
    """Verify that a checkpoint was trained on today's released real rows plus its matched file."""
    metrics_path = base / "05_results/metrics" / condition / setting / f"seed_{seed}.json"
    require(metrics_path.is_file(), f"Missing validation result; train before test reporting: {metrics_path}")
    result = json.loads(metrics_path.read_text())
    require(result.get("status") == "completed"
            and (result.get("setting"), result.get("seed"), result.get("condition")) == (setting, seed, condition),
            f"Incomplete or mislabelled synthetic result: {metrics_path}")
    require(result.get("shared_training_policy_sha256") == policy_sha256,
            f"Synthetic result does not match the frozen training policy: {metrics_path}")
    folder = base / "04_checkpoints" / condition / setting / f"seed_{seed}"
    run_path, source_path = folder / "run_config.json", folder / "results.json"
    require(run_path.is_file() and source_path.is_file(), f"Missing completed checkpoint: {folder}")
    run, source = json.loads(run_path.read_text()), json.loads(source_path.read_text())
    specification = run["specification"]
    require(rows_digest([specification]) == run["fingerprint"] == source.get("fingerprint") == result.get("fingerprint"),
            f"Checkpoint provenance differs: {folder}")
    require((specification.get("mode"), specification.get("condition"), specification.get("setting"),
             specification.get("seed"), specification.get("protocol_id"))
            == ("main", condition, setting, seed, config["protocol_id"]), f"Unexpected checkpoint identity: {folder}")
    require(specification["training"] == training._effective_training(config, condition, setting, seed, None)[0],
            f"Checkpoint training policy differs: {folder}")
    require(specification.get("trainer_source_sha256") == file_sha256(Path(training.__file__)),
            f"src/training.py changed after this checkpoint was trained: {folder}")
    real, validation = _released_real(config, base, setting, seed)
    num_classes = config["dataset"]["num_classes"]
    ratio = float(config.get("augmentation", {}).get("primary_ratio", 1.0))
    synthetic, _, synthetic_sha256 = load_matched(base, condition, setting, seed, real, validation, num_classes, ratio)
    require(synthetic_sha256 == result.get("matched_data_sha256"),
            f"Matched data changed after training: {matched_path(base, condition, setting, seed)}")
    require(specification.get("training_rows_sha256") == rows_digest(real + synthetic)
            and specification.get("validation_rows_sha256") == rows_digest(validation),
            f"Checkpoint was not trained on the released real subset plus its matched file: {folder}")
    best = folder / "best_model"
    required = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"}
    require(required <= {item.name for item in best.glob("*") if item.is_file()}, f"Incomplete checkpoint: {best}")
    files = sorted(item for item in best.rglob("*") if item.is_file()) + [run_path, source_path]
    return {"condition": condition, "setting": setting, "seed": seed, "reused": False, "folder": folder,
            "validation_result": result, "shared_training_policy_sha256": policy_sha256,
            "checkpoint_sha256": {item.relative_to(folder).as_posix(): file_sha256(item) for item in files}}


def evaluate_synthetic_test(config_path=None, base_path=None, execute=False, conditions=None, settings=None, seeds=None):
    """Report-only test evaluation for completed F runs; test scores never select anything."""
    config = load_config(config_path)
    base = resolve_base_path(config, base_path)
    plan = _plan(config, conditions, settings, seeds)
    if not execute:
        for setting, seed, condition in plan:
            print(f"Test reporting: {condition}/{setting}/seed_{seed}")
        print(f"Preview only: {len(plan)} entries. Every requested validation result must exist before the test split opens.")
        return []
    require(config.get("status") == "frozen", "Freeze the classifier policy before test reporting.")
    policy_sha256 = inference._shared_policy(config, base)
    entries = [_synthetic_entry(config, base, setting, seed, condition, policy_sha256) for setting, seed, condition in plan]
    rows, test_sha256, manifest_sha256 = inference._official_test(config, base)
    completed = [inference._report_entry(entry, rows, config, base, test_sha256, manifest_sha256, {}) for entry in entries]
    summary = []
    for path in sorted((base / "05_results/metrics/test").glob("F*/*/seed_*.json")):
        result = json.loads(path.read_text())
        summary.append({key: result.get(key) for key in (
            "setting", "condition", "seed", "macro_f1", "accuracy", "rare_class_f1", "rare_class_recall",
            "test_rows", "real_count", "synthetic_count", "training_rows", "best_epoch", "epochs_run",
            "additional_inference_seconds",
        )})
    if summary:
        write_csv(base / "05_results/metrics/synthetic_test_results.csv", summary)
        write_csv(PROJECT_ROOT / "results/metrics" / config["protocol_id"] / "synthetic_test_results.csv", summary)
    return completed
