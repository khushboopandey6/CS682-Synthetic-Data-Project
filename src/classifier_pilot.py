"""Run and summarize the master plan's shared classifier pilot."""

import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .project_config import PROJECT_ROOT, file_sha256, load_config, resolve_base_path, write_json


def pilot_plan(config):
    pilot = config["classifier_pilot"]
    return [(float(rate), setting, condition, int(pilot["development_seed"]))
            for rate in pilot["learning_rate_candidates"]
            for setting in pilot["settings"] for condition in pilot["conditions"]]


def select_learning_rate(records, config):
    expected = set(pilot_plan(config))
    indexed = {}
    for row in records:
        key = (float(row["learning_rate"]), row["setting"], row["condition"], int(row["seed"]))
        if key in indexed or key not in expected:
            raise ValueError(f"Unexpected or duplicate pilot result: {key}")
        score = float(row["macro_f1"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid pilot macro-F1: {score}")
        indexed[key] = score
    if set(indexed) != expected:
        raise ValueError(f"Pilot is incomplete: {len(indexed)}/{len(expected)} fits")
    summaries = []
    for rate in sorted({key[0] for key in expected}):
        scores = [score for key, score in indexed.items() if key[0] == rate]
        summaries.append({"learning_rate": rate, "mean_macro_f1": sum(scores) / len(scores),
                          "completed_fits": len(scores), "minimum_macro_f1": min(scores)})
    winner = min(summaries, key=lambda row: (-row["mean_macro_f1"], row["learning_rate"]))
    return winner, summaries


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def run_classifier_pilot(config_path=None, base_path=None, execute=False, freeze=False):
    config_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "experiment.yaml"
    config = load_config(config_path)
    base_path = resolve_base_path(config, base_path)
    plan = pilot_plan(config)
    if not config["training"].get("model_revision"):
        raise ValueError("Pin the pretrained model revision before the pilot")
    if not execute and not freeze:
        for rate, setting, condition, seed in plan:
            print(f"{setting}/seed_{seed}/{condition}: learning_rate={rate:g}")
        print(f"Preview only: {len(plan)} fits. Use --run to execute the pilot.")
        return {"status": "preview", "planned_fits": len(plan)}

    data_root = base_path / "01_data"
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Prepare and verify the official data release before the pilot")
    report_root = base_path / "05_results" / "classifier_pilot"
    report_root.mkdir(parents=True, exist_ok=True)
    records = []
    from .training import train_classifier

    for rate, setting, condition, seed in plan:
        data_path = (data_root / setting / f"seed_{seed}.jsonl" if condition == "B1" else
                     data_root / "baselines" / condition / setting / f"seed_{seed}.jsonl")
        if not data_path.is_file():
            raise ValueError(f"Missing prepared pilot dataset: {data_path}")
        run_path = (base_path / "04_checkpoints" / "classifier_pilot" / f"lr_{rate:.0e}"
                    / condition / setting / f"seed_{seed}")
        if execute:
            result = train_classifier(data_path, data_root / "validation.jsonl", condition, setting,
                                      seed, config, output_dir=run_path, base_path=base_path,
                                      pilot_learning_rate=rate)
        else:
            result_path = run_path / "results.json"
            if not result_path.is_file():
                raise ValueError(f"Missing completed pilot fit: {run_path}")
            result = train_classifier(data_path, data_root / "validation.jsonl", condition, setting,
                                      seed, config, output_dir=run_path, base_path=base_path,
                                      pilot_learning_rate=rate)
        if result.get("status") != "completed":
            raise ValueError(f"Pilot fit has not completed: {run_path}")
        row = {"learning_rate": rate, "setting": setting, "condition": condition, "seed": seed,
               "macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
               "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
               "real_rows": result.get("real_rows", result.get("real_count")),
               "train_rows": result.get("train_rows", result.get("training_rows")),
               "optimizer_steps": result.get("optimizer_steps"),
               "total_seconds": result.get("total_seconds", result.get("training_seconds")),
               "device": result["device"], "run_path": run_path.relative_to(base_path).as_posix(),
               "training_data_sha256": file_sha256(data_path)}
        records.append(row)
        write_csv(report_root / "validation_trials.csv", records)

    winner, summaries = select_learning_rate(records, config)
    if config.get("status") == "frozen":
        previous = config["training"].get("learning_rate")
        if previous is None or float(previous) != float(winner["learning_rate"]):
            raise ValueError("The frozen learning rate differs; review a new protocol version")
    write_csv(report_root / "learning_rates.csv", summaries)
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    selection = {**winner, "planned_fits": len(plan), "completed_fits": len(records),
                 "selection_metric": "equal-weight mean best validation macro-F1 across B1/B3 and both settings",
                 "tie_policy": "lower learning rate on exact tie", "official_test_used": False,
                 "data_manifest_sha256": file_sha256(manifest_path),
                 "pilot_trials_sha256": file_sha256(report_root / "validation_trials.csv"),
                 "model_revision": config["training"]["model_revision"], "code_commit": code_commit}
    write_json(report_root / "selection.json", selection)
    git_reports = PROJECT_ROOT / "results" / "metrics" / config["protocol_id"]
    write_csv(git_reports / "classifier_pilot_trials.csv", records)
    write_csv(git_reports / "classifier_learning_rates.csv", summaries)
    write_json(git_reports / "classifier_selection.json", selection)

    if freeze:
        import yaml

        selected = float(winner["learning_rate"])
        if config.get("status") != "frozen":
            config["training"]["learning_rate"] = selected
            config["status"] = "frozen"
            config["classifier_pilot"]["selection"] = selection
            config["classifier_pilot"]["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
            temporary = config_path.with_name(config_path.name + ".tmp")
            temporary.write_text(yaml.safe_dump(config, sort_keys=False))
            temporary.replace(config_path)
        print(f"Frozen shared learning rate: {selected:g}")
    print(json.dumps(selection, indent=2))
    return selection
