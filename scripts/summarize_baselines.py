"""Summarize verified baseline scores and completed-run costs across three seeds."""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier_pilot import write_csv
from src.project_config import PROJECT_ROOT, file_sha256, load_config, resolve_base_path, write_json


def summarize():
    config = load_config()
    base = resolve_base_path(config)
    report_root = PROJECT_ROOT / "results/metrics" / config["protocol_id"]
    expected = {(setting, condition, seed) for setting in config["settings"]
                for condition in ("B1", "B2", "B3") for seed in config["seeds"]}
    summaries, timings, inputs, split_rows = [], [], {}, {}
    for split in ("validation", "test"):
        audit_path = report_root / f"baseline_{split}_audit.json"
        audit = json.loads(audit_path.read_text())
        assert audit["status"] == "complete" and audit["verified_entries"] == 18
        if split == "validation":
            assert audit["frozen_config_sha256"] == file_sha256(PROJECT_ROOT / "configs/experiment.yaml")
        path = report_root / f"baseline_{split}_results.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(expected)
        keys = {(row["setting"], row["condition"], int(row["seed"])) for row in rows}
        assert keys == expected
        audited = {(row["setting"], row["condition"], int(row["seed"])): row for row in audit["entries"]}
        for row in rows:
            evidence = audited[(row["setting"], row["condition"], int(row["seed"]))]
            for metric in ("macro_f1", "accuracy", "rare_class_f1", "rare_class_recall"):
                key = metric.replace("rare_class_", "rare_macro_") if metric not in evidence else metric
                # Validation audits contain primary scores; rare scores are bound to the verified run JSON.
                if key in evidence:
                    assert math.isclose(float(row[metric]), evidence[key], rel_tol=0, abs_tol=1e-12)
        split_rows[split] = rows
        inputs[path.name] = file_sha256(path)
        inputs[audit_path.name] = file_sha256(audit_path)
        for setting in config["settings"]:
            for condition in ("B1", "B2", "B3"):
                group = [row for row in rows if row["setting"] == setting and row["condition"] == condition]
                assert sorted(int(row["seed"]) for row in group) == sorted(config["seeds"])
                aggregate = {"split": split, "setting": setting, "condition": condition,
                             "n": len(group), "seeds": ",".join(str(seed) for seed in config["seeds"])}
                for metric in ("macro_f1", "accuracy", "rare_class_f1", "rare_class_recall"):
                    values = [float(row[metric]) for row in group]
                    aggregate[f"{metric}_mean"] = statistics.mean(values)
                    aggregate[f"{metric}_std"] = statistics.stdev(values)
                summaries.append(aggregate)

    for setting in config["settings"]:
        for condition in ("B1", "B2", "B3"):
            runs = [json.loads((base / "05_results/metrics" / condition / setting / f"seed_{seed}.json").read_text())
                    for seed in config["seeds"]]
            for seed in config["seeds"]:
                path = base / "05_results/metrics" / condition / setting / f"seed_{seed}.json"
                inputs[path.relative_to(base).as_posix()] = file_sha256(path)
            for run in runs:
                row = next(row for row in split_rows["validation"] if
                           (row["setting"], row["condition"], int(row["seed"])) ==
                           (setting, condition, run["seed"]))
                for metric, key in (("macro_f1", "macro_f1"), ("accuracy", "accuracy"),
                                    ("rare_class_f1", "rare_macro_f1"), ("rare_class_recall", "rare_macro_recall")):
                    assert math.isclose(float(row[metric]), run[key], abs_tol=1e-12, rel_tol=0)
            timing = {"setting": setting, "condition": condition, "n": len(runs),
                      "independent_fits": sum(not run["training_reused"] for run in runs),
                      "reused_entries": sum(run["training_reused"] for run in runs),
                      "training_devices": ",".join(f"{device}:{sum(r['device'] == device for r in runs)}"
                                                   for device in sorted({r["device"] for r in runs}))}
            measures = {
                "optimization_seconds": [0.0 if r["training_reused"] else r["training_seconds"] for r in runs],
                "validation_seconds": [0.0 if r["training_reused"] else r["validation_seconds"] for r in runs],
                "fit_elapsed_seconds": [r["additional_training_seconds"] for r in runs],
                "test_inference_seconds": [float(row["additional_inference_seconds"]) for row in split_rows["test"]
                                           if row["setting"] == setting and row["condition"] == condition],
                "epochs_run": [r["epochs_run"] for r in runs],
                "best_epoch": [r["best_epoch"] for r in runs],
            }
            for name, values in measures.items():
                timing.update({f"{name}_mean": statistics.mean(values), f"{name}_std": statistics.stdev(values),
                               f"{name}_total": sum(values)})
            timings.append(timing)
    assert sum(row["independent_fits"] for row in timings) == 15
    assert sum(row["reused_entries"] for row in timings) == 3

    def formatted(row, metric, scale=1):
        return f"{row[metric + '_mean'] / scale:.4f} ± {row[metric + '_std'] / scale:.4f}"

    lines = ["# Baseline results summary", "",
             "Means and sample standard deviations (n−1 denominator) across training seeds 42, 123 and 456. "
             "All seeds use the same fixed validation and test splits. These are seed-to-seed standard deviations, not confidence intervals.", ""]
    for split in ("test", "validation"):
        lines += [f"## {split.title()} scores", "", "| Setting | Baseline | Macro-F1 | Accuracy | Rare-class F1 | Rare-class recall |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in summaries:
            if row["split"] == split:
                lines.append(f"| {row['setting']} | {row['condition']} | " + " | ".join(
                    formatted(row, key) for key in ("macro_f1", "accuracy", "rare_class_f1", "rare_class_recall")) + " |")
        lines.append("")
    lines += ["## Completed-run timing", "", "Times are mean ± sample SD per seed. Optimization excludes validation; "
              "fit elapsed time includes setup, optimization, validation and checkpoint work.", "",
              "| Setting | Baseline | Optimization minutes | Fit elapsed minutes | Test inference seconds | Independent fits | Training devices |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in timings:
        lines.append(f"| {row['setting']} | {row['condition']} | {formatted(row, 'optimization_seconds', 60)} | "
                     f"{formatted(row, 'fit_elapsed_seconds', 60)} | {formatted(row, 'test_inference_seconds')} | "
                     f"{row['independent_fits']} | {row['training_devices']} |")
    lines += ["", "Balanced B2 is identical to B1 and reuses its scores and models. Its zero incremental time means "
              "B1's cost is counted once. This is not an independently trained B2 cohort.", "",
              f"Total completed main-fit elapsed time: {sum(r['fit_elapsed_seconds_total'] for r in timings)/60:.2f} minutes. "
              f"Total test inference elapsed time: {sum(r['test_inference_seconds_total'] for r in timings):.2f} seconds.", "",
              "Pilot experiments, interrupted attempts, EDA preparation, generation and filtering are excluded from these completed-main-fit totals. "
              "They must be accounted for separately in the full project cost analysis. Timing is elapsed time, not accelerator utilization.", "",
              "Low-resource seed 42 used CPU for B1/B3; remaining main fits used MPS. All test inference used MPS. "
              "Hardware and training-sample variation accompany seed variation; this table is descriptive and does not establish statistical significance.", ""]
    metadata = {"status": "complete", "seeds": config["seeds"], "standard_deviation": "sample (ddof=1)",
                "score_groups": len(summaries), "independent_main_fits": 15, "balanced_B2_reuses": 3,
                "inputs_sha256": inputs, "frozen_config_sha256": file_sha256(PROJECT_ROOT / "configs/experiment.yaml")}
    for folder in (report_root, base / "05_results/metrics"):
        write_csv(folder / "baseline_score_summary.csv", summaries)
        write_csv(folder / "baseline_timing_summary.csv", timings)
        write_json(folder / "baseline_summary_manifest.json", metadata)
        (folder / "baseline_summary.md").write_text("\n".join(lines))
    print(report_root / "baseline_summary.md")
    for row in summaries:
        if row["split"] == "test":
            print(row["setting"], row["condition"], formatted(row, "macro_f1"))


if __name__ == "__main__":
    summarize()
