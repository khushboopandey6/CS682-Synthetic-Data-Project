"""Index local baseline checkpoints without adding model weights to Git."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(project_root, experiment):
    run_root = project_root / "results" / "baselines" / experiment
    records = []
    for result_path in sorted(run_root.glob("*/B*/seed_*/results.json")):
        result = json.loads(result_path.read_text())
        if result["status"] != "completed":
            continue
        checkpoint = result_path.parent / "best_model"
        weights = checkpoint / "model.safetensors"
        for required in (weights, checkpoint / "config.json", checkpoint / "tokenizer_config.json"):
            if not required.is_file():
                raise FileNotFoundError(f"Completed run is missing {required}")
        records.append({
            "setting": result["setting"],
            "baseline": result["condition"],
            "seed": result["seed"],
            "checkpoint_path": checkpoint.relative_to(project_root).as_posix(),
            "weights_path": weights.relative_to(project_root).as_posix(),
            "run_config_path": (result_path.parent / "run_config.json").relative_to(project_root).as_posix(),
            "weights_bytes": weights.stat().st_size,
            "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()),
            "weights_sha256": sha256(weights),
            "validation_macro_f1": result["macro_f1"],
            "storage": "local_only",
        })
    if not records:
        raise ValueError(f"No completed baseline checkpoints found in {run_root}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="distilbert_validated")
    args = parser.parse_args()
    if args.experiment in {".", ".."} or Path(args.experiment).name != args.experiment:
        parser.error("Use an experiment folder name, without a path.")
    project_root = Path(__file__).resolve().parents[1]
    records = build_index(project_root, args.experiment)
    output = project_root / "results" / "metrics" / "checkpoint_index.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output)
    total = sum(record["checkpoint_bytes"] for record in records)
    print(f"Indexed {len(records)} local checkpoints ({total / 1_000_000_000:.2f} GB).")
    print(f"Saved {output.relative_to(project_root)}; model files remain excluded from Git.")


if __name__ == "__main__":
    main()
