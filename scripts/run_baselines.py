"""Preview or execute the official B1/B2/B3 matrix after the classifier pilot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_runs import run_baselines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--base-path")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--settings", nargs="+", choices=("low_resource", "imbalanced"))
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    run_baselines(args.config, args.base_path, execute=args.run, settings=args.settings, seeds=args.seeds)
