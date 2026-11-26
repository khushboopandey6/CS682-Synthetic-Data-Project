"""Prepare Module A's official B2/B3 datasets from the released real subsets."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import prepare_baseline_data
from src.project_config import load_config, resolve_base_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--settings", nargs="+", choices=("low_resource", "imbalanced"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--conditions", nargs="+", choices=("B2", "B3"), default=("B2", "B3"))
    args = parser.parse_args()
    config = load_config(args.config)
    base_path = resolve_base_path(config, args.base_path)
    reports = prepare_baseline_data(config, base_path, args.settings, args.seeds, args.conditions)
    for report in reports:
        print(f"{report['condition']} {report['setting']} seed {report['seed']}: "
              f"{report['rows']} rows; B1-identical={report['b1_identical_inputs']}; {report['data_path']}")


if __name__ == "__main__":
    main()
