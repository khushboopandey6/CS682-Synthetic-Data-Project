"""Preview, train or test-report the F0/F1/F2 matched-synthetic runs with the shared trainer.

Examples (from the repository root, with CS682_BASE_PATH pointing at the shared Drive folder):
    python scripts/run_synthetic.py --conditions F0 F2 --settings low_resource --seeds 42          # preview
    python scripts/run_synthetic.py --conditions F0 F2 --settings low_resource --seeds 42 --run    # train
    python scripts/run_synthetic.py --conditions F0 F2 --settings low_resource --seeds 42 --test --run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synthetic_runs import evaluate_synthetic_test, run_synthetic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--base-path")
    parser.add_argument("--conditions", nargs="+", choices=("F0", "F1", "F2"))
    parser.add_argument("--settings", nargs="+", choices=("low_resource", "imbalanced"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--test", action="store_true", help="report-only test evaluation of completed runs")
    parser.add_argument("--run", action="store_true", help="execute instead of previewing")
    args = parser.parse_args()
    action = evaluate_synthetic_test if args.test else run_synthetic
    action(args.config, args.base_path, execute=args.run, conditions=args.conditions,
           settings=args.settings, seeds=args.seeds)
