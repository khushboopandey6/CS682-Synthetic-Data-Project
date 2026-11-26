"""Preview or export reporting-only baseline test predictions after all main fits."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import evaluate_baselines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--base-path")
    parser.add_argument("--run", action="store_true", help="Report test results; never use them for model selection")
    args = parser.parse_args()
    evaluate_baselines(args.config, args.base_path, execute=args.run)
