"""Preview or run the shared 12-fit classifier pilot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier_pilot import run_classifier_pilot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--base-path")
    parser.add_argument("--run", action="store_true", help="Execute the 12 model fits; otherwise preview")
    parser.add_argument("--freeze", action="store_true", help="Freeze the winner after all 12 fits complete")
    args = parser.parse_args()
    run_classifier_pilot(args.config, args.base_path, execute=args.run, freeze=args.freeze)
