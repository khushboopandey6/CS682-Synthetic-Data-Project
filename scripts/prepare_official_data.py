"""Export the shared master-protocol Banking77 data release from a local cache."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import prepare_official_data
from src.project_config import load_config, resolve_base_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Shared YAML configuration (default: configs/experiment.yaml)")
    parser.add_argument("--base-path", type=Path, default=None, help="Override the shared artifact base directory")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data/reproducible",
                        help="Local source root containing raw/banking77 and raw/source.json; offline cached fallback only")
    args = parser.parse_args()
    config = load_config(args.config)
    base_path = resolve_base_path(config, args.base_path)
    manifest = prepare_official_data(config, base_path, args.source_root)
    print(f"Official data release verified: {base_path / '01_data'}")
    print("Experiment seeds:", ", ".join(map(str, manifest["config"]["seeds"])))
    print("Frozen rare-class IDs:", ", ".join(map(str, manifest["rare_class_ids"])))
    for name, artifact in manifest["datasets"].items():
        print(f"  {name}: {artifact['rows']} rows; SHA256 {artifact['file_sha256']}")
    print("Existing cached sources and historical experiments were preserved.")


if __name__ == "__main__":
    main()
