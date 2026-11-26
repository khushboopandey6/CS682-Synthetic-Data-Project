"""Resume compatible main baselines on MPS; preserve interrupted CPU artifacts."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/private/tmp/cs682-baseline-packages")
project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
os.chdir(project)

from src.baseline_runs import run_baselines
from src.project_config import load_config, write_json
from src.training import select_device

if "--preview" in sys.argv:
    run_baselines(config_path=project / "artifacts/CS682_Project/handoff/frozen_experiment.yaml", execute=False)
    raise SystemExit(0)

device = select_device()
if device.type != "mps":
    raise RuntimeError(f"MPS GPU required; selected {device}. CPU training will not start.")
config_path = project / "artifacts/CS682_Project/handoff/frozen_experiment.yaml"
config = load_config(config_path)
assert (config["training"]["learning_rate"], config["training"]["max_epochs"],
        config["training"]["early_stopping_patience"]) == (1e-4, 30, 5)
base = project / config["paths"]["base_path"]
partial = base / "04_checkpoints/B1/low_resource/seed_123"
archive = base / "04_checkpoints/main_epoch30_interrupted_cpu/B1/low_resource/seed_123"
if partial.exists() and not (partial / "results.json").exists():
    runtime = json.loads((partial / "runtime.json").read_text())
    if runtime["device"] == "cpu":
        if archive.exists():
            raise RuntimeError("Interrupted-run archive already exists; refusing to overwrite it.")
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial.rename(archive)
        write_json(base / "handoff/interrupted_main_cpu_archive.json", {
            "reason": "Resume remaining main fits on MPS; interrupted fit restarts from seed.",
            "original_path": partial.relative_to(base).as_posix(),
            "archive_path": archive.relative_to(base).as_posix(),
        })
        print("Preserved interrupted CPU seed-123 run in its archive.", flush=True)
print(f"Resuming main matrix on {device}; completed matching fits will be reused.", flush=True)
run_baselines(config_path=config_path, execute=True)

from scripts.export_baseline_handoff import export_baseline_handoff
export_baseline_handoff(config_path=config_path)
print("Main baselines complete. Validation summary: results/metrics/master_v1/baseline_validation_results.csv", flush=True)
