#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname -- "$script_dir")"
python_bin="${CS682_PYTHON:-/Users/dundimanikanta/sem2/IR_646/Assignment1/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Set CS682_PYTHON to the Python environment with the project dependencies."
    exit 1
fi
export PYTHONPATH="/private/tmp/cs682-baseline-packages${PYTHONPATH:+:$PYTHONPATH}"
log_dir="$project_root/artifacts/CS682_Project/05_results"
mkdir -p "$log_dir"
/usr/bin/caffeinate -i "$python_bin" -u "$script_dir/resume_baselines_mps.py" "$@" \
    2>&1 | tee -a "$log_dir/main_baselines_console.log"
