"""Bootstrap the pinned public source explicitly on a fresh Colab runtime."""

import tempfile
from pathlib import Path

from .data_preparation import _load_official_source, _source_dataset_digest
from .project_config import write_json


def ensure_source_cache(config, source_root, allow_download=False):
    source_root = Path(source_root)
    target = source_root / "raw" / "banking77"
    if target.exists():
        _load_official_source(config, source_root)
        return target
    if not allow_download:
        raise ValueError("The pinned source cache is absent. Enable its explicit download on a fresh runtime.")
    from datasets import DatasetDict, load_dataset

    dataset = config["dataset"]
    raw = load_dataset(dataset["dataset_id"], revision=dataset["dataset_revision"], token=False,
                       cache_dir=str(source_root / "_cache" / "datasets"))
    prepared = DatasetDict({
        split: data.add_column("source_row", list(range(len(data)))).add_column(
            "source_id", [f"{split}:{index}" for index in range(len(data))])
        for split, data in raw.items()
    })
    metadata = {"dataset_id": dataset["dataset_id"], "revision": dataset["dataset_revision"],
                "class_names": prepared["train"].features["label"].names,
                "rows": {split: len(data) for split, data in prepared.items()},
                "sha256": {split: _source_dataset_digest(data) for split, data in prepared.items()}}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source-", dir=target.parent) as staging:
        staged = Path(staging) / "banking77"
        prepared.save_to_disk(str(staged))
        if target.exists():
            raise ValueError("Another source-cache writer appeared; verify it before retrying")
        staged.rename(target)
        write_json(target.parent / "source.json", metadata)
    _load_official_source(config, source_root)
    return target
