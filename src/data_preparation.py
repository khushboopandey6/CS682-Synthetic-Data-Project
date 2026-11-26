"""Prepare the master's official Banking77 splits without changing source data."""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .project_config import (
    PROJECT_ROOT, file_sha256, read_jsonl, require, rows_digest, write_json,
    write_jsonl,
)


def _source_dataset_digest(data):
    """Verify the content checksum written by the existing source-cache notebook."""
    digest = hashlib.sha256()
    digest.update(json.dumps(data.features.to_dict(), sort_keys=True).encode("utf-8"))
    for row in data:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_official_source(config, source_root):
    from datasets import DownloadConfig, config as datasets_config, load_dataset, load_from_disk
    from huggingface_hub import constants as hub_constants

    specification = config["dataset"]
    dataset_id = specification["dataset_id"]
    revision = specification["dataset_revision"]
    require(dataset_id == "PolyAI/banking77", "Expected the official PolyAI/banking77 source.")
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
            "Use an immutable Banking77 commit revision.")
    raw_path = source_root / "raw" / "banking77"
    saved_source = None
    if raw_path.exists():
        source_path = source_root / "raw" / "source.json"
        require(source_path.is_file(), f"Missing provenance for cached source: {source_path}")
        saved_source = json.loads(source_path.read_text())
        require(saved_source["dataset_id"] == dataset_id and saved_source["revision"] == revision,
                "Cached raw dataset differs from the configured dataset ID/revision.")
        raw = load_from_disk(str(raw_path))
    else:
        # An existing Hugging Face cache is an offline fallback, never a download.
        previous = datasets_config.HF_DATASETS_OFFLINE, hub_constants.HF_HUB_OFFLINE
        datasets_config.HF_DATASETS_OFFLINE = hub_constants.HF_HUB_OFFLINE = True
        try:
            raw = load_dataset(
                dataset_id, revision=revision, cache_dir=str(source_root / "_cache" / "datasets"),
                token=False, download_config=DownloadConfig(local_files_only=True),
                download_mode="reuse_dataset_if_exists",
            )
        except Exception as error:
            raise ValueError(
                f"Pinned Banking77 source is unavailable locally at {raw_path}. "
                "Prepare the pinned source cache first; this command does not download data."
            ) from error
        finally:
            datasets_config.HF_DATASETS_OFFLINE, hub_constants.HF_HUB_OFFLINE = previous

    require(set(raw) == {"train", "test"}, "Expected the official train and test source splits.")
    names = raw["train"].features["label"].names
    require(names == raw["test"].features["label"].names, "Official train/test label mappings differ.")
    if saved_source is not None:
        require(saved_source["class_names"] == names, "Cached label mapping differs from its source manifest.")
    canonical = {}
    for split in ("train", "test"):
        data = raw[split]
        if saved_source is not None:
            require(len(data) == saved_source["rows"][split], f"Cached {split} source row count changed.")
            require(_source_dataset_digest(data) == saved_source["sha256"][split],
                    f"Cached {split} source content fails its provenance checksum.")
        rows = []
        for index, row in enumerate(data):
            identifier = f"{split}:{index}"
            require(row.get("source_id", identifier) == identifier,
                    f"Invalid cached source ID at {identifier}.")
            require(row.get("source_row", index) == index,
                    f"Invalid cached source row number at {identifier}.")
            rows.append({"id": identifier, "text": row["text"], "label": row["label"]})
        canonical[split] = rows
    return canonical["train"], canonical["test"], list(names)


def _validate_source(train, test, names, num_classes):
    require(num_classes == len(names) == len(set(names)) == 77, "Banking77 requires 77 distinct official class names.")
    require(len(train) == 10003 and len(test) == 3080, "Official Banking77 split sizes must be 10003/3080.")
    for split, rows in (("train", train), ("test", test)):
        require([row["id"] for row in rows] == [f"{split}:{i}" for i in range(len(rows))],
                f"{split} source IDs must preserve original official row indices.")
        require(all(isinstance(row["text"], str) and row["text"].strip() for row in rows),
                f"Invalid or empty customer query in {split}.")
        require(all(type(row["label"]) is int and 0 <= row["label"] < num_classes for row in rows),
                f"Invalid official integer label in {split}.")
        require(set(row["label"] for row in rows) == set(range(num_classes)),
                f"Missing intent classes in official {split}.")


def _group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    return groups


def sample_per_class(rows, counts, seed):
    """Match notebook 01: local RNG; classes in source encounter order."""
    rng = random.Random(seed)
    selected = []
    for label, population in _group_rows(rows).items():
        require(len(population) >= counts[label], f"Not enough training examples for label {label}.")
        selected.extend(rng.sample(population, counts[label]))
    return selected


def _dataset_statistics(rows):
    counts = Counter(row["label"] for row in rows)
    return {
        "rows": len(rows),
        "per_class": {str(label): count for label, count in sorted(counts.items())},
        "unique_ids": len({row["id"] for row in rows}),
    }


def _serialized_bytes(kind, value):
    # Match the shared write_json/write_jsonl helpers, including the final newline.
    if kind == "jsonl":
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value).encode("utf-8")
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _artifact_sha256(kind, value):
    return hashlib.sha256(_serialized_bytes(kind, value)).hexdigest()


def prepare_official_data(config, base_path, source_root=None):
    """Export verified JSONL contracts, refusing to overwrite conflicting artifacts.

    Only dataset/sampling settings enter the data manifest, so subsequently
    freezing a classifier pilot does not change an already released data split.
    The original cached source, historical datasets and checkpoints stay intact.
    """
    source_root = Path(source_root) if source_root is not None else PROJECT_ROOT / "data" / "reproducible"
    base_path = Path(base_path).expanduser().resolve()
    output = base_path / "01_data"
    require(config.get("schema_version") == 1 and bool(config.get("protocol_id")), "Invalid shared protocol configuration.")
    dataset_config = config["dataset"]
    settings = config["settings"]
    seeds = config["seeds"]
    require(isinstance(seeds, list) and len(seeds) == len(set(seeds)) == 3
            and all(type(seed) is int for seed in seeds), "Provide three distinct integer experiment seeds.")
    require(dataset_config["sampling_algorithm"] == "python_random_sample_source_class_encounter_order",
            "Unsupported validation/training sampling algorithm.")
    imbalance = settings["imbalanced"]
    require(imbalance["rare_class_selection_algorithm"] == "python_random_sample_sorted_label_ids",
            "Unsupported rare-class selection algorithm.")
    for value in (dataset_config["validation_per_class"], settings["low_resource"]["samples_per_class"],
                  imbalance["num_rare_classes"], imbalance["rare_samples_per_class"], imbalance["majority_samples_per_class"]):
        require(type(value) is int and value > 0, "Dataset counts must be positive integers.")
    require(type(dataset_config["validation_seed"]) is int and type(imbalance["rare_class_seed"]) is int,
            "Validation and rare-class selection seeds must be integers.")

    train, test, names = _load_official_source(config, source_root)
    _validate_source(train, test, names, dataset_config["num_classes"])
    labels = list(range(len(names)))
    require(imbalance["num_rare_classes"] < len(labels), "Rare classes must be a proper subset of all classes.")
    rare_ids = random.Random(imbalance["rare_class_seed"]).sample(sorted(labels), imbalance["num_rare_classes"])
    rare_set = set(rare_ids)
    validation = sample_per_class(train, {label: dataset_config["validation_per_class"] for label in labels},
                                  dataset_config["validation_seed"])
    validation_ids = {row["id"] for row in validation}
    remaining = [row for group in _group_rows(train).values() for row in group if row["id"] not in validation_ids]
    datasets = {"validation.jsonl": validation, "test.jsonl": test}
    for setting in ("low_resource", "imbalanced"):
        counts = {label: (settings["low_resource"]["samples_per_class"] if setting == "low_resource" else
                          imbalance["rare_samples_per_class"] if label in rare_set else
                          imbalance["majority_samples_per_class"]) for label in labels}
        for seed in seeds:
            rows = sample_per_class(remaining, counts, seed)
            require(not {row["id"] for row in rows} & validation_ids, "Training and validation source IDs overlap.")
            require(len({row["id"] for row in rows}) == len(rows), "Duplicate selected training source IDs.")
            require(Counter(row["label"] for row in rows) == Counter(counts), "Sampled class distribution differs.")
            datasets[f"{setting}/seed_{seed}.jsonl"] = rows

    label_mapping = {str(label): name for label, name in enumerate(names)}
    rare_classes = {
        "selection_seed": imbalance["rare_class_seed"],
        "selection_algorithm": imbalance["rare_class_selection_algorithm"],
        "rare_class_ids": rare_ids,
        "classes": [{"id": label, "name": names[label]} for label in rare_ids],
    }
    normalize = lambda text: " ".join(text.casefold().split())
    test_texts = {normalize(row["text"]) for row in test}
    statistics = {
        "official_train_rows": len(train), "official_test_rows": len(test),
        "remaining_training_pool_rows": len(remaining), "num_classes": len(names),
        "datasets": {name: _dataset_statistics(rows) for name, rows in datasets.items()},
        "integrity": {
            "status": "passed", "training_validation_source_overlap": 0,
            "train_test_source_overlap": 0,
            "original_train_test_shared_normalized_texts": len({normalize(row["text"]) for row in train} & test_texts),
            "validation_test_shared_normalized_texts": len({normalize(row["text"]) for row in validation} & test_texts),
            "note": "Original official train/test text overlap is reported, never removed or used to change sampling.",
        },
    }
    artifacts = {name: ("jsonl", rows) for name, rows in datasets.items()}
    artifacts.update({"label_mapping.json": ("json", label_mapping), "rare_classes.json": ("json", rare_classes),
                      "dataset_statistics.json": ("json", statistics)})
    manifest = {
        "schema_version": 1, "protocol_id": config["protocol_id"],
        "config": {"dataset": dataset_config, "seeds": list(seeds), "settings": settings},
        "source": {"dataset_id": dataset_config["dataset_id"], "revision": dataset_config["dataset_revision"],
                   "official_train_rows": len(train), "official_test_rows": len(test),
                   "train_rows_sha256": rows_digest(train), "test_rows_sha256": rows_digest(test)},
        "rare_class_ids": rare_ids,
        "datasets": {name: {**_dataset_statistics(rows), "ids": [row["id"] for row in rows],
                             "rows_sha256": rows_digest(rows), "file_sha256": _artifact_sha256("jsonl", rows)}
                     for name, rows in datasets.items()},
        "metadata_files": {name: _artifact_sha256(kind, value) for name, (kind, value) in artifacts.items()
                           if kind == "json"},
    }
    artifacts["manifest.json"] = ("json", manifest)
    # Check the entire release before any write; a manifest is the completion marker.
    completed_release = (output / "manifest.json").exists()
    for name, (kind, value) in artifacts.items():
        path = output / name
        if path.exists():
            require(path.is_file() and file_sha256(path) == _artifact_sha256(kind, value),
                    f"Existing official artifact differs: {path}. Choose a new base path; do not overwrite a release.")
        elif completed_release:
            raise ValueError(f"Completed official data release is missing {path}; restore it or choose a new base path.")
    for name, (kind, value) in artifacts.items():
        path = output / name
        if not path.exists():
            (write_jsonl if kind == "jsonl" else write_json)(path, value)
        require(file_sha256(path) == _artifact_sha256(kind, value), f"Save/load checksum failed: {path}")
        if kind == "jsonl":
            require(rows_digest(read_jsonl(path)) == manifest["datasets"][name]["rows_sha256"],
                    f"Reloaded row content differs: {path}")
    return manifest
