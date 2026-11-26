"""Shared, validation-selected DistilBERT training for the master protocol."""

import importlib.metadata
import json
import math
import numbers
import os
import platform
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .project_config import (
    file_sha256, read_jsonl, require, resolve_base_path, rows_digest,
    write_json, write_jsonl,
)


TRAINER_VERSION = 1
CONDITIONS = frozenset({"B1", "B2", "B3", "F0", "F1", "F2"})


def select_device(preference="auto", torch_module=None):
    """Choose CUDA, then MPS, then CPU, or honor an explicit available device."""
    require(preference in {"auto", "cuda", "mps", "cpu"}, "Unknown training device.")
    if torch_module is None:
        import torch as torch_module
    cuda = torch_module.cuda.is_available()
    mps = bool(getattr(torch_module.backends, "mps", None)) and torch_module.backends.mps.is_available()
    available = {"cuda": cuda, "mps": mps, "cpu": True}
    if preference == "auto":
        preference = next(name for name in ("cuda", "mps", "cpu") if available[name])
    require(available[preference], f"Requested device is unavailable: {preference}")
    return torch_module.device(preference)


def _libraries():
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from transformers import get_linear_schedule_with_warmup
    from types import SimpleNamespace

    return SimpleNamespace(**locals())


def _effective_training(config, condition, setting, seed, pilot_learning_rate):
    require(config.get("schema_version") == 1, "Expected schema-version-1 configuration.")
    require(condition in CONDITIONS, f"Unknown training condition: {condition}")
    require(setting in config["settings"], f"Unknown experimental setting: {setting}")
    require(isinstance(seed, numbers.Integral) and not isinstance(seed, bool), "Seed must be an integer.")
    training = dict(config["training"])
    pilot = pilot_learning_rate is not None
    if pilot:
        plan = config["classifier_pilot"]
        require(condition in plan["conditions"] and setting in plan["settings"], "Condition or setting is outside the classifier pilot.")
        require(seed == plan["development_seed"], "The classifier pilot uses only its development seed.")
        require(pilot_learning_rate in plan["learning_rate_candidates"], "Learning rate is outside the declared pilot grid.")
        learning_rate = pilot_learning_rate
    else:
        require(config.get("status") == "frozen", "Main training requires the completed, frozen classifier pilot.")
        require(seed in config["seeds"], "Seed is outside the frozen experimental protocol.")
        learning_rate = training.get("learning_rate")
    require(isinstance(learning_rate, numbers.Real) and not isinstance(learning_rate, bool)
            and math.isfinite(learning_rate) and learning_rate > 0, "A positive finite learning rate is required.")
    require(training.get("optimizer", "adamw").lower() == "adamw", "Only AdamW is supported.")
    require(training.get("full_fine_tuning") is True, "The protocol requires full fine-tuning.")
    require(training.get("precision") == "float32", "The protocol requires float32 training.")
    require(training.get("lr_scheduler") == "linear", "The protocol requires linear learning-rate decay.")
    require(training.get("validation_interval_epochs") == 1, "Validate once after each epoch.")
    require(training.get("checkpoint_metric") == "macro_f1", "Select checkpoints by validation macro-F1.")
    require(training.get("restore_best_checkpoint") is True, "The best validation checkpoint must be restored.")
    for key in ("num_labels", "batch_size", "eval_batch_size", "max_epochs", "early_stopping_patience", "max_length"):
        require(isinstance(training[key], numbers.Integral) and not isinstance(training[key], bool)
                and training[key] > 0, f"{key} must be a positive integer.")
    require(training["num_labels"] == config["dataset"]["num_classes"], "Classifier and dataset label counts differ.")
    require(0 <= training["warmup_ratio"] < 1, "Warmup ratio must lie in [0, 1).")
    require(training["weight_decay"] >= 0 and training["max_grad_norm"] > 0, "Invalid optimizer regularization settings.")
    training["learning_rate"] = float(learning_rate)
    require(training.get("model") and training.get("model_revision"), "Model name and pinned revision are required.")
    return training, pilot


def _rows(data, name):
    if isinstance(data, (str, Path)):
        path = Path(data)
        require(not (path.stem == "test" or path.stem.startswith("test_") or path.stem.endswith("_test")),
                "Official test data must not be supplied to the trainer.")
        data = read_jsonl(path)
    rows = [dict(row) for row in data]
    require(rows, f"{name} data is empty.")
    for row in rows:
        require(isinstance(row.get("id"), str) and bool(row["id"].strip()), f"{name} rows need nonempty string IDs.")
        require(not row["id"].startswith("test:"), "Official test IDs are not training or validation inputs.")
        require(isinstance(row.get("text"), str) and bool(row["text"].strip()), f"{name} contains blank or invalid text.")
        require(isinstance(row.get("label"), numbers.Integral) and not isinstance(row["label"], bool),
                f"{name} labels must be integers.")
        require(str(row.get("split", "")).lower() != "test", "Test examples are not training or validation inputs.")
        row["label"] = int(row["label"])
    return rows


def _validate_rows(training_rows, validation_rows, condition, num_labels):
    expected = set(range(num_labels))
    require(set(row["label"] for row in training_rows) == expected, "Training must contain every configured intent class.")
    require(set(row["label"] for row in validation_rows) == expected, "Validation must contain every configured intent class.")
    validation_ids = [row["id"] for row in validation_rows]
    require(len(set(validation_ids)) == len(validation_ids), "Validation contains repeated IDs.")
    source_rows = {}
    for row in training_rows:
        if row["id"] in source_rows:
            require(condition == "B2", "Only B2 may repeat real-example IDs for oversampling.")
            require(source_rows[row["id"]] == (row["text"], row["label"]), "Repeated training IDs disagree on text or label.")
        source_rows[row["id"]] = (row["text"], row["label"])
    require(not set(source_rows) & set(validation_ids), "Training and validation IDs overlap.")
    validation_id_set = set(validation_ids)
    for row in training_rows:
        parents = [row.get("parent_id"), row.get("source_id")]
        require(all(parent is None or isinstance(parent, str) for parent in parents),
                "Parent and source IDs must be strings.")
        declared_parents = row.get("parent_ids") or []
        if isinstance(declared_parents, str):
            declared_parents = [declared_parents]
        require(isinstance(declared_parents, (list, tuple)) and all(isinstance(parent, str) for parent in declared_parents),
                "parent_ids must contain string IDs.")
        parents.extend(declared_parents)
        require(not any(parent in validation_id_set for parent in parents if isinstance(parent, str)),
                "Training provenance references a validation parent ID.")
    normalize = lambda text: " ".join(text.casefold().split())
    validation_text_ids = {}
    for row in validation_rows:
        validation_text_ids.setdefault(normalize(row["text"]), []).append(row["id"])
    synthetic_matches = []
    for row in training_rows:
        matches = validation_text_ids.get(normalize(row["text"]), [])
        if row.get("source") in {None, "", "real", "oversampled"}:
            require(not matches, "Real training and validation text overlap.")
        elif matches:
            synthetic_matches.append({"training_id": row["id"], "validation_ids": matches})
    if condition == "B2":
        require(len(set(Counter(row["label"] for row in training_rows).values())) == 1,
                "B2 inputs must already be balanced by random oversampling.")
    _training_counts(training_rows)
    return {
        "normalization": "casefold and collapse whitespace",
        "real_validation_id_overlap_count": 0,
        "real_validation_text_overlap_count": 0,
        "synthetic_validation_text_overlap_count": len(synthetic_matches),
        "synthetic_validation_text_overlap_ids": [row["training_id"] for row in synthetic_matches],
        "synthetic_validation_text_overlap_pairs": synthetic_matches,
        "policy": "Generated-text coincidences are recorded without removing or changing examples; real rows and provenance remain disjoint.",
    }


def _training_counts(rows):
    """Separate unique real parents, generated examples, and repeated real exposures."""
    real = {row["id"]: row for row in rows if row.get("source") in {None, "", "real"}}
    synthetic_count = 0
    for row in rows:
        source = row.get("source")
        if source == "oversampled":
            parent = real.get(row.get("parent_id"))
            require(parent is not None and (row["text"], row["label"]) == (parent["text"], parent["label"]),
                    "Oversampling rows must preserve a selected real parent's text and label.")
        elif source not in {None, "", "real"}:
            synthetic_count += 1
            if source == "eda":
                parent = real.get(row.get("parent_id"))
                require(parent is not None and row["label"] == parent["label"],
                        "EDA rows must preserve their selected real parent's label.")
    real_count = len(real)
    return {"real_count": real_count, "real_rows": real_count, "real_unique_ids": real_count,
            "synthetic_count": synthetic_count, "synthetic_rows": synthetic_count,
            "training_rows": len(rows), "train_rows": len(rows),
            "additional_draws": len(rows) - real_count - synthetic_count}


def _rare_ids(base_path, num_labels):
    path = base_path / "01_data" / "rare_classes.json"
    require(path.is_file(), f"Missing fixed rare-class metadata: {path}")
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("rare_class_ids", value.get("rare_classes", value.get("classes")))
    require(isinstance(value, list) and value, "Rare-class metadata must contain IDs.")
    ids = [item["id"] if isinstance(item, dict) else item for item in value]
    require(all(isinstance(label, int) and not isinstance(label, bool) and 0 <= label < num_labels for label in ids)
            and len(set(ids)) == len(ids), "Invalid rare-class IDs.")
    return ids


def _class_names(base_path, num_labels):
    path = base_path / "01_data" / "label_mapping.json"
    if not path.exists():
        return [str(label) for label in range(num_labels)]
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("id2label", value.get("class_names", value.get("label_names", value)))
    if isinstance(value, dict):
        value = [value[str(label)] for label in range(num_labels)]
    require(isinstance(value, list) and len(value) == num_labels and all(isinstance(name, str) for name in value),
            "Invalid label mapping.")
    return value


def _write_csv(path, rows):
    import csv

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _checkpoint_files(folder):
    best = folder / "best_model"
    files = [folder / name for name in (
        "run_config.json", "runtime.json", "history.csv", "results.json", "validation_predictions.jsonl",
        "validation_probabilities.npz", "per_class_metrics.csv", "stage_log.csv",
    )]
    files.extend([best / "config.json", best / "tokenizer_config.json"])
    require(all(path.is_file() for path in files) and bool(list(best.glob("*.safetensors"))),
            f"Completed run has missing artifacts: {folder}")


def _synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _release(torch, device):
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _seed(torch, np, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def _tokenize(rows, tokenizer, max_length):
    encoded = tokenizer([row["text"] for row in rows], truncation=True, max_length=max_length)
    features = [{**{key: values[index] for key, values in encoded.items()}, "labels": row["label"]}
                for index, row in enumerate(rows)]
    return features, sum(len(ids) for ids in encoded["input_ids"])


def _evaluate(model, loader, libraries, device, num_labels, rare_ids):
    torch, np = libraries.torch, libraries.np
    labels, probabilities, loss_sum, count = [], [], 0.0, 0
    model.eval()
    _synchronize(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            gold = batch.pop("labels")
            logits = model(**batch).logits
            loss_sum += float(torch.nn.functional.cross_entropy(logits, gold, reduction="sum").cpu())
            count += len(gold)
            labels.extend(gold.cpu().tolist())
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    _synchronize(torch, device)
    elapsed = time.perf_counter() - started
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.concatenate(probabilities)
    require(np.isfinite(probabilities).all() and math.isfinite(loss_sum), "Validation produced non-finite outputs.")
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "validation_loss": loss_sum / count,
        "accuracy": float(libraries.accuracy_score(labels, predictions)),
        "macro_f1": float(libraries.f1_score(labels, predictions, labels=list(range(num_labels)), average="macro", zero_division=0)),
        "rare_macro_f1": float(libraries.f1_score(labels, predictions, labels=rare_ids, average="macro", zero_division=0)),
        "rare_macro_recall": float(libraries.recall_score(labels, predictions, labels=rare_ids, average="macro", zero_division=0)),
    }
    return metrics, labels, predictions, probabilities, elapsed


def _runtime(libraries, device):
    packages = {}
    for name in ("torch", "transformers", "numpy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": str(device), "hardware": _hardware_name(libraries.torch, device),
            "precision": "float32", "packages": packages}


def _hardware_name(torch, device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip()
            if chip:
                return chip
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _code_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _train_new_run(training_rows, validation_rows, condition, setting, seed, training,
                   base_path, folder, rare_ids, class_names, fingerprint, validation_overlap):
    libraries = _libraries()
    torch, np = libraries.torch, libraries.np
    device = select_device(training.get("device", "auto"), torch)
    runtime = _runtime(libraries, device)
    runtime_path = folder / "runtime.json"
    if runtime_path.exists():
        require(json.loads(runtime_path.read_text()) == runtime,
                f"Interrupted run has a different runtime; choose another output directory: {folder}")
    else:
        write_json(runtime_path, runtime)
    print(f"Training {condition}/{setting}/seed_{seed} on {device}; incomplete matching runs restart from the original seed.", flush=True)
    start_time = datetime.now(timezone.utc).isoformat()
    provenance = {"code_commit": _code_commit(), "trainer_source_sha256": file_sha256(Path(__file__))}
    started = time.perf_counter()
    _seed(torch, np, seed, device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cache = Path(__file__).resolve().parents[1] / "data" / "reproducible" / "_cache" / "hub"
    tokenizer = libraries.AutoTokenizer.from_pretrained(
        training["model"], revision=training["model_revision"], cache_dir=str(cache), token=False,
    )
    features, training_tokens = _tokenize(training_rows, tokenizer, training["max_length"])
    validation_features, validation_tokens = _tokenize(validation_rows, tokenizer, training["max_length"])
    collator = libraries.DataCollatorWithPadding(tokenizer, return_tensors="pt")
    loader = libraries.DataLoader(features, batch_size=training["batch_size"], shuffle=True,
                                 generator=torch.Generator().manual_seed(seed), collate_fn=collator, num_workers=0)
    validation_loader = libraries.DataLoader(validation_features, batch_size=training["eval_batch_size"],
                                            shuffle=False, collate_fn=collator, num_workers=0)
    model = optimizer = scheduler = None
    history, stages = [], []
    best_score, best_epoch, stale_epochs, steps = -1.0, 0, 0, 0
    training_seconds = validation_seconds = 0.0
    best_path = folder / "best_model"
    try:
        model = libraries.AutoModelForSequenceClassification.from_pretrained(
            training["model"], revision=training["model_revision"], cache_dir=str(cache), token=False,
            num_labels=training["num_labels"], id2label=dict(enumerate(class_names)),
            label2id={name: index for index, name in enumerate(class_names)}, attn_implementation="eager",
        ).float().to(device)
        require(all(parameter.requires_grad for parameter in model.parameters()), "Full fine-tuning requires every model parameter to be trainable.")
        criterion = torch.nn.CrossEntropyLoss()
        # Keep the working baseline's decay policy: matrices decay, vector parameters do not.
        groups = [
            {"params": [parameter for parameter in model.parameters() if parameter.ndim >= 2], "weight_decay": training["weight_decay"]},
            {"params": [parameter for parameter in model.parameters() if parameter.ndim < 2], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=training["learning_rate"], foreach=False)
        planned_steps = training["max_epochs"] * len(loader)
        scheduler = libraries.get_linear_schedule_with_warmup(
            optimizer, int(planned_steps * training["warmup_ratio"]), planned_steps,
        )
        for epoch in range(1, training["max_epochs"] + 1):
            model.train()
            _synchronize(torch, device)
            epoch_started = time.perf_counter()
            loss_sum, count = 0.0, 0
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                gold = batch.pop("labels")
                optimizer.zero_grad(set_to_none=True)
                logits = model(**batch).logits
                loss = criterion(logits, gold)
                require(bool(torch.isfinite(loss).cpu()), "Training loss is non-finite.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                loss_sum += float(loss.detach().cpu()) * len(gold)
                count += len(gold)
                steps += 1
            _synchronize(torch, device)
            elapsed = time.perf_counter() - epoch_started
            metrics, _, _, _, evaluation_seconds = _evaluate(
                model, validation_loader, libraries, device, training["num_labels"], rare_ids,
            )
            training_seconds += elapsed
            validation_seconds += evaluation_seconds
            history.append({"epoch": epoch, "train_loss": loss_sum / count, **metrics,
                            "training_seconds": elapsed, "validation_seconds": evaluation_seconds,
                            "optimizer_steps": steps, "learning_rate": scheduler.get_last_lr()[0]})
            for stage, seconds, tokens in (("training", elapsed, training_tokens), ("validation", evaluation_seconds, validation_tokens)):
                stages.append({"stage": stage, "epoch": epoch, "condition": condition, "setting": setting, "seed": seed,
                               "device": str(device), "hardware": runtime["hardware"], **provenance,
                               "model_name": training["model"], "model_revision": training["model_revision"], "wall_seconds": seconds,
                               "accelerator_synchronized_wall_seconds": seconds if device.type != "cpu" else 0.0,
                               "nonpadding_tokens": tokens, "optimizer_steps": len(loader) if stage == "training" else 0})
            _write_csv(folder / "history.csv", history)
            _write_csv(folder / "stage_log.csv", stages)
            print(f"{condition}/{setting}/seed_{seed} epoch {epoch}: validation macro-F1={metrics['macro_f1']:.4f}; training={elapsed:.1f}s", flush=True)
            if metrics["macro_f1"] > best_score:
                best_score, best_epoch, stale_epochs = metrics["macro_f1"], epoch, 0
                model.save_pretrained(str(best_path), safe_serialization=True)
                tokenizer.save_pretrained(str(best_path))
            else:
                stale_epochs += 1
                if stale_epochs >= training["early_stopping_patience"]:
                    break
        del batch, gold, logits, loss, criterion, groups
        model = optimizer = scheduler = None
        _release(torch, device)
        model = libraries.AutoModelForSequenceClassification.from_pretrained(
            str(best_path), local_files_only=True, attn_implementation="eager",
        ).float().to(device)
        metrics, labels, predictions, probabilities, seconds = _evaluate(
            model, validation_loader, libraries, device, training["num_labels"], rare_ids,
        )
        require(np.isclose(metrics["macro_f1"], best_score), "Reloaded best-checkpoint macro-F1 differs from its recorded epoch.")
        validation_seconds += seconds
        stages.append({"stage": "best_checkpoint_validation", "epoch": best_epoch, "condition": condition,
                       "setting": setting, "seed": seed, "device": str(device), "hardware": runtime["hardware"], **provenance,
                       "model_name": training["model"], "model_revision": training["model_revision"], "wall_seconds": seconds,
                       "accelerator_synchronized_wall_seconds": seconds if device.type != "cpu" else 0.0,
                       "nonpadding_tokens": validation_tokens, "optimizer_steps": 0})
        prediction_rows = [
            {"id": row["id"], "text": row["text"], "true_label": int(labels[index]),
             "predicted_label": int(predictions[index]), "confidence": float(probabilities[index].max()),
             "condition": condition, "setting": setting, "seed": int(seed), "split": "validation"}
            for index, row in enumerate(validation_rows)
        ]
        write_jsonl(folder / "validation_predictions.jsonl", prediction_rows)
        probability_path = folder / "validation_probabilities.npz"
        temporary = probability_path.with_name(probability_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, probabilities=probabilities, labels=labels,
                                ids=np.asarray([row["id"] for row in validation_rows]))
        temporary.replace(probability_path)
        precision, recall, f1, support = libraries.precision_recall_fscore_support(
            labels, predictions, labels=list(range(training["num_labels"])), zero_division=0,
        )
        _write_csv(folder / "per_class_metrics.csv", [
            {"label": label, "intent": class_names[label], "rare_class": label in rare_ids,
             "precision": float(precision[label]), "recall": float(recall[label]),
             "f1": float(f1[label]), "support": int(support[label])}
            for label in range(training["num_labels"])
        ])
        _write_csv(folder / "stage_log.csv", stages)
        result = {
            "status": "completed", "fingerprint": fingerprint, "condition": condition, "setting": setting, "seed": int(seed),
            **metrics, "best_epoch": best_epoch, "epochs_run": len(history), **_training_counts(training_rows),
            "validation_rows": len(validation_rows), "optimizer_steps": steps, "planned_optimizer_steps": planned_steps,
            "training_tokens": training_tokens * len(history), "validation_tokens": validation_tokens * (len(history) + 1),
            "training_seconds": training_seconds, "validation_seconds": validation_seconds,
            "total_seconds": time.perf_counter() - started, "device": str(device), "precision": "float32",
            "hardware": runtime["hardware"], "start_time": start_time,
            "end_time": datetime.now(timezone.utc).isoformat(), **provenance,
            "model_path": str(best_path), "run_path": str(folder),
            "validation_overlap": validation_overlap,
            "timing_note": "Classifier elapsed time only; synchronized accelerator intervals are not measured hardware utilization. Upstream generation/EDA/filter costs are separate.",
        }
        write_json(folder / "results.json", result)
        return result
    finally:
        model = optimizer = scheduler = None
        _release(torch, device)


def train_classifier(train_data, validation_data, condition, setting, seed, config, *,
                     output_dir=None, base_path=None, pilot_learning_rate=None):
    """Train once, select on validation macro-F1, and safely reuse matching results.

    Main runs require ``status: frozen`` and a positive configured learning rate.
    An explicit pilot rate is restricted to the declared development grid. Incomplete
    matching runs restart from their seed; optimizer or mid-epoch state is not resumed.
    """
    training, pilot = _effective_training(config, condition, setting, seed, pilot_learning_rate)
    base = resolve_base_path(config, base_path)
    training_rows, validation_rows = _rows(train_data, "Training"), _rows(validation_data, "Validation")
    validation_overlap = _validate_rows(training_rows, validation_rows, condition, training["num_labels"])
    rare_ids = _rare_ids(base, training["num_labels"])
    class_names = _class_names(base, training["num_labels"])
    specification = {
        "trainer_version": TRAINER_VERSION, "protocol_id": config.get("protocol_id"),
        "trainer_source_sha256": file_sha256(Path(__file__)),
        "mode": "classifier_pilot" if pilot else "main", "condition": condition, "setting": setting, "seed": int(seed),
        "training": training, "training_rows_sha256": rows_digest(training_rows),
        "validation_rows_sha256": rows_digest(validation_rows), "rare_class_ids": rare_ids, "class_names": class_names,
        "validation_overlap": validation_overlap,
    }
    fingerprint = rows_digest([specification])
    if output_dir is None:
        if pilot:
            folder = base / "04_checkpoints" / "classifier_pilot" / f"lr_{training['learning_rate']:.0e}" / condition / setting / f"seed_{seed}"
        else:
            folder = base / "04_checkpoints" / condition / setting / f"seed_{seed}"
    else:
        folder = Path(output_dir).expanduser().resolve()
    config_path, result_path = folder / "run_config.json", folder / "results.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        require(saved.get("fingerprint") == fingerprint and saved.get("specification") == specification,
                f"Existing run configuration or data differs; choose another output directory: {folder}")
    else:
        require(not folder.exists() or not any(folder.iterdir()), f"Existing run has no configuration: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        write_json(config_path, {"fingerprint": fingerprint, "specification": specification,
                                 "code_commit": _code_commit(),
                                 "config_provenance_sha256": rows_digest([config])})
    if result_path.exists():
        result = json.loads(result_path.read_text())
        require(result.get("status") == "completed" and result.get("fingerprint") == fingerprint,
                f"Invalid completed result: {folder}")
        _checkpoint_files(folder)
        print(f"Reusing completed {condition}/{setting}/seed_{seed}: validation macro-F1={result['macro_f1']:.4f}", flush=True)
        return result
    return _train_new_run(training_rows, validation_rows, condition, setting, int(seed), training,
                          base, folder, rare_ids, class_names, fingerprint, validation_overlap)
