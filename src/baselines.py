"""Prepare reproducible oversampling and EDA datasets without accessing evaluation data."""

import importlib.metadata as metadata
import json
import time
from pathlib import Path

from .project_config import (file_sha256, read_jsonl, require, rows_digest,
                             write_json, write_jsonl)

import random
import re
from collections import Counter
from functools import lru_cache


EDA_STOP_WORDS = frozenset("""i me my myself we our ours ourselves you your yours yourself yourselves he him his
himself she her hers herself it its itself they them their theirs themselves what
which who whom this that these those am is are was were be been being have has had
having do does did doing a an the and but if or because as until while of at by for
with about against between into through during before after above below to from up
down in out on off over under again further then once here there when where why how
all any both each few more most other some such no nor not only own same so than too
very can will just should now
""".split())
EDA_NEGATION = frozenset({"no", "not", "never", "nor", "neither", "cannot"})
EDA_METHODS = ("synonym_replacement", "random_insertion", "random_swap", "random_deletion")


def eda_tokens(text):
    return re.findall(r"\w+(?:['’]\w+)*(?:[.,:/-]\d+)*|[^\w\s]", text, flags=re.UNICODE)


def eda_normalize(text):
    return " ".join(token.casefold() for token in eda_tokens(text))


def eda_protected(token):
    word = token.casefold().replace("’", "'")
    return word in EDA_NEGATION or word.endswith("n't") or any(char.isdigit() for char in token)


def eda_join(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"([$£€])\s+(?=\d)", r"\1", text)
    return text.strip()


@lru_cache(maxsize=10000)
def eda_synonyms(word):
    from nltk.corpus import wordnet

    choices = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            candidate = " ".join(lemma.name().replace("_", " ").replace("-", " ").casefold().split())
            if candidate != word.casefold() and candidate and all(part.isalpha() for part in candidate.split()):
                choices.add(candidate)
    # WordNet and set traversal order must not decide the augmentation seed.
    return tuple(sorted(choices))


def eda_variant(text, method, rng, alpha=0.1, deletion_probability=0.1):
    if alpha == 0 and method in EDA_METHODS[:3]:
        return text
    tokens = eda_tokens(text)
    eligible = [index for index, token in enumerate(tokens) if token.isalpha() and not eda_protected(token)]
    synonym_words = sorted({tokens[index].casefold() for index in eligible
                            if tokens[index].casefold() not in EDA_STOP_WORDS})
    n_edits = max(1, int(alpha * len([token for token in tokens if any(char.isalnum() for char in token)])))

    if method == "synonym_replacement":
        rng.shuffle(synonym_words)
        replaced = 0
        for word in synonym_words:
            choices = eda_synonyms(word)
            if choices:
                replacement = rng.choice(choices).split()
                tokens = [part for token in tokens for part in
                          (replacement if token.casefold() == word else [token])]
                replaced += 1
                if replaced == n_edits:
                    break
    elif method == "random_insertion":
        candidates = [(word, eda_synonyms(word)) for word in synonym_words]
        candidates = [(word, choices) for word, choices in candidates if choices]
        for _ in range(n_edits):
            if not candidates:
                break
            _, choices = rng.choice(candidates)
            position = rng.randrange(len(tokens) + 1)
            tokens[position:position] = rng.choice(choices).split()
    elif method == "random_swap":
        if len(eligible) >= 2:
            for _ in range(n_edits):
                first, second = rng.sample(eligible, 2)
                tokens[first], tokens[second] = tokens[second], tokens[first]
    elif method == "random_deletion":
        original = tokens
        tokens = [token for token in tokens if not token.isalpha() or eda_protected(token)
                  or rng.random() >= deletion_probability]
        if not any(any(char.isalnum() for char in token) for token in tokens):
            tokens = original
    else:
        raise ValueError(f"Unknown EDA method: {method}")
    return eda_join(tokens)


def build_eda_rows(real_data, seed, augmentations_per_real=1, attempts_per_example=16,
                   alpha=0.1, deletion_probability=0.1):
    if not isinstance(augmentations_per_real, int) or augmentations_per_real < 0:
        raise ValueError("augmentations_per_real must be a nonnegative integer")
    if not isinstance(attempts_per_example, int) or attempts_per_example < 1:
        raise ValueError("attempts_per_example must be a positive integer")
    if not 0 <= alpha <= 1 or not 0 <= deletion_probability <= 1:
        raise ValueError("EDA edit and deletion rates must lie in [0, 1]")

    real_rows = [dict(row) for row in real_data]
    rng = random.Random(seed)
    seen = {eda_normalize(row["text"]) for row in real_rows}
    records = [{**row, "is_augmented": False, "augmentation_id": f"real:{row['source_id']}",
                "augmentation_method": "original"} for row in real_rows]
    accepted_by_class = Counter()
    attempts_by_method = Counter({method: 0 for method in EDA_METHODS})
    accepted_by_method = Counter({method: 0 for method in EDA_METHODS})
    rejected = Counter()

    for row in real_rows:
        methods = list(EDA_METHODS)
        rng.shuffle(methods)
        accepted = 0
        attempt = 0
        while accepted < augmentations_per_real and attempt < attempts_per_example * augmentations_per_real:
            method = methods[attempt % len(methods)]
            candidate = eda_variant(row["text"], method, rng, alpha, deletion_probability)
            attempt += 1
            attempts_by_method[method] += 1
            normalized = eda_normalize(candidate)
            if not normalized or not any(char.isalnum() for char in candidate):
                rejected["empty"] += 1
                continue
            if normalized == eda_normalize(row["text"]):
                rejected["unchanged"] += 1
                continue
            if normalized in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(normalized)
            records.append({**row, "text": candidate, "is_augmented": True,
                            "augmentation_id": f"eda:{seed}:{row['source_id']}:{accepted}",
                            "augmentation_method": method})
            accepted += 1
            accepted_by_class[int(row["label"])] += 1
            accepted_by_method[method] += 1

    original_counts = Counter(int(row["label"]) for row in real_rows)
    requested = len(real_rows) * augmentations_per_real
    accepted = sum(accepted_by_class.values())
    report = {
        "seed": seed,
        "alpha": alpha,
        "deletion_probability": deletion_probability,
        "augmentations_per_real": augmentations_per_real,
        "attempts_per_example": attempts_per_example,
        "real_rows": len(real_rows),
        "requested_augmented_rows": requested,
        "accepted_augmented_rows": accepted,
        "shortfall": requested - accepted,
        "training_rows": len(records),
        "attempts_by_method": dict(attempts_by_method),
        "accepted_by_method": dict(accepted_by_method),
        "rejections": dict(rejected),
        "per_class": {str(label): {"requested": count * augmentations_per_real,
                                   "accepted": accepted_by_class[label],
                                   "shortfall": count * augmentations_per_real - accepted_by_class[label]}
                      for label, count in sorted(original_counts.items())},
        "policy": "SR/RI/RS/RD; preserve punctuation, negation and numeric tokens; deduplicate against real training and accepted EDA only",
    }
    return records, report


def _validate_real_rows(rows, num_labels=None):
    require(bool(rows), "The selected real training subset is empty.")
    require(all(isinstance(row.get("id"), str) and row["id"] for row in rows),
            "Every real training row needs a nonempty string id.")
    require(len({row["id"] for row in rows}) == len(rows), "Real training IDs must be unique.")
    require(all(isinstance(row.get("text"), str) and row["text"].strip() for row in rows),
            "Real training text must be nonempty.")
    require(all(type(row.get("label")) is int and row["label"] >= 0 for row in rows),
            "Labels must be nonnegative integer class IDs.")
    if num_labels is not None:
        require(set(row["label"] for row in rows) == set(range(num_labels)),
                "The selected real subset must contain every official label.")


def build_random_oversampling(real_data, seed, num_labels=77, setting="imbalanced"):
    """Append replacement draws while preserving the original records unchanged."""
    original = [dict(row) for row in real_data]
    _validate_real_rows(original, num_labels)
    groups = {label: [] for label in range(num_labels)}
    for row in original:
        groups[row["label"]].append(row)
    target = max(len(rows) for rows in groups.values())
    rng = random.Random(seed)
    records = [dict(row) for row in original]
    multiplicities = Counter(row["id"] for row in original)
    for label in range(num_labels):
        for _ in range(target - len(groups[label])):
            parent = rng.choice(groups[label])
            multiplicities[parent["id"]] += 1
            record_id = f"oversample:{setting}:{seed}:{parent['id']}:{multiplicities[parent['id']] - 1}"
            records.append({**parent, "id": record_id, "occurrence_id": record_id,
                            "parent_id": parent["id"], "source": "oversampled",
                            "setting": setting, "seed": int(seed)})
    require(len({row["id"] for row in records}) == len(records), "An oversampling ID collides with a real ID.")
    before = Counter(row["label"] for row in original)
    after = Counter(row["label"] for row in records)
    require(after == Counter({label: target for label in range(num_labels)}),
            "Oversampling did not balance all classes.")
    report = {
        "method": "random_oversampling", "implementation_version": 1,
        "seed": int(seed), "target_per_class": target,
        "real_unique_rows": len(original), "training_rows": len(records),
        "additional_draws": len(records) - len(original),
        "no_op": len(records) == len(original),
        "class_counts_before": {str(label): before[label] for label in range(num_labels)},
        "class_counts_after": {str(label): after[label] for label in range(num_labels)},
        "source_multiplicities": dict(sorted(multiplicities.items())),
        "policy": "Preserve original rows; append seeded replacement draws to the largest class count; ordinary cross-entropy.",
    }
    return records, report


def build_eda_baseline(real_data, setting, seed, **eda_config):
    """Adapt the verified EDA sampler to the shared JSONL record contract."""
    original = [dict(row) for row in real_data]
    _validate_real_rows(original)
    internal = [{**row, "source_id": row["id"]} for row in original]
    legacy_rows, report = build_eda_rows(internal, seed, **eda_config)
    parents = {row["id"]: row for row in original}
    records = [dict(row) for row in original]
    for row in legacy_rows[len(original):]:
        parent_id = row["source_id"]
        record_id = f"eda:{setting}:{seed}:{row['augmentation_id']}"
        records.append({**parents[parent_id], "id": record_id, "text": row["text"],
                        "parent_id": parent_id, "augmentation_id": record_id,
                        "source": "eda", "setting": setting, "seed": int(seed),
                        "is_augmented": True, "augmentation_method": row["augmentation_method"]})
    require(len({row["id"] for row in records}) == len(records), "An EDA ID collides with a real ID.")
    return records, report


def wordnet_metadata(cache_path=None):
    """Use installed corpora only; preparing baselines never downloads resources."""
    import nltk
    from nltk.corpus import wordnet

    if cache_path is not None and str(cache_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(cache_path))
    try:
        wordnet.ensure_loaded()
    except LookupError as error:
        raise RuntimeError("Cached NLTK WordNet is required for B3; install the corpus before preparing baselines.") from error
    return {"nltk_version": metadata.version("nltk"), "wordnet_version": wordnet.get_version()}


def _validate_output(original, records, condition, setting, seed, statistics):
    require(records[:len(original)] == original, "Baseline preparation changed the original real records.")
    require(len({row["id"] for row in records}) == len(records), "Baseline IDs are not unique.")
    parents = {row["id"]: row for row in original}
    added = records[len(original):]
    for row in added:
        parent = parents.get(row.get("parent_id"))
        require(parent is not None and row["label"] == parent["label"], "Baseline parent or intended label differs.")
        require(row.get("setting") == setting and row.get("seed") == seed, "Baseline setting or seed differs.")
        if condition == "B2":
            require(row["text"] == parent["text"], "Oversampling changed a real query.")
            require(row.get("occurrence_id") == row["id"], "Missing unique oversampling occurrence ID.")
        else:
            require(row.get("source") == "eda" and row.get("augmentation_id") == row["id"], "Invalid EDA provenance.")
            require(isinstance(row.get("text"), str) and row["text"].strip(), "EDA produced blank text.")
    require(statistics["training_rows"] == len(records), "Baseline report row count differs.")
    if condition == "B2":
        counts = Counter(row["label"] for row in records)
        require({str(label): count for label, count in sorted(counts.items())} == statistics["class_counts_after"],
                "Oversampling class counts differ from its report.")
        require(statistics["additional_draws"] == len(added), "Oversampling report has incorrect draw count.")
        require(statistics["no_op"] == (not added), "Oversampling no-op status differs.")
    else:
        normalized = [eda_normalize(row["text"]) for row in added]
        require(len(set(normalized)) == len(normalized), "EDA contains repeated normalized text.")
        require(not set(normalized) & {eda_normalize(row["text"]) for row in original}, "EDA duplicates real training text.")
        require(statistics["accepted_augmented_rows"] == len(added), "EDA accepted count differs from its report.")
        require(statistics["shortfall"] == statistics["requested_augmented_rows"] - len(added), "EDA shortfall differs.")
        accepted_counts = Counter(row["label"] for row in added)
        original_counts = Counter(row["label"] for row in original)
        for label, count in original_counts.items():
            requested = count * statistics["augmentations_per_real"]
            require(statistics["per_class"][str(label)] == {
                "requested": requested, "accepted": accepted_counts[label], "shortfall": requested - accepted_counts[label],
            }, "EDA per-class counts differ from its report.")


def _load_official_subset(base_path, setting, seed, num_labels):
    relative = f"{setting}/seed_{seed}.jsonl"
    path = base_path / "01_data" / relative
    manifest_path = base_path / "01_data" / "manifest.json"
    require(path.is_file() and manifest_path.is_file(),
            "Release the official 01_data artifacts before preparing baselines.")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["datasets"].get(relative)
    require(expected is not None, f"Training subset is missing from the official manifest: {relative}")
    rows = read_jsonl(path)
    _validate_real_rows(rows, num_labels)
    require(file_sha256(path) == expected["file_sha256"], f"Official training file hash differs: {relative}")
    require(rows_digest(rows) == expected["rows_sha256"], f"Official training content differs: {relative}")
    require(len(rows) == expected["rows"], f"Official training row count differs: {relative}")
    return path, rows


def _eda_parameters(config):
    baseline = config["baselines"]
    return {
        "alpha": baseline["eda_alpha"],
        "deletion_probability": baseline["eda_deletion_probability"],
        "augmentations_per_real": baseline["eda_augmentations_per_real"],
        "attempts_per_example": baseline["eda_attempts_per_example"],
    }


def prepare_baseline_data(config, base_path, settings=None, seeds=None, conditions=("B2", "B3")):
    """Export baseline JSONL and strict, reusable per-dataset preparation reports."""
    base_path = Path(base_path)
    settings = list(settings) if settings is not None else list(config["settings"])
    seeds = list(seeds) if seeds is not None else list(config["seeds"])
    conditions = list(conditions)
    require(settings and len(set(settings)) == len(settings) and set(settings) <= set(config["settings"]), "Unknown or repeated setting.")
    require(seeds and len(set(seeds)) == len(seeds) and set(seeds) <= set(config["seeds"]), "Unknown or repeated seed.")
    require(conditions and len(set(conditions)) == len(conditions) and set(conditions) <= {"B2", "B3"}, "Conditions must be B2 and/or B3.")
    require(config["baselines"]["B2"] == "random_oversampling", "B2 must use random oversampling.")
    require(config["baselines"]["B3"] == "easy_data_augmentation", "B3 must use EDA.")
    reports = []
    corpus = wordnet_metadata() if "B3" in conditions else None
    for setting in settings:
        for seed in seeds:
            source_path, original = _load_official_subset(base_path, setting, seed, config["dataset"]["num_classes"])
            for condition in conditions:
                specification = {
                    "schema_version": 1, "implementation_version": 1,
                    "protocol_id": config["protocol_id"], "condition": condition,
                    "setting": setting, "seed": int(seed), "num_classes": config["dataset"]["num_classes"],
                    "dataset_source": {"id": config["dataset"]["dataset_id"],
                                       "revision": config["dataset"]["dataset_revision"]},
                    "source_path": str(source_path.relative_to(base_path)),
                    "source_file_sha256": file_sha256(source_path), "source_rows_sha256": rows_digest(original),
                    "method": config["baselines"][condition],
                    "parameters": _eda_parameters(config) if condition == "B3" else {"target": "largest_selected_class_count"},
                    "corpus": corpus if condition == "B3" else None,
                }
                config_sha = rows_digest([specification])
                output_path = base_path / "01_data" / "baselines" / condition / setting / f"seed_{seed}.jsonl"
                report_path = base_path / "01_data" / "baselines" / "reports" / condition / setting / f"seed_{seed}.json"
                if report_path.exists():
                    saved = json.loads(report_path.read_text())
                    require(saved["specification"] == specification and saved["config_sha256"] == config_sha,
                            f"Baseline preparation configuration differs: {report_path}")
                    require(output_path.is_file() and file_sha256(output_path) == saved["file_sha256"],
                            f"Saved baseline content differs: {output_path}")
                    records = read_jsonl(output_path)
                    require(rows_digest(records) == saved["rows_sha256"], f"Saved baseline row hash differs: {output_path}")
                    _validate_output(original, records, condition, setting, seed, saved["statistics"])
                    require(saved["rows"] == len(records) and saved["per_class"] == {
                        str(label): count for label, count in sorted(Counter(row["label"] for row in records).items())
                    }, f"Saved baseline report counts differ: {report_path}")
                    require(saved["b1_identical_inputs"] == (condition == "B2" and saved["statistics"]["no_op"]),
                            f"Saved B1 identity status differs: {report_path}")
                    if saved["b1_identical_inputs"]:
                        require(output_path.read_bytes() == source_path.read_bytes(), "Balanced B2 bytes differ from B1 input.")
                    reports.append(saved)
                    continue
                require(not output_path.exists(), f"Baseline data exists without a preparation report: {output_path}")
                started = time.perf_counter()
                if condition == "B2":
                    records, statistics = build_random_oversampling(original, seed, config["dataset"]["num_classes"], setting)
                else:
                    records, statistics = build_eda_baseline(original, setting, seed, **_eda_parameters(config))
                _validate_output(original, records, condition, setting, seed, statistics)
                identical = condition == "B2" and statistics["no_op"]
                if identical:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output_path.with_name(output_path.name + ".tmp")
                    temporary.write_bytes(source_path.read_bytes())
                    temporary.replace(output_path)
                else:
                    write_jsonl(output_path, records)
                report = {
                    "schema_version": 1, "status": "prepared", "condition": condition,
                    "setting": setting, "seed": int(seed), "specification": specification,
                    "config_sha256": config_sha, "source_file_sha256": specification["source_file_sha256"],
                    "source_rows_sha256": specification["source_rows_sha256"],
                    "data_path": str(output_path.relative_to(base_path)),
                    "report_path": str(report_path.relative_to(base_path)),
                    "file_sha256": file_sha256(output_path), "rows_sha256": rows_digest(records),
                    "rows": len(records), "per_class": {str(label): count for label, count in sorted(Counter(row["label"] for row in records).items())},
                    "b1_identical_inputs": identical, "statistics": statistics,
                    "preparation_seconds": time.perf_counter() - started,
                    "method_caveat": "EDA preserves intended labels, not guaranteed semantic correctness." if condition == "B3" else None,
                }
                write_json(report_path, report)
                reports.append(report)
    return reports
