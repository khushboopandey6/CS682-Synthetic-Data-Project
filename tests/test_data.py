"""Meaningful release-contract and reproducibility tests; no downloads or GPU."""

import copy
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src.data_preparation import prepare_official_data, sample_per_class
from src.project_config import file_sha256, read_jsonl


class OfficialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Interleave labels: encounter-order grouping must work independently of
        # the original dataset being physically sorted by intent.
        cls.names = [f"intent_{label}" for label in range(77)]
        labels = [(i + 11) % 77 for i in range(10003)]
        cls.train = [{"id": f"train:{i}", "text": f"Original query {i} for intent {label}.", "label": label}
                     for i, label in enumerate(labels)]
        cls.test = [{"id": f"test:{i}", "text": f"Held-out query {i}.", "label": i % 77} for i in range(3080)]
        cls.configuration = {
            "schema_version": 1, "protocol_id": "master_v1",
            "dataset": {"name": "banking77", "dataset_id": "PolyAI/banking77",
                        "dataset_revision": "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
                        "num_classes": 77, "validation_per_class": 10, "validation_seed": 42,
                        "sampling_algorithm": "python_random_sample_source_class_encounter_order"},
            "seeds": [42, 123, 456],
            "settings": {"low_resource": {"samples_per_class": 10},
                         "imbalanced": {"rare_class_seed": 42,
                                        "rare_class_selection_algorithm": "python_random_sample_sorted_label_ids",
                                        "num_rare_classes": 15, "rare_samples_per_class": 5,
                                        "majority_samples_per_class": 20}},
        }

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cs682-data-test-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.loader = patch("src.data_preparation._load_official_source",
                            return_value=(copy.deepcopy(self.train), copy.deepcopy(self.test), list(self.names)))
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.config = copy.deepcopy(self.configuration)

    def prepare(self, **kwargs):
        return prepare_official_data(self.config, kwargs.get("base_path", self.base), source_root=self.base / "source")

    def test_contract_counts_labels_ids_and_split_isolation(self):
        manifest = self.prepare()
        root = self.base / "01_data"
        rare = random.Random(42).sample(list(range(77)), 15)
        self.assertEqual(manifest["rare_class_ids"], rare)
        self.assertEqual(json.loads((root / "rare_classes.json").read_text())["rare_class_ids"], rare)
        self.assertEqual(json.loads((root / "label_mapping.json").read_text()), {str(i): name for i, name in enumerate(self.names)})
        validation = read_jsonl(root / "validation.jsonl")
        validation_ids = {row["id"] for row in validation}
        self.assertEqual(Counter(row["label"] for row in validation), Counter({label: 10 for label in range(77)}))
        self.assertEqual(read_jsonl(root / "test.jsonl"), self.test)
        original_by_id = {row["id"]: row for row in self.train}
        for setting in ("low_resource", "imbalanced"):
            for seed in self.config["seeds"]:
                name = f"{setting}/seed_{seed}.jsonl"
                rows = read_jsonl(root / name)
                counts = {label: 10 if setting == "low_resource" else 5 if label in rare else 20 for label in range(77)}
                self.assertEqual(Counter(row["label"] for row in rows), Counter(counts))
                self.assertEqual(len(rows), 770 if setting == "low_resource" else 1315)
                self.assertEqual(len({row["id"] for row in rows}), len(rows))
                self.assertFalse(validation_ids & {row["id"] for row in rows})
                for row in rows:
                    self.assertEqual(set(row), {"id", "text", "label"})
                    self.assertEqual(row, original_by_id[row["id"]])
                self.assertEqual(manifest["datasets"][name]["ids"], [row["id"] for row in rows])
                self.assertEqual(manifest["datasets"][name]["file_sha256"], file_sha256(root / name))

    def test_same_seed_matches_independent_index_sampling_and_is_portable(self):
        # Independently sample index lists like the historical notebook.
        groups = {}
        for i, row in enumerate(self.train):
            groups.setdefault(row["label"], []).append(i)
        rng = random.Random(42)
        selected = [i for indices in groups.values() for i in rng.sample(indices, 10)]
        expected = [self.train[i] for i in selected]
        self.assertEqual(sample_per_class(self.train, {label: 10 for label in range(77)}, 42), expected)
        first = self.prepare(base_path=self.base / "one")
        second = self.prepare(base_path=self.base / "two")
        self.assertEqual(first, second)
        self.assertEqual(read_jsonl(self.base / "one/01_data/validation.jsonl"), expected)
        self.assertNotEqual(first["datasets"]["low_resource/seed_42.jsonl"]["ids"],
                            first["datasets"]["low_resource/seed_123.jsonl"]["ids"])

    def test_idempotent_reuse_preserves_every_artifact(self):
        first = self.prepare()
        before = {p.relative_to(self.base): (file_sha256(p), p.stat().st_mtime_ns)
                  for p in self.base.rglob("*") if p.is_file()}
        # Training settings are not data dependencies.
        self.config["training"] = {"learning_rate": 1e-4, "max_epochs": 30}
        self.assertEqual(self.prepare(), first)
        after = {p.relative_to(self.base): (file_sha256(p), p.stat().st_mtime_ns)
                 for p in self.base.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_changed_records_or_sampling_refuse_to_overwrite_release(self):
        self.prepare()
        path = self.base / "01_data/low_resource/seed_42.jsonl"
        path.write_text(path.read_text().replace("Original query", "Altered query", 1))
        before = {p.relative_to(self.base): file_sha256(p) for p in self.base.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ValueError, "Existing official artifact differs"):
            self.prepare()
        self.assertEqual(before, {p.relative_to(self.base): file_sha256(p) for p in self.base.rglob("*") if p.is_file()})

    def test_preflight_conflict_makes_no_partial_new_release(self):
        root = self.base / "01_data"
        root.mkdir()
        mapping = root / "label_mapping.json"
        mapping.write_text('{"0":"wrong_mapping"}\n')
        with self.assertRaisesRegex(ValueError, "Existing official artifact differs"):
            self.prepare()
        self.assertEqual(list(root.iterdir()), [mapping])

    def test_changed_seed_set_and_missing_completed_file_refused(self):
        self.prepare()
        self.config["seeds"] = [42, 123, 789]
        with self.assertRaisesRegex(ValueError, "missing|differs"):
            self.prepare()
        self.assertFalse((self.base / "01_data/low_resource/seed_789.jsonl").exists())
        self.config["seeds"] = [42, 123, 456]
        (self.base / "01_data/low_resource/seed_456.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.prepare()

    def test_malformed_source_and_unknown_sampling_algorithm_refused(self):
        self.loader.stop()
        malformed = copy.deepcopy(self.train)
        malformed[0]["id"] = "train:99999"
        with patch("src.data_preparation._load_official_source", return_value=(malformed, self.test, self.names)):
            with self.assertRaisesRegex(ValueError, "source IDs"):
                self.prepare()
        self.config["dataset"]["sampling_algorithm"] = "unrecorded_sampler"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
