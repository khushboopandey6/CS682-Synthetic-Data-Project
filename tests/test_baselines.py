import copy
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src import baselines
from src.project_config import file_sha256, read_jsonl, rows_digest, write_json, write_jsonl


LEXICON = {"card": ("payment card",), "working": ("functioning",),
           "transfer": ("remittance",), "money": ("cash",)}


def fixture_rows(counts):
    rows = []
    for label, count in enumerate(counts):
        for number in range(count):
            rows.append({"id": f"train:{label}:{number}",
                         "text": f"My card is not working for {label * 100 + number} dollars.",
                         "label": label, "extra_metadata": f"kept-{label}-{number}"})
    return rows


def fixture_config():
    return {"schema_version": 1, "protocol_id": "baseline_fixture", "seeds": [42, 123],
            "dataset": {"num_classes": 2, "dataset_id": "fixture", "dataset_revision": "fixture-v1"},
            "settings": {"low_resource": {}, "imbalanced": {}},
            "baselines": {"B2": "random_oversampling", "B3": "easy_data_augmentation",
                          "eda_alpha": 0.1, "eda_deletion_probability": 0.1,
                          "eda_augmentations_per_real": 1, "eda_attempts_per_example": 16}}


def release_fixture(root):
    datasets = {}
    for setting, counts in (("low_resource", (2, 2)), ("imbalanced", (1, 2))):
        for seed in (42, 123):
            rows = fixture_rows(counts)
            relative = f"{setting}/seed_{seed}.jsonl"
            path = root / "01_data" / relative
            write_jsonl(path, rows)
            datasets[relative] = {"rows": len(rows), "rows_sha256": rows_digest(rows),
                                  "file_sha256": file_sha256(path)}
    write_json(root / "01_data" / "manifest.json", {"datasets": datasets})
    return datasets


class BaselineBuildersTest(unittest.TestCase):
    def test_oversampling_is_seeded_and_keeps_originals(self):
        rows = fixture_rows((2, 4))
        rng_state = random.getstate()
        first, report = baselines.build_random_oversampling(rows, 42, num_labels=2)
        second, repeated = baselines.build_random_oversampling(rows, 42, num_labels=2)
        self.assertEqual((first, report), (second, repeated))
        self.assertEqual(random.getstate(), rng_state)
        self.assertEqual(first[:len(rows)], rows)
        self.assertEqual(Counter(row["label"] for row in first), {0: 4, 1: 4})
        self.assertEqual(len({row["id"] for row in first}), len(first))
        source = {row["id"]: row for row in rows}
        for row in first[len(rows):]:
            parent = source[row["parent_id"]]
            self.assertEqual(row["text"], parent["text"])
            self.assertEqual(row["label"], parent["label"])
            self.assertEqual(row["extra_metadata"], parent["extra_metadata"])
            self.assertEqual(row["occurrence_id"], row["id"])
        self.assertFalse(report["no_op"])
        self.assertEqual(report["additional_draws"], 2)

    def test_balanced_oversampling_is_exact_identity(self):
        original = fixture_rows((2, 2))
        rows, report = baselines.build_random_oversampling(original, 42, num_labels=2, setting="low_resource")
        self.assertEqual(rows, original)
        self.assertTrue(report["no_op"])
        self.assertEqual(report["additional_draws"], 0)

    def test_duplicate_real_ids_and_missing_classes_are_rejected(self):
        original = fixture_rows((1, 1))
        original[1]["id"] = original[0]["id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            baselines.build_random_oversampling(original, 42, num_labels=2)
        with self.assertRaisesRegex(ValueError, "every official label"):
            baselines.build_random_oversampling(fixture_rows((1,)), 42, num_labels=2)

    @patch("src.baselines.eda_synonyms", side_effect=lambda word: LEXICON.get(word, ()))
    def test_eda_repeat_provenance_and_training_only_dedup(self, unused):
        original = fixture_rows((2, 2))
        state = random.getstate()
        rows, report = baselines.build_eda_baseline(original, "low_resource", 42)
        repeated, repeated_report = baselines.build_eda_baseline(original, "low_resource", 42)
        self.assertEqual((rows, report), (repeated, repeated_report))
        self.assertEqual(random.getstate(), state)
        self.assertEqual(rows[:len(original)], original)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertEqual(len({baselines.eda_normalize(row["text"]) for row in rows}), len(rows))
        source = {row["id"]: row for row in original}
        for row in rows[len(original):]:
            parent = source[row["parent_id"]]
            self.assertEqual(row["label"], parent["label"])
            self.assertEqual(row["extra_metadata"], parent["extra_metadata"])
            self.assertEqual(row["source"], "eda")
            self.assertEqual(row["augmentation_id"], row["id"])
            self.assertNotEqual(row["text"], parent["text"])
        self.assertEqual(report["accepted_augmented_rows"], len(rows) - len(original))
        self.assertLessEqual(sum(report["attempts_by_method"].values()), len(original) * 16)

    @patch("src.baselines.eda_synonyms", side_effect=lambda word: LEXICON.get(word, ()))
    def test_every_eda_operator_keeps_negation_and_numeric_tokens(self, unused):
        text = "My card isn't working for $123.45 on 12/03/2026 with account 123-456."
        for method in baselines.EDA_METHODS:
            changed = baselines.eda_variant(text, method, random.Random(42))
            tokens = baselines.eda_tokens(changed)
            for protected in ("isn't", "123.45", "12/03/2026", "123-456"):
                self.assertIn(protected, tokens)
        for method in baselines.EDA_METHODS[:3]:
            self.assertEqual(baselines.eda_variant(text, method, random.Random(42), alpha=0), text)

    def test_shortfall_is_recorded_instead_of_forcing_duplicates(self):
        original = [{"id": "a", "text": "not 123", "label": 0},
                    {"id": "b", "text": "never 456", "label": 1}]
        rows, report = baselines.build_eda_baseline(original, "low_resource", 42)
        self.assertEqual(rows, original)
        self.assertEqual(report["accepted_augmented_rows"], 0)
        self.assertEqual(report["shortfall"], 2)
        self.assertEqual(sum(report["attempts_by_method"].values()), 32)
        self.assertEqual(report["per_class"]["0"], {"requested": 1, "accepted": 0, "shortfall": 1})


class BaselineExportsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = fixture_config()
        self.datasets = release_fixture(self.root)
        self.synonyms = patch("src.baselines.eda_synonyms", side_effect=lambda word: LEXICON.get(word, ()))
        self.corpus = patch("src.baselines.wordnet_metadata", return_value={"wordnet_version": "fixture", "nltk_version": "fixture"})
        self.synonyms.start()
        self.corpus.start()
        self.addCleanup(self.synonyms.stop)
        self.addCleanup(self.corpus.stop)
        self.addCleanup(self.directory.cleanup)

    def test_exports_reuse_and_validation_test_are_not_loaded(self):
        (self.root / "01_data" / "validation.jsonl").write_text("not valid JSON")
        (self.root / "01_data" / "test.jsonl").write_text("not valid JSON")
        reports = baselines.prepare_baseline_data(self.config, self.root)
        self.assertEqual(len(reports), 8)
        file_states = {path: (file_sha256(path), path.stat().st_mtime_ns)
                       for path in (self.root / "01_data" / "baselines").rglob("*") if path.is_file()}
        repeated = baselines.prepare_baseline_data(self.config, self.root)
        self.assertEqual(repeated, reports)
        self.assertEqual(file_states, {path: (file_sha256(path), path.stat().st_mtime_ns) for path in file_states})
        for report in reports:
            output = self.root / report["data_path"]
            self.assertEqual(file_sha256(output), report["file_sha256"])
            self.assertEqual(rows_digest(read_jsonl(output)), report["rows_sha256"])
            if report["condition"] == "B2" and report["setting"] == "low_resource":
                self.assertTrue(report["b1_identical_inputs"])

    def test_balanced_export_preserves_even_noncanonical_file_bytes(self):
        source = self.root / "01_data/low_resource/seed_42.jsonl"
        rows = read_jsonl(source)
        source.write_text("\n".join(json.dumps(row, ensure_ascii=True, separators=(", ", ":  ")) for row in rows) + "\n\n")
        manifest_path = self.root / "01_data/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["datasets"]["low_resource/seed_42.jsonl"]["file_sha256"] = file_sha256(source)
        write_json(manifest_path, manifest)
        reports = baselines.prepare_baseline_data(self.config, self.root, ["low_resource"], [42], ["B2"])
        self.assertEqual((self.root / reports[0]["data_path"]).read_bytes(), source.read_bytes())

    def test_config_and_content_conflicts_refuse_to_overwrite(self):
        reports = baselines.prepare_baseline_data(self.config, self.root, ["low_resource"], [42], ["B3"])
        output = self.root / reports[0]["data_path"]
        saved = output.read_bytes()
        changed = copy.deepcopy(self.config)
        changed["baselines"]["eda_alpha"] = 0.2
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            baselines.prepare_baseline_data(changed, self.root, ["low_resource"], [42], ["B3"])
        self.assertEqual(output.read_bytes(), saved)
        output.write_bytes(saved + b"\n")
        with self.assertRaisesRegex(ValueError, "content differs"):
            baselines.prepare_baseline_data(self.config, self.root, ["low_resource"], [42], ["B3"])
        self.assertEqual(output.read_bytes(), saved + b"\n")

    def test_training_lr_changes_do_not_invalidate_dataset_cache(self):
        reports = baselines.prepare_baseline_data(self.config, self.root, ["low_resource"], [42])
        changed = copy.deepcopy(self.config)
        changed["training"] = {"learning_rate": 0.0001}
        changed["status"] = "frozen"
        self.assertEqual(baselines.prepare_baseline_data(changed, self.root, ["low_resource"], [42]), reports)

    def test_source_change_is_rejected_against_official_manifest(self):
        source = self.root / "01_data/low_resource/seed_42.jsonl"
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Official training file hash differs"):
            baselines.prepare_baseline_data(self.config, self.root, ["low_resource"], [42], ["B2"])
        self.assertFalse((self.root / "01_data/baselines").exists())


if __name__ == "__main__":
    unittest.main()
