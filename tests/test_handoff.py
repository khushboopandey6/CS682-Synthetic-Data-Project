"""The transfer package must never legitimize stale or tampered source artifacts."""

import contextlib
import hashlib
import io
import json
import random
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts.package_data_release import package_data_release, verify_data_release
from src.baselines import prepare_baseline_data
from src.project_config import file_sha256, rows_digest, write_json, write_jsonl


class HandoffTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="cs682-handoff-test-")
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.config = {
            "schema_version": 1, "protocol_id": "fixture_v1", "seeds": [42],
            "dataset": {"dataset_id": "fixture", "dataset_revision": "fixture-v1", "num_classes": 2},
            "settings": {"low_resource": {}, "imbalanced": {}},
            "baselines": {"B2": "random_oversampling", "B3": "easy_data_augmentation", "eda_alpha": 0.1,
                          "eda_deletion_probability": 0.1, "eda_augmentations_per_real": 1, "eda_attempts_per_example": 16},
        }
        real = [{"id": f"train:{i}", "text": f"My card is working for {i} dollars.", "label": i // 2} for i in range(4)]
        datasets = {"low_resource/seed_42.jsonl": real, "imbalanced/seed_42.jsonl": [real[0], *real[2:]],
                    "validation.jsonl": [{"id": "train:4", "text": "Validation question.", "label": 0}],
                    "test.jsonl": [{"id": "test:0", "text": "Test question.", "label": 0}]}
        descriptions = {}
        for name, rows in datasets.items():
            path = self.base / "01_data" / name
            write_jsonl(path, rows)
            descriptions[name] = {"rows": len(rows), "file_sha256": file_sha256(path), "rows_sha256": rows_digest(rows),
                                  "per_class": {str(k): v for k, v in sorted(Counter(r["label"] for r in rows).items())},
                                  "ids": [r["id"] for r in rows]}
        metadata = {"label_mapping.json": {"0": "intent_a", "1": "intent_b"},
                    "rare_classes.json": {"selection_seed": 42, "rare_class_ids": random.Random(42).sample([0, 1], 1)},
                    "dataset_statistics.json": {"datasets": descriptions}}
        for name, value in metadata.items():
            write_json(self.base / "01_data" / name, value)
        manifest = {"schema_version": 1, "protocol_id": self.config["protocol_id"],
                    "config": {key: self.config[key] for key in ("dataset", "settings", "seeds")},
                    "source": {"dataset_id": "fixture", "revision": "fixture-v1"}, "datasets": descriptions,
                    "metadata_files": {name: file_sha256(self.base / "01_data" / name) for name in metadata}}
        write_json(self.base / "01_data/manifest.json", manifest)
        with patch("src.baselines.wordnet_metadata", return_value={"wordnet_version": "fixture", "nltk_version": "fixture"}), \
             patch("src.baselines.eda_synonyms", return_value=("payment card",)):
            prepare_baseline_data(self.config, self.base)

    def package(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return package_data_release(self.config, self.base)

    def assert_rejected_without_package(self):
        with self.assertRaises(ValueError):
            self.package()
        self.assertFalse((self.base / "handoff").exists())

    def test_archive_members_match_inventory_and_valid_reuse_is_unchanged(self):
        manifest = verify_data_release(self.base)
        self.assertEqual(manifest["protocol_id"], "fixture_v1")
        self.assertFalse((self.base / "handoff").exists())
        inventory = self.package()
        archive = self.base / "handoff" / inventory["archive_name"]
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(set(bundle.namelist()), {row["path"] for row in inventory["files"]})
            for row in inventory["files"]:
                payload = bundle.read(row["path"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), row["sha256"])
                self.assertEqual(len(payload), row["bytes"])
        before = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in self.base.rglob("*") if p.is_file()}
        self.assertEqual(self.package(), inventory)
        self.assertEqual(before, {p: (file_sha256(p), p.stat().st_mtime_ns) for p in self.base.rglob("*") if p.is_file()})

    def test_tampered_official_data_or_metadata_rejected_before_zip_creation(self):
        for relative in ("low_resource/seed_42.jsonl", "label_mapping.json"):
            with self.subTest(relative=relative):
                path = self.base / "01_data" / relative
                previous = path.read_bytes()
                path.write_bytes(previous + b" ")
                self.assert_rejected_without_package()
                path.write_bytes(previous)

    def test_tampered_baseline_output_or_source_hash_rejected_before_zip_creation(self):
        output = self.base / "01_data/baselines/B3/imbalanced/seed_42.jsonl"
        previous = output.read_bytes()
        output.write_bytes(previous.replace(b"working", b"corrupted", 1))
        self.assert_rejected_without_package()
        output.write_bytes(previous)
        report_path = self.base / "01_data/baselines/reports/B3/imbalanced/seed_42.json"
        report = json.loads(report_path.read_text())
        report["source_file_sha256"] = "0" * 64
        write_json(report_path, report)
        self.assert_rejected_without_package()

    def test_stale_configuration_hash_or_missing_report_rejected(self):
        path = self.base / "01_data/baselines/reports/B3/low_resource/seed_42.json"
        previous = path.read_bytes()
        report = json.loads(previous)
        report["specification"]["parameters"]["alpha"] = 0.2
        write_json(path, report)
        self.assert_rejected_without_package()
        path.write_bytes(previous)
        path.unlink()
        self.assert_rejected_without_package()

    def test_extra_unverified_file_is_not_silently_packaged(self):
        (self.base / "01_data" / "unverified.txt").write_text("not in the release")
        self.assert_rejected_without_package()

    def test_existing_archive_does_not_bypass_source_verification(self):
        inventory = self.package()
        archive = self.base / "handoff" / inventory["archive_name"]
        before = (archive.read_bytes(), archive.stat().st_mtime_ns)
        source = self.base / "01_data/low_resource/seed_42.jsonl"
        source.write_bytes(source.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.package()
        self.assertEqual(before, (archive.read_bytes(), archive.stat().st_mtime_ns))


if __name__ == "__main__":
    unittest.main()
