"""Checkpoint handoffs must distinguish pending fits from verified B1 models."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from scripts.export_baseline_handoff import export_baseline_handoff
from src.project_config import PROJECT_ROOT, file_sha256, read_jsonl, rows_digest, write_json, write_jsonl


class BaselineHandoffTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="cs682-b1-handoff-")
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.config_path = self.base / "experiment.yaml"
        self.reports = self.base / "git_reports"
        self.config = {
            "schema_version": 1, "protocol_id": "fixture_v1", "status": "frozen", "seeds": [42, 123, 456],
            "paths": {"base_path": str(self.base)}, "dataset": {"num_classes": 2},
            "settings": {"low_resource": {}, "imbalanced": {}},
            "training": {"model": "fixture", "model_revision": "pinned-fixture", "learning_rate": 1e-4,
                         "num_labels": 2, "precision": "float32"},
        }
        self.config_path.write_text(yaml.safe_dump(self.config))
        descriptions = {}
        datasets = {"validation.jsonl": [{"id": "val0", "text": "Validation zero", "label": 0},
                                          {"id": "val1", "text": "Validation one", "label": 1}]}
        for setting in self.config["settings"]:
            for seed in self.config["seeds"]:
                datasets[f"{setting}/seed_{seed}.jsonl"] = [
                    {"id": f"{setting}:{seed}:{label}", "text": f"Training {setting} {seed} {label}", "label": label}
                    for label in range(2)]
        for name, rows in datasets.items():
            path = self.base / "01_data" / name
            write_jsonl(path, rows)
            descriptions[name] = {"rows": len(rows), "file_sha256": file_sha256(path),
                                  "rows_sha256": rows_digest(rows), "ids": [row["id"] for row in rows]}
        metadata = {"label_mapping.json": {"0": "zero", "1": "one"}, "rare_classes.json": {"rare_class_ids": [1]}}
        for name, value in metadata.items():
            write_json(self.base / "01_data" / name, value)
        write_json(self.base / "01_data/manifest.json", {
            "protocol_id": self.config["protocol_id"], "config": {key: self.config[key] for key in ("dataset", "settings", "seeds")},
            "rare_class_ids": [1],
            "datasets": descriptions, "metadata_files": {name: file_sha256(self.base / "01_data" / name) for name in metadata}})
        self.config["classifier_pilot"] = {"selection": {"data_manifest_sha256": file_sha256(self.base / "01_data/manifest.json")}}
        self.config_path.write_text(yaml.safe_dump(self.config))

    def export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return export_baseline_handoff(self.config_path, self.base, self.reports)

    def create_run(self, setting="low_resource", seed=42):
        folder = self.base / "04_checkpoints/B1" / setting / f"seed_{seed}"
        train = read_jsonl(self.base / "01_data" / setting / f"seed_{seed}.jsonl")
        validation = read_jsonl(self.base / "01_data/validation.jsonl")
        source = file_sha256(PROJECT_ROOT / "src/training.py")
        spec = {"protocol_id": self.config["protocol_id"], "mode": "main", "condition": "B1", "setting": setting, "seed": seed,
                "training": copy.deepcopy(self.config["training"]), "training_rows_sha256": rows_digest(train),
                "validation_rows_sha256": rows_digest(validation), "rare_class_ids": [1], "class_names": ["zero", "one"],
                "trainer_source_sha256": source}
        fingerprint = rows_digest([spec])
        write_json(folder / "run_config.json", {"specification": spec, "fingerprint": fingerprint,
                                                "config_provenance_sha256": rows_digest([self.config])})
        write_json(folder / "results.json", {"status": "completed", "condition": "B1", "setting": setting, "seed": seed,
                                             "fingerprint": fingerprint, "trainer_source_sha256": source, "code_commit": "a" * 40,
                                             "real_count": 2, "training_rows": 2, "synthetic_count": 0,
                                             "macro_f1": 1.0, "accuracy": 1.0, "rare_macro_f1": 1.0, "rare_macro_recall": 1.0,
                                             "epochs_run": 1, "best_epoch": 1, "device": "cpu", "precision": "float32"})
        write_json(folder / "runtime.json", {"device": "cpu", "precision": "float32"})
        (folder / "history.csv").write_text("epoch,macro_f1\n1,1.0\n")
        (folder / "stage_log.csv").write_text("stage,wall_seconds\ntraining,1.0\n")
        (folder / "per_class_metrics.csv").write_text("label,f1\n0,1.0\n1,1.0\n")
        probabilities = np.asarray([[.9, .1], [.2, .8]], dtype=np.float32)
        predictions = [{"id": row["id"], "text": row["text"], "true_label": row["label"], "predicted_label": row["label"],
                        "confidence": float(probabilities[index].max()), "condition": "B1", "setting": setting,
                        "seed": seed, "split": "validation"} for index, row in enumerate(validation)]
        write_jsonl(folder / "validation_predictions.jsonl", predictions)
        np.savez_compressed(folder / "validation_probabilities.npz", probabilities=probabilities,
                            labels=np.asarray([0, 1]), ids=np.asarray(["val0", "val1"]))
        best = folder / "best_model"
        write_json(best / "config.json", {"id2label": {"0": "zero", "1": "one"}})
        write_json(best / "tokenizer_config.json", {})
        write_json(best / "tokenizer.json", {"fixture": True})
        (best / "model.safetensors").write_bytes(b"unit-test fixture weights")
        return folder

    def test_no_runs_exports_six_pending_without_creating_models(self):
        index = self.export()
        self.assertEqual((index["status"], index["completed_runs"], index["expected_runs"]), ("partial", 0, 6))
        self.assertEqual(len(index["missing_runs"]), 6)
        self.assertTrue(all(row["status"] == "pending" and row["reason"] == "not_started" for row in index["missing_runs"]))
        self.assertFalse((self.base / "04_checkpoints").exists())
        self.assertEqual(index, json.loads((self.reports / "b1_checkpoint_index.json").read_text()))

    def test_progressive_export_is_portable_and_complete_after_six_runs(self):
        folder = self.create_run()
        first = self.export()
        self.assertEqual((first["completed_runs"], len(first["missing_runs"])), (1, 5))
        files = first["checkpoints"][0]["files"]
        self.assertTrue(any(row["path"].endswith("model.safetensors") for row in files))
        for row in files:
            self.assertFalse(Path(row["path"]).is_absolute())
            self.assertEqual(row["sha256"], file_sha256(self.base / row["path"]))
            self.assertEqual(row["bytes"], (self.base / row["path"]).stat().st_size)
        local = self.base / "handoff/fixture_v1_b1_checkpoints.json"
        before = (local.read_bytes(), local.stat().st_mtime_ns)
        self.assertEqual(self.export(), first)
        self.assertEqual((local.read_bytes(), local.stat().st_mtime_ns), before)
        for setting in self.config["settings"]:
            for seed in self.config["seeds"]:
                if setting != "low_resource" or seed != 42:
                    self.create_run(setting, seed)
        final = self.export()
        self.assertEqual((final["status"], final["completed_runs"], final["missing_runs"]), ("complete", 6, []))
        self.assertEqual(list((self.base / "handoff").iterdir()), [local])
        self.assertTrue((folder / "best_model/model.safetensors").exists())

    def test_confidence_and_wrong_subset_are_rejected_before_publishing(self):
        folder = self.create_run()
        predictions_path = folder / "validation_predictions.jsonl"
        predictions = read_jsonl(predictions_path)
        predictions[0]["confidence"] = .2
        write_jsonl(predictions_path, predictions)
        with self.assertRaisesRegex(ValueError, "confidence"):
            self.export()
        self.assertFalse((self.base / "handoff").exists())
        folder = self.create_run()
        run = json.loads((folder / "run_config.json").read_text())
        wrong_train = read_jsonl(self.base / "01_data/low_resource/seed_123.jsonl")
        run["specification"]["training_rows_sha256"] = rows_digest(wrong_train)
        run["fingerprint"] = rows_digest([run["specification"]])
        write_json(folder / "run_config.json", run)
        result = json.loads((folder / "results.json").read_text())
        result["fingerprint"] = run["fingerprint"]
        write_json(folder / "results.json", result)
        with self.assertRaisesRegex(ValueError, "official subset"):
            self.export()
        self.assertFalse((self.base / "handoff").exists())

    def test_verified_weights_cannot_silently_change_or_disappear(self):
        folder = self.create_run()
        self.export()
        local = self.base / "handoff/fixture_v1_b1_checkpoints.json"
        before = local.read_bytes()
        weights = folder / "best_model/model.safetensors"
        weights.write_bytes(b"changed fixture weights")
        with self.assertRaisesRegex(ValueError, "changed or disappeared"):
            self.export()
        self.assertEqual(local.read_bytes(), before)
        weights.unlink()
        with self.assertRaisesRegex(ValueError, "model or tokenizer"):
            self.export()
        self.assertEqual(local.read_bytes(), before)

    def test_unfrozen_configuration_and_tampered_data_are_rejected(self):
        self.config_path.write_text(yaml.safe_dump({**self.config, "status": "classifier_pilot_pending"}))
        with self.assertRaisesRegex(ValueError, "Freeze"):
            self.export()
        self.config_path.write_text(yaml.safe_dump(self.config))
        validation = self.base / "01_data/validation.jsonl"
        validation.write_text(validation.read_text() + " \n")
        with self.assertRaisesRegex(ValueError, "Official data hash"):
            self.export()
        self.assertFalse((self.base / "handoff").exists())

    def test_self_consistent_other_release_cannot_replace_frozen_handoff(self):
        self.export()
        local = self.base / "handoff/fixture_v1_b1_checkpoints.json"
        git_copy = self.reports / "b1_checkpoint_index.json"
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (local, git_copy)}
        training = self.base / "01_data/low_resource/seed_42.jsonl"
        rows = read_jsonl(training)
        rows[0]["text"] = "A different but internally consistent official release"
        write_jsonl(training, rows)
        manifest_path = self.base / "01_data/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["datasets"]["low_resource/seed_42.jsonl"].update(
            file_sha256=file_sha256(training), rows_sha256=rows_digest(rows))
        write_json(manifest_path, manifest)
        with patch("scripts.export_baseline_handoff.read_jsonl") as reader:
            with self.assertRaisesRegex(ValueError, "frozen pilot release"):
                self.export()
            reader.assert_not_called()
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (local, git_copy)})

    def test_rare_metadata_must_match_the_pinned_manifest_ids(self):
        manifest_path = self.base / "01_data/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rare_class_ids"] = [0]
        write_json(manifest_path, manifest)
        self.config["classifier_pilot"]["selection"]["data_manifest_sha256"] = file_sha256(manifest_path)
        self.config_path.write_text(yaml.safe_dump(self.config))
        with self.assertRaisesRegex(ValueError, "rare-class IDs"):
            self.export()
        self.assertFalse((self.base / "handoff").exists())


if __name__ == "__main__":
    unittest.main()
