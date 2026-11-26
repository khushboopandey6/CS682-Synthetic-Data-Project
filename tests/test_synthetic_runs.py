"""F0/F1/F2 run gates: matched-data contract, equal per-class counts and shared-trainer calls."""

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src import synthetic_runs
from src.project_config import file_sha256, read_jsonl, rows_digest, write_json, write_jsonl


def _describe(path, rows):
    return {"file_sha256": file_sha256(path), "rows_sha256": rows_digest(rows), "rows": len(rows),
            "ids": [row["id"] for row in rows],
            "per_class": {str(label): count for label, count in sorted(Counter(row["label"] for row in rows).items())}}


class SyntheticRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base = self.root / "artifacts"
        self.real = [{"id": "train:0", "text": "card arrived late", "label": 0},
                     {"id": "train:1", "text": "card not here", "label": 0},
                     {"id": "train:2", "text": "send money abroad", "label": 1},
                     {"id": "train:3", "text": "transfer failed", "label": 1}]
        self.validation = [{"id": "train:10", "text": "where is my card", "label": 0},
                           {"id": "train:11", "text": "transfer pending", "label": 1}]
        real_path = self.base / "01_data/low_resource/seed_42.jsonl"
        validation_path = self.base / "01_data/validation.jsonl"
        write_jsonl(real_path, self.real)
        write_jsonl(validation_path, self.validation)
        write_json(self.base / "01_data/label_mapping.json", {"0": "card_arrival", "1": "transfer"})
        write_json(self.base / "01_data/rare_classes.json", {"rare_class_ids": [0]})
        write_json(self.base / "01_data/manifest.json", {
            "protocol_id": "fixture",
            "datasets": {"low_resource/seed_42.jsonl": _describe(real_path, self.real),
                         "validation.jsonl": _describe(validation_path, self.validation)}})
        self.config = {
            "schema_version": 1, "protocol_id": "fixture", "status": "frozen", "paths": {"base_path": "unused"},
            "dataset": {"num_classes": 2}, "seeds": [42], "settings": {"low_resource": {}}, "baselines": {},
            "augmentation": {"primary_ratio": 1.0},
            "training": {"learning_rate": 1e-4},
            "classifier_pilot": {"selection": {"data_manifest_sha256": file_sha256(self.base / "01_data/manifest.json")}},
        }
        for condition in ("F0", "F1", "F2"):
            self.write_matched(condition, [self.synthetic(condition, 0, 0), self.synthetic(condition, 1, 1)])
        self.calls = []

    def synthetic(self, condition, index, label, **overrides):
        row = {"id": f"synthetic_low_resource_42_{label}_{condition}_{index}", "text": f"generated {condition} {index}",
               "label": label, "source": "qwen", "setting": "low_resource", "seed": 42}
        row.update(overrides)
        return row

    def write_matched(self, condition, rows):
        write_jsonl(synthetic_runs.matched_path(self.base, condition, "low_resource", 42), rows)

    def fake_trainer(self, train, validation, condition, setting, seed, config, **kwargs):
        self.calls.append((condition, list(train), list(validation)))
        folder = self.base / "04_checkpoints" / condition / setting / f"seed_{seed}"
        write_jsonl(folder / "validation_predictions.jsonl", [{"id": "train:10", "true_label": 0, "predicted_label": 0,
                                                                "confidence": 0.9, "split": "validation"}])
        (folder / "stage_log.csv").write_text("stage,wall_seconds\ntraining,1\n")
        synthetic = sum(1 for row in train if row.get("source") == "qwen")
        return {"status": "completed", "fingerprint": "f", "condition": condition, "setting": setting, "seed": seed,
                "macro_f1": 0.7, "accuracy": 0.7, "real_count": len(train) - synthetic, "synthetic_count": synthetic,
                "training_rows": len(train), "best_epoch": 1, "epochs_run": 1, "optimizer_steps": 1,
                "training_seconds": 1.0, "total_seconds": 2.0, "device": "cpu"}

    def run_f(self, **kwargs):
        with patch("src.synthetic_runs.load_config", return_value=self.config), \
             patch("src.synthetic_runs.PROJECT_ROOT", self.root), \
             patch("src.training.train_classifier", side_effect=self.fake_trainer):
            return synthetic_runs.run_synthetic(base_path=self.base, execute=True, **kwargs)

    def test_trains_real_plus_matched_rows_and_records_counts(self):
        results = self.run_f(conditions=["F0", "F2"])
        self.assertEqual([call[0] for call in self.calls], ["F0", "F2"])
        condition, train, validation = self.calls[0]
        self.assertEqual(train[:4], self.real)
        self.assertEqual([row["id"] for row in train[4:]], [row["id"] for row in read_jsonl(
            synthetic_runs.matched_path(self.base, "F0", "low_resource", 42))])
        self.assertEqual(validation, self.validation)
        self.assertEqual(results[0]["matched_counts_per_class"], {"0": 1, "1": 1})
        self.assertEqual(results[0]["actual_augmentation_ratio"], 0.5)
        self.assertEqual(results[0]["classes_below_target"], 2)
        saved = json.loads((self.base / "05_results/metrics/F2/low_resource/seed_42.json").read_text())
        self.assertEqual(saved["condition"], "F2")
        self.assertTrue((self.base / "05_results/validation_predictions/F2/low_resource/seed_42.jsonl").is_file())
        self.assertTrue((self.base / "06_cost_logs/training/F0/low_resource/seed_42.csv").is_file())
        summary = (self.root / "results/metrics/fixture/synthetic_validation_results.csv").read_text()
        self.assertEqual(summary.count("\n"), 3)

    def test_preview_does_not_train(self):
        with patch("src.synthetic_runs.load_config", return_value=self.config), \
             patch("src.training.train_classifier", side_effect=self.fake_trainer):
            self.assertEqual(synthetic_runs.run_synthetic(base_path=self.base), [])
        self.assertEqual(self.calls, [])

    def test_requires_all_three_matched_files(self):
        synthetic_runs.matched_path(self.base, "F1", "low_resource", 42).unlink()
        with self.assertRaisesRegex(ValueError, "Missing matched synthetic data"):
            self.run_f(conditions=["F0"])
        self.assertEqual(self.calls, [])

    def test_rejects_unequal_per_class_counts(self):
        self.write_matched("F2", [self.synthetic("F2", 0, 0)])
        with self.assertRaisesRegex(ValueError, "per-class counts differ"):
            self.run_f(conditions=["F0"])

    def test_rejects_counts_above_augmentation_target(self):
        rows = [self.synthetic("F0", index, 0) for index in range(3)]
        for condition in ("F0", "F1", "F2"):
            self.write_matched(condition, [{**row, "id": row["id"] + condition} for row in rows])
        with self.assertRaisesRegex(ValueError, "exceed the 1.0x augmentation target"):
            self.run_f(conditions=["F0"])

    def test_rejects_contract_violations(self):
        cases = {
            "belongs to": self.synthetic("F0", 0, 0, setting="imbalanced"),
            "must be generated": self.synthetic("F0", 0, 0, source="real"),
            "collides": self.synthetic("F0", 0, 0, id="train:10"),
            "Invalid synthetic label": self.synthetic("F0", 0, 5),
        }
        for message, row in cases.items():
            with self.subTest(message=message):
                self.write_matched("F0", [row, self.synthetic("F0", 1, 1)])
                with self.assertRaisesRegex(ValueError, message):
                    self.run_f(conditions=["F0"])

    def test_rejects_unreleased_real_data(self):
        write_jsonl(self.base / "01_data/low_resource/seed_42.jsonl", self.real[:3] + [
            {**self.real[3], "text": "edited"}])
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.run_f(conditions=["F0"])

    def test_refuses_conditions_outside_f(self):
        with patch("src.synthetic_runs.load_config", return_value=self.config):
            with self.assertRaisesRegex(ValueError, "only F0, F1 and F2"):
                synthetic_runs.run_synthetic(base_path=self.base, conditions=["B1"])

    def test_test_reporting_requires_validation_results_first(self):
        config = copy.deepcopy(self.config)
        with patch("src.synthetic_runs.load_config", return_value=config), \
             patch("src.inference._official_test") as official_test:
            with self.assertRaisesRegex(ValueError, "train before test reporting"):
                synthetic_runs.evaluate_synthetic_test(base_path=self.base, execute=True, conditions=["F0"])
        official_test.assert_not_called()

    def build_checkpoint(self, condition="F0"):
        """Write a checkpoint folder that looks exactly like a completed shared-trainer run."""
        from src import inference, training
        config = self.full_config()
        real, validation = self.real, self.validation
        synthetic = read_jsonl(synthetic_runs.matched_path(self.base, condition, "low_resource", 42))
        specification = {
            "mode": "main", "condition": condition, "setting": "low_resource", "seed": 42, "protocol_id": "fixture",
            "training": training._effective_training(config, condition, "low_resource", 42, None)[0],
            "trainer_source_sha256": file_sha256(Path(training.__file__)),
            "training_rows_sha256": rows_digest(real + synthetic), "validation_rows_sha256": rows_digest(validation),
        }
        fingerprint = rows_digest([specification])
        folder = self.base / "04_checkpoints" / condition / "low_resource/seed_42"
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            (folder / "best_model").mkdir(parents=True, exist_ok=True)
            (folder / "best_model" / name).write_text(name)
        write_json(folder / "run_config.json", {"fingerprint": fingerprint, "specification": specification})
        result = {"status": "completed", "fingerprint": fingerprint, "condition": condition, "setting": "low_resource",
                  "seed": 42, "best_epoch": 1, "epochs_run": 1, "real_count": 4, "synthetic_count": len(synthetic),
                  "training_rows": 4 + len(synthetic)}
        write_json(folder / "results.json", result)
        write_json(self.base / "05_results/metrics" / condition / "low_resource/seed_42.json", {
            **result, "shared_training_policy_sha256": inference._shared_policy(config, self.base),
            "matched_data_sha256": file_sha256(synthetic_runs.matched_path(self.base, condition, "low_resource", 42))})
        return config

    def full_config(self):
        config = copy.deepcopy(self.config)
        config["training"] = {
            "model": "fixture-model", "model_revision": "pinned", "num_labels": 2, "full_fine_tuning": True,
            "optimizer": "adamw", "learning_rate": 1e-4, "batch_size": 2, "eval_batch_size": 2, "max_epochs": 2,
            "validation_interval_epochs": 1, "early_stopping_patience": 1, "max_length": 8, "weight_decay": 0.01,
            "lr_scheduler": "linear", "warmup_ratio": 0.1, "max_grad_norm": 1.0, "precision": "float32",
            "device": "cpu", "checkpoint_metric": "macro_f1", "restore_best_checkpoint": True}
        return config

    def evaluate(self, config):
        import numpy as np
        test_rows = [{"id": "test:0", "text": "test zero", "label": 0}, {"id": "test:1", "text": "test one", "label": 1}]

        def fake_evaluator(entry, rows, config, base):
            probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8]])
            metrics = {"macro_f1": 1.0, "accuracy": 1.0, "rare_macro_f1": 1.0, "rare_macro_recall": 1.0, "test_loss": 0.1}
            per_class = [{"label": 0, "f1": 1.0}, {"label": 1, "f1": 1.0}]
            timing = {"inference_seconds": 0.1, "total_seconds": 0.2, "nonpadding_tokens": 4,
                      "start_time": None, "end_time": None, "runtime": {"device": "cpu"}}
            return metrics, np.asarray([0, 1]), np.asarray([0, 1]), probabilities, per_class, timing

        with patch("src.synthetic_runs.load_config", return_value=config), \
             patch("src.synthetic_runs.PROJECT_ROOT", self.root), \
             patch("src.inference.PROJECT_ROOT", self.root), \
             patch("src.inference._official_test", return_value=(test_rows, "test-sha", "manifest-sha")), \
             patch("src.inference._evaluate_checkpoint", side_effect=fake_evaluator):
            return synthetic_runs.evaluate_synthetic_test(base_path=self.base, execute=True, conditions=["F0"])

    def test_test_reporting_exports_predictions_for_verified_checkpoint(self):
        results = self.evaluate(self.build_checkpoint())
        self.assertEqual(results[0]["condition"], "F0")
        self.assertFalse(results[0]["test_used_for_selection"])
        predictions = read_jsonl(self.base / "05_results/predictions/F0/low_resource/seed_42.jsonl")
        self.assertEqual([row["condition"] for row in predictions], ["F0", "F0"])
        self.assertTrue((self.base / "05_results/metrics/synthetic_test_results.csv").is_file())

    def test_test_reporting_rejects_matched_data_changed_after_training(self):
        config = self.build_checkpoint()
        self.write_matched("F0", [self.synthetic("F0", 7, 0), self.synthetic("F0", 8, 1)])
        with self.assertRaisesRegex(ValueError, "Matched data changed after training"):
            self.evaluate(config)


if __name__ == "__main__":
    unittest.main()
