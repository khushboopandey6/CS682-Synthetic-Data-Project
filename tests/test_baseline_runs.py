import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.baseline_runs import run_baselines
from src.project_config import write_json, write_jsonl


class BaselineExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base = self.root / "artifacts"
        self.config = {"status": "frozen", "protocol_id": "fixture", "dataset": {},
                       "training": {"learning_rate": 1e-4}, "settings": {"low_resource": {}},
                       "seeds": [42], "baselines": {}}
        real = [{"id": "train:0", "text": "card help", "label": 0},
                {"id": "train:1", "text": "transfer help", "label": 1}]
        write_jsonl(self.base / "01_data/low_resource/seed_42.jsonl", real)
        write_jsonl(self.base / "01_data/baselines/B2/low_resource/seed_42.jsonl", real)
        write_jsonl(self.base / "01_data/baselines/B3/low_resource/seed_42.jsonl", real)
        write_json(self.base / "01_data/label_mapping.json", {"0": "card", "1": "transfer"})
        write_json(self.base / "01_data/rare_classes.json", {"rare_class_ids": [0]})
        self.new_fits = set()

    def fake_trainer(self, train, validation, condition, setting, seed, config, **kwargs):
        folder = self.base / "04_checkpoints" / condition / setting / f"seed_{seed}"
        self.new_fits.add((condition, setting, seed))
        write_jsonl(folder / "validation_predictions.jsonl", [
            {"id": "validation:0", "true_label": 0, "predicted_label": 0, "confidence": 0.8,
             "condition": condition, "setting": setting, "seed": seed, "split": "validation"}])
        (folder / "stage_log.csv").write_text("stage,wall_seconds\ntraining,1\n")
        return {"condition": condition, "setting": setting, "seed": seed,
                "macro_f1": 0.8, "accuracy": 0.8, "real_count": 2, "synthetic_count": 0,
                "training_rows": 2, "best_epoch": 1, "epochs_run": 1, "optimizer_steps": 1,
                "training_seconds": 1.0, "total_seconds": 2.0, "device": "cpu"}

    def test_balanced_identity_has_b2_predictions_and_no_extra_fit(self):
        with patch("src.baseline_runs.load_config", return_value=self.config), \
             patch("src.baseline_runs.PROJECT_ROOT", self.root), \
             patch("src.training.train_classifier", side_effect=self.fake_trainer):
            rows = run_baselines(base_path=self.base, execute=True)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(self.new_fits), 2)
        b2 = next(row for row in rows if row["condition"] == "B2")
        self.assertTrue(b2["training_reused"])
        self.assertEqual(b2["additional_training_seconds"], 0)
        predictions = self.base / "05_results/validation_predictions/B2/low_resource/seed_42.jsonl"
        saved = json.loads(predictions.read_text().strip())
        self.assertEqual(saved["condition"], "B2")
        self.assertEqual(saved["reused_from_condition"], "B1")

    def test_main_matrix_refuses_unfrozen_policy(self):
        config = copy.deepcopy(self.config)
        config["status"] = "classifier_pilot_pending"
        with patch("src.baseline_runs.load_config", return_value=config):
            with self.assertRaisesRegex(ValueError, "freeze"):
                run_baselines(base_path=self.base, execute=True)


if __name__ == "__main__":
    unittest.main()
