"""Contract and CPU-only checks for the shared trainer; no model downloads."""

import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import training
from src.project_config import write_json


def fixture_config():
    return {
        "schema_version": 1, "protocol_id": "master_v1", "status": "frozen",
        "paths": {"base_path": "unused"}, "seeds": [42, 123, 456],
        "dataset": {"num_classes": 2}, "settings": {"low_resource": {}, "imbalanced": {}},
        "training": {
            "model": "tiny-local-fixture", "model_revision": "pinned-fixture", "num_labels": 2,
            "learning_rate": 0.01, "full_fine_tuning": True, "optimizer": "adamw",
            "precision": "float32", "device": "cpu", "lr_scheduler": "linear",
            "validation_interval_epochs": 1, "checkpoint_metric": "macro_f1", "restore_best_checkpoint": True,
            "batch_size": 2, "eval_batch_size": 2, "max_epochs": 2, "early_stopping_patience": 1,
            "max_length": 8, "weight_decay": 0.01, "warmup_ratio": 0.0, "max_grad_norm": 1.0,
        },
        "classifier_pilot": {
            "conditions": ["B1", "B3"], "settings": ["low_resource", "imbalanced"],
            "development_seed": 42, "learning_rate_candidates": [2e-5, 5e-5, 1e-4],
        },
    }


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        write_json(self.base / "01_data" / "rare_classes.json", {"rare_class_ids": [1]})
        self.config = fixture_config()
        self.train = [{"id": "train0", "text": "training zero", "label": 0},
                      {"id": "train1", "text": "training one", "label": 1}]
        self.validation = [{"id": "val0", "text": "validation zero", "label": 0},
                           {"id": "val1", "text": "validation one", "label": 1}]
        self.addCleanup(self.temporary.cleanup)

    def call(self, **options):
        return training.train_classifier(self.train, self.validation, "B1", "low_resource", 42,
                                         self.config, base_path=self.base, **options)

    def complete_fixture(self, *arguments):
        folder, fingerprint = arguments[7], arguments[10]
        result = {"status": "completed", "fingerprint": fingerprint, "macro_f1": 0.5}
        for name in ("runtime.json", "history.csv", "validation_predictions.jsonl", "validation_probabilities.npz",
                     "per_class_metrics.csv", "stage_log.csv"):
            (folder / name).write_text("fixture")
        best = folder / "best_model"
        best.mkdir()
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
            (best / name).write_text("fixture")
        write_json(folder / "results.json", result)
        return result

    def snapshot(self):
        return {str(path.relative_to(self.base)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.base.rglob("*") if path.is_file()}

    def test_main_requires_frozen_positive_rate_before_writing(self):
        self.config["status"] = "classifier_pilot_pending"
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.call()
        self.config["status"] = "frozen"
        self.config["training"]["learning_rate"] = None
        with self.assertRaisesRegex(ValueError, "positive"):
            self.call()
        self.assertFalse((self.base / "04_checkpoints").exists())

    def test_explicit_pilot_rate_can_reuse_after_config_freezes(self):
        self.config["status"] = "classifier_pilot_pending"
        self.config["training"]["learning_rate"] = None
        with patch.object(training, "_train_new_run", side_effect=self.complete_fixture) as train:
            first = self.call(pilot_learning_rate=2e-5)
            self.assertEqual(train.call_args.args[5]["learning_rate"], 2e-5)
            before = self.snapshot()
            self.config["status"] = "frozen"
            self.config["training"]["learning_rate"] = 1e-4
            second = self.call(pilot_learning_rate=2e-5)
            self.assertEqual(first, second)
            self.assertEqual(train.call_count, 1)
            self.assertEqual(before, self.snapshot())

    def test_pilot_rejects_undeclared_rate_condition_and_seed(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            self.call(pilot_learning_rate=0.02)
        with self.assertRaisesRegex(ValueError, "pilot"):
            training.train_classifier(self.train, self.validation, "F0", "low_resource", 42,
                                      self.config, base_path=self.base, pilot_learning_rate=2e-5)
        with self.assertRaisesRegex(ValueError, "development seed"):
            training.train_classifier(self.train, self.validation, "B1", "low_resource", 123,
                                      self.config, base_path=self.base, pilot_learning_rate=2e-5)

    def test_completed_config_or_data_conflict_does_not_modify_files(self):
        with patch.object(training, "_train_new_run", side_effect=self.complete_fixture) as train:
            self.call()
            before = self.snapshot()
            self.config["training"]["learning_rate"] *= 2
            with self.assertRaisesRegex(ValueError, "differs"):
                self.call()
            self.config["training"]["learning_rate"] /= 2
            self.train[0]["text"] += " changed"
            with self.assertRaisesRegex(ValueError, "differs"):
                self.call()
            self.assertEqual(before, self.snapshot())
            self.assertEqual(train.call_count, 1)

    def test_changed_trainer_source_refuses_cached_reuse_but_commit_only_change_does_not(self):
        with patch.object(training, "_train_new_run", side_effect=self.complete_fixture) as train:
            with patch.object(training, "_code_commit", return_value="a" * 40):
                self.call()
            saved = json.loads((self.base / "04_checkpoints/B1/low_resource/seed_42/run_config.json").read_text())
            self.assertEqual(saved["code_commit"], "a" * 40)
            self.assertEqual(saved["specification"]["trainer_source_sha256"],
                             training.file_sha256(Path(training.__file__)))
            before = self.snapshot()
            with patch.object(training, "_code_commit", return_value="b" * 40):
                self.call()
            with patch.object(training, "file_sha256", return_value="0" * 64):
                with self.assertRaisesRegex(ValueError, "differs"):
                    self.call()
            self.assertEqual(before, self.snapshot())
            self.assertEqual(train.call_count, 1)

    def test_missing_completed_artifact_is_not_retrained(self):
        with patch.object(training, "_train_new_run", side_effect=self.complete_fixture) as train:
            self.call()
            (self.base / "04_checkpoints" / "B1" / "low_resource" / "seed_42" / "validation_probabilities.npz").unlink()
            with self.assertRaisesRegex(ValueError, "missing artifacts"):
                self.call()
            self.assertEqual(train.call_count, 1)

    def test_validation_overlap_and_duplicate_training_ids_are_rejected(self):
        self.validation[0]["id"] = "train0"
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            self.call()
        self.validation[0]["id"] = "val0"
        self.validation[0]["text"] = " TRAINING   ZERO "
        with self.assertRaisesRegex(ValueError, "text overlap"):
            self.call()
        self.validation[0]["text"] = "validation zero"
        self.train += copy.deepcopy(self.train)
        with self.assertRaisesRegex(ValueError, "Only B2"):
            self.call()
        with patch.object(training, "_train_new_run", return_value={"ok": True}) as train:
            training.train_classifier(self.train, self.validation, "B2", "low_resource", 42,
                                      self.config, base_path=self.base)
            self.assertEqual(len(train.call_args.args[0]), 4)

    def test_test_input_is_rejected_without_reading_it(self):
        with patch.object(training, "read_jsonl") as reader:
            with self.assertRaisesRegex(ValueError, "test data"):
                training.train_classifier("test.jsonl", self.validation, "B1", "low_resource", 42,
                                          self.config, base_path=self.base)
            reader.assert_not_called()

    def test_master_source_contract_counts_real_eda_and_replacement_rows(self):
        eda = [{**row, "id": f"eda:{row['id']}", "text": row["text"] + " edited",
                "parent_id": row["id"], "source": "eda"} for row in self.train]
        counts = training._training_counts(self.train + eda)
        self.assertEqual((counts["real_count"], counts["synthetic_count"], counts["training_rows"]), (2, 2, 4))
        self.assertEqual(counts["additional_draws"], 0)
        replacement = {**self.train[0], "id": "oversample:train0:1", "parent_id": "train0", "source": "oversampled"}
        counts = training._training_counts(self.train + [replacement])
        self.assertEqual((counts["real_rows"], counts["synthetic_count"], counts["train_rows"]), (2, 0, 3))
        self.assertEqual(counts["additional_draws"], 1)
        generated = {"id": "qwen:1", "text": "a generated query", "label": 0, "source": "qwen"}
        counts = training._training_counts(self.train + [generated])
        self.assertEqual((counts["real_count"], counts["synthetic_count"]), (2, 1))
        counts = training._training_counts(self.train + copy.deepcopy(self.train))
        self.assertEqual((counts["real_count"], counts["additional_draws"]), (2, 2))

    def test_augmented_provenance_must_refer_to_selected_real_parents(self):
        wrong = {**self.train[0], "id": "oversample:wrong", "parent_id": "val0", "source": "oversampled"}
        with self.assertRaisesRegex(ValueError, "selected real parent"):
            training._training_counts(self.train + [wrong])

    def test_generated_validation_text_coincidence_is_recorded_without_filtering(self):
        generated = {"id": "eda:train0", "text": " VALIDATION   ZERO ", "label": 0,
                     "source": "eda", "parent_id": "train0"}
        rows = self.train + [generated]
        before = copy.deepcopy(rows)
        overlap = training._validate_rows(rows, self.validation, "B3", 2)
        self.assertEqual(rows, before)
        self.assertEqual(overlap["synthetic_validation_text_overlap_count"], 1)
        self.assertEqual(overlap["synthetic_validation_text_overlap_ids"], ["eda:train0"])
        self.assertEqual(overlap["synthetic_validation_text_overlap_pairs"],
                         [{"training_id": "eda:train0", "validation_ids": ["val0"]}])
        with patch.object(training, "_train_new_run", side_effect=self.complete_fixture):
            training.train_classifier(rows, self.validation, "B3", "low_resource", 42,
                                      self.config, base_path=self.base)
        specification = json.loads((self.base / "04_checkpoints/B3/low_resource/seed_42/run_config.json").read_text())
        self.assertEqual(specification["specification"]["validation_overlap"], overlap)

    def test_generated_id_and_parent_cannot_reference_validation(self):
        generated = {"id": "qwen:1", "text": "new generated query", "label": 0, "source": "qwen"}
        for field in ("id", "parent_id", "source_id"):
            wrong = {**generated, field: "val0"}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "validation"):
                training._validate_rows(self.train + [wrong], self.validation, "F0", 2)

    def test_oversampled_real_text_match_remains_forbidden(self):
        rows = [dict(row) for row in self.train]
        rows[0]["text"] = "validation zero"
        with self.assertRaisesRegex(ValueError, "Real training and validation text overlap"):
            training._validate_rows(rows, self.validation, "B1", 2)

    def test_device_preference_and_order(self):
        def fake(cuda, mps):
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                                   backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
                                   device=lambda name: name)
        self.assertEqual(training.select_device("auto", fake(True, True)), "cuda")
        self.assertEqual(training.select_device("auto", fake(False, True)), "mps")
        self.assertEqual(training.select_device("auto", fake(False, False)), "cpu")
        self.assertEqual(training.select_device("cpu", fake(True, True)), "cpu")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            training.select_device("mps", fake(False, False))

    def test_hardware_names_use_actual_accelerator_or_documented_fallback(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda device: "NVIDIA T4"))
        self.assertEqual(training._hardware_name(torch, SimpleNamespace(type="cuda")), "NVIDIA T4")
        with patch.object(training.platform, "system", return_value="Darwin"):
            with patch.object(training.subprocess, "check_output", return_value="Apple M2 Pro\n"):
                self.assertEqual(training._hardware_name(torch, SimpleNamespace(type="mps")), "Apple M2 Pro")
            with patch.object(training.subprocess, "check_output", side_effect=OSError), \
                 patch.object(training.platform, "processor", return_value=""), \
                 patch.object(training.platform, "machine", return_value="arm64"):
                self.assertEqual(training._hardware_name(torch, SimpleNamespace(type="mps")), "arm64")


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "sklearn", "transformers", "safetensors")),
                     "Optional CPU integration check needs the existing ML environment")
class TinyCPUTrainingTests(unittest.TestCase):
    setUp = TrainingContractTests.setUp
    call = TrainingContractTests.call

    def test_tiny_model_training_checkpoint_reload_and_earliest_tie(self):
        import torch
        from safetensors.torch import load_file, save_file
        from types import SimpleNamespace

        libraries = training._libraries()

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def __call__(self, texts, **kwargs):
                return {"input_ids": [[int(text.endswith("one"))] for text in texts],
                        "attention_mask": [[1] for text in texts]}

            def save_pretrained(self, folder):
                Path(folder).mkdir(exist_ok=True)
                write_json(Path(folder) / "tokenizer_config.json", {})
                write_json(Path(folder) / "tokenizer.json", {})

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 2)
                torch.nn.init.zeros_(self.linear.weight)
                torch.nn.init.zeros_(self.linear.bias)

            @classmethod
            def from_pretrained(cls, name, **kwargs):
                instance = cls()
                path = Path(name) / "model.safetensors"
                if path.exists():
                    instance.load_state_dict(load_file(str(path), device="cpu"))
                return instance

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.linear(input_ids.float()))

            def save_pretrained(self, folder, **kwargs):
                Path(folder).mkdir(exist_ok=True)
                save_file(self.state_dict(), str(Path(folder) / "model.safetensors"))
                write_json(Path(folder) / "config.json", {"num_labels": 2})

        class Collator:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, rows):
                return {key: torch.tensor([row[key] for row in rows], dtype=torch.long) for key in rows[0]}

        libraries.AutoTokenizer = Tokenizer
        libraries.AutoModelForSequenceClassification = Model
        libraries.DataCollatorWithPadding = Collator
        with patch.object(training, "_libraries", return_value=libraries), \
             patch.object(training, "_code_commit", return_value="a" * 40):
            result = self.call()
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["epochs_run"], 2)
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertEqual((result["real_count"], result["synthetic_count"], result["training_rows"]), (2, 0, 2))
            self.assertEqual(result["validation_overlap"]["synthetic_validation_text_overlap_count"], 0)
            self.assertEqual(result["macro_f1"], 1.0)
            self.assertEqual(result["code_commit"], "a" * 40)
            self.assertEqual(result["trainer_source_sha256"], training.file_sha256(Path(training.__file__)))
            self.assertTrue(result["hardware"])
            start, end = datetime.fromisoformat(result["start_time"]), datetime.fromisoformat(result["end_time"])
            self.assertIsNotNone(start.utcoffset())
            self.assertLessEqual(start, end)
            run_folder = Path(result["run_path"])
            runtime = json.loads((run_folder / "runtime.json").read_text())
            self.assertEqual(runtime["hardware"], result["hardware"])
            self.assertNotIn("start_time", runtime)
            with (run_folder / "stage_log.csv").open() as handle:
                stages = list(csv.DictReader(handle))
            self.assertTrue(all(row["code_commit"] == result["code_commit"] and
                                row["trainer_source_sha256"] == result["trainer_source_sha256"]
                                for row in stages))
            rows = training.read_jsonl(Path(result["run_path"]) / "validation_predictions.jsonl")
            self.assertEqual([row["id"] for row in rows], ["val0", "val1"])
            self.assertTrue(all(row["split"] == "validation" and row["condition"] == "B1" for row in rows))
            probabilities = libraries.np.load(Path(result["run_path"]) / "validation_probabilities.npz")
            self.assertEqual(probabilities["probabilities"].shape, (2, 2))
            self.assertTrue(libraries.np.allclose(probabilities["probabilities"].sum(axis=1), 1))


if __name__ == "__main__":
    unittest.main()
