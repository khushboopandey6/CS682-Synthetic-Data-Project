"""Reporting gates, checkpoint integrity, and CPU-only inference export checks."""

import copy
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import inference, training
from src.project_config import file_sha256, read_jsonl, rows_digest, write_json, write_jsonl


class BaselineInferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base = self.root / "artifacts"
        self.config = {
            "schema_version": 1, "protocol_id": "fixture", "status": "frozen",
            "paths": {"base_path": "unused"}, "dataset": {
                "dataset_id": "fixture/banking", "dataset_revision": "pinned-dataset", "num_classes": 2,
            }, "seeds": [42], "settings": {"low_resource": {}},
            "baselines": {"B2": "random_oversampling", "B3": "easy_data_augmentation", "eda_alpha": 0.1,
                          "eda_deletion_probability": 0.1, "eda_augmentations_per_real": 1,
                          "eda_attempts_per_example": 16},
            "training": {"learning_rate": 1e-4, "precision": "float32", "num_labels": 2,
                         "max_length": 8, "eval_batch_size": 2, "device": "cpu", "model_revision": "pinned-model"},
            "classifier_pilot": {"selection": {}},
        }
        self.rows = [{"id": "test:0", "text": "test zero", "label": 0},
                     {"id": "test:1", "text": "test one", "label": 1}]
        real = [{"id": "train:0", "text": "training zero", "label": 0},
                {"id": "train:1", "text": "training one", "label": 1}]
        write_jsonl(self.base / "01_data/low_resource/seed_42.jsonl", real)
        write_jsonl(self.base / "01_data/baselines/B2/low_resource/seed_42.jsonl", real)
        write_json(self.base / "01_data/label_mapping.json", {"0": "zero", "1": "one"})
        write_json(self.base / "01_data/rare_classes.json", {"rare_class_ids": [1]})
        validation = [{"id": "train:2", "text": "validation zero", "label": 0},
                      {"id": "train:3", "text": "validation one", "label": 1}]
        validation_path = self.base / "01_data/validation.jsonl"
        write_jsonl(validation_path, validation)
        eda = real + [{**row, "id": f"eda:{index}", "text": row["text"] + " edited",
                       "parent_id": row["id"], "source": "eda"} for index, row in enumerate(real)]
        eda_path = self.base / "01_data/baselines/B3/low_resource/seed_42.jsonl"
        write_jsonl(eda_path, eda)
        test_path = self.base / "01_data/test.jsonl"
        write_jsonl(test_path, self.rows)
        self.manifest = {"protocol_id": "fixture", "config": {
            key: copy.deepcopy(self.config[key]) for key in ("dataset", "settings", "seeds")}, "source": {
            "dataset_id": "fixture/banking", "revision": "pinned-dataset", "official_test_rows": 2,
        }, "rare_class_ids": [1], "metadata_files": {
            name: file_sha256(self.base / "01_data" / name) for name in ("label_mapping.json", "rare_classes.json")},
        "datasets": {"test.jsonl": self.describe(test_path, self.rows),
                     "validation.jsonl": self.describe(validation_path, validation),
                     "low_resource/seed_42.jsonl": self.describe(self.base / "01_data/low_resource/seed_42.jsonl", real)}}
        original = self.manifest["datasets"]["low_resource/seed_42.jsonl"]
        for condition, data in (("B2", real), ("B3", eda)):
            data_relative = f"01_data/baselines/{condition}/low_resource/seed_42.jsonl"
            report_relative = f"01_data/baselines/reports/{condition}/low_resource/seed_42.json"
            specification = {"protocol_id": "fixture", "condition": condition, "setting": "low_resource",
                             "seed": 42, "num_classes": 2, "dataset_source": {"id": "fixture/banking", "revision": "pinned-dataset"},
                             "source_path": "01_data/low_resource/seed_42.jsonl", "source_file_sha256": original["file_sha256"],
                             "source_rows_sha256": original["rows_sha256"], "method": self.config["baselines"][condition],
                             "parameters": inference._eda_parameters(self.config) if condition == "B3"
                             else {"target": "largest_selected_class_count"}}
            write_json(self.base / report_relative, {
                "status": "prepared", "condition": condition, "setting": "low_resource", "seed": 42,
                "specification": specification, "config_sha256": rows_digest([specification]),
                "source_file_sha256": original["file_sha256"], "source_rows_sha256": original["rows_sha256"],
                "data_path": data_relative, "report_path": report_relative,
                **self.describe(self.base / data_relative, data),
            })
        self.pin_manifest()
        policy = inference._shared_policy(self.config, self.base)
        for condition in ("B1", "B3"):
            folder = self.base / "04_checkpoints" / condition / "low_resource/seed_42"
            best = folder / "best_model"
            best.mkdir(parents=True)
            for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
                (best / name).write_text("fixture")
            specification = {"mode": "main", "condition": condition, "setting": "low_resource", "seed": 42,
                             "protocol_id": "fixture", "training": copy.deepcopy(self.config["training"]),
                             "training_rows_sha256": rows_digest(real if condition == "B1" else eda),
                             "validation_rows_sha256": rows_digest(validation), "rare_class_ids": [1],
                             "class_names": ["zero", "one"],
                             "trainer_source_sha256": file_sha256(Path(training.__file__))}
            fingerprint = rows_digest([specification])
            write_json(folder / "run_config.json", {"fingerprint": fingerprint, "specification": specification})
            result = {"status": "completed", "fingerprint": fingerprint, "condition": condition,
                      "setting": "low_resource", "seed": 42, "best_epoch": 1, "epochs_run": 2,
                      "real_count": 2, "synthetic_count": 2 if condition == "B3" else 0,
                      "training_rows": 4 if condition == "B3" else 2}
            write_json(folder / "results.json", result)
            write_json(self.base / "05_results/metrics" / condition / "low_resource/seed_42.json",
                       {**result, "shared_training_policy_sha256": policy, "training_reused": False})
            if condition == "B1":
                write_json(self.base / "05_results/metrics/B2/low_resource/seed_42.json", {
                    **result, "condition": "B2", "training_reused": True,
                    "shared_training_policy_sha256": policy,
                    "reused_from": "04_checkpoints/B1/low_resource/seed_42",
                })

    @staticmethod
    def describe(path, rows):
        return {"rows": len(rows), "file_sha256": file_sha256(path), "rows_sha256": rows_digest(rows),
                "ids": [row["id"] for row in rows],
                "per_class": {str(label): count for label, count in sorted(Counter(row["label"] for row in rows).items())}}

    def pin_manifest(self):
        path = self.base / "01_data/manifest.json"
        write_json(path, self.manifest)
        self.config["classifier_pilot"]["selection"]["data_manifest_sha256"] = file_sha256(path)

    def call(self, execute=True):
        with patch.object(inference, "load_config", return_value=self.config), \
             patch.object(inference, "PROJECT_ROOT", self.root):
            return inference.evaluate_baselines(base_path=self.base, execute=execute)

    def fake_evaluator(self, entry, rows, config, base):
        import numpy as np

        probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        labels = np.asarray([row["label"] for row in rows])
        metrics = {"macro_f1": 1.0, "accuracy": 1.0, "rare_macro_f1": 1.0,
                   "rare_macro_recall": 1.0, "test_loss": 0.1}
        per_class = [{"label": label, "intent": str(label), "f1": 1.0, "support": 1} for label in range(2)]
        timing = {"inference_seconds": 1.0, "total_seconds": 2.0, "nonpadding_tokens": 4,
                  "start_time": "2026-11-26T00:00:00+00:00", "end_time": "2026-11-26T00:00:02+00:00",
                  "runtime": {"device": "cpu", "precision": "float32", "hardware": "fixture"}}
        return metrics, labels, probabilities.argmax(axis=1), probabilities, per_class, timing

    def snapshot_reports(self):
        return {str(path.relative_to(self.base)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.base.rglob("*") if path.is_file()
                and "01_data" not in path.parts and "04_checkpoints" not in path.parts}

    def test_preview_and_incomplete_main_gates_do_not_open_test(self):
        with patch.object(inference, "_official_test") as test_reader:
            self.config["status"] = "classifier_pilot_pending"
            self.assertEqual(self.call(execute=False), [])
            with self.assertRaisesRegex(ValueError, "Freeze"):
                self.call()
            self.config["status"] = "frozen"
            (self.base / "05_results/metrics/B3/low_resource/seed_42.json").unlink()
            with self.assertRaisesRegex(ValueError, "every baseline"):
                self.call()
            test_reader.assert_not_called()

    def test_changed_frozen_training_policy_is_rejected_before_test(self):
        self.config["training"]["learning_rate"] = 2e-5
        with patch.object(inference, "_official_test") as test_reader:
            with self.assertRaisesRegex(ValueError, "frozen training policy"):
                self.call()
            test_reader.assert_not_called()

    def test_self_consistent_wrong_subset_or_metadata_is_rejected_before_test(self):
        for condition, key, wrong in (("B1", "training_rows_sha256", "wrong-real-subset"),
                                      ("B3", "training_rows_sha256", "wrong-eda-subset"),
                                      ("B1", "validation_rows_sha256", "wrong-validation"),
                                      ("B1", "rare_class_ids", [0]),
                                      ("B1", "class_names", ["one", "zero"]),
                                      ("B1", "trainer_source_sha256", "wrong-trainer")):
            folder = self.base / "04_checkpoints" / condition / "low_resource/seed_42"
            conditions = ("B1", "B2") if condition == "B1" else ("B3",)
            paths = [folder / "run_config.json", folder / "results.json"] + [
                self.base / "05_results/metrics" / name / "low_resource/seed_42.json" for name in conditions]
            original = {path: json.loads(path.read_text()) for path in paths}
            run = copy.deepcopy(original[paths[0]])
            run["specification"][key] = wrong
            fingerprint = rows_digest([run["specification"]])
            run["fingerprint"] = fingerprint
            write_json(paths[0], run)
            for path in paths[1:]:
                write_json(path, {**original[path], "fingerprint": fingerprint})
            with self.subTest(condition=condition, field=key), \
                 patch.object(inference, "_official_test") as test_reader, \
                 patch.object(inference, "read_jsonl", wraps=read_jsonl) as reader:
                with self.assertRaisesRegex(ValueError, "released"):
                    self.call()
                test_reader.assert_not_called()
                self.assertTrue(all(Path(call.args[0]).name != "test.jsonl" for call in reader.call_args_list))
            for path, value in original.items():
                write_json(path, value)

    def test_actual_training_files_and_preparation_reports_are_verified_before_test(self):
        for relative in ("01_data/validation.jsonl", "01_data/low_resource/seed_42.jsonl",
                         "01_data/baselines/B3/low_resource/seed_42.jsonl"):
            path = self.base / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(path=relative), patch.object(inference, "_official_test") as test_reader:
                with self.assertRaisesRegex(ValueError, "Released file checksum"):
                    self.call()
                test_reader.assert_not_called()
            path.write_bytes(original)
        report_path = self.base / "01_data/baselines/reports/B2/low_resource/seed_42.json"
        report = json.loads(report_path.read_text())
        report["specification"]["parameters"] = {"target": "different-policy"}
        report["config_sha256"] = rows_digest([report["specification"]])
        write_json(report_path, report)
        with patch.object(inference, "_official_test") as test_reader:
            with self.assertRaisesRegex(ValueError, "preparation policy differs"):
                self.call()
            test_reader.assert_not_called()

    def test_official_test_file_and_ordered_ids_are_verified(self):
        test_path = self.base / "01_data/test.jsonl"
        test_path.write_text(test_path.read_text() + "\n")
        with patch.object(inference, "_evaluate_checkpoint") as evaluate:
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.call()
            write_jsonl(test_path, self.rows)
            self.manifest["datasets"]["test.jsonl"]["ids"].reverse()
            self.pin_manifest()
            with self.assertRaisesRegex(ValueError, "ordered IDs"):
                self.call()
            evaluate.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Probability exports require NumPy")
    def test_exports_reuse_b2_predictions_and_charge_inference_once(self):
        import numpy as np

        with patch.object(inference, "_evaluate_checkpoint", side_effect=self.fake_evaluator) as evaluate:
            rows = self.call()
            self.assertEqual(evaluate.call_count, 2)
            b2 = next(row for row in rows if row["condition"] == "B2")
            self.assertEqual(b2["rare_class_f1"], b2["rare_macro_f1"])
            self.assertEqual(b2["rare_class_recall"], b2["rare_macro_recall"])
            self.assertTrue(b2["inference_reused"])
            self.assertEqual(b2["additional_inference_seconds"], 0.0)
            self.assertEqual(sum(row["additional_inference_seconds"] for row in rows), 4.0)
            saved = read_jsonl(self.base / "05_results/predictions/B2/low_resource/seed_42.jsonl")
            self.assertEqual([row["id"] for row in saved], ["test:0", "test:1"])
            self.assertTrue(all(row["condition"] == "B2" and row["split"] == "test"
                                and row["reused_from_condition"] == "B1" for row in saved))
            b1_probs = self.base / "05_results/probabilities/test/B1/low_resource/seed_42.npz"
            b2_probs = self.base / "05_results/probabilities/test/B2/low_resource/seed_42.npz"
            self.assertEqual(b1_probs.read_bytes(), b2_probs.read_bytes())
            with np.load(b2_probs) as arrays:
                self.assertEqual(arrays["probabilities"].shape, (2, 2))
                self.assertTrue(np.allclose(arrays["probabilities"].sum(axis=1), 1.0))
            before = self.snapshot_reports()
            self.assertEqual(self.call(), rows)
            self.assertEqual(evaluate.call_count, 2)
            after = self.snapshot_reports()
            # Summary CSVs may be regenerated, while detailed completed reports are reused unchanged.
            before.pop("05_results/metrics/baseline_test_results.csv")
            after.pop("05_results/metrics/baseline_test_results.csv")
            self.assertEqual(before, after)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Probability exports require NumPy")
    def test_changed_checkpoint_and_tampered_exports_refuse_reuse(self):
        with patch.object(inference, "_evaluate_checkpoint", side_effect=self.fake_evaluator) as evaluate:
            self.call()
            before = self.snapshot_reports()
            weights = self.base / "04_checkpoints/B1/low_resource/seed_42/best_model/model.safetensors"
            weights.write_text("changed weights")
            with self.assertRaisesRegex(ValueError, "reporting inputs differ"):
                self.call()
            self.assertEqual(before, self.snapshot_reports())
            weights.write_text("fixture")
            predictions = self.base / "05_results/predictions/B1/low_resource/seed_42.jsonl"
            predictions.write_text(predictions.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                self.call()
            self.assertEqual(evaluate.call_count, 2)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "sklearn", "transformers", "safetensors")),
                     "CPU checkpoint inference needs the existing ML environment")
class TinyCPUInferenceTests(unittest.TestCase):
    setUp = BaselineInferenceTests.setUp
    pin_manifest = BaselineInferenceTests.pin_manifest
    call = BaselineInferenceTests.call
    describe = staticmethod(BaselineInferenceTests.describe)

    def test_saved_tiny_checkpoints_are_loaded_locally_and_reported(self):
        import torch
        from safetensors.torch import load_file, save_file

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                if not kwargs.get("local_files_only"):
                    raise AssertionError("Inference must remain local")
                return cls()

            def __call__(self, texts, **kwargs):
                return {"input_ids": [[int(text.endswith("one"))] for text in texts],
                        "attention_mask": [[1] for text in texts]}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 2)
                self.config = SimpleNamespace(num_labels=2)

            @classmethod
            def from_pretrained(cls, name, **kwargs):
                if not kwargs.get("local_files_only"):
                    raise AssertionError("Inference must load the saved checkpoint")
                model = cls()
                model.load_state_dict(load_file(str(Path(name) / "model.safetensors")))
                return model

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.linear(input_ids.float()))

        class Collator:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, rows):
                return {key: torch.tensor([row[key] for row in rows], dtype=torch.long) for key in rows[0]}

        for path in (self.base / "04_checkpoints").glob("*/*/*/best_model/model.safetensors"):
            save_file({"linear.weight": torch.tensor([[-1.0], [1.0]]),
                       "linear.bias": torch.tensor([1.0, 0.0])}, str(path))
        libraries = training._libraries()
        libraries.AutoTokenizer = Tokenizer
        libraries.AutoModelForSequenceClassification = Model
        libraries.DataCollatorWithPadding = Collator
        with patch.object(training, "_libraries", return_value=libraries):
            rows = self.call()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["macro_f1"] == 1.0 for row in rows))
        saved = json.loads((self.base / "05_results/metrics/test/B1/low_resource/seed_42.json").read_text())
        self.assertEqual(saved["runtime"]["device"], "cpu")
        self.assertTrue(saved["reporting_only"])
        self.assertFalse(saved["test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
