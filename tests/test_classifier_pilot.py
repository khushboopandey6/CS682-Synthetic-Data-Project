import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.classifier_pilot import pilot_plan, run_classifier_pilot, select_learning_rate
from src.project_config import write_json


class PilotSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = {"classifier_pilot": {"development_seed": 42,
                       "settings": ["low_resource", "imbalanced"], "conditions": ["B1", "B3"],
                       "learning_rate_candidates": [2e-5, 5e-5, 1e-4]}}

    def records(self, scores):
        return [{"learning_rate": rate, "setting": setting, "condition": condition,
                 "seed": seed, "macro_f1": scores[rate][index % 4]}
                for index, (rate, setting, condition, seed) in enumerate(pilot_plan(self.config))]

    def test_equal_weight_selection_does_not_select_best_single_fit(self):
        records = self.records({2e-5: [0.9, 0.4, 0.4, 0.4],
                                5e-5: [0.7] * 4, 1e-4: [0.6] * 4})
        winner, _ = select_learning_rate(records, self.config)
        self.assertEqual(winner["learning_rate"], 5e-5)

    def test_exact_tie_prefers_lower_rate(self):
        records = self.records({2e-5: [0.5] * 4, 5e-5: [0.5] * 4, 1e-4: [0.4] * 4})
        winner, _ = select_learning_rate(records, self.config)
        self.assertEqual(winner["learning_rate"], 2e-5)

    def test_incomplete_and_duplicate_results_are_rejected(self):
        records = self.records({2e-5: [0.5] * 4, 5e-5: [0.6] * 4, 1e-4: [0.4] * 4})
        for invalid in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                select_learning_rate(invalid, self.config)


class PilotDriverIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.base = self.root / "artifacts"
        self.config_path = self.root / "configs" / "experiment.yaml"
        self.config = {
            "schema_version": 1, "protocol_id": "fixture_master", "status": "classifier_pilot_pending",
            "paths": {"base_path": str(self.base)},
            "training": {"model_revision": "pinned-fixture", "learning_rate": None},
            "classifier_pilot": {
                "development_seed": 42, "settings": ["low_resource", "imbalanced"],
                "conditions": ["B1", "B3"], "learning_rate_candidates": [2e-5, 5e-5, 1e-4],
            },
        }
        write_json(self.config_path, self.config)
        write_json(self.base / "01_data" / "manifest.json", {"fixture": "immutable release"})
        for setting in self.config["classifier_pilot"]["settings"]:
            for condition in self.config["classifier_pilot"]["conditions"]:
                path = (self.base / "01_data" / setting / "seed_42.jsonl" if condition == "B1" else
                        self.base / "01_data" / "baselines" / condition / setting / "seed_42.jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"id": "fixture", "text": f"{setting} {condition}", "label": 0}) + "\n")
        self.scores = {2e-5: [0.95, 0.4, 0.4, 0.4], 5e-5: [0.75] * 4, 1e-4: [0.7] * 4}
        self.fit_keys = set()
        self.new_fit_count = 0
        self.dump_yaml = Mock(side_effect=lambda value, sort_keys=False: json.dumps(value, sort_keys=sort_keys, indent=2))

    def fake_trainer(self, train_path, validation_path, condition, setting, seed, config, **kwargs):
        rate = kwargs["pilot_learning_rate"]
        self.assertEqual(seed, 42)
        self.assertEqual(validation_path, self.base / "01_data" / "validation.jsonl")
        self.assertTrue(Path(train_path).is_file())
        index = config["classifier_pilot"]["settings"].index(setting) * 2 + config["classifier_pilot"]["conditions"].index(condition)
        result = {
            "status": "completed", "macro_f1": self.scores[rate][index], "accuracy": 0.8,
            "best_epoch": 7, "epochs_run": 12, "real_count": 2, "training_rows": 4 if condition == "B3" else 2,
            "optimizer_steps": 12, "total_seconds": 1.0, "device": "cpu",
        }
        folder = Path(kwargs["output_dir"])
        if not (folder / "results.json").exists():
            write_json(folder / "results.json", result)
            self.new_fit_count += 1
        self.fit_keys.add((rate, setting, condition, seed))
        return result

    def patches(self):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("src.classifier_pilot.PROJECT_ROOT", self.root))
        stack.enter_context(patch("src.classifier_pilot.load_config", side_effect=lambda path: json.loads(Path(path).read_text())))
        stack.enter_context(patch("src.classifier_pilot.subprocess.check_output", return_value="a" * 40 + "\n"))
        stack.enter_context(patch("src.training.train_classifier", side_effect=self.fake_trainer))
        # JSON is valid YAML; isolate orchestration from optional PyYAML installation.
        stack.enter_context(patch.dict("sys.modules", {"yaml": SimpleNamespace(safe_dump=self.dump_yaml)}))
        return stack

    def test_complete_twelve_fit_driver_freezes_mean_winner_only_once(self):
        with self.patches():
            selection = run_classifier_pilot(self.config_path, self.base, execute=True, freeze=True)
            frozen = json.loads(self.config_path.read_text())
            first_bytes = self.config_path.read_bytes()
            first_mtime = self.config_path.stat().st_mtime_ns
            second_selection = run_classifier_pilot(self.config_path, self.base, execute=False, freeze=True)
        self.assertEqual(len(self.fit_keys), 12)
        self.assertEqual(self.new_fit_count, 12)
        self.assertEqual(selection["learning_rate"], 5e-5)
        self.assertEqual(selection["completed_fits"], 12)
        self.assertEqual(frozen["status"], "frozen")
        self.assertEqual(frozen["training"]["learning_rate"], 5e-5)
        self.assertEqual(frozen["classifier_pilot"]["selection"], selection)
        self.assertEqual(selection, second_selection)
        self.assertEqual(self.dump_yaml.call_count, 1)
        self.assertEqual(first_bytes, self.config_path.read_bytes())
        self.assertEqual(first_mtime, self.config_path.stat().st_mtime_ns)
        report = self.base / "05_results" / "classifier_pilot" / "selection.json"
        git_report = self.root / "results" / "metrics" / "fixture_master" / "classifier_selection.json"
        self.assertEqual(json.loads(report.read_text()), selection)
        self.assertEqual(json.loads(git_report.read_text()), selection)

    def test_conflicting_frozen_winner_preserves_config_and_published_reports(self):
        self.config["status"] = "frozen"
        self.config["training"]["learning_rate"] = 2e-5
        write_json(self.config_path, self.config)
        reports = [
            self.base / "05_results/classifier_pilot/selection.json",
            self.base / "05_results/classifier_pilot/learning_rates.csv",
            self.root / "results/metrics/fixture_master/classifier_selection.json",
            self.root / "results/metrics/fixture_master/classifier_learning_rates.csv",
            self.root / "results/metrics/fixture_master/classifier_pilot_trials.csv",
        ]
        for path in reports:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("previous frozen report\n")
        protected = [self.config_path, *reports]
        before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in protected}
        for freeze in (True, False):
            with self.subTest(freeze=freeze), self.patches():
                with self.assertRaisesRegex(ValueError, "frozen learning rate differs"):
                    run_classifier_pilot(self.config_path, self.base, execute=True, freeze=freeze)
        after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in protected}
        self.assertEqual(before, after)
        self.assertEqual(self.dump_yaml.call_count, 0)
        self.assertTrue((self.base / "05_results/classifier_pilot/validation_trials.csv").is_file())


if __name__ == "__main__":
    unittest.main()
