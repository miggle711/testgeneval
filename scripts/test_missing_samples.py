"""Harness-shaped fixtures; no dataset, container, GPU or log writes required."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location("missing_samples", Path(__file__).with_name("missing_samples.py"))
missing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(missing)

from swebench_docker.constants import (
    APPLY_PATCH_FAIL, INSTALL_FAIL, TESTS_CONFIG, TESTS_ERROR, TESTS_TIMEOUT,
    UNFILTERED_TESTS_FAILED, UNFILTERED_TESTS_PASSED,
)

PREFIX = "[django__django__4.1] [django__django-15380]"


def block(*messages, setting="full", kind="pred"):
    # LogWrapper.write appends a space/newline, and embedded newlines have no prefix.
    return f"{PREFIX} {TESTS_CONFIG}{setting} {kind}\n \n" + "".join(
        f"{PREFIX} {message} \n" for message in messages)


CLEAN = block("TestsTime: 0.0", UNFILTERED_TESTS_FAILED)


class TestEndings(unittest.TestCase):
    def test_only_full_prediction_starts_count_and_count_metadata_is_not_a_sample(self):
        text = ("TargetMutantCount: 20\n" + block("done", setting="first") +
                block("done", kind="baseline") + CLEAN + block("unfinished"))
        blocks = missing.full_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(sum(missing.end_marker(b) >= 0 for b in blocks), 1)
        self.assertEqual(missing.ending(blocks, text), "cut_off_mid_block")

    def test_final_markers_distinguish_timeout_error_cutoff_and_clean_end(self):
        cases = [(block(TESTS_TIMEOUT), "timeout_marker"),
                 (block("MutationTimeout"), "timeout_marker"),
                 (block(TESTS_ERROR + ": boom"), "harness_error"),
                 (block("Running mutation testing"), "cut_off_mid_block"),
                 (CLEAN, "cleanly"),
                 (block(UNFILTERED_TESTS_PASSED), "cleanly"),
                 (block(TESTS_TIMEOUT, UNFILTERED_TESTS_FAILED), "cleanly"),
                 ("Task Metadata:\n", "no_full_block"),
                 (APPLY_PATCH_FAIL, "harness_error"),
                 (INSTALL_FAIL, "harness_error")]
        for text, expected in cases:
            with self.subTest(expected=expected, text=text):
                self.assertEqual(missing.ending(missing.full_blocks(text), text), expected)

    def test_timeout_in_an_earlier_sample_does_not_describe_the_last_sample(self):
        for final, expected in ((CLEAN, "cleanly"), (block("started"), "cut_off_mid_block")):
            text = block(TESTS_TIMEOUT, UNFILTERED_TESTS_FAILED) + final
            self.assertEqual(missing.ending(missing.full_blocks(text), text), expected)


class TestReports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.a = "django__django-15380-1"
        self.b = "scikit-learn__scikit-learn-11578-2"
        self.c = "django__django-15380-3"

    def log(self, instance, text, folder="model"):
        path = self.logs / folder / f"{instance}.model.full.eval.log"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        return path

    def ids(self, *ids):
        path = self.root / "ids.txt"
        path.write_text("\n".join(ids) + "\n")
        return path

    def preds(self, rows):
        path = self.root / "predictions.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def test_histogram_repo_names_missing_logs_and_per_instance_prediction_counts(self):
        self.log(self.a, CLEAN * 4)
        self.log(self.b, "Task Metadata:\nTargetMutantCount: 12\n")
        ids = self.ids(self.a, self.b, self.c)
        pred = self.preds([{"id": self.a, "instance_id": "different", "preds": {"full": ["t"] * 5}},
                           {"id": self.b, "preds": {"full": ["t"] * 2}}])
        report = missing.collect(self.logs, instances_file=ids, predictions_file=pred)
        self.assertEqual(report["overall"]["logs_by_full_blocks"],
                         {"0": 1, "1": 0, "2": 0, "3": 0, "4": 1, "5": 0})
        self.assertEqual(report["overall"]["no_log"], 1)
        self.assertEqual(report["overall"]["short_log_endings"]["cleanly"], 1)
        self.assertEqual(report["overall"]["short_log_endings"]["no_full_block"], 1)
        self.assertEqual(report["overall"]["short_or_absent_prediction_counts"],
                         {"2": 1, "5": 1, "no_record": 1})
        self.assertEqual(report["by_repo"]["scikit-learn__scikit-learn"]["logs"], 1)
        self.assertEqual(report["by_repo"]["django__django"]["no_log"], 1)
        rows = {r["instance"]: r for r in report["instances"]}
        self.assertEqual(rows[self.a]["prediction_count"], 5)
        self.assertEqual(rows[self.c]["ending"], "no_log")

    def test_no_instance_list_does_not_claim_a_dataset_wide_count(self):
        self.log(self.a, CLEAN * 5)
        report = missing.collect(self.logs)
        self.assertIn("absent logs unknown", report["universe"])
        self.assertEqual(report["overall"]["no_log"], 0)
        self.assertEqual(report["overall"]["short_log_endings"]["cleanly"], 0)
        pred = self.preds([{"id": self.b, "preds": {"full": ["t"] * 5}}])
        self.assertEqual(missing.collect(self.logs, predictions_file=pred)["overall"]["no_log"], 1)

    def test_host_kill_is_external_evidence_not_inferred_from_a_cutoff(self):
        self.log(self.a, CLEAN + block("Running mutation testing"))
        runner = self.root / "slurm.out"
        # Exact pre-#99 run_apptainer.py warning; no corresponding write to tcm.log.
        runner.write_text(f"WARNING [{self.a}][/images/django.sif]  Container exceeded host timeout of 900s, killed.\n"
                          f"WARNING [{self.b}][/images/sklearn.sif]  Error running container:\n")
        before = missing.collect(self.logs)
        self.assertEqual(before["instances"][0]["runner_events"], [])
        after = missing.collect(self.logs, runner_logs=[runner], instances_file=self.ids(self.a, self.b))
        rows = {r["instance"]: r for r in after["instances"]}
        self.assertEqual(rows[self.a]["ending"], "cut_off_mid_block")
        self.assertEqual(rows[self.a]["runner_events"], ["host_timeout"])
        self.assertEqual(rows[self.b]["runner_events"], ["container_error"])
        self.assertEqual(after["overall"]["short_or_absent_runner_events"],
                         {"container_error": 1, "host_timeout": 1})

    def test_partial_files_and_other_settings_do_not_inflate_counts(self):
        path = self.log(self.a, CLEAN * 6)
        path.with_suffix(path.suffix + ".partial").write_text(CLEAN * 3)
        path.with_name(f"{self.a}.model.first.eval.log").write_text(CLEAN)
        report = missing.collect(self.logs)
        self.assertEqual(report["overall"]["logs"], 1)
        self.assertEqual(report["overall"]["logs_over_k"], 1)

    def test_duplicate_model_logs_or_prediction_ids_fail_instead_of_selecting_one(self):
        self.log(self.a, CLEAN)
        self.log(self.a, CLEAN, folder="other_model")
        with self.assertRaisesRegex(ValueError, "multiple full logs"):
            missing.collect(self.logs)
        pred = self.preds([{"id": self.a}, {"id": self.a}])
        with self.assertRaisesRegex(ValueError, "duplicate prediction id"):
            missing.predictions(pred)

    def test_empty_predictions_and_missing_full_setting_remain_distinct(self):
        pred = self.preds([{"id": self.a, "preds": {"full": []}}, {"id": self.b, "preds": {}}])
        report = missing.collect(self.logs, predictions_file=pred)
        self.assertEqual(report["overall"]["short_or_absent_prediction_counts"],
                         {"0": 1, "no_full_setting": 1})
        pred = self.preds([{"id": self.a, "preds": {"full": "not a list"}}])
        with self.assertRaisesRegex(ValueError, "list of strings"):
            missing.predictions(pred)

    def test_cli_is_read_only_and_returns_json(self):
        path = self.log(self.a, CLEAN)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            missing.main([str(self.logs), "--k", "2"])
        report = json.loads(output.getvalue())
        self.assertEqual(report["overall"]["logs_by_full_blocks"]["1"], 1)
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)
        # Exercise direct script execution without PYTHONPATH or third-party packages.
        result = subprocess.run([sys.executable, "-I", str(Path(missing.__file__)), str(self.logs)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["overall"]["logs"], 1)

    def test_mistyped_universe_does_not_silently_drop_observed_instances(self):
        self.log(self.a, CLEAN)
        with self.assertRaisesRegex(ValueError, "outside instance list"):
            missing.collect(self.logs, instances_file=self.ids(self.b))
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            missing.collect(self.logs, instances_file=self.ids(self.a, self.a))


if __name__ == "__main__":
    unittest.main()
