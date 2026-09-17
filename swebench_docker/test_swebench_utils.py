# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import tempfile
import unittest

from swebench_docker.swebench_utils import get_eval_report, get_logs_eval

TESTS_CONFIG_MARKER = ">>>>> Tests config"


class TestGetLogsEvalFunctionMetrics(unittest.TestCase):
    def _write_log(self, content, instance_id):
        # get_repo_from_lp derives repo from the filename shape
        # "<instance_id>.<model>.<setting>.eval.log", so the real
        # filename needs to look like that, not a random temp name.
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, f"{instance_id}.model.full.eval.log")
        with open(path, "w") as f:
            f.write(content)
        # addCleanup runs LIFO, so register rmdir first, remove second.
        self.addCleanup(os.rmdir, tmp_dir)
        self.addCleanup(os.remove, path)
        return path

    def test_function_and_whole_file_metrics_do_not_collide(self):
        content = (
            f"{TESTS_CONFIG_MARKER} full\n"
            "TestsTime: 1.5\n"
            ">>>>> All Tests Passed\n"
            "\nCoverageLOG: 42.0%\n"
            "\nFunctionCoverageLOG: 88.0%\n"
            "\nMutationLOG: 30.0%"
            "\nMutationUncertainty: 5.0"
            "\nMutationNum: 100"
            "\nFunctionMutationLOG: 75.0%"
            "\nFunctionMutationNum: 8"
        )
        log_fp = self._write_log(content, "django__django-1-1")

        results = get_logs_eval(log_fp)

        self.assertIn("full", results)
        r = results["full"]
        self.assertEqual(r["coverage"], [42.0])
        self.assertEqual(r["function_coverage"], [88.0])
        self.assertEqual(r["mutation_score"], [30.0])
        self.assertEqual(r["mutation_uncertainty"], [5.0])
        self.assertEqual(r["mutation_num"], [100.0])
        self.assertEqual(r["function_mutation_score"], [75.0])
        self.assertEqual(r["function_mutation_num"], [8.0])

    def test_missing_function_metrics_default_to_negative_one(self):
        content = (
            f"{TESTS_CONFIG_MARKER} full\n"
            "TestsTime: 1.5\n"
            ">>>>> All Tests Passed\n"
            "\nCoverageLOG: 42.0%\n"
            "\nMutationLOG: 30.0%"
            "\nMutationUncertainty: 5.0"
            "\nMutationNum: 100"
        )
        log_fp = self._write_log(content, "django__django-2-2")

        results = get_logs_eval(log_fp)
        r = results["full"]
        self.assertEqual(r["function_coverage"], [-1])
        self.assertEqual(r["function_mutation_score"], [-1])
        self.assertEqual(r["function_mutation_num"], [-1])

    def test_any_tests_passed_marker_is_captured(self):
        # RQ3 instrumentation: ANY_TESTS_PASSED is a real, distinct
        # marker from TESTS_PASSED (which is really "All Tests Passed"
        # semantics, see constants.py) -- this test just confirms
        # get_logs_eval actually reads it into a parallel list, same
        # shape as tests_passed.
        content = (
            f"{TESTS_CONFIG_MARKER} full\n"
            "TestsTime: 1.5\n"
            ">>>>> All Tests Passed\n"
            ">>>>> Any Test Passed\n"
        )
        log_fp = self._write_log(content, "django__django-3-3")

        results = get_logs_eval(log_fp)
        r = results["full"]
        self.assertEqual(r["tests_passed"], [True])
        self.assertEqual(r["any_tests_passed"], [True])

    def test_any_tests_passed_distinct_from_whole_suite_pass(self):
        # The real behaviour this instrumentation exists for: a whole
        # suite that FAILS (some real assertion elsewhere in the file)
        # can still have had at least one individual test case pass.
        # TESTS_FAILED and ANY_TESTS_PASSED are not mutually exclusive.
        content = (
            f"{TESTS_CONFIG_MARKER} full\n"
            "TestsTime: 1.5\n"
            ">>>>> Some Tests Failed\n"
            ">>>>> Any Test Passed\n"
        )
        log_fp = self._write_log(content, "django__django-4-4")

        results = get_logs_eval(log_fp)
        r = results["full"]
        self.assertEqual(r["tests_passed"], [False])
        self.assertEqual(r["any_tests_passed"], [True])

    def test_any_tests_passed_absent_marker_defaults_false(self):
        # A log written before this instrumentation existed (or any
        # real run where no individual test passed, e.g. django's
        # runtests.py, which never gets a --junitxml report to derive
        # a real per-test-case signal from) never contains the marker
        # at all -- must default to False, not raise.
        content = (
            f"{TESTS_CONFIG_MARKER} full\n"
            "TestsTime: 1.5\n"
            ">>>>> Some Tests Failed\n"
        )
        log_fp = self._write_log(content, "django__django-5-5")

        results = get_logs_eval(log_fp)
        r = results["full"]
        self.assertEqual(r["any_tests_passed"], [False])


class TestGetEvalReportAveraging(unittest.TestCase):
    def test_negative_one_sentinels_excluded_from_average(self):
        # -1 sentinels (from rows where the target range couldn't be
        # resolved) must be excluded from both the sum and the count,
        # not just the count.
        eval_sm = {
            "full": {
                "tests_passed": [True],
                "tests_compiled": [True],
                "coverage": [42.0],
                "test_time": [1.5],
                "test_error": ["Success"],
                "mutation_score": [30.0],
                "function_coverage": [90.0, 90.0, 90.0, -1],
            }
        }
        report = get_eval_report(
            eval_sm, {"inst-1": {"baseline_covs": {}}}, "inst-1", is_baseline=False
        )
        self.assertEqual(report["full_av_function_coverage"], 90.0)

    def test_any_pass_at_k_diverges_from_whole_suite_pass_at_k(self):
        # RQ3's real reason for existing: a sample whose whole test
        # suite fails (full_pass_at_1 -- really "All Tests Passed"@1,
        # see constants.py) can still have full_any_pass_at_1 True, if
        # at least one individual test case in that sample passed.
        # These two metrics answering differently for the same sample
        # is the entire point, not a bug to reconcile.
        eval_sm = {
            "full": {
                "tests_passed": [False],
                "any_tests_passed": [True],
                "tests_compiled": [True],
                "coverage": [42.0],
                "test_time": [1.5],
                "test_error": ["AssertionError"],
                "mutation_score": [30.0],
                "function_coverage": [90.0],
            }
        }
        report = get_eval_report(
            eval_sm, {"inst-1": {"baseline_covs": {}}}, "inst-1", is_baseline=False
        )
        self.assertEqual(report["full_pass_at_1"], False)
        self.assertEqual(report["full_any_pass_at_1"], True)

    def test_any_pass_at_k_absent_when_no_any_tests_passed_data(self):
        # Old eval_sm data (pre-instrumentation) has no "any_tests_passed"
        # key at all -- get_eval_report must not raise, and must simply
        # not produce any full_any_pass_at_k keys, rather than a
        # misleading default.
        eval_sm = {
            "full": {
                "tests_passed": [True],
                "tests_compiled": [True],
                "coverage": [42.0],
                "test_time": [1.5],
                "test_error": ["Success"],
                "mutation_score": [30.0],
                "function_coverage": [90.0],
            }
        }
        report = get_eval_report(
            eval_sm, {"inst-1": {"baseline_covs": {}}}, "inst-1", is_baseline=False
        )
        self.assertEqual(report["full_pass_at_1"], True)
        self.assertNotIn("full_any_pass_at_1", report)


if __name__ == "__main__":
    unittest.main()
