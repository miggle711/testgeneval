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


if __name__ == "__main__":
    unittest.main()
