# Copyright (c) Meta Platforms, Inc. and affiliates.

import importlib.util
import os
import sys
import tempfile
import unittest

# scripts/ isn't a package (matches merge_and_validate.py's own
# convention, run standalone), so import by path rather than
# `from scripts.merge_any_pass_logs import ...`.
_SPEC = importlib.util.spec_from_file_location(
    "merge_any_pass_logs",
    os.path.join(os.path.dirname(__file__), "merge_any_pass_logs.py"),
)
merge_any_pass_logs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge_any_pass_logs)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swebench_docker.swebench_utils import get_logs_eval  # noqa: E402

TESTS_CONFIG = ">>>>> Tests config"

# Real, trimmed fixture derived directly from an actual M3 evaluation
# log (job under np8_logs/qwen34b_instruct/, a real pass@5 run),
# rather than a hand-guessed shape -- confirmed the real production
# constants.py marker strings this way (caught and fixed two wrong
# guesses in the merge script's own dev before this was written: the
# real TESTS_CONFIG line has no space before the setting name and ends
# in " pred"/" baseline", and ANY_TESTS_FAILED is
# ">>>>> No Tests Passed", not ">>>>> Any Test Failed"). Trimmed of
# the real subprocess/cosmic-ray output noise (hundreds of KB in the
# genuine log), keeping the real preamble/marker/summary lines intact.
REAL_PREAMBLE = (
    "[django__django__4.1] [django__django-15380] Task Metadata:\n"
    "\t- Instance ID: django__django-15380\n"
)

REAL_FAILING_BLOCK = (
    "full pred\n"
    " \n"
    "[django__django__4.1] [django__django-15380] TestsTime: 0.0 \n"
    "[django__django__4.1] [django__django-15380] >>>>> Some Tests Failed \n"
    "[django__django__4.1] [django__django-15380] >>>>> Unfiltered Tests Failed \n"
)

REAL_PASSING_BLOCK_WITH_MUTATION = (
    "full pred\n"
    " \n"
    "[astropy__astropy__4.3] [astropy__astropy-12057] TestsTime: 12.3 \n"
    "[astropy__astropy__4.3] [astropy__astropy-12057] >>>>> All Tests Passed \n"
    "[astropy__astropy__4.3] [astropy__astropy-12057] \n"
    "CoverageLOG: 39.61661341853035% \n"
    "[astropy__astropy__4.3] [astropy__astropy-12057] \n"
    "FunctionCoverageLOG: 41.935483870967744% \n"
    "[astropy__astropy__4.3] [astropy__astropy-12057] \n"
    "MutationLOG: 19.6% \n"
    "[astropy__astropy__4.3] [astropy__astropy-12057] \n"
    "MutationNum: 153 \n"
)


class TestMergeAnyPassLogs(unittest.TestCase):
    def _write_and_merge(self, original_content, backfill_content):
        # Real merged log needs to outlive this helper (callers run
        # get_logs_eval on the real path afterward), so the temp dir
        # is registered for cleanup rather than used as a context
        # manager that deletes everything on return.
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))

        original_dir = os.path.join(tmp, "original")
        backfill_dir = os.path.join(tmp, "backfill")
        out_dir = os.path.join(tmp, "merged")
        os.makedirs(original_dir)
        os.makedirs(backfill_dir)
        filename = "inst-1.model.full.eval.log"
        with open(os.path.join(original_dir, filename), "w") as f:
            f.write(original_content)
        with open(os.path.join(backfill_dir, filename), "w") as f:
            f.write(backfill_content)

        sys.argv = [
            "merge_any_pass_logs.py",
            "--original-log-dir", original_dir,
            "--backfill-log-dir", backfill_dir,
            "--out-dir", out_dir,
        ]
        merge_any_pass_logs.main()

        merged_path = os.path.join(out_dir, filename)
        with open(merged_path) as f:
            merged_content = f.read()
        return merged_content, merged_path

    def test_real_5_sample_pass5_log_merges_cleanly(self):
        # Real shape: a pass@5 file has 5 real "full pred" blocks per
        # instance, same as this genuine M3 log had.
        original = REAL_PREAMBLE + TESTS_CONFIG.join(
            [""] + [REAL_FAILING_BLOCK] * 5
        )
        backfill = REAL_PREAMBLE + TESTS_CONFIG.join(
            [""] + [REAL_FAILING_BLOCK + ">>>>> No Tests Passed\n"] * 5
        )
        merged_content, merged_path = self._write_and_merge(original, backfill)

        results = get_logs_eval(merged_path)
        self.assertEqual(results["full"]["any_tests_passed"], [False] * 5)
        self.assertEqual(results["full"]["tests_passed"], [False] * 5)

    def test_real_passing_block_keeps_mutation_data_and_gets_any_pass(self):
        original = REAL_PREAMBLE + TESTS_CONFIG + REAL_PASSING_BLOCK_WITH_MUTATION
        backfill = REAL_PREAMBLE + TESTS_CONFIG + (
            "full pred\n"
            "[astropy__astropy__4.3] [astropy__astropy-12057] TestsTime: 3.1 \n"
            "[astropy__astropy__4.3] [astropy__astropy-12057] >>>>> All Tests Passed \n"
            "[astropy__astropy__4.3] [astropy__astropy-12057] >>>>> Any Test Passed \n"
        )
        merged_content, merged_path = self._write_and_merge(original, backfill)

        results = get_logs_eval(merged_path)
        r = results["full"]
        self.assertEqual(r["any_tests_passed"], [True])
        # Real mutation/coverage data from the original must survive
        # the merge unchanged.
        self.assertAlmostEqual(r["coverage"][0], 39.61661341853035)
        self.assertAlmostEqual(r["mutation_score"][0], 19.6)
        self.assertEqual(r["mutation_num"][0], 153.0)


if __name__ == "__main__":
    unittest.main()
