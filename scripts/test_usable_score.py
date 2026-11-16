# Copyright (c) Meta Platforms, Inc. and affiliates.

import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict

# scripts/ is not a package, so import by path (as test_merge_any_pass_logs.py does).
_SPEC = importlib.util.spec_from_file_location(
    "usable_score", os.path.join(os.path.dirname(__file__), "usable_score.py")
)
usable_score = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(usable_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swebench_docker.constants import (  # noqa: E402
    ANY_TESTS_FAILED,
    ANY_TESTS_PASSED,
    TESTS_CONFIG,
    TESTS_ERROR,
    TESTS_FAILED,
    TESTS_PASSED,
    TESTS_TIMEOUT,
    UNFILTERED_TESTS_FAILED,
    UNFILTERED_TESTS_PASSED,
)

PREFIX = "[django__django__4.1] [django__django-15380]"


def block(*messages, setting="full"):
    """One sample's part of an eval log, built the way the harness builds it:
    every message goes through LogWrapper.write (swebench_docker/
    context_manager.py), which writes "<prefix> <message> \\n". The messages
    below are the strings the harness passes to it."""
    return f"{setting} pred\n \n" + "".join(f"{PREFIX} {m} \n" for m in messages)


# evaluate_instance.py, when no test passed alone: nothing is run.
NO_TEST_PASSED = block("TestsTime: 0.0", TESTS_FAILED, UNFILTERED_TESTS_FAILED)

# run_tests_task, when the filtered suite fails once its tests run together.
SUITE_FAILED = block(
    "TestsTime: 3.2",
    f"\n{TESTS_FAILED}\n",
    f"\n{ANY_TESTS_PASSED}\n",
    "Test script run successful",
    UNFILTERED_TESTS_FAILED,
)

TEST_TIMEOUT = block(
    "Test script run timed out", f"{TESTS_TIMEOUT} after 300 seconds\n", UNFILTERED_TESTS_FAILED
)

HARNESS_ERROR = block("Test script run failed", f"{TESTS_ERROR}: boom")


def passing(*after_coverage, ended=True, started=True):
    """A sample whose filtered suite passed. `after_coverage` is what
    mutation testing logged."""
    messages = ["TestsTime: 12.3", f"\n{TESTS_PASSED}\n", "\nCoverageLOG: 39.6%\n"]
    if started:
        messages.append("Running mutation testing")
    messages.extend(after_coverage)
    if ended:
        messages += [f"\n{ANY_TESTS_PASSED}\n", "Test script run successful", UNFILTERED_TESTS_FAILED]
    return block(*messages)


def scored(score, done, total):
    """Function-scope mutation as testgeneval#102 logs it."""
    return passing(
        f"\nFunctionMutationLOG: {score}%",
        f"\nFunctionMutationNum: {done}",
        f"\nFunctionMutationTotal: {total}",
    )


class TestClassify(unittest.TestCase):
    def outcome(self, text):
        return usable_score.classify(text)[0]

    def test_a_sample_that_never_ran_its_suite_scores_zero(self):
        self.assertEqual(self.outcome(NO_TEST_PASSED), "zero_no_test_passed")
        self.assertEqual(self.outcome(SUITE_FAILED), "zero_suite_failed")
        self.assertEqual(self.outcome(TEST_TIMEOUT), "zero_test_timeout")

    def test_harness_error_and_cut_off_log_are_unknown_not_zero(self):
        self.assertEqual(self.outcome(HARNESS_ERROR), "unknown_harness_error")
        self.assertEqual(self.outcome("full pred\n \n"), "unknown_cut_off")
        self.assertEqual(self.outcome(passing(ended=False, started=False)), "unknown_cut_off")

    def test_finished_function_mutation_is_a_score(self):
        self.assertEqual(usable_score.classify(scored(50.0, 4, 4)), ("score", 50.0, 4.0, 4.0))
        # A score of 0 is still a score.
        self.assertEqual(self.outcome(scored(0.0, 4, 4)), "score")

    def test_fewer_results_than_target_mutants_is_incomplete(self):
        cut_short = passing(
            "MutationTimeout",
            "\nFunctionMutationLOG: 50.0%",
            "\nFunctionMutationNum: 2",
            "\nFunctionMutationTotal: 4",
        )
        self.assertEqual(usable_score.classify(cut_short), ("unknown_mutation_incomplete", 50.0, 2.0, 4.0))
        nothing_finished = passing("\nFunctionMutationNum: 0", "\nFunctionMutationTotal: 4")
        self.assertEqual(self.outcome(nothing_finished), "unknown_mutation_incomplete")

    def test_no_mutant_and_no_target_range_are_told_apart(self):
        self.assertEqual(self.outcome(passing("\nFunctionMutationTotal: 0")), "no_target_mutants")
        self.assertEqual(
            self.outcome(passing("\nFunctionMutationFAIL: no target function range")),
            "target_unresolved",
        )
        self.assertEqual(
            self.outcome(passing("\nFunctionMutationFAIL: subprocess exited 1, stderr: x")),
            "unknown_function_score_failed",
        )

    def test_passing_sample_with_no_mutation_run_is_unknown(self):
        # As with --skip_mutation, or when the file is missing from the coverage data.
        self.assertEqual(self.outcome(passing(started=False)), "unknown_no_mutation")

    def test_logs_written_before_function_mutation_total_existed(self):
        whole = ["\nMutationNum: 153", "\nMutationLOG: 19.6%", "\nMutationUncertainty: 3.1"]
        func = ["\nFunctionMutationLOG: 25.0%", "\nFunctionMutationNum: 8"]
        self.assertEqual(usable_score.classify(passing(*whole, *func)), ("score", 25.0, 8.0, None))
        self.assertEqual(
            self.outcome(passing("MutationTimeout", *whole, *func)), "unknown_mutation_incomplete"
        )
        # A whole-file score and no function score: no mutant fell in the target.
        self.assertEqual(self.outcome(passing(*whole)), "no_target_mutants")
        self.assertEqual(self.outcome(passing("\nMutationFAIL")), "unknown_mutation_failed")
        # Mutation started and the block ends: the container was killed.
        self.assertEqual(self.outcome(passing(ended=False)), "unknown_mutation_killed")


class TestRows(unittest.TestCase):
    def test_short_run_in_an_old_log_is_re_marked_incomplete(self):
        def row(done, total=None):
            return {"run": "R", "instance": "i-1-1", "outcome": "score",
                    "func_score": 50.0, "func_done": done, "func_total": total}

        rows = [row(6.0), row(4.0), row(6.0)]
        self.assertEqual(usable_score.flag_short_runs(rows), 1)
        self.assertEqual([r["outcome"] for r in rows],
                         ["score", "unknown_mutation_incomplete", "score"])
        # With a logged total, classify() has already decided; nothing is re-marked.
        rows = [row(6.0, 6.0), row(4.0, 4.0)]
        self.assertEqual(usable_score.flag_short_runs(rows), 0)

    def test_value_of_a_sample_under_each_way_of_handling_unknowns(self):
        score = {"outcome": "score", "func_score": 40.0}
        zero = {"outcome": "zero_suite_failed", "func_score": None}
        partial = {"outcome": "unknown_mutation_incomplete", "func_score": 60.0}
        missing = {"outcome": "unknown_missing", "func_score": None}
        for variant in ("known", "unknown_as_zero", "unknown_as_partial"):
            self.assertEqual(usable_score.value(score, variant), 40.0)
            self.assertEqual(usable_score.value(zero, variant), 0.0)
        self.assertIsNone(usable_score.value(partial, "known"))
        self.assertEqual(usable_score.value(partial, "unknown_as_zero"), 0.0)
        self.assertEqual(usable_score.value(partial, "unknown_as_partial"), 60.0)
        self.assertIsNone(usable_score.value(missing, "known"))
        self.assertIsNone(usable_score.value(missing, "unknown_as_partial"))
        self.assertEqual(usable_score.value(missing, "unknown_as_zero"), 0.0)

    def test_interval_of_identical_values_is_that_value(self):
        self.assertEqual(usable_score.boot_ci([5.0] * 10, n=200), (5.0, 5.0))


class TestFromLogs(unittest.TestCase):
    """Three instances, 2 samples each, written as eval logs.

                  instruct                 kg_only
    A (11111-1)   score 50, no test passed score 100, score 50
    B (22222-2)   score 40, (missing)      suite failed, score 100
    C (33333-3)   no test passed x2        no test passed x2
    """

    A, B, C = "django__django-11111-1", "django__django-22222-2", "django__django-33333-3"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        other_setting = block("TestsTime: 1.0", f"\n{TESTS_PASSED}\n", setting="first")
        logs = {
            ("instruct", self.A): [scored(50.0, 4, 4), other_setting, NO_TEST_PASSED],
            ("kg_only", self.A): [scored(100.0, 4, 4), scored(50.0, 4, 4)],
            ("instruct", self.B): [scored(40.0, 5, 5)],
            ("kg_only", self.B): [SUITE_FAILED, scored(100.0, 5, 5)],
            ("instruct", self.C): [NO_TEST_PASSED, NO_TEST_PASSED],
            ("kg_only", self.C): [NO_TEST_PASSED, NO_TEST_PASSED],
        }
        for (arm, instance), blocks in logs.items():
            # Logs sit one directory down, as run_evaluation.py writes them.
            folder = os.path.join(self.tmp, arm, "model")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, f"{instance}.model.full.eval.log"), "w") as f:
                f.write("Task Metadata:\n" + TESTS_CONFIG.join([""] + blocks))
        self.specs = [f"M:{arm}:{os.path.join(self.tmp, arm)}" for arm in usable_score.ARMS]

    def test_every_sample_gets_a_row_and_other_settings_are_ignored(self):
        rows, _ = usable_score.collect(self.specs, k=2)
        self.assertEqual(len(rows), 12)
        got = {(r["arm"], r["instance"], r["sample"]): r["outcome"] for r in rows}
        self.assertEqual(got[("instruct", self.A, 1)], "zero_no_test_passed")
        self.assertEqual(got[("instruct", self.B, 0)], "score")
        self.assertEqual(got[("instruct", self.B, 1)], "unknown_missing")
        self.assertEqual(got[("kg_only", self.B, 0)], "zero_suite_failed")
        self.assertEqual({r["repo"] for r in rows}, {"django__django"})

    def test_paired_instance_means(self):
        rows, _ = usable_score.collect(self.specs, k=2)
        per = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["instance"] != self.C:
                per[r["instance"]][r["arm"]].append(r)
        # Unknown left out: B's instruct mean is its one known sample.
        self.assertEqual(sorted(usable_score.paired(per, "known", 2)), [(25.0, 75.0), (40.0, 50.0)])
        # The protocol's rule: B is dropped, because one of its samples is unknown.
        self.assertEqual(usable_score.paired(per, "known", 2, strict=True), [(25.0, 75.0)])
        # Sensitivity analysis: the unknown sample counts as 0.
        self.assertEqual(sorted(usable_score.paired(per, "unknown_as_zero", 2)),
                         [(20.0, 50.0), (25.0, 75.0)])

    def run_main(self, *argv):
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["usable_score.py", *argv]
        try:
            with contextlib.redirect_stdout(out):
                usable_score.main()
        finally:
            sys.argv = old
        return out.getvalue()

    def test_report_and_rerun_from_csv(self):
        csv_path = os.path.join(self.tmp, "usable_score.csv")
        report = self.run_main("--out", csv_path, "--k", "2", *self.specs)
        lines = [" ".join(line.split()) for line in report.splitlines()]
        self.assertIn("2 eligible", lines)
        self.assertIn("1 undetermined (no sample reached mutation)", lines)
        # A: 75 - 25 = +50; B: 50 - 40 = +10; the mean is +30.
        known = next(l for l in lines if l.startswith("left out (mean of the known samples)"))
        self.assertIn("2 32.5 62.5 +30.0", known)
        # With C (0 in both arms): (50 + 10 + 0) / 3 = +20.
        with_c = next(l for l in lines if l.startswith("left out, plus undetermined instances"))
        self.assertIn("3 21.7 41.7 +20.0", with_c)
        self.assertIn("instances with all 2 samples known, both arms too few instances (1)", lines)

        again = self.run_main("--from-csv", csv_path, "--k", "2")
        self.assertEqual(again, report[report.index("\nSamples re-marked"):])


if __name__ == "__main__":
    unittest.main()
