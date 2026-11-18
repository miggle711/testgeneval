# Regression tests for pass@k evaluation timeouts: one container evaluates all
# of an instance's samples, but its host-side kill cap was TIMEOUT + 300s
# whatever the sample count, so pass@5's later samples were killed partway,
# mostly passing ones since only those run mutation. --skip_existing then
# treated those short logs as done.

import os
import tempfile
import unittest
from unittest.mock import patch

import run_evaluation
from swebench_docker.constants import (
    MUTATION_TIMEOUT,
    UNFILTERED_TESTS_FAILED,
    UNFILTERED_TESTS_PASSED,
)
from swebench_docker.run_apptainer import APPTAINER_HOST_TIMEOUT_MARGIN, filter_runs, host_timeout_for


class HostTimeoutTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("APPTAINER_HOST_TIMEOUT", None)

    def test_scales_with_the_sample_count(self):
        one = host_timeout_for(3600, 1, skip_mutation=False)
        five = host_timeout_for(3600, 5, skip_mutation=False)
        self.assertEqual(one, 3600 + MUTATION_TIMEOUT + APPTAINER_HOST_TIMEOUT_MARGIN)
        self.assertEqual(five, 5 * (3600 + MUTATION_TIMEOUT) + APPTAINER_HOST_TIMEOUT_MARGIN)

    def test_no_mutation_budget_when_mutation_is_skipped(self):
        self.assertEqual(host_timeout_for(3600, 5, skip_mutation=True),
                         5 * 3600 + APPTAINER_HOST_TIMEOUT_MARGIN)

    def test_fixed_override_still_wins(self):
        os.environ["APPTAINER_HOST_TIMEOUT"] = "1234"
        self.assertEqual(host_timeout_for(3600, 5, skip_mutation=False), 1234)


SAMPLE = """import pytest
from pkg import f

def test_one():
    assert f(1) == 1

@pytest.mark.parametrize("x", [1, 2])
def test_two(x):
    assert f(x)

class TestThing(unittest.TestCase):
    def setUp(self):
        self.x = 1

    def test_a(self):
        assert self.x

    def test_b(self):
        assert not f(0)
"""


class HostTimeoutFromTheTestsTest(unittest.TestCase):
    """fyp-paper#7 open item 4: filtering runs every generated test on its own,
    each up to TIMEOUT, so a cap of one TIMEOUT per sample killed sphinx-7305
    (5 samples, TIMEOUT 3600, MUTATION_TIMEOUT 7200: cap 54,300 s) before its
    last samples were evaluated."""

    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("APPTAINER_HOST_TIMEOUT", None)

    def test_counts_the_runs_filtering_makes(self):
        # 2 test functions + 2 test methods (setUp isn't a test)
        self.assertEqual(filter_runs(SAMPLE), 4)
        self.assertEqual(filter_runs(""), 0)
        self.assertEqual(filter_runs(None), 0)

    def test_the_cap_is_the_sum_of_the_inner_limits(self):
        t, mt = 3600, MUTATION_TIMEOUT
        cap = host_timeout_for(t, 2, skip_mutation=False, samples=[SAMPLE, SAMPLE])
        per_sample = (4 + 1) * t + t + mt          # filtering, combined run, init, mutation
        self.assertEqual(cap, (t + t) + 2 * per_sample + APPTAINER_HOST_TIMEOUT_MARGIN)

    def test_without_mutation_only_the_test_runs_count(self):
        t = 3600
        cap = host_timeout_for(t, 1, skip_mutation=True, samples=[SAMPLE])
        self.assertEqual(cap, t + (4 + 1) * t + APPTAINER_HOST_TIMEOUT_MARGIN)

    def test_never_below_the_earlier_estimate(self):
        for samples in ([SAMPLE] * 5, [""] * 5, ["def test_x():\n    pass\n"] * 5):
            with self.subTest(n_tests=filter_runs(samples[0])):
                self.assertGreaterEqual(host_timeout_for(3600, 5, False, samples=samples),
                                        host_timeout_for(3600, 5, False))

    def test_a_worst_case_the_old_cap_cut_now_fits(self):
        # Every sample's filtering runs and combined run each reach TIMEOUT,
        # and three samples' mutation runs time out (as on sphinx-7305): more
        # than the old cap of 54,300 s, and within the new one.
        t, mt = 3600, MUTATION_TIMEOUT
        worst = (t + t) + 5 * ((4 + 1) * t) + 3 * (t + mt)
        self.assertGreater(worst, host_timeout_for(t, 5, False))
        self.assertLessEqual(worst, host_timeout_for(t, 5, False, samples=[SAMPLE] * 5))

    def test_the_override_still_wins(self):
        os.environ["APPTAINER_HOST_TIMEOUT"] = "1234"
        self.assertEqual(host_timeout_for(3600, 5, False, samples=[SAMPLE] * 5), 1234)


class SkipExistingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = os.path.join(self._tmp.name, "x.full.eval.log")

    def _write(self, n_finished, path=None):
        with open(path or self.log, "w") as f:
            for i in range(n_finished):
                marker = UNFILTERED_TESTS_PASSED if i % 2 else UNFILTERED_TESTS_FAILED
                f.write(f">>>>> Tests config full pred\n...\n{marker}\n")
            f.write(">>>>> Tests config full pred\nkilled partway\n")

    def test_counts_finished_samples(self):
        self._write(3)
        self.assertEqual(run_evaluation.completed_samples(self.log), 3)

    def test_partial_log_is_set_aside_not_overwritten(self):
        self._write(2)
        first = run_evaluation.set_aside_partial_log(self.log)
        self.assertTrue(first.endswith(".partial"))
        self.assertFalse(os.path.exists(self.log))
        self._write(1)
        second = run_evaluation.set_aside_partial_log(self.log)
        self.assertTrue(second.endswith(".partial.1"))
        self.assertTrue(os.path.exists(first) and os.path.exists(second))


if __name__ == "__main__":
    unittest.main()
