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
from swebench_docker.run_apptainer import APPTAINER_HOST_TIMEOUT_MARGIN, host_timeout_for


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
