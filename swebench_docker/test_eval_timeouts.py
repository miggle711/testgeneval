# Regression tests for pass@k evaluation timeouts: one container evaluates all
# of an instance's samples, but its host-side kill cap was TIMEOUT + 300s
# whatever the sample count, so pass@5's later samples were killed partway,
# mostly passing ones since only those run mutation. --skip_existing then
# treated those short logs as done.

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_evaluation
from swebench_docker.constants import (
    MUTATION_TIMEOUT,
    OVERHEAD_TIMEOUT,
    UNFILTERED_TESTS_FAILED,
    UNFILTERED_TESTS_PASSED,
)
from swebench_docker.context_manager import ExecWrapper
from swebench_docker.run_apptainer import (
    APPTAINER_HOST_TIMEOUT_MARGIN,
    OVERHEAD_STEPS_PER_INSTANCE,
    OVERHEAD_STEPS_PER_MUTATION,
    OVERHEAD_STEPS_PER_RUN,
    filter_runs,
    host_timeout_for,
    parse_slurm_time,
    slurm_budget_warning,
    slurm_seconds_left,
)


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
    each up to TIMEOUT, so a budget of one TIMEOUT per sample killed sphinx-7305
    and sphinx-10353 (5 samples, TIMEOUT 3600, MUTATION_TIMEOUT 7200: 54,300 s)
    before their last samples were evaluated."""

    def setUp(self):
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("APPTAINER_HOST_TIMEOUT", None)
        self.t, self.mt, self.o = 3600, MUTATION_TIMEOUT, OVERHEAD_TIMEOUT

    def expected(self, run_counts, mutation=True):
        t, mt, o = self.t, self.mt, self.o
        b_instance = t + OVERHEAD_STEPS_PER_INSTANCE * o + (t if mutation else 0)
        per_run = t + OVERHEAD_STEPS_PER_RUN * o
        mut = (t + mt + OVERHEAD_STEPS_PER_MUTATION * o) if mutation else 0
        return b_instance + sum(n * per_run + mut for n in run_counts) + APPTAINER_HOST_TIMEOUT_MARGIN

    def test_counts_the_runs_filtering_makes(self):
        # 2 test functions + 2 test methods (setUp isn't a test)
        self.assertEqual(filter_runs(SAMPLE), 4)
        self.assertEqual(filter_runs(""), 0)
        self.assertEqual(filter_runs(None), 0)

    def test_each_sample_counts_its_own_tests(self):
        one = "def test_x():\n    pass\n"
        cap = host_timeout_for(self.t, 3, False, samples=[SAMPLE, one, ""])
        self.assertEqual(cap, self.expected([4 + 1, 1 + 1, 0 + 1]))

    def test_without_mutation_only_test_runs_and_setup_count(self):
        cap = host_timeout_for(self.t, 1, True, samples=[SAMPLE])
        self.assertEqual(cap, self.expected([4 + 1], mutation=False))

    def test_settings_without_filtering_run_once_per_sample(self):
        # first/last/extra (completion_processing) run each sample once
        cap = host_timeout_for(self.t, 2, False, samples=[SAMPLE, SAMPLE], filtered=False)
        self.assertEqual(cap, self.expected([1, 1]))

    def test_never_below_the_earlier_estimate(self):
        for samples in ([SAMPLE] * 5, [""] * 5, ["def test_x():\n    pass\n"] * 5):
            with self.subTest(n_tests=filter_runs(samples[0])):
                self.assertGreaterEqual(host_timeout_for(self.t, 5, False, samples=samples),
                                        host_timeout_for(self.t, 5, False))

    def test_a_worst_case_the_old_budget_cut_now_fits(self):
        # Every test run reaches TIMEOUT and three samples' mutation runs time
        # out (as on sphinx-7305): more than the old 54,300 s, within the new.
        t, mt = self.t, self.mt
        worst = 2 * t + 5 * ((4 + 1) * t) + 3 * (t + mt)
        self.assertGreater(worst, host_timeout_for(t, 5, False))
        self.assertLessEqual(worst, host_timeout_for(t, 5, False, samples=[SAMPLE] * 5))

    def test_the_mutation_limit_is_not_changed(self):
        # The fix is to the outer watchdog only.
        self.assertEqual(MUTATION_TIMEOUT, int(os.environ.get("MUTATION_TIMEOUT", "3600")))

    def test_the_override_still_wins(self):
        os.environ["APPTAINER_HOST_TIMEOUT"] = "1234"
        self.assertEqual(host_timeout_for(self.t, 5, False, samples=[SAMPLE] * 5), 1234)


class OverheadTimeoutTest(unittest.TestCase):
    """Every harness subprocess without its own limit gets OVERHEAD_TIMEOUT, so
    the host budget can account for it; explicit limits are left alone."""

    def run_with(self, **kwargs):
        seen = {}
        def fake_run(cmd, **args):
            seen.update(args)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        log = SimpleNamespace(write=lambda *a, **k: None)
        wrapper = ExecWrapper(subprocess_args={"check": False}, logger=log, default_timeout=600)
        with patch("swebench_docker.context_manager.subprocess.run", fake_run):
            wrapper(["true"], **kwargs)
        return seen

    def test_default_applies_when_no_limit_is_given(self):
        self.assertEqual(self.run_with()["timeout"], 600)

    def test_an_explicit_limit_is_kept(self):
        self.assertEqual(self.run_with(timeout=3600)["timeout"], 3600)

    def test_no_default_means_no_limit(self):
        seen = {}
        def fake_run(cmd, **args):
            seen.update(args)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        log = SimpleNamespace(write=lambda *a, **k: None)
        with patch("swebench_docker.context_manager.subprocess.run", fake_run):
            ExecWrapper(subprocess_args={}, logger=log)(["true"])
        self.assertNotIn("timeout", seen)


class SlurmTimeTest(unittest.TestCase):
    def test_parses_slurm_time_strings(self):
        self.assertEqual(parse_slurm_time("2-03:04:05"), 2 * 86400 + 3 * 3600 + 4 * 60 + 5)
        self.assertEqual(parse_slurm_time("03:04:05"), 3 * 3600 + 4 * 60 + 5)
        self.assertEqual(parse_slurm_time("04:05"), 4 * 60 + 5)
        self.assertEqual(parse_slurm_time("45"), 45)
        self.assertIsNone(parse_slurm_time("UNLIMITED"))
        self.assertIsNone(parse_slurm_time(""))

    def squeue_says(self, left):
        def fake_run(cmd, **kwargs):
            self.assertEqual(cmd[:4], ["squeue", "-h", "-j", "123"])
            return SimpleNamespace(stdout=left + "\n", returncode=0)
        return patch("swebench_docker.run_apptainer.subprocess.run", fake_run)

    def test_warns_when_the_budget_exceeds_the_jobs_time_left(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), self.squeue_says("1-00:00:00"):
            warning = slurm_budget_warning(200000, "inst-1")
        self.assertIn("host timeout 200000s exceeds the 86400s left in Slurm job 123", warning)

    def test_no_warning_when_the_job_has_time(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), self.squeue_says("3-00:00:00"):
            self.assertIsNone(slurm_budget_warning(200000, "inst-1"))

    def test_no_warning_for_an_unlimited_job(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), self.squeue_says("UNLIMITED"):
            self.assertIsNone(slurm_budget_warning(200000, "inst-1"))

    def test_outside_slurm_there_is_no_limit(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLURM_JOB_ID", None)
            self.assertIsNone(slurm_seconds_left())


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
