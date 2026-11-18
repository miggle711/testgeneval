"""#125: a harness exception in one sample must not end the instance.

run_tests_task used to catch any exception, write `Tests Errored` and call
exit(), which ended the container with status 0 and lost every remaining
sample. These call the real method on a stand-in for the context manager, so
no container is needed.
"""

import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

import importlib.util
import json

from swebench_docker.constants import MAP_VERSION_TO_INSTALL, TESTS_ERROR, TESTS_PASSED, TESTS_TIMEOUT
from swebench_docker.context_manager import TaskEnvContextManager

REPO = "django/django"
VERSION = sorted(MAP_VERSION_TO_INSTALL[REPO])[0]


class FakeLog:
    def __init__(self):
        self.lines = []

    def write(self, message, mode="a", level=None):
        self.lines.append(message)

    def text(self):
        return "\n".join(self.lines)


def fake_manager(exec_fn):
    return SimpleNamespace(
        instance={"repo": REPO, "version": VERSION},
        timeout=60,
        cmd_conda_run="conda run -n testbed",
        exec=exec_fn,
        log=FakeLog(),
        _any_test_passed=lambda path, returncode: returncode == 0,
        _log_function_coverage=lambda instance, data: None,
        run_mutation_testing=lambda *args: None,
    )


INSTANCE = {"test_cmd": "./tests/runtests.py x", "code_file": "django/x.py"}


class RunTestsTaskErrors(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.dir = tempfile.TemporaryDirectory()
        os.chdir(self.dir.name)
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(os.chdir, self.cwd)

    def run_task(self, manager, log_data=True):
        return TaskEnvContextManager.run_tests_task(
            manager, INSTANCE, log_data=log_data, skip_mutation=False
        )

    def test_missing_coverage_json_is_a_coverage_failure(self):
        # The real case in #125: the tests passed, `coverage json` wrote no
        # file, and the unguarded open() raised.
        manager = fake_manager(lambda *a, **k: subprocess.CompletedProcess(a, 0))
        ran, passed = self.run_task(manager)
        self.assertEqual((ran, passed), (True, True))
        self.assertIn(TESTS_PASSED, manager.log.text())
        self.assertIn("CoverageFAIL:django/x.py no coverage.json written", manager.log.text())
        self.assertNotIn(TESTS_ERROR, manager.log.text())

    def test_an_error_in_a_samples_run_is_recorded_and_returns(self):
        def boom(*a, **k):
            raise RuntimeError("boom")
        manager = fake_manager(boom)
        self.assertEqual(self.run_task(manager), (False, False))
        self.assertIn(f"{TESTS_ERROR}: boom", manager.log.text())

    def test_an_error_while_filtering_tests_still_stops_the_instance(self):
        # Returning "failed" here would drop a generated test for a harness
        # fault and score it as a generation failure. It must raise, and not
        # with exit()'s status 0.
        def boom(*a, **k):
            raise RuntimeError("boom")
        manager = fake_manager(boom)
        with self.assertRaises(RuntimeError):
            self.run_task(manager, log_data=False)
        self.assertNotIn(TESTS_ERROR, manager.log.text())


def load_usable_score():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "usable_score.py")
    spec = importlib.util.spec_from_file_location("usable_score_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcome(log_text):
    """usable_score.py's classification of one sample's block."""
    return load_usable_score().classify(log_text)[0]


class WhichTimeoutIsWhich(unittest.TestCase):
    """fyp-paper#7 open item 4: with OVERHEAD_TIMEOUT, harness steps can time
    out too. Only the generated tests' own command may be recorded as a test
    timeout (scored 0); any other step's timeout is a harness fault (unknown)."""

    def setUp(self):
        self.cwd = os.getcwd()
        self.dir = tempfile.TemporaryDirectory()
        os.chdir(self.dir.name)
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(os.chdir, self.cwd)

    def run_task(self, manager, log_data=True):
        manager.log.write(">>>>> Tests config full pred\n")
        return TaskEnvContextManager.run_tests_task(manager, INSTANCE, log_data=log_data, skip_mutation=False)

    @staticmethod
    def passing_exec(timeout_on=None):
        """Tests pass; `coverage json` writes the code file's coverage; the
        command whose text contains `timeout_on` times out instead."""
        def run(cmd, **kwargs):
            text = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            if timeout_on and timeout_on in text:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 600))
            if "coverage json" in text:
                with open("coverage.json", "w") as f:
                    json.dump({"files": {"django/x.py": {"summary": {"percent_covered": 50}}}}, f)
            return subprocess.CompletedProcess(cmd, 0)
        return run

    def test_the_test_commands_own_timeout_is_a_test_timeout(self):
        manager = fake_manager(self.passing_exec(timeout_on="runtests.py"))
        self.assertEqual(self.run_task(manager), (False, False))
        self.assertIn(f"{TESTS_TIMEOUT} after 60 seconds", manager.log.text())

    def test_a_coverage_json_timeout_is_not_a_test_timeout(self):
        manager = fake_manager(self.passing_exec(timeout_on="coverage json"))
        ran, passed = self.run_task(manager)
        text = manager.log.text()
        self.assertEqual((ran, passed), (True, True))
        self.assertIn("CoverageFAIL:django/x.py harness step timed out after 600s: coverage json", text)
        self.assertNotIn(TESTS_TIMEOUT, text)
        self.assertNotIn("no coverage.json written", text)
        self.assertNotIn("Running mutation testing", text)
        self.assertTrue(outcome(text).startswith("unknown"), outcome(text))

    def test_a_mutation_step_timeout_is_not_a_test_timeout(self):
        manager = fake_manager(self.passing_exec())
        manager.mutation_scope = "function"
        def init_hangs(*args):
            raise subprocess.TimeoutExpired(["cosmic-ray", "init", "mutation.toml", "mutation.sqlite"], 60)
        manager.run_mutation_testing = init_hangs
        ran, passed = self.run_task(manager)
        text = manager.log.text()
        self.assertEqual((ran, passed), (True, True))
        self.assertIn("FunctionMutationFAIL: harness step timed out after 60s: cosmic-ray init", text)
        self.assertNotIn(TESTS_TIMEOUT, text)
        self.assertTrue(outcome(text).startswith("unknown"), outcome(text))

    def test_any_other_timeout_is_a_harness_error(self):
        manager = fake_manager(self.passing_exec())
        def hangs(path, returncode):
            raise subprocess.TimeoutExpired(["junit-read"], 600)
        manager._any_test_passed = hangs
        self.assertEqual(self.run_task(manager), (False, False))
        text = manager.log.text()
        self.assertIn(f"{TESTS_ERROR}: harness step timed out after 600s", text)
        self.assertNotIn(TESTS_TIMEOUT, text)

    def test_a_filtering_runs_test_timeout_drops_that_test(self):
        # A per-test filtering run has only the test command; its timeout
        # drops the test (failed), as before, and writes no log markers.
        manager = fake_manager(self.passing_exec(timeout_on="runtests.py"))
        self.assertEqual(TaskEnvContextManager.run_tests_task(
            manager, INSTANCE, log_data=False, skip_mutation=False), (False, False))
        self.assertNotIn(TESTS_TIMEOUT, manager.log.text())
        self.assertNotIn(TESTS_ERROR, manager.log.text())

if __name__ == "__main__":
    unittest.main()
