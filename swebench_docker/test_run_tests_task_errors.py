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

from swebench_docker.constants import MAP_VERSION_TO_INSTALL, TESTS_ERROR, TESTS_PASSED
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


if __name__ == "__main__":
    unittest.main()
