# The per-instance target-mutant count (fyp-paper#7, eligibility rule 3):
# what it logs for each outcome, where in the log it lands, and that the
# per-sample results read from the log do not change. None of these need
# cosmic-ray; the count itself is checked against a real cosmic-ray run in
# test_mutation_scope.py. Run with:
#   python -m unittest swebench_docker.test_target_mutant_count -v

import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swebench_docker import evaluate_instance
from swebench_docker.constants import (
    TESTS_CONFIG,
    TESTS_FAILED,
    TESTS_PASSED,
    UNFILTERED_TESTS_FAILED,
)
from swebench_docker.context_manager import (
    COUNT_CONFIG,
    LogWrapper,
    TaskEnvContextManager,
)
from swebench_docker.mutation_scope import COUNT_DB, count_script
from swebench_docker.swebench_utils import get_logs_eval

INSTANCE = {"code_file": "calc.py", "repo": "django/django", "version": "3.0"}


class FakeLog:
    def __init__(self):
        self.text = ""

    def write(self, message, level=None, mode=None):
        if level is None:
            self.text += message


class FakeExec:
    """Stands in for ExecWrapper: answers each command from `answers`, a
    list of (returncode, stdout) or an exception to raise."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append((cmd, kwargs))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(returncode=answer[0], stdout=answer[1])


def _manager(answers, ranges=((5, 8),), conda="conda run -n testbed "):
    tcm = TaskEnvContextManager.__new__(TaskEnvContextManager)
    tcm.cmd_conda_run = conda
    tcm.timeout = 60
    tcm.log = FakeLog()
    tcm.exec = FakeExec(answers)
    tcm._resolve_target_ranges = lambda instance: list(ranges) if ranges else None
    return tcm


class CountLogTest(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        tmp = tempfile.TemporaryDirectory()
        os.chdir(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, cwd)
        with open("calc.py", "w") as source:
            source.write("def target(x):\n    return x + 1\n")

    def _count(self, tcm, specifications=None):
        tcm.log_target_mutant_count(INSTANCE, specifications or {})
        return tcm.log.text

    def test_a_count_is_logged_with_the_time_it_took(self):
        tcm = _manager([(0, ""), (0, "noise\nTARGET_MUTANT_COUNT 7\n")])
        log = self._count(tcm)
        self.assertRegex(log, r"^\nTargetMutantCount: 7\nTargetMutantCountTime: \d+\.\d$")

    def test_zero_mutants_is_a_count_not_a_failure(self):
        log = self._count(_manager([(0, ""), (0, "TARGET_MUTANT_COUNT 0\n")]))
        self.assertIn("\nTargetMutantCount: 0\n", log)
        self.assertNotIn("FAIL", log)

    def test_an_unresolved_target_runs_nothing(self):
        tcm = _manager([], ranges=None)
        log = self._count(tcm)
        self.assertIn("\nTargetMutantCountFAIL: no target function range\n", log)
        self.assertEqual(tcm.exec.commands, [])
        self.assertNotIn("\nTargetMutantCount: ", log)

    def test_a_failed_init_is_not_logged_as_zero(self):
        tcm = _manager([(1, "FileNotFoundError: gone.py")])
        log = self._count(tcm)
        self.assertIn(
            "\nTargetMutantCountFAIL: cosmic-ray init exited 1: FileNotFoundError: gone.py", log
        )
        self.assertEqual(len(tcm.exec.commands), 1)
        self.assertNotIn("no target function range", log)
        self.assertNotIn("\nTargetMutantCount: ", log)

    def test_a_missing_source_fails_before_target_resolution_or_cosmic_ray(self):
        os.remove("calc.py")
        tcm = _manager([])
        with patch.object(tcm, "_resolve_target_ranges") as resolve:
            log = self._count(tcm)
        self.assertIn("\nTargetMutantCountFAIL: source file not found: calc.py", log)
        self.assertNotIn("\nTargetMutantCount: ", log)
        resolve.assert_not_called()
        self.assertEqual(tcm.exec.commands, [])

    def test_a_count_script_that_prints_no_count_is_not_logged_as_zero(self):
        tcm = _manager([(0, ""), (1, "Traceback: no such table")])
        log = self._count(tcm)
        self.assertIn("\nTargetMutantCountFAIL: subprocess exited 1", log)
        self.assertIn("no such table", log)
        self.assertNotIn("\nTargetMutantCount: ", log)

    def test_a_timeout_is_logged_and_does_not_raise(self):
        tcm = _manager([subprocess.TimeoutExpired("cosmic-ray init", 60)])
        log = self._count(tcm)
        self.assertIn("\nTargetMutantCountFAIL: cosmic-ray init timed out after 60 seconds", log)

    def test_any_other_error_is_logged_and_does_not_raise(self):
        tcm = _manager([OSError("no conda")])
        self.assertIn("\nTargetMutantCountFAIL: OSError('no conda')", self._count(tcm))

    def test_it_uses_its_own_config_and_db_and_removes_them(self):
        tcm = _manager([(0, ""), (0, "TARGET_MUTANT_COUNT 3\n")])

        def exec_and_create(cmd, **kwargs):
            # cosmic-ray init would create the db; the config is there by then.
            if "init" in cmd:
                self.assertTrue(os.path.exists(COUNT_CONFIG))
                open(COUNT_DB, "w").close()
            return FakeExec.__call__(tcm.exec, cmd, **kwargs)

        with patch.object(tcm, "exec", side_effect=exec_and_create) as run:
            self._count(tcm)
        init = run.call_args_list[0].args[0]
        self.assertEqual(init[-3:], ["init", COUNT_CONFIG, COUNT_DB])
        self.assertEqual(run.call_args_list[0].kwargs["timeout"], 60)
        self.assertNotIn("mutation.sqlite", " ".join(init))
        self.assertNotEqual(COUNT_DB, "mutation.sqlite")
        self.assertNotEqual(COUNT_CONFIG, "mutation.toml")
        self.assertEqual(os.listdir("."), ["calc.py"])

    def test_the_script_reads_the_count_db_only(self):
        script = count_script("calc.py", [(5, 8)])
        self.assertIn(f"WorkDB({COUNT_DB!r}, WorkDB.Mode.open)", script)
        self.assertNotIn("set_multiple_results", script)

    def test_conda_images_go_through_the_testbed_env_and_python_images_do_not(self):
        answers = [(0, ""), (0, "TARGET_MUTANT_COUNT 1\n")]
        conda = _manager(answers)
        self._count(conda)
        self.assertEqual(conda.exec.commands[0][0][:5],
                         ["conda", "run", "-n", "testbed", "cosmic-ray"])
        plain = _manager(answers)
        self._count(plain, {"image": "python"})
        self.assertEqual(plain.exec.commands[0][0][0], "cosmic-ray")
        self.assertEqual(plain.exec.commands[1][0][:2], ["python", "-c"])


class FakeManager:
    """The context manager main() opens, recording what main() asks of it."""

    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def apply_patch(self, patch_text, patch_type):
        self.calls.append("apply_patch")
        return True

    def log_target_mutant_count(self, instance, specifications):
        self.calls.append(("count", specifications))


class MainOrderTest(unittest.TestCase):
    def _main(self, skip_mutation):
        calls = []
        instance = {
            "instance_id": "django__django-1", "id": "django__django-1-1",
            "repo": "django/django", "version": "3.0",
            "patch": "p", "test_patch": "t", "preds": {"full": ["a", "b"]},
        }
        with patch.object(evaluate_instance, "TaskEnvContextManager",
                          lambda *a, **k: FakeManager(calls)), \
             patch.object(evaluate_instance, "full_processing",
                          lambda *a: calls.append("samples")), \
             patch.dict(evaluate_instance.MAP_VERSION_TO_INSTALL,
                        {"django/django": {"3.0": {"marker": 1}}}), \
             self.assertLogs(evaluate_instance.logger):
            evaluate_instance.main(instance, "testbed", "full", ".", ".", 60,
                                   skip_mutation=skip_mutation)
        return calls

    def test_counted_once_after_the_patches_and_before_any_sample(self):
        self.assertEqual(
            self._main(skip_mutation=False),
            ["apply_patch", "apply_patch", ("count", {"marker": 1}), "samples"],
        )

    def test_not_counted_when_mutation_is_skipped(self):
        self.assertEqual(self._main(skip_mutation=True),
                         ["apply_patch", "apply_patch", "samples"])


class LogReadersTest(unittest.TestCase):
    """The count sits before the first sample block, which get_logs_eval
    does not read, so every per-sample result stays as it was."""

    def _log(self, with_count):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "django__django-1-1.model.full.eval.log")
        log = LogWrapper(path, prefix="[testbed] [django__django-1]")
        log.write("Task Metadata:")
        if with_count:
            log.write("\nTargetMutantCount: 8")
            log.write("\nTargetMutantCountTime: 1.2")
        log.write(f"{TESTS_CONFIG}full pred\n")
        log.write("TestsTime: 1.5")
        log.write(f"\n{TESTS_PASSED}\n")
        log.write("\nCoverageLOG: 42.0%\n")
        log.write("\nFunctionMutationLOG: 75.0%")
        log.write("\nFunctionMutationNum: 8")
        log.write("\nFunctionMutationTotal: 8")
        log.write(f"{TESTS_CONFIG}full pred\n")
        log.write("TestsTime: 0.0")
        log.write(TESTS_FAILED)
        log.write(UNFILTERED_TESTS_FAILED)
        return path

    def test_sample_results_are_the_same_with_and_without_the_count(self):
        without = get_logs_eval(self._log(with_count=False))
        with_count = get_logs_eval(self._log(with_count=True))
        self.assertEqual(with_count, without)
        self.assertEqual(with_count["full"]["tests_passed"], [True, False])
        self.assertEqual(with_count["full"]["function_mutation_num"][0], 8.0)

    def test_the_count_lands_before_the_first_sample_block(self):
        with open(self._log(with_count=True)) as f:
            head = f.read().split(TESTS_CONFIG)[0]
        self.assertIn("\nTargetMutantCount: 8 \n", head)


if __name__ == "__main__":
    unittest.main()
