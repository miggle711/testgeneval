# Target-function mutation (testgeneval#10): with mutation_scope "function",
# only the target function's mutants run, and the score is over those. Runs
# real cosmic-ray on a small module, so it is skipped where cosmic_ray is not
# installed. Testbed images may contain older releases. Run with:
#   python -m unittest swebench_docker.test_mutation_scope -v

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from swebench_docker.context_manager import ExecWrapper, TaskEnvContextManager

try:
    import cosmic_ray  # noqa: F401
    HAVE_COSMIC_RAY = True
except ImportError:
    HAVE_COSMIC_RAY = False

CALC = """def add(a, b):
    return a + b


def target(x):
    if x > 0:
        return x * 2
    return -x
"""
TARGET = [(5, 8)]

TEST = """from calc import target
def test_target():
    assert target(3) == 6
    assert target(-2) == 2
"""


class FakeLog:
    def __init__(self):
        self.text = ""

    def write(self, message, level=None, mode=None):
        if level is None:
            self.text += message


def _manager(scope, mutation_timeout=600):
    tcm = TaskEnvContextManager.__new__(TaskEnvContextManager)
    tcm.cmd_conda_run = ""
    tcm.timeout = 600
    tcm.mutation_timeout = mutation_timeout
    tcm.mutation_scope = scope
    tcm.log = FakeLog()
    tcm.exec = ExecWrapper(
        subprocess_args={"capture_output": True, "text": True}, logger=tcm.log
    )
    tcm._resolve_target_ranges = lambda instance: TARGET
    return tcm


def _value(log, key):
    m = re.search(rf"\n{key}: ([\d.]+)", log)
    return float(m.group(1)) if m else None


@unittest.skipUnless(HAVE_COSMIC_RAY, "cosmic_ray is not installed")
class MutationScopeTest(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.dir = tempfile.TemporaryDirectory()
        os.chdir(self.dir.name)
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(os.chdir, self.cwd)
        with open("calc.py", "w") as f:
            f.write(CALC)
        with open("test_calc.py", "w") as f:
            f.write(TEST)
        # cosmic-ray, cr-rate and cr-report sit next to this interpreter.
        bin_dir = os.path.dirname(sys.executable)
        env = patch.dict(os.environ, {"PATH": bin_dir + os.pathsep + os.environ["PATH"]})
        env.start()
        self.addCleanup(env.stop)

    def _run(self, scope, test_cmd=None, mutation_timeout=600):
        tcm = _manager(scope, mutation_timeout)
        tcm.run_mutation_testing(
            {"code_file": "calc.py"}, {}, test_time=1,
            test_cmd=test_cmd or f"{sys.executable} -m pytest -q -x -p no:randomly test_calc.py",
        )
        return tcm.log.text

    def _outcomes(self):
        out = subprocess.run(
            [sys.executable, "-c",
             "from cosmic_ray.work_db import WorkDB\n"
             "db = WorkDB('mutation.sqlite', WorkDB.Mode.open)\n"
             "r = dict(db.results)\n"
             "for item in db.work_items:\n"
             "    res = r.get(item.job_id)\n"
             "    print(item.mutations[0].start_pos[0], res.worker_outcome.value if res else 'pending')\n"],
            capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        return [(int(l.split()[0]), l.split()[1]) for l in out if l]

    def test_function_scope_runs_only_the_target_mutants(self):
        log = self._run("function")
        outcomes = self._outcomes()
        outside = [o for line, o in outcomes if not 5 <= line <= 8]
        inside = [o for line, o in outcomes if 5 <= line <= 8]
        self.assertTrue(outside and set(outside) == {"skipped"})
        self.assertTrue(inside and "skipped" not in inside and "pending" not in inside)
        self.assertEqual(_value(log, "FunctionMutationNum"), len(inside))
        self.assertEqual(_value(log, "FunctionMutationTotal"), len(inside))
        self.assertNotIn("\nMutationLOG", log)

    def test_function_score_matches_the_score_filtered_from_a_whole_file_run(self):
        func_log = self._run("function")
        os.remove("mutation.sqlite")
        file_log = self._run("file")
        self.assertIsNotNone(_value(file_log, "MutationLOG"))
        for key in ("FunctionMutationLOG", "FunctionMutationNum", "FunctionMutationTotal"):
            self.assertEqual(_value(func_log, key), _value(file_log, key), key)

    def test_a_timed_out_run_reports_fewer_finished_than_total(self):
        # cosmic-ray runs the test command without a shell, so the delay
        # goes in the test file itself.
        with open("test_slow.py", "w") as f:
            f.write("import time\ntime.sleep(0.5)\n" + TEST)
        slow = f"{sys.executable} -m pytest -q -x -p no:randomly test_slow.py"
        log = self._run("function", test_cmd=slow, mutation_timeout=4)
        self.assertIn("MutationTimeout", log)
        done = _value(log, "FunctionMutationNum")
        if done is not None:
            self.assertLess(done, _value(log, "FunctionMutationTotal"))

    def test_no_target_range_runs_nothing(self):
        tcm = _manager("function")
        tcm._resolve_target_ranges = lambda instance: None
        tcm.run_mutation_testing({"code_file": "calc.py"}, {}, test_time=1, test_cmd="true")
        self.assertIn("FunctionMutationFAIL: no target function range", tcm.log.text)
        self.assertTrue(all(o == "pending" for _, o in self._outcomes()))


    def _file_scope_with_ranges(self, ranges):
        tcm = _manager("file")
        tcm._resolve_target_ranges = lambda instance: ranges
        tcm.run_mutation_testing(
            {"code_file": "calc.py"}, {}, test_time=1,
            test_cmd=f"{sys.executable} -m pytest -q -x -p no:randomly test_calc.py",
        )
        return tcm.log.text

    def test_unresolved_target_and_empty_target_are_logged_apart(self):
        # Whole-file scope: an unresolved target and a target with no mutant
        # used to both log nothing.
        unresolved = self._file_scope_with_ranges(None)
        self.assertIn("FunctionMutationFAIL: no target function range", unresolved)
        os.remove("mutation.sqlite")
        empty = self._file_scope_with_ranges([(3, 4)])  # blank lines in calc.py
        self.assertEqual(_value(empty, "FunctionMutationTotal"), 0)
        self.assertNotIn("FunctionMutationFAIL", empty)


class ScopeValueTest(unittest.TestCase):
    def test_unknown_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            TaskEnvContextManager({}, "full", "t", ".", ".", mutation_scope="files")


class LegacySkipApiTest(unittest.TestCase):
    def test_skip_script_works_without_bulk_api_and_preserves_target_and_completed_jobs(self):
        from swebench_docker.mutation_scope import skip_outside_script

        def item(job_id, line):
            return SimpleNamespace(job_id=job_id, mutations=[SimpleNamespace(
                module_path="calc.py", start_pos=(line, 0))])

        class LegacyDB:
            class Mode:
                open = "open"

            def __init__(self, path, mode):
                self.work_items = [item("target", 6), item("outside", 2), item("completed", 1)]
                self.pending_work_items = self.work_items[:2]
                self.saved = {}
                self.closed = False

            def set_result(self, job_id, result):
                self.saved[job_id] = result

            def close(self):
                self.closed = True

        db = LegacyDB("mutation.sqlite", "open")
        work_db = ModuleType("cosmic_ray.work_db")
        work_db.WorkDB = lambda *args: db
        work_db.WorkDB.Mode = LegacyDB.Mode
        work_item = ModuleType("cosmic_ray.work_item")
        work_item.WorkResult = SimpleNamespace
        work_item.WorkerOutcome = SimpleNamespace(SKIPPED="skipped")
        modules = {"cosmic_ray": ModuleType("cosmic_ray"),
                   "cosmic_ray.work_db": work_db, "cosmic_ray.work_item": work_item}
        output = io.StringIO()
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(output):
            exec(skip_outside_script("calc.py", TARGET), {})
        self.assertEqual(set(db.saved), {"outside"})
        self.assertEqual(db.saved["outside"].worker_outcome, "skipped")
        self.assertTrue(db.closed)
        self.assertIn("FUNC_MUTANTS_IN_RANGE 1", output.getvalue())


class UntestedMutantsTest(unittest.TestCase):
    """cosmic-ray's is_killed is `test_outcome != SURVIVED`, so a mutant it
    never tested (worker outcome no-test, exception, abnormal) reads as
    killed. The function score counts only "normal" outcomes and reports the
    rest as untested."""

    @staticmethod
    def _score(outcomes):
        """Run score_script over target mutants with these
        (worker_outcome, is_killed) results, plus one outside the target."""
        from swebench_docker.mutation_scope import score_script

        def item(job_id, line):
            return SimpleNamespace(job_id=job_id, mutations=[SimpleNamespace(
                module_path="calc.py", start_pos=(line, 0))])

        class DB:
            class Mode:
                open = "open"

            def __init__(self, path, mode):
                self.work_items = [item(f"t{i}", 6) for i in range(len(outcomes))] + [item("out", 2)]
                self.results = [(f"t{i}", SimpleNamespace(worker_outcome=w, is_killed=k))
                                for i, (w, k) in enumerate(outcomes) if w is not None]
                self.results.append(("out", SimpleNamespace(worker_outcome="skipped", is_killed=True)))

            def close(self):
                pass

        work_db = ModuleType("cosmic_ray.work_db")
        work_db.WorkDB = DB
        modules = {"cosmic_ray": ModuleType("cosmic_ray"), "cosmic_ray.work_db": work_db}
        output = io.StringIO()
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(output):
            exec(score_script("calc.py", TARGET), {})
        return output.getvalue().split("\n")

    def test_untested_mutants_are_not_counted_as_kills(self):
        lines = self._score([("normal", True), ("normal", False),
                             ("no-test", True), ("exception", True), ("abnormal", True)])
        self.assertIn("FUNC_MUTATION_UNTESTED 3", lines)
        self.assertIn("FUNC_MUTATION_RESULT 50.0 2 5", lines)

    def test_only_untested_mutants_reports_none_tested(self):
        lines = self._score([("no-test", True), ("exception", True)])
        self.assertIn("FUNC_MUTATION_UNTESTED 2", lines)
        self.assertIn("FUNC_MUTATION_NO_MUTANTS_IN_RANGE 2", lines)

    def test_unfinished_mutants_are_neither_tested_nor_untested(self):
        lines = self._score([("normal", True), (None, None)])
        self.assertIn("FUNC_MUTATION_UNTESTED 0", lines)
        self.assertIn("FUNC_MUTATION_RESULT 100.0 1 2", lines)

    def test_the_log_records_the_untested_count(self):
        tcm = _manager("function")
        tcm.exec = lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="FUNC_MUTATION_UNTESTED 3\nFUNC_MUTATION_RESULT 50.0 2 5\n")
        tcm._log_function_mutation_score({"code_file": "calc.py"}, TARGET)
        self.assertEqual(_value(tcm.log.text, "FunctionMutationUntested"), 3)
        self.assertEqual(_value(tcm.log.text, "FunctionMutationLOG"), 50.0)
        self.assertEqual(_value(tcm.log.text, "FunctionMutationNum"), 2)
        self.assertEqual(_value(tcm.log.text, "FunctionMutationTotal"), 5)


if __name__ == "__main__":
    unittest.main()
