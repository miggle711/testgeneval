# Target-function mutation (testgeneval#10): with mutation_scope "function",
# only the target function's mutants run, and the score is over those. Runs
# real cosmic-ray on a small module, so it is skipped where cosmic_ray is not
# installed (the testbed images pin cosmic-ray==8.3.14). Run with:
#   python -m unittest swebench_docker.test_mutation_scope -v

import os
import re
import subprocess
import sys
import tempfile
import unittest
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


class ScopeValueTest(unittest.TestCase):
    def test_unknown_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            TaskEnvContextManager({}, "full", "t", ".", ".", mutation_scope="files")


if __name__ == "__main__":
    unittest.main()
