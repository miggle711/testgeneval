"""#145: a timed-out `cosmic-ray exec` must not leave the code file mutated.

cosmic-ray mutates the code file in place while it tests each mutant. A
timeout used to kill only the direct child, so on a plain python image the
file stayed mutated (cosmic-ray killed part-way through a mutant), and on a
conda image cosmic-ray kept running under the killed `conda run` wrapper,
mutating the file while later samples ran.

These run the real run_mutation_testing on a stand-in for the context manager,
with stub `cosmic-ray`, `cr-rate` and `cr-report` executables on PATH and a
stub `conda run` that, like the real one, starts the command as its child and
waits. No container or cosmic-ray install is needed.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

from swebench_docker.context_manager import ExecWrapper, TaskEnvContextManager

PATCHED = "def f(a, b):\n    return a + b  # patched, not committed\n"
MUTATED = "def f(a, b):\n    return a - b  # patched, not committed\n"

COSMIC_RAY = r'''#!{python}
import os, sys, time
mode, code, pidfile = os.environ["STUB_MODE"], os.environ["STUB_CODE"], os.environ["STUB_PIDFILE"]
if sys.argv[1:2] != ["exec"]:
    sys.exit(0)                      # init: nothing to do
with open(pidfile, "w") as f:
    f.write(str(os.getpid()))
original = open(code).read()
mutated = original.replace("a + b", "a - b")
if mode == "killed_mid_mutant":      # applies a mutant, then hangs in its test
    open(code, "w").write(mutated)
    time.sleep(60)
elif mode == "keeps_cycling":        # tests mutant after mutant, restoring each
    end = time.time() + 60
    while time.time() < end:
        open(code, "w").write(mutated); time.sleep(0.05)
        open(code, "w").write(original); time.sleep(0.05)
elif mode == "leaves_it_mutated":    # finishes, but without restoring
    open(code, "w").write(mutated)
# "clean": finishes and leaves the file alone
'''

CR_RATE = "#!{python}\nprint('1.0 2.0 3.0')\n"      # low, value, high: what the parser reads
CR_REPORT = "#!{python}\nprint('total jobs: 1')\n"

CONDA_RUN = r'''#!{python}
# Like `conda run -n testbed cmd...`: start the command as a child and wait.
import subprocess, sys
args = sys.argv[1:]
if args[:1] == ["run"]:
    args = args[3:]                  # drop "run -n testbed"
sys.exit(subprocess.call(args))
'''


class FakeLog:
    def __init__(self):
        self.lines = []

    def write(self, message, mode="a", level=None):
        self.lines.append(message)

    def text(self):
        return "\n".join(self.lines)


class MutationTimeoutCleanup(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(os.chdir, self.cwd)
        root = self.dir.name
        self.repo = os.path.join(root, "repo")
        self.bin = os.path.join(root, "bin")
        os.makedirs(os.path.join(self.repo, "pkg"))
        os.makedirs(self.bin)
        self.code = os.path.join(self.repo, "pkg", "m.py")
        with open(self.code, "w") as f:
            f.write(PATCHED)
        for name, text in (("cosmic-ray", COSMIC_RAY), ("cr-rate", CR_RATE), ("cr-report", CR_REPORT),
                           ("conda", CONDA_RUN)):
            p = os.path.join(self.bin, name)
            with open(p, "w") as f:
                f.write(text.format(python=sys.executable))
            os.chmod(p, 0o755)
        self.pidfile = os.path.join(root, "cosmic-ray.pid")
        env = {"PATH": self.bin + os.pathsep + os.environ["PATH"], "STUB_CODE": self.code,
               "STUB_PIDFILE": self.pidfile}
        old = {k: os.environ.get(k) for k in list(env) + ["STUB_MODE"]}
        os.environ.update(env)
        self.addCleanup(self._restore_env, old)
        self.addCleanup(self._kill_leftover)
        os.chdir(self.repo)

    @staticmethod
    def _restore_env(old):
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _pid(self):
        try:
            with open(self.pidfile) as f:
                return int(f.read())
        except (OSError, ValueError):
            return None

    def _alive(self, pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        # A killed process can linger as a zombie until it is reaped.
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
        return bool(state.strip()) and not state.strip().startswith("Z")

    def _kill_leftover(self):
        pid = self._pid()
        if pid and self._alive(pid):
            os.kill(pid, 9)

    def manager(self, conda, mutation_timeout):
        m = SimpleNamespace(
            repo_dir=self.repo,
            timeout=60,
            mutation_timeout=mutation_timeout,
            cmd_conda_run="conda run -n testbed " if conda else "",
            log=FakeLog(),
            _log_function_mutation_score=lambda instance: None,
        )
        m.exec = ExecWrapper(subprocess_args={"cwd": self.repo, "check": True, "shell": False,
                                              "universal_newlines": True, "stdout": subprocess.PIPE,
                                              "stderr": subprocess.STDOUT}, logger=m.log)
        m._run_cosmic_ray_exec = lambda cmd, code_file: TaskEnvContextManager._run_cosmic_ray_exec(m, cmd, code_file)
        return m

    def run_mutation(self, mode, conda, mutation_timeout):
        os.environ["STUB_MODE"] = mode
        m = self.manager(conda, mutation_timeout)
        spec = {} if conda else {"image": "python"}
        TaskEnvContextManager.run_mutation_testing(m, {"code_file": "pkg/m.py"}, spec, 1.0, "pytest")
        return m

    def read(self):
        with open(self.code) as f:
            return f.read()

    def assert_left_clean(self):
        pid = self._pid()
        self.assertIsNotNone(pid, "the stub cosmic-ray never ran")
        deadline = time.time() + 3
        while self._alive(pid) and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(self._alive(pid), "cosmic-ray is still running after the timeout")
        self.assertEqual(self.read(), PATCHED)
        time.sleep(0.3)   # nothing goes on changing it afterwards
        self.assertEqual(self.read(), PATCHED)

    def test_python_image_timeout_mid_mutant_restores_the_file(self):
        m = self.run_mutation("killed_mid_mutant", conda=False, mutation_timeout=1)
        self.assertIn("MutationTimeout", m.log.text())
        self.assertIn("MutationCleanup: restored pkg/m.py", m.log.text())
        self.assert_left_clean()

    def test_conda_image_timeout_kills_cosmic_ray_under_conda_run(self):
        m = self.run_mutation("keeps_cycling", conda=True, mutation_timeout=1)
        self.assertIn("MutationTimeout", m.log.text())
        self.assert_left_clean()

    def test_conda_image_timeout_mid_mutant_restores_the_file(self):
        m = self.run_mutation("killed_mid_mutant", conda=True, mutation_timeout=1)
        self.assertIn("MutationTimeout", m.log.text())
        self.assertIn("MutationCleanup: restored pkg/m.py", m.log.text())
        self.assert_left_clean()

    def test_restores_the_uncommitted_patched_file_not_a_git_version(self):
        # The file under test carries the patch, uncommitted: restoring from
        # git would bring back the pre-patch code. The copy is taken first.
        self.run_mutation("killed_mid_mutant", conda=False, mutation_timeout=1)
        self.assertEqual(self.read(), PATCHED)

    def test_a_run_that_finishes_cleanly_changes_nothing(self):
        m = self.run_mutation("clean", conda=True, mutation_timeout=30)
        self.assertNotIn("MutationTimeout", m.log.text())
        self.assertNotIn("MutationCleanup", m.log.text())
        self.assertEqual(self.read(), PATCHED)

    def test_a_run_that_finishes_but_leaves_the_file_mutated_is_restored(self):
        m = self.run_mutation("leaves_it_mutated", conda=False, mutation_timeout=30)
        self.assertNotIn("MutationTimeout", m.log.text())
        self.assertIn("MutationCleanup: restored pkg/m.py", m.log.text())
        self.assertEqual(self.read(), PATCHED)


if __name__ == "__main__":
    unittest.main()
