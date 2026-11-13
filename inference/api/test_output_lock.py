# testgeneval#124: two jobs for the same model, arm, temperature and k used
# to read the same existing ids and both append every instance. Now one
# predictions file has one writer at a time, through <file>.lock.
# The first test is Miguel's repro from #124: the real run_api.main() twice
# in parallel, only the HTTP call replaced.

import json
import os
import subprocess
import tempfile
import threading
import unittest
from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from inference.api import run_api

ROWS = [{"id": f"row-{i}", "instance_id": f"row-{i}", "code_src": "def add(a, b):\n    return a + b\n",
         "code_file": "pkg/calc.py", "local_imports": [],
         "preds_context": {"preamble": "", "last_minus_one": "", "last": ""}} for i in range(3)]


def reply(*a, n=1, **kw):
    usage = NS(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    return NS(choices=[NS(message=NS(content="```python\npass\n```"), finish_reason="stop")] * n,
              usage=usage), 0


class OutputLockTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ds = os.path.join(self.dir, "ds")
        DatasetDict({"test": Dataset.from_list(ROWS)}).save_to_disk(self.ds)
        self.env = patch.dict(os.environ, {"LOCAL_MODEL_BASE_URL": "http://x/v1/"})
        self.env.start()
        os.environ.pop("SLURM_JOB_ID", None)

    def tearDown(self):
        self.env.stop()

    def job(self):
        run_api.main(self.ds, "test", "Qwen/Qwen3-4B-Instruct-2507", None, None, self.dir,
                     False, "temperature=0.8", None, 1, False, skip_completion=True)

    def output(self):
        return [os.path.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith(".jsonl")]

    def rows_per_id(self):
        out = self.output()
        return Counter(json.loads(l)["id"] for l in open(out[0])) if out else Counter()

    def lock_path(self):
        return os.path.join(self.dir, "Qwen3-4B-Instruct-2507__ds__0.8__k1__test.jsonl.lock")

    def test_two_parallel_jobs_do_not_duplicate_rows(self):
        both_started = threading.Barrier(2)
        first = threading.local()
        errors = []

        def fake_call_chat(*a, **kw):
            if not getattr(first, "done", False):
                first.done = True
                both_started.wait(timeout=5)
            return reply(*a, **kw)

        def guarded():
            try:
                self.job()
            except RuntimeError as e:
                errors.append(e)
                try:
                    both_started.wait(timeout=5)  # let the other job go on
                except threading.BrokenBarrierError:
                    pass

        with patch.object(run_api, "call_chat", fake_call_chat):
            jobs = [threading.Thread(target=guarded) for _ in range(2)]
            [j.start() for j in jobs]
            [j.join() for j in jobs]
        self.assertEqual(len(errors), 1)
        self.assertIn("being written by another run", str(errors[0]))
        self.assertEqual(dict(self.rows_per_id()), {"row-0": 1, "row-1": 1, "row-2": 1})

    def test_lock_is_released_after_a_normal_run(self):
        with patch.object(run_api, "call_chat", reply):
            self.job()
        self.assertFalse(os.path.exists(self.lock_path()))

    def test_lock_of_a_dead_process_on_this_host_is_replaced(self):
        dead = subprocess.Popen(["true"])
        dead.wait()
        with open(self.lock_path(), "w") as f:
            json.dump({"job": None, "host": run_api.socket.gethostname(), "pid": dead.pid}, f)
        with patch.object(run_api, "call_chat", reply):
            self.job()
        self.assertEqual(sum(self.rows_per_id().values()), 3)

    def test_lock_of_a_running_slurm_job_refuses(self):
        with open(self.lock_path(), "w") as f:
            json.dump({"job": "111", "host": "m3-gpu1", "pid": 1}, f)
        running = NS(returncode=0, stdout="RUNNING\n")
        with patch.object(run_api.subprocess, "run", return_value=running), \
                patch.object(run_api, "call_chat", reply):
            with self.assertRaises(RuntimeError):
                self.job()
        self.assertEqual(self.output(), [])

    def test_lock_of_a_finished_slurm_job_is_replaced(self):
        with open(self.lock_path(), "w") as f:
            json.dump({"job": "111", "host": "m3-gpu1", "pid": 1}, f)
        gone = NS(returncode=1, stdout="", stderr="slurm_load_jobs error: Invalid job id specified")
        with patch.object(run_api.subprocess, "run", return_value=gone), \
                patch.object(run_api, "call_chat", reply):
            self.job()
        self.assertEqual(sum(self.rows_per_id().values()), 3)

    def test_unreadable_lock_refuses(self):
        with open(self.lock_path(), "w") as f:
            f.write("not json")
        with patch.object(run_api, "call_chat", reply):
            with self.assertRaises(RuntimeError):
                self.job()


if __name__ == "__main__":
    unittest.main()
