import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_token_counts",
    os.path.join(os.path.dirname(__file__), "aggregate_token_counts.py"),
)
aggregate_token_counts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aggregate_token_counts)


def _line(inp, out, samples=None):
    tail = f", samples={samples}" if samples is not None else ""
    return f"INFO run_api: input_tokens={inp}, output_tokens={out}, cost=0.00{tail}\n"


class AggregateTokenCountsTest(unittest.TestCase):
    def _run(self, logs, extra_args=()):
        with tempfile.TemporaryDirectory() as d:
            for name, lines in logs.items():
                with open(os.path.join(d, name), "w") as f:
                    f.writelines(lines)
            out = os.path.join(d, "out.csv")
            argv = ["aggregate_token_counts.py", "--logs", os.path.join(d, "slurm-*.out"),
                    "--out", out, *extra_args]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                aggregate_token_counts.main()
            with open(out, newline="") as f:
                return {r["job_id"]: r for r in csv.DictReader(f)}

    def test_logged_samples_give_output_tokens_per_sample(self):
        # A k=5 call logs the five completions' total; per sample is a fifth.
        rows = self._run({"slurm-1.out": [_line(1000, 5000, 5), _line(2000, 15000, 5)]})
        r = rows["1"]
        self.assertEqual(r["mean_input_tokens"], "1500.0")
        self.assertEqual(r["mean_output_tokens"], "10000.0")
        self.assertEqual(r["total_samples"], "10")
        self.assertEqual(r["mean_output_tokens_per_sample"], "2000.0")
        self.assertEqual(r["samples_source"], "logged")

    def test_old_lines_without_samples_count_as_one_unless_told(self):
        rows = self._run({"slurm-2.out": [_line(1000, 5000)]})
        self.assertEqual(rows["2"]["mean_output_tokens_per_sample"], "5000.0")
        self.assertEqual(rows["2"]["samples_source"], "assumed 1")

        rows = self._run({"slurm-2.out": [_line(1000, 5000)]}, ["--num_samples", "5"])
        self.assertEqual(rows["2"]["mean_output_tokens_per_sample"], "1000.0")
        self.assertEqual(rows["2"]["samples_source"], "--num_samples")

    def test_logged_samples_win_over_num_samples(self):
        # A resumed job can mix old and new lines; only the old ones are filled in.
        rows = self._run(
            {"slurm-3.out": [_line(1000, 4000, 4), _line(1000, 5000)]}, ["--num_samples", "5"]
        )
        self.assertEqual(rows["3"]["total_samples"], "9")
        self.assertEqual(rows["3"]["mean_output_tokens_per_sample"], "1000.0")
        self.assertEqual(rows["3"]["samples_source"], "logged + --num_samples")


if __name__ == "__main__":
    unittest.main()
