"""scripts/capped_completion_audit.py: capped samples are classified by the
same rules in each arm, and nothing is turned into a score."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import capped_completion_audit as audit  # noqa: E402

TESTS = "import pytest\n\n" + "".join(f"def test_{i}():\n    assert {i} == {i}\n\n" for i in range(3))
LOOP_LINE = TESTS + "def test_loop():\n" + "    assert x == 1\n" * 80
LOOP_BLOCK = TESTS + "def test_loop():\n" + "    a = f(1)\n    b = g(a)\n    assert b\n" * 20
PROGRESSING = TESTS + "def test_cut():\n    value = compute(\n        1,"   # cut mid-call
UNCLEAR = "import pytest\n\nclass TestThing:\n    def helper(self):\n        return [\n            1,"
STOPPED_BAD = "def test_a(:\n    pass\n"


def write(rows):
    path = os.path.join(tempfile.mkdtemp(), "preds.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def row(i, texts, reasons):
    return {"id": i, "preds": {"full": texts},
            "usage": {"full": [{"finish_reason": r, "completion_tokens": 8000} for r in reasons]}}


class ClassifyTest(unittest.TestCase):
    def test_repeated_line_and_repeated_block_are_loops(self):
        self.assertEqual(audit.classify(LOOP_LINE, "length")["category"], "loop")
        self.assertEqual(audit.classify(LOOP_BLOCK, "length")["category"], "loop")

    def test_capped_but_still_writing_tests_is_progressing(self):
        c = audit.classify(PROGRESSING, "length")
        self.assertEqual(c["category"], "progressing")
        self.assertEqual(c["complete_tests"], 3)
        self.assertFalse(c["parses"])

    def test_capped_with_no_complete_test_is_unclear(self):
        c = audit.classify(UNCLEAR, "length")
        self.assertEqual((c["category"], c["complete_tests"]), ("unclear", 0))

    def test_stopped_samples_are_not_classified_but_parse_status_is_kept(self):
        self.assertEqual(audit.classify(TESTS, "stop"), {"capped": False, "category": "stopped", "parses": True})
        self.assertFalse(audit.classify(STOPPED_BAD, "stop")["parses"])


class MainTest(unittest.TestCase):
    def test_per_arm_summary_uses_the_same_rules(self):
        a = write([row("x", [LOOP_LINE, PROGRESSING, TESTS], ["length", "length", "stop"])])
        b = write([row("x", [UNCLEAR, STOPPED_BAD], ["length", "stop"]),
                   {"id": "y", "preds": {"full": ["t", "u"]}, "usage": {"full": [{"finish_reason": "stop"}]}}])
        out = os.path.join(tempfile.mkdtemp(), "audit.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            s = audit.main([f"instruct:{a}", f"kg_only:{b}", "--out", out])
        self.assertEqual(s["instruct"]["capped"], 2)
        self.assertEqual(s["instruct"]["capped_by_category"], {"loop": 1, "progressing": 1})
        self.assertEqual(s["instruct"]["stopped_not_parsing"], 0)
        self.assertEqual(s["kg_only"]["capped_by_category"], {"unclear": 1})
        self.assertEqual(s["kg_only"]["stopped_not_parsing"], 1)
        self.assertEqual(s["kg_only"]["rows_skipped_usage_not_per_sample"], 1)
        with open(out) as f:
            self.assertEqual(len(f.read().strip().splitlines()), 1 + 3 + 2)


if __name__ == "__main__":
    unittest.main()
