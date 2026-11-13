"""scripts/capped_review_sheet.py: a seeded sample per (arm, category), with
blank label fields, and the full completion on request."""

import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import capped_completion_audit as audit  # noqa: E402
import capped_review_sheet as sheet  # noqa: E402
from test_capped_completion_audit import ENUMERATION, LOOP_LINE, PROGRESSING, UNCLEAR, write  # noqa: E402


class ReviewSheetTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        texts = [PROGRESSING, ENUMERATION, LOOP_LINE, UNCLEAR]
        row = {"id": "x", "preds": {"full": texts},
               "usage": {"full": [{"finish_reason": "length", "completion_tokens": 8000}] * 4}}
        self.pred = write([row])
        self.audit_csv = os.path.join(self.dir, "audit.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            audit.main([f"instruct:{self.pred}", "--out", self.audit_csv])

    def test_one_sample_per_category_with_blank_labels(self):
        out = os.path.join(self.dir, "review")
        with contextlib.redirect_stdout(io.StringIO()):
            items = sheet.main([f"instruct:{self.pred}", "--audit-csv", self.audit_csv,
                                "--per-cell", "3", "--out", out])
        self.assertEqual(sorted(i["category"] for i in items),
                         ["enumeration", "loop", "progressing", "unclear"])
        rows = list(csv.DictReader(open(out + ".csv")))
        self.assertEqual({r["label"] for r in rows}, {""})
        self.assertEqual({r["seed"] for r in rows}, {"none recorded (no per-request seed)"})
        md = open(out + ".md").read()
        self.assertIn("Last 60 lines", md)
        self.assertIn("instruct / enumeration: x sample 1", md)

    def test_show_prints_the_whole_completion(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sheet.main(["--show", f"instruct:{self.pred}", "x", "3"])
        self.assertEqual(buf.getvalue().rstrip("\n"), UNCLEAR)

    def test_recorded_seed_is_reported(self):
        self.assertEqual(sheet.seed_of({"sampling": {"full": {"seed": 42}}}, 0), "42")


if __name__ == "__main__":
    unittest.main()
