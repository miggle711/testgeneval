"""scripts/remap_ids_to_lite.py matches whole tasks, not instance ids
(testgeneval#143)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import remap_ids_to_lite as remap  # noqa: E402


def row(rid, iid, test_file, code_file="pkg/mod.py"):
    return {"id": rid, "instance_id": iid, "code_file": code_file, "test_file": test_file}


FULL = [row("a-100", "a", "tests/test_a.py"),
        row("s-200", "sympy__sympy-17022", "sympy/printing/tests/test_pycode.py",
            "sympy/printing/pycode.py"),
        row("z-300", "z", "tests/test_z.py")]
LITE = [row("a-1", "a", "tests/test_a.py"),
        row("s-23", "sympy__sympy-17022", "sympy/printing/tests/test_numpy.py",
            "sympy/printing/pycode.py")]


def write(rows):
    d = Path(tempfile.mkdtemp())
    src = d / "preds.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return src, d / "out.jsonl"


class RemapTest(unittest.TestCase):
    def setUp(self):
        self.maps = remap.build_lite_id_map(FULL, LITE)

    def test_same_task_is_remapped(self):
        src, dst = write([{"id": "a-100", "instance_id": "a"}])
        self.assertEqual(remap.remap_file(src, dst, self.maps), (1, []))
        self.assertEqual(json.loads(dst.read_text())["id"], "a-1")

    def test_same_instance_different_test_file_is_refused(self):
        src, dst = write([{"id": "s-200", "instance_id": "sympy__sympy-17022"}])
        with self.assertRaises(ValueError) as cm:
            remap.remap_file(src, dst, self.maps)
        self.assertIn("different task", str(cm.exception))

    def test_skip_mismatched_leaves_it_out_and_reports_it(self):
        src, dst = write([{"id": "a-100", "instance_id": "a"},
                          {"id": "s-200", "instance_id": "sympy__sympy-17022"}])
        self.assertEqual(remap.remap_file(src, dst, self.maps, skip_mismatched=True), (1, ["s-200"]))
        self.assertEqual([json.loads(l)["id"] for l in dst.read_text().splitlines()], ["a-1"])

    def test_prediction_already_carrying_a_lite_id_is_kept(self):
        src, dst = write([{"id": "s-23", "instance_id": "sympy__sympy-17022"}])
        self.assertEqual(remap.remap_file(src, dst, self.maps), (1, []))
        self.assertEqual(json.loads(dst.read_text())["id"], "s-23")

    def test_task_not_in_lite_is_refused(self):
        src, dst = write([{"id": "z-300", "instance_id": "z"}])
        with self.assertRaises(ValueError):
            remap.remap_file(src, dst, self.maps)


if __name__ == "__main__":
    unittest.main()
