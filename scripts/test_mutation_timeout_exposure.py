"""scripts/mutation_timeout_exposure.py: samples after a mutation timeout (#145)."""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mutation_timeout_exposure as exposure  # noqa: E402


def block(outcome=">>>>> All Tests Passed", num=None, total=None, timeout=False, score=50.0):
    lines = [">>>>> Tests config full pred", outcome]
    if timeout:
        lines.append("[x] [y] MutationTimeout ")
    if num is not None:
        lines += [f"MutationNum: {num}", f"MutationLOG: {score}%"]
    if total is not None:
        lines += [f"FunctionMutationTotal: {total}", f"FunctionMutationLOG: {score}%"]
    return "\n" + "\n".join(lines) + "\n"


class Exposure(unittest.TestCase):
    def test_no_timeout_exposes_nothing(self):
        samples = exposure.parse_log("head" + block(num=10) + block(num=10))
        self.assertEqual(exposure.exposed_samples(samples), [])

    def test_samples_after_the_first_timeout_are_exposed(self):
        text = "head" + block(num=10) + block(num=10, timeout=True) + block(num=12) + block(num=10)
        got = exposure.exposed_samples(exposure.parse_log(text))
        self.assertEqual([g["sample"] for g in got], [2, 3])
        self.assertEqual([g["count_changed"] for g in got], [True, False])
        self.assertTrue(all(g["after_timeout_in_sample"] == 1 for g in got))

    def test_function_scope_counts_and_samples_without_a_count(self):
        text = ("head" + block(total=96) + block(total=96, timeout=True)
                + block(outcome=">>>>> Some Tests Failed") + block(total=100))
        got = exposure.exposed_samples(exposure.parse_log(text))
        self.assertEqual([(g["sample"], g["count_changed"]) for g in got], [(2, None), (3, True)])
        self.assertEqual(got[0]["outcome"], "failed")
        self.assertEqual(got[1]["count_kind"], "FunctionMutationTotal")

    def test_main_writes_rows_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            logs = os.path.join(d, "scout_instruct")
            os.makedirs(logs)
            with open(os.path.join(logs, "a__a-1-2.M_t=0.8.full.eval.log"), "w") as f:
                f.write("head" + block(num=10, timeout=True) + block(num=11))
            with open(os.path.join(logs, "b__b-1-2.M_t=0.8.full.eval.log"), "w") as f:
                f.write("head" + block(num=5) + block(num=5))
            out = os.path.join(d, "x.csv")
            rows = exposure.main([logs, "--out", out])
            self.assertEqual(len(rows), 1)
            with open(out) as f:
                got = list(csv.DictReader(f))
            self.assertEqual((got[0]["instance"], got[0]["sample"], got[0]["count_changed"]),
                             ("a__a-1-2", "1", "True"))


if __name__ == "__main__":
    unittest.main()
