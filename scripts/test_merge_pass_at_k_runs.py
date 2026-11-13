# Copyright (c) Meta Platforms, Inc. and affiliates.

# Tests for merge_pass_at_k_runs.py, the stage-1 (k=1) + stage-2 (k=4)
# predictions merge for the staged pass@5 plan (fyp-paper#5). The
# properties under test are the ones the plan's guarantees rest on:
# stage-1 samples stay FIRST (frozen pass@1), config-freeze violations
# are hard errors, and stage-2-only ids stay out unless opted in.

import importlib.util
import json
import os
import tempfile
import unittest

# scripts/ is not a package, so load the module by path, same approach as
# test_merge_any_pass_logs.py.
_SPEC = importlib.util.spec_from_file_location(
    "merge_pass_at_k_runs",
    os.path.join(os.path.dirname(__file__), "merge_pass_at_k_runs.py"),
)
merge_pass_at_k_runs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge_pass_at_k_runs)


def make_record(curr_id, preds, model="TestModel_t=0.8", prompt="PROMPT", usage="auto"):
    rec = {
        "id": curr_id,
        "instance_id": curr_id.rsplit("-", 1)[0],
        "model_name_or_path": model,
        "preds_prompts": {"full": prompt},
        "preds": {"full": preds},
    }
    if usage == "auto":
        rec["usage"] = {
            "full": [
                {"finish_reason": "stop", "completion_tokens": 10, "reasoning_tokens": None}
                for _ in preds
            ]
        }
    elif usage is not None:
        rec["usage"] = usage
    return rec


class MergePassAtKRunsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _write(self, name, records):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            for rec in records:
                print(json.dumps(rec), file=f)
        return path

    def _run(self, stage1, stage2, extra_args=(), output_name="merged.jsonl"):
        out = os.path.join(self.dir, output_name)
        rc = merge_pass_at_k_runs.main(
            ["--stage1", stage1, "--stage2", stage2, "--output", out, *extra_args]
        )
        return rc, out

    def _read(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f]

    def test_merged_row_identity_counts_both_stages(self):
        # A merged row copied from stage 1 would otherwise still say one
        # sample while holding five.
        def with_run(rec, k):
            rec["run"] = {"arm": "kg_only", "num_samples": k, "temperature": 0.8,
                          "kg_prompts_md5": "abc", "testgeneval_commit": "c1"}
            return rec

        s1 = self._write("k1.jsonl", [with_run(make_record("a-1", ["s1"]), 1)])
        s2 = self._write("k4.jsonl", [with_run(make_record("a-1", ["s2", "s3", "s4", "s5"]), 4)])
        rc, out = self._run(s1, s2)
        self.assertEqual(rc, 0)
        run = self._read(out)[0]["run"]
        self.assertEqual((run["arm"], run["num_samples"]), ("kg_only", 5))
        self.assertEqual([r["num_samples"] for r in run["merged_from"]], [1, 4])

    def test_happy_path_stage1_samples_first(self):
        s1 = self._write("k1.jsonl", [
            make_record("django__django-1-1", ["S1_A"]),
            make_record("django__django-2-1", ["S1_B"]),
        ])
        s2 = self._write("k4.jsonl", [
            make_record("django__django-1-1", ["S2_A1", "S2_A2", "S2_A3", "S2_A4"]),
            make_record("django__django-2-1", ["S2_B1", "S2_B2", "S2_B3", "S2_B4"]),
        ])
        rc, out = self._run(s1, s2)
        self.assertEqual(rc, 0)
        records = self._read(out)
        self.assertEqual(len(records), 2)
        # Stage-1 sample first: this is what freezes pass@1 (pass_at_1
        # reads tests_passed[:1]).
        self.assertEqual(
            records[0]["preds"]["full"],
            ["S1_A", "S2_A1", "S2_A2", "S2_A3", "S2_A4"],
        )
        self.assertEqual(len(records[0]["usage"]["full"]), 5)
        # No bookkeeping key leaks into the output.
        self.assertNotIn("_line", records[0])

    def test_short_stage2_is_merged_not_fatal(self):
        # The none-content guard can leave < 4 samples in the k4 file;
        # the merge keeps what exists and report generation's
        # len(...) >= k guard handles the rest.
        s1 = self._write("k1.jsonl", [make_record("x-1", ["A"])])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B", "C", "D"])])
        rc, out = self._run(s1, s2)
        self.assertEqual(rc, 0)
        self.assertEqual(self._read(out)[0]["preds"]["full"], ["A", "B", "C", "D"])

    def test_model_name_mismatch_is_fatal(self):
        # model_name_or_path embeds the temperature, so this doubles as
        # the between-stages config-freeze check.
        s1 = self._write("k1.jsonl", [make_record("x-1", ["A"], model="M_t=0.8")])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B"], model="M_t=0.6")])
        rc, _ = self._run(s1, s2)
        self.assertEqual(rc, 1)

    def test_prompt_mismatch_is_fatal_unless_allowed(self):
        s1 = self._write("k1.jsonl", [make_record("x-1", ["A"], prompt="OLD PROMPT")])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B"], prompt="NEW PROMPT")])
        rc, _ = self._run(s1, s2)
        self.assertEqual(rc, 1)
        rc, out = self._run(s1, s2, extra_args=["--allow_prompt_mismatch", "--force"])
        self.assertEqual(rc, 0)
        self.assertEqual(self._read(out)[0]["preds"]["full"], ["A", "B"])

    def test_stage2_only_id_excluded_by_default(self):
        # A stage-2-only id would take its pass@1 from a stage-2 sample,
        # changing the pass@1 population after stage 1 was reported.
        s1 = self._write("k1.jsonl", [make_record("x-1", ["A"])])
        s2 = self._write("k4.jsonl", [
            make_record("x-1", ["B", "C", "D", "E"]),
            make_record("x-2", ["F", "G", "H", "I"]),
        ])
        rc, out = self._run(s1, s2)
        self.assertEqual(rc, 0)
        self.assertEqual([r["id"] for r in self._read(out)], ["x-1"])

        rc, out = self._run(
            s1, s2, extra_args=["--include_stage2_only", "--force"]
        )
        self.assertEqual(rc, 0)
        records = self._read(out)
        self.assertEqual([r["id"] for r in records], ["x-1", "x-2"])
        self.assertEqual(records[1]["preds"]["full"], ["F", "G", "H", "I"])

    def test_stage1_only_id_kept_with_one_sample(self):
        s1 = self._write("k1.jsonl", [
            make_record("x-1", ["A"]),
            make_record("x-2", ["B"]),
        ])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["C", "D", "E", "F"])])
        rc, out = self._run(s1, s2)
        self.assertEqual(rc, 0)
        records = self._read(out)
        self.assertEqual(records[1]["preds"]["full"], ["B"])

    def test_empty_stage1_preds_get_no_stage2_samples(self):
        # run_api.py writes preds=[] when the stage-1 sample's content was
        # None. Merging stage 2 in would put a stage-2 sample at position 0
        # and give this id a pass@1 it did not have after stage 1.
        s1 = self._write("k1.jsonl", [
            make_record("x-1", []),
            make_record("x-2", ["A"]),
        ])
        s2 = self._write("k4.jsonl", [
            make_record("x-1", ["B", "C", "D", "E"]),
            make_record("x-2", ["F", "G", "H", "I"]),
        ])
        rc, out = self._run(s1, s2)
        self.assertEqual(rc, 0)
        records = self._read(out)
        self.assertEqual(records[0]["preds"]["full"], [])
        self.assertEqual(records[1]["preds"]["full"], ["A", "F", "G", "H", "I"])

    def test_one_sided_usage_is_dropped_not_misaligned(self):
        # Pre-instrumentation files have no usage at all; a one-sided
        # concat would look aligned to preds while covering only one
        # stage, so it must be dropped.
        s1 = self._write("k1.jsonl", [make_record("x-1", ["A"], usage=None)])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B", "C", "D", "E"])])
        rc, out = self._run(s1, s2)
        self.assertEqual(rc, 0)
        rec = self._read(out)[0]
        self.assertEqual(rec["preds"]["full"], ["A", "B", "C", "D", "E"])
        self.assertNotIn("full", rec.get("usage", {}))

    def test_duplicate_id_within_a_file_is_fatal(self):
        s1 = self._write("k1.jsonl", [
            make_record("x-1", ["A"]),
            make_record("x-1", ["A2"]),
        ])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B", "C", "D", "E"])])
        rc, _ = self._run(s1, s2)
        self.assertEqual(rc, 1)

    def test_existing_output_requires_force(self):
        s1 = self._write("k1.jsonl", [make_record("x-1", ["A"])])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B", "C", "D", "E"])])
        out = self._write("merged.jsonl", [{"id": "stale"}])
        rc = merge_pass_at_k_runs.main(
            ["--stage1", s1, "--stage2", s2, "--output", out]
        )
        self.assertEqual(rc, 1)
        # Stale content untouched.
        self.assertEqual(self._read(out), [{"id": "stale"}])

    def test_output_must_not_be_an_input(self):
        s1 = self._write("k1.jsonl", [make_record("x-1", ["A"])])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B", "C", "D", "E"])])
        rc = merge_pass_at_k_runs.main(
            ["--stage1", s1, "--stage2", s2, "--output", s1, "--force"]
        )
        self.assertEqual(rc, 1)

    def test_too_many_samples_in_a_stage_is_fatal(self):
        # A k=5 file given as stage 1 (or a k=5 file as stage 2) would push
        # the merged list past 5; evaluation would silently use the first 5.
        ok1 = self._write("k1.jsonl", [make_record("x-1", ["A"])])
        ok2 = self._write("k4.jsonl", [make_record("x-1", ["B", "C", "D", "E"])])
        big = self._write("k5.jsonl", [make_record("x-1", ["A", "B", "C", "D", "E"])])
        self.assertEqual(self._run(big, ok2)[0], 1)
        self.assertEqual(self._run(ok1, big, output_name="m2.jsonl")[0], 1)

    def test_output_aliasing_an_input_is_fatal(self):
        # A symlink or hard link to an input would be overwritten by --force.
        s1 = self._write("k1.jsonl", [make_record("x-1", ["A"])])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B", "C", "D", "E"])])
        for make_alias in (os.symlink, os.link):
            alias = os.path.join(self.dir, f"alias_{make_alias.__name__}.jsonl")
            make_alias(s1, alias)
            rc = merge_pass_at_k_runs.main(
                ["--stage1", s1, "--stage2", s2, "--output", alias, "--force"]
            )
            self.assertEqual(rc, 1)
            self.assertEqual(self._read(s1)[0]["preds"]["full"], ["A"])

    def test_prompt_mismatch_is_fatal_even_with_empty_stage1(self):
        s1 = self._write("k1.jsonl", [make_record("x-1", [], prompt="OLD PROMPT")])
        s2 = self._write("k4.jsonl", [make_record("x-1", ["B"], prompt="NEW PROMPT")])
        self.assertEqual(self._run(s1, s2)[0], 1)


if __name__ == "__main__":
    unittest.main()
