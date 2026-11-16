"""scripts/prompt_budget_check.py: rule 4's limit applied to both versions
of a kg_only prompt, paired by instance."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import prompt_budget_check as pbc  # noqa: E402


class CompareTest(unittest.TestCase):
    def test_budget_is_inclusive_and_pairs_exclusions(self):
        tokens = {
            "fits": {"instruct": 90, "with": 100, "without": 95},          # exactly at budget
            "only_with": {"instruct": 90, "with": 101, "without": 99},      # the change tips it over
            "instruct_over": {"instruct": 150, "with": 50, "without": 40},  # out either way
        }
        rows, summary = pbc.compare(tokens, budget=100)
        self.assertEqual(summary["excluded_without"], 1)
        self.assertEqual(summary["excluded_with"], 2)
        self.assertEqual(summary["excluded_only_with"], ["only_with"])
        self.assertEqual(summary["excluded_only_without"], [])
        self.assertEqual(summary["kg_only_tokens_added"], {"instances_changed": 3, "max": 10})
        self.assertEqual(summary["smallest_kg_only_margin_with"], -1)
        self.assertEqual([r["id"] for r in rows], ["fits", "instruct_over", "only_with"])

    def test_messages_match_call_chat(self):
        self.assertEqual(pbc.messages("S", "U"), [{"role": "system", "content": "S"},
                                                  {"role": "user", "content": "U"}])
        self.assertEqual(pbc.messages("S", "U", True), [{"role": "user", "content": "S\n\nU"}])


class MainTest(unittest.TestCase):
    """The real prompt classes on a two-row dataset; only the token counter
    is replaced (one token per character of the user turn)."""

    def setUp(self):
        from datasets import Dataset, DatasetDict
        self.dir = tempfile.mkdtemp()
        rows = [{"id": i, "instance_id": i, "code_src": "LIMIT = 10\ndef f(x):\n    return x < LIMIT\n",
                 "code_file": "pkg/mod.py", "local_imports": ["from pkg.mod import f"],
                 "preds_context": {"preamble": "", "last_minus_one": "", "last": ""}}
                for i in ("a", "b")]
        self.ds = os.path.join(self.dir, "ds")
        DatasetDict({"test": Dataset.from_list(rows)}).save_to_disk(self.ds)
        section = "Module constants:\npkg.mod:\n```python\nLIMIT = 10\n```\n"
        base = "Seed: def f(x): ...\n" + "# context line\n" * 100   # longer than instruct

        def entry(ctx):
            # "context" for the shared-instructions prompt classes (#103),
            # "prompt" for the ones before it.
            return {"context": ctx, "prompt": ctx, "target_functions": ["f"], "target_classes": [None]}
        self.with_path = os.path.join(self.dir, "with.json")
        self.without_path = os.path.join(self.dir, "without.json")
        json.dump({"a": entry(base + section), "b": entry(base)}, open(self.with_path, "w"))
        json.dump({"a": entry(base), "b": entry(base)}, open(self.without_path, "w"))

    def run_main(self, budget_extra):
        counter = lambda msgs: len(msgs[-1]["content"])
        args = ["--dataset-path", self.ds, "--kg-prompts", self.with_path,
                "--kg-prompts-without", self.without_path, "--model", "m",
                "--max-model-len", str(1000 + budget_extra), "--output-limit", "1000",
                "--out", os.path.join(self.dir, "out.csv")]
        with contextlib.redirect_stdout(io.StringIO()):
            return pbc.main(args, counter=counter)

    def test_section_adds_tokens_only_where_it_is_present(self):
        summary = self.run_main(10 ** 6)
        self.assertEqual(summary["kg_only_tokens_added"]["instances_changed"], 1)
        self.assertGreater(summary["kg_only_tokens_added"]["max"], 0)
        self.assertEqual(summary["excluded_with"], 0)

    def test_budget_between_the_two_versions_excludes_only_with_the_section(self):
        first = self.run_main(10 ** 6)
        # A budget that the version without the section fits and the version with it does not.
        tight = first["max_tokens"]["without"]
        if first["max_tokens"]["instruct"] > tight:
            self.skipTest("instruct prompt longer than kg_only here; nothing to separate")
        summary = self.run_main(tight)
        self.assertEqual(summary["excluded_only_with"], ["a"])
        self.assertEqual(summary["excluded_only_without"], [])

    def test_files_covering_different_instances_are_refused(self):
        json.dump({"a": {"context": "x", "prompt": "x", "target_functions": ["f"]}},
                  open(self.without_path, "w"))
        with self.assertRaises(SystemExit):
            self.run_main(10 ** 6)


if __name__ == "__main__":
    unittest.main()
