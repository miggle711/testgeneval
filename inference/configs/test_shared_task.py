# Both arms get the same instructions (fyp-paper#5, decision B): the system
# message and everything after the context block must be identical, so the
# arms differ only in the context they show. Run with:
#   python -m unittest inference.configs.test_shared_task -v

import json
import os
import tempfile
import unittest

from datasets import Dataset, DatasetDict

from inference.configs import shared_task
from inference.configs.instruct_prompt import InstructPrompt
from inference.configs.kg_only_prompt import KGOnlyPrompt

KG_CONTEXT = "Function under test: area\nModule: geo.shapes\n\nSource:\n```python\ndef area(self): ...\n```\n"


def _row(row_id="geo-1-1"):
    return {
        "id": row_id, "code_src": "class Square:\n    def area(self): ...\n",
        "code_file": "geo/shapes.py", "local_imports": ["from geo.shapes import Square"],
        "preds_context": {"preamble": "", "last_minus_one": "", "last": ""},
    }


def _prompts(entry, row=None):
    row = row or _row()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "kg_prompts.json")
        with open(path, "w") as f:
            json.dump({row["id"]: entry}, f)
        ds = DatasetDict({"test": Dataset.from_list([row])})
        instruct, kg_only = InstructPrompt(kg_prompts_path=path), KGOnlyPrompt(prompts_path=path)
        return (instruct, instruct.add_prompts_to_dataset(ds)["test"][0]["preds_prompts"]["full"],
                kg_only, kg_only.add_prompts_to_dataset(ds)["test"][0]["preds_prompts"]["full"])


ENTRY = {"context": KG_CONTEXT, "target_functions": ["area"], "target_classes": ["Square"]}


class SharedTaskTest(unittest.TestCase):
    def test_both_arms_end_with_the_same_task_text(self):
        _, instruct, _, kg_only = _prompts(ENTRY)
        task = shared_task.task_text(["Square.area"])
        self.assertTrue(instruct.endswith(task))
        self.assertTrue(kg_only.endswith(task))
        self.assertIn("tests `Square.area`, which was recently modified", task)
        self.assertIn("including error cases and boundary conditions", task)

    def test_the_arms_differ_only_in_context(self):
        _, instruct, _, kg_only = _prompts(ENTRY)
        task = shared_task.task_text(["Square.area"])
        self.assertIn("class Square:", instruct[: -len(task)])
        self.assertTrue(kg_only.startswith(KG_CONTEXT.rstrip()))
        self.assertNotIn("class Square:", kg_only)

    def test_both_arms_share_the_system_message(self):
        instruct_cfg, _, kg_cfg, _ = _prompts(ENTRY)
        self.assertEqual(instruct_cfg.system_message_full, kg_cfg.system_message_full)
        self.assertEqual(kg_cfg.system_message_full, shared_task.SYSTEM_MESSAGE)

    def test_several_targets(self):
        entry = dict(ENTRY, target_functions=["area", "scale"], target_classes=["Square", None])
        _, instruct, _, kg_only = _prompts(entry)
        task = shared_task.task_text(["Square.area", "scale"])
        self.assertIn("tests `Square.area`, `scale`, which were recently modified", task)
        self.assertTrue(instruct.endswith(task) and kg_only.endswith(task))

    def test_an_old_kg_prompts_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no 'context' key"):
            _prompts({"prompt": "old", "target_functions": ["area"], "target_classes": [None]})

    def test_instruct_without_a_target_uses_the_shared_fallback(self):
        instruct = InstructPrompt()
        ds = DatasetDict({"test": Dataset.from_list([_row()])})
        full = instruct.add_prompts_to_dataset(ds)["test"][0]["preds_prompts"]["full"]
        self.assertTrue(full.endswith(shared_task.task_text([])))
        self.assertIn("tests the code above", full)


if __name__ == "__main__":
    unittest.main()
