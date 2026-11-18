# Generation follows the rerun protocol (fyp-paper#7) unless told otherwise:
#   testgeneval#117: a kg_prompts path that was given but does not resolve,
#     or a row it has no entry for, is an error in both arms. instruct used
#     to run on without naming the target while kg_only refused.
#   testgeneval#118: temperature defaults to 0.8, not the rejected 0.2, and
#     Qwen2.5-Coder-7B is registered with its real context and output cap.
# The real run_api.main() runs; openai_inference is replaced to capture
# what it would have sent.

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from inference.api import run_api

ROWS = [{"id": i, "instance_id": i, "code_src": "def add(a, b):\n    return a + b\n",
         "code_file": "pkg/calc.py", "local_imports": ["from pkg.calc import add"],
         "preds_context": {"preamble": "", "last_minus_one": "", "last": ""}}
        for i in ("row-1", "row-2")]
# How the prompt names the target: in backticks, both in instruct_prompt.py's
# focus line and in #103's shared task text. Unfocused prompts never do.
FOCUS = "`add`"
TARGET = {"prompt": "KG context", "target_functions": ["add"], "target_classes": [None]}


class ProtocolDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ds = os.path.join(self.dir, "ds")
        DatasetDict({"test": Dataset.from_list(ROWS)}).save_to_disk(self.ds)
        self.kg_all = self.write_kg({"row-1": TARGET, "row-2": TARGET})

    def write_kg(self, entries, name="kg_prompts.json"):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            json.dump(entries, f)
        return path

    def run_main(self, prompt_config="instruct", model_args="temperature=0.8", **kw):
        seen = {}

        def fake_inference(test_dataset, **kwargs):
            seen["output_file"] = str(kwargs["output_file"])
            seen["model_args"] = dict(kwargs["model_args"])
            for r in test_dataset:
                seen[r["id"]] = r["preds_prompts"]["full"]

        with patch.object(run_api, "openai_inference", fake_inference), \
                patch.dict(os.environ, {"LOCAL_MODEL_BASE_URL": "http://x/v1/"}):
            run_api.main(self.ds, "test", "Qwen/Qwen3-4B-Instruct-2507", None, None, self.dir,
                         False, model_args, None, 5, False, skip_completion=True,
                         prompt_config=prompt_config, **kw)
        return seen

    def test_instruct_with_a_complete_kg_prompts_file_names_the_target(self):
        seen = self.run_main(kg_prompts_path=self.kg_all)
        self.assertIn(FOCUS, seen["row-1"])
        self.assertIn(FOCUS, seen["row-2"])

    def test_mistyped_kg_prompts_path_is_an_error_for_instruct(self):
        with self.assertRaises(FileNotFoundError):
            self.run_main(kg_prompts_path=os.path.join(self.dir, "kg_prompt.json"))

    def test_row_without_an_entry_is_an_error_for_instruct(self):
        partial = self.write_kg({"row-1": TARGET}, "partial.json")
        with self.assertRaises(ValueError) as cm:
            self.run_main(kg_prompts_path=partial)
        self.assertIn("row-2", str(cm.exception))

    def test_kg_only_without_a_path_is_an_error(self):
        with self.assertRaises(ValueError):
            self.run_main(prompt_config="kg_only")

    def test_instruct_without_a_path_still_runs_unfocused(self):
        seen = self.run_main()
        self.assertNotIn(FOCUS, seen["row-1"])
        self.assertNotIn(FOCUS, seen["row-2"])

    def test_temperature_defaults_to_the_protocols(self):
        seen = self.run_main(model_args=None, kg_prompts_path=self.kg_all)
        self.assertIn("__0.8__", os.path.basename(seen["output_file"]))
        self.assertEqual(run_api.DEFAULT_TEMPERATURE, 0.8)

    def test_qwen25_coder_7b_is_registered(self):
        model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        self.assertEqual(run_api.MODEL_LIMITS[model], 32_768)
        self.assertEqual(run_api.OUTPUT_LIMITS[model], 8_000)
        self.assertEqual(run_api.prompt_token_budget(model), 32_768 - 8_000 - run_api.EPSILON)


if __name__ == "__main__":
    unittest.main()
