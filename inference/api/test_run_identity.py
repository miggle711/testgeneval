"""Every prediction row says which run produced it. The arm and sample
count were only in the path the file was saved under, and nothing tied a
file to the kg_prompts.json it was generated from."""

import hashlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from inference.api import run_api

ROW = {
    "id": "row-1", "instance_id": "row-1", "code_src": "def add(a, b):\n    return a + b\n",
    "code_file": "pkg/calc.py", "local_imports": [],
    "preds_context": {"preamble": "", "last_minus_one": "", "last": ""},
}
KG_PROMPTS = {"row-1": {"prompt": "KG context", "target_functions": ["add"], "target_classes": [None]}}


def fake_call_chat(*args, n=1, **kwargs):
    usage = NS(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    choice = NS(message=NS(content="```python\npass\n```"), finish_reason="stop")
    return NS(choices=[choice] * n, usage=usage), 0


class TestRunIdentityInRows(unittest.TestCase):
    def generate(self, prompt_config, num_samples):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        dataset = os.path.join(d.name, "ds")
        DatasetDict({"test": Dataset.from_list([ROW])}).save_to_disk(dataset)
        kg_prompts = os.path.join(d.name, "kg_prompts.json")
        with open(kg_prompts, "w") as f:
            json.dump(KG_PROMPTS, f)
        with patch.object(run_api, "call_chat", fake_call_chat), \
             patch.dict(os.environ, {"LOCAL_MODEL_BASE_URL": "http://x/v1/"}):
            run_api.main(
                dataset, "test", "Qwen/Qwen3-4B-Instruct-2507", None, None, d.name, False,
                "temperature=0.8", None, num_samples, False, skip_completion=True,
                prompt_config=prompt_config, kg_prompts_path=kg_prompts,
            )
        out = [f for f in os.listdir(d.name) if f.endswith(".jsonl")]
        with open(os.path.join(d.name, out[0])) as f:
            row = json.loads(f.readline())
        with open(kg_prompts, "rb") as f:
            return row, hashlib.md5(f.read()).hexdigest()

    def test_row_records_arm_sample_count_temperature_and_prompts_file(self):
        for prompt_config, num_samples in [("instruct", 1), ("kg_only", 5)]:
            with self.subTest(prompt_config=prompt_config, num_samples=num_samples):
                row, kg_md5 = self.generate(prompt_config, num_samples)
                run = row["run"]
                self.assertEqual(
                    (run["arm"], run["num_samples"], run["temperature"], run["kg_prompts_md5"]),
                    (prompt_config, num_samples, 0.8, kg_md5),
                )
                self.assertIn("testgeneval_commit", run)


if __name__ == "__main__":
    unittest.main()
