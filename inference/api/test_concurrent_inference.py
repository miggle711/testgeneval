# Verifies max_concurrency > 1 doesn't corrupt output or drop/duplicate
# instances -- the real risk with concurrent writes to one output file
# and shared total_cost/cost_exceeded state across worker threads.

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset

from inference.api import run_api


def fake_call_chat(model_name_or_path, prompt_text, *args, **kwargs):
    # Small sleep so concurrent calls actually overlap in wall time
    # instead of the GIL serializing them so tightly the test can't
    # distinguish max_concurrency=1 from >1.
    time.sleep(0.01)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=f"```{prompt_text}```"))]
    )
    return response, 0.0


class TestConcurrentInference(unittest.TestCase):
    def _run(self, num_instances, max_concurrency):
        dataset = Dataset.from_dict(
            {
                "id": [f"item-{i}" for i in range(num_instances)],
                "instance_id": [f"inst-{i}" for i in range(num_instances)],
                "preds_prompts": [{"full": f"prompt-{i}"} for i in range(num_instances)],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "out.jsonl"
            with patch.object(run_api, "call_chat", side_effect=fake_call_chat):
                run_api.openai_inference(
                    test_dataset=dataset,
                    model_name_or_path="fake-model",
                    output_file=str(output_file),
                    model_args={"temperature": 0},
                    existing_ids=set(),
                    max_cost=None,
                    num_samples=1,
                    postprocess_fn=lambda text, is_full: text,
                    system_message="sys",
                    system_message_full="sys_full",
                    skip_full=False,
                    skip_completion=True,
                    max_concurrency=max_concurrency,
                )
            lines = output_file.read_text().splitlines()
        return lines

    def test_sequential_produces_all_instances(self):
        lines = self._run(num_instances=10, max_concurrency=1)
        self.assertEqual(len(lines), 10)
        for line in lines:
            json.loads(line)  # each line is valid, unmangled JSON

    def test_concurrent_produces_all_instances_no_corruption(self):
        lines = self._run(num_instances=20, max_concurrency=5)
        self.assertEqual(len(lines), 20)
        ids = set()
        for line in lines:
            data = json.loads(line)  # would raise if writes interleaved
            ids.add(data["id"])
        self.assertEqual(ids, {f"item-{i}" for i in range(20)})

    def test_concurrent_no_duplicate_or_dropped_ids(self):
        lines = self._run(num_instances=37, max_concurrency=8)
        ids = [json.loads(line)["id"] for line in lines]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {f"item-{i}" for i in range(37)})


if __name__ == "__main__":
    unittest.main()
