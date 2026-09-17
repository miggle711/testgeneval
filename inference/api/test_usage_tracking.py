# Real regression protection for the per-sample usage tracking added in
# testgeneval#63: output_dict["usage"][prompt_name] should hold one real
# entry per sample (finish_reason/completion_tokens/reasoning_tokens),
# parallel to output_dict["preds"][prompt_name]. Nothing else in this
# repo's test suite asserts this output shape directly.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset

from inference.api import run_api


def fake_call_chat_with_reasoning(model_name_or_path, prompt_text, *args, **kwargs):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=f"```{prompt_text}```"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=7,
            completion_tokens=42,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=13),
        ),
    )
    return response, 0.0


class TestUsageTracking(unittest.TestCase):
    def _run(self, fake, num_samples=1):
        dataset = Dataset.from_dict(
            {
                "id": ["item-0"],
                "instance_id": ["inst-0"],
                "preds_prompts": [{"full": "prompt-0"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "out.jsonl"
            with patch.object(run_api, "call_chat", side_effect=fake):
                run_api.openai_inference(
                    test_dataset=dataset,
                    model_name_or_path="fake-model",
                    output_file=str(output_file),
                    model_args={"temperature": 0.8},
                    existing_ids=set(),
                    max_cost=None,
                    num_samples=num_samples,
                    postprocess_fn=lambda text, is_full: text,
                    system_message="sys",
                    system_message_full="sys_full",
                    skip_full=False,
                    skip_completion=True,
                    max_concurrency=1,
                )
            lines = output_file.read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_usage_present_and_shaped_per_sample(self):
        records = self._run(fake_call_chat_with_reasoning, num_samples=1)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIn("usage", record)
        self.assertIn("full", record["usage"])
        self.assertEqual(len(record["usage"]["full"]), len(record["preds"]["full"]))
        sample = record["usage"]["full"][0]
        self.assertEqual(sample["finish_reason"], "stop")
        self.assertEqual(sample["completion_tokens"], 42)
        self.assertEqual(sample["reasoning_tokens"], 13)

    def test_usage_absent_reasoning_details_is_none_not_a_crash(self):
        def fake_no_reasoning(model_name_or_path, prompt_text, *args, **kwargs):
            response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=f"```{prompt_text}```"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=7, completion_tokens=42, completion_tokens_details=None
                ),
            )
            return response, 0.0

        records = self._run(fake_no_reasoning, num_samples=1)
        self.assertIsNone(records[0]["usage"]["full"][0]["reasoning_tokens"])


if __name__ == "__main__":
    unittest.main()
