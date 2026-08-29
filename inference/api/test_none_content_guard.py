# Regression test: response.choices[i].message.content can come back
# None rather than an empty string, confirmed real for gpt-oss under a
# too-low output token budget (its harmony response format can exhaust
# max_tokens on a separate reasoning channel before ever writing the
# final answer, see testgeneval#40). process_instance() used to pass
# that None straight into postprocess_fn, which crashes on
# None.replace(...); with num_samples > 1, that exception discarded the
# whole instance's predictions, including any real, good completions
# among the n requested. Fixed to skip only the None sample and keep the
# rest.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset

from inference.api import run_api


def fake_call_chat_with_none_content(model_name_or_path, prompt_text, *args, **kwargs):
    # 5 samples requested, 2 come back None (simulating gpt-oss losing
    # the final answer channel to a too-low output budget), 3 come back
    # real content.
    n = kwargs.get("n", 1)
    choices = []
    for i in range(n):
        content = None if i in (1, 3) else f"```{prompt_text}-sample{i}```"
        choices.append(SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="length"))
    response = SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10 * n),
    )
    return response, 0.0


def fake_call_chat_all_none(model_name_or_path, prompt_text, *args, **kwargs):
    n = kwargs.get("n", 1)
    choices = [
        SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="length")
        for _ in range(n)
    ]
    response = SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10 * n),
    )
    return response, 0.0


class TestNoneContentGuard(unittest.TestCase):
    def _run(self, num_samples, fake):
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

    def test_none_samples_are_skipped_not_crashed_on(self):
        records = self._run(num_samples=5, fake=fake_call_chat_with_none_content)
        # Instance is not dropped just because some of its 5 samples came
        # back None.
        self.assertEqual(len(records), 1)
        preds = records[0]["preds"]["full"]
        # Only the 3 real completions survive, the 2 None ones are
        # skipped rather than crashing postprocess_fn.
        self.assertEqual(len(preds), 3)

    def test_all_none_samples_yields_empty_predictions_not_a_crash(self):
        records = self._run(num_samples=5, fake=fake_call_chat_all_none)
        # The instance still gets written (not silently dropped as
        # "failed"), just with no predictions for this prompt.
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["preds"]["full"], [])


if __name__ == "__main__":
    unittest.main()
