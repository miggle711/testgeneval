# Regression test: num_samples now applies to the "full" setting, not
# just completion settings. process_instance() used to hardcode
# num_samples_curr = 1 for prompt_name == "full" regardless of what
# --num_samples requested, silently blocking pass@k evaluation (k > 1)
# for this fork's full-only scope. Fixed to request num_samples
# completions in one call via call_chat's n parameter, unpacking every
# response.choices entry instead of just choices[0].

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset

from inference.api import run_api


def fake_call_chat_multi_sample(model_name_or_path, prompt_text, *args, **kwargs):
    # Real behavior mocked here: n completions requested, n choices
    # returned, each with distinguishable content so a test can tell
    # them apart (real sampling at temperature > 0 wouldn't produce
    # identical completions either).
    n = kwargs.get("n", 1)
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content=f"```{prompt_text}-sample{i}```")
        )
        for i in range(n)
    ]
    response = SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10 * n),
    )
    return response, 0.0


def fake_call_chat_records_n(calls_seen):
    def _fake(model_name_or_path, prompt_text, *args, **kwargs):
        calls_seen.append(kwargs.get("n"))
        return fake_call_chat_multi_sample(model_name_or_path, prompt_text, *args, **kwargs)

    return _fake


class TestPassAtKForFullSetting(unittest.TestCase):
    def _run(self, num_samples, skip_full=False, skip_completion=True, fake=None):
        dataset = Dataset.from_dict(
            {
                "id": ["item-0"],
                "instance_id": ["inst-0"],
                "preds_prompts": [{"full": "prompt-0"}],
            }
        )
        fake = fake or fake_call_chat_multi_sample
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
                    skip_full=skip_full,
                    skip_completion=skip_completion,
                    max_concurrency=1,
                )
            lines = output_file.read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_num_samples_5_produces_5_predictions_for_full(self):
        records = self._run(num_samples=5)
        self.assertEqual(len(records), 1)
        preds = records[0]["preds"]["full"]
        self.assertEqual(len(preds), 5)
        # 5 genuinely distinct entries, not the same completion repeated.
        self.assertEqual(len(set(preds)), 5)

    def test_num_samples_1_still_produces_1_prediction(self):
        records = self._run(num_samples=1)
        preds = records[0]["preds"]["full"]
        self.assertEqual(len(preds), 1)

    def test_n_param_passed_through_to_call_chat_for_full(self):
        calls_seen = []
        self._run(num_samples=5, fake=fake_call_chat_records_n(calls_seen))
        # Exactly one call_chat invocation for the full setting (n
        # requested in one call), not 5 separate invocations.
        self.assertEqual(calls_seen, [5])


if __name__ == "__main__":
    unittest.main()
