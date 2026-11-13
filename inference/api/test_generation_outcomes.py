# Protocol decisions on generation outcomes (fyp-paper#7, Miguel's three
# questions of 13 Nov):
#   1. A None completion is a generation failure, scored 0: the sample is
#      kept as an empty reply and flagged, not dropped.
#   2. An instance that gets no prediction row is recorded, with a reason,
#      in a skipped file next to the predictions.
#   3. Rule 4: a prompt that does not fit the server's context is excluded
#      and recorded as prompt_too_long, not truncated.
# Only call_chat (the HTTP call) is replaced; openai_inference runs for real.

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from datasets import Dataset

from inference.api import run_api
from inference.configs.config_utils import postprocess_python_output

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
LONG_PROMPT = "FIRST LINE OF THE CODE FILE\n" + "x = 1\n" * 40000 + "LAST LINE\n"
VLLM_TOO_LONG = (
    "Error code: 400 - {'object': 'error', 'message': \"This model's maximum context "
    "length is 32768 tokens. However, you requested 48000 tokens (40000 in the "
    "messages, 8000 in the completion).\"}"
)


def response(contents):
    usage = NS(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    return NS(choices=[NS(message=NS(content=c), finish_reason="length") for c in contents],
              usage=usage), 0.0


class GenerationOutcomesTest(unittest.TestCase):
    def run_inference(self, rows, fake, out_dir, postprocess=postprocess_python_output):
        output_file = os.path.join(out_dir, "preds.jsonl")
        sent = []

        def recording_fake(model, prompt, *args, **kwargs):
            sent.append(prompt)
            return fake(prompt, kwargs.get("n", 1))

        env = {"LOCAL_MODEL_BASE_URL": "http://x/v1/", "SERVER_MAX_MODEL_LEN": "32768"}
        with patch.dict(os.environ, env), patch.object(run_api, "call_chat", recording_fake):
            run_api.openai_inference(
                Dataset.from_list(rows), MODEL, output_file, {"temperature": 0.8}, set(),
                None, 5, postprocess, "sys", "sys", False, skip_completion=True,
            )
        preds = [json.loads(line) for line in Path(output_file).read_text().splitlines()] \
            if os.path.exists(output_file) else []
        skipped_file = os.path.join(out_dir, "preds.skipped.tsv")
        skipped = Path(skipped_file).read_text().splitlines() if os.path.exists(skipped_file) else []
        return preds, skipped, sent

    @staticmethod
    def row(id_, prompt):
        return {"id": id_, "instance_id": id_, "preds_prompts": {"full": prompt}}

    def test_none_completion_is_kept_and_scores_like_a_reply_without_code(self):
        fake = lambda prompt, n: response([None, "```python\ndef test_a(): pass\n```", None, None, None])
        with tempfile.TemporaryDirectory() as d:
            preds, skipped, _ = self.run_inference([self.row("none-row", "p")], fake, d)
        self.assertEqual(len(preds[0]["preds"]["full"]), 5)
        self.assertEqual(preds[0]["preds"]["full"][0], postprocess_python_output("no code here", True))
        self.assertEqual([u.get("empty_content", False) for u in preds[0]["usage"]["full"]],
                         [True, False, True, True, True])
        self.assertEqual(skipped, [])

    def test_too_long_prompt_is_sent_whole_not_truncated(self):
        fake = lambda prompt, n: response(["```python\npass\n```"] * n)
        with tempfile.TemporaryDirectory() as d:
            _, _, sent = self.run_inference([self.row("long-row", LONG_PROMPT)], fake, d)
        self.assertEqual(sent, [LONG_PROMPT])

    def test_prompt_refused_as_too_long_is_recorded_and_gets_no_row(self):
        def fake(prompt, n):
            raise RuntimeError(VLLM_TOO_LONG)
        with tempfile.TemporaryDirectory() as d:
            preds, skipped, _ = self.run_inference([self.row("long-row", LONG_PROMPT)], fake, d)
        self.assertEqual(preds, [])
        self.assertEqual(skipped[0].split("\t"), list(run_api.SKIPPED_COLUMNS))
        cols = skipped[1].split("\t")
        self.assertEqual(cols[:5], ["long-row", "long-row", MODEL, "full", "prompt_too_long"])
        self.assertIn("maximum context length", cols[5])

    def test_vllm_engine_wording_is_also_too_long(self):
        error = ValueError("The decoder prompt (length 70000) is longer than the "
                           "maximum model length of 65536.")
        self.assertEqual(run_api.skip_reason(error), "prompt_too_long")

    def test_other_errors_are_request_failed(self):
        def fake(prompt, n):
            raise TimeoutError("Request timed out.")
        with tempfile.TemporaryDirectory() as d:
            preds, skipped, _ = self.run_inference([self.row("slow-row", "p")], fake, d)
        self.assertEqual(preds, [])
        self.assertEqual(skipped[1].split("\t")[4], "request_failed")

    def test_resubmit_does_not_repeat_a_skipped_row(self):
        def fake(prompt, n):
            raise RuntimeError(VLLM_TOO_LONG)
        with tempfile.TemporaryDirectory() as d:
            self.run_inference([self.row("long-row", LONG_PROMPT)], fake, d)
            _, skipped, _ = self.run_inference([self.row("long-row", LONG_PROMPT)], fake, d)
        self.assertEqual(len(skipped), 2)

    def test_skipped_file_is_not_a_jsonl_prediction_file(self):
        self.assertEqual(run_api.skipped_path("/r/qwen__kg_only__k5__test.jsonl"),
                         "/r/qwen__kg_only__k5__test.skipped.tsv")


if __name__ == "__main__":
    unittest.main()
