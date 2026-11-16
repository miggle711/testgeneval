# A run can replace the model's OUTPUT_LIMITS entry as max_tokens
# (--output_limit; OUTPUT_LIMIT in m3_run_inference.slurm), for the Qwen3-4B
# 8k against 16k comparison (fyp-paper#5). Everything else about the request
# stays the same, so two runs that differ only in the limit send the same
# prompt with the same seed.

import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from datasets import Dataset

from inference.api import run_api

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SLURM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3_run_inference.slurm")


def fake_response(n):
    usage = NS(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    return NS(choices=[NS(message=NS(content="```python\npass\n```"), finish_reason="stop")] * n,
              usage=usage, model="served-model", system_fingerprint="fp-1"), 0.0


class OutputLimitTest(unittest.TestCase):
    def run_inference(self, output_limit, env=None):
        sent = []

        def fake_call_chat(model, prompt, temperature, top_p, max_tokens, *a, n=1, seed=None, **kw):
            sent.append({"max_tokens": max_tokens, "seed": seed, "prompt": prompt})
            return fake_response(n)

        out = os.path.join(tempfile.mkdtemp(), "preds.jsonl")
        rows = [{"id": "a", "instance_id": "a", "preds_prompts": {"full": "p"}}]
        with patch.object(run_api, "call_chat", fake_call_chat), \
                patch.dict(os.environ, {"LOCAL_MODEL_BASE_URL": "http://x/v1/", **(env or {})}):
            run_api.openai_inference(Dataset.from_list(rows), MODEL, out, {"temperature": 0.8}, set(),
                                     None, 5, lambda t, f: t, "sys", "sys", False,
                                     skip_completion=True, seed_base=0, output_limit=output_limit)
        return sent, json.loads(open(out).readline())

    def test_default_is_the_model_entry(self):
        sent, row = self.run_inference(None)
        self.assertEqual(sent[0]["max_tokens"], run_api.OUTPUT_LIMITS[MODEL])
        self.assertEqual(row["sampling"]["full"]["max_tokens"], run_api.OUTPUT_LIMITS[MODEL])

    def test_override_is_sent_and_recorded(self):
        sent, row = self.run_inference(16_000)
        self.assertEqual(sent[0]["max_tokens"], 16_000)
        self.assertEqual(row["sampling"]["full"]["max_tokens"], 16_000)

    def test_only_the_limit_differs_between_two_runs(self):
        (a,), _ = self.run_inference(8_000)
        (b,), _ = self.run_inference(16_000)
        self.assertEqual((a["seed"], a["prompt"]), (b["seed"], b["prompt"]))
        self.assertNotEqual(a["max_tokens"], b["max_tokens"])

    def test_budget_check_uses_the_override(self):
        self.assertEqual(run_api.prompt_token_budget(MODEL, 32_768, 16_000),
                         32_768 - 16_000 - run_api.EPSILON)
        with self.assertRaises(ValueError):
            run_api.prompt_token_budget(MODEL, 32_768, 32_000)

    def test_a_limit_that_leaves_no_room_stops_before_any_request(self):
        with self.assertRaises(ValueError):
            self.run_inference(32_000, env={"SERVER_MAX_MODEL_LEN": "32768"})

    def test_non_positive_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            run_api.output_limit_for(MODEL, 0)


class SlurmPassesOutputLimitTest(unittest.TestCase):
    """The slurm script's OUTPUT_LIMIT block, run on its own."""

    def block(self):
        text = open(SLURM).read()
        start = text.index('OUTPUT_LIMIT="${OUTPUT_LIMIT:-}"')
        end = text.index('echo "output limit:', start)
        return text[start:text.index("\n", end) + 1]

    def run_block(self, value):
        script = "EXTRA_ARGS=()\n" + self.block() + 'printf "%s\\n" "${EXTRA_ARGS[@]}"\n'
        env = {k: v for k, v in os.environ.items() if k != "OUTPUT_LIMIT"}
        if value is not None:
            env["OUTPUT_LIMIT"] = value
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True,
                              check=True).stdout.split("\n")

    def test_set(self):
        out = self.run_block("16000")
        self.assertIn("output limit: 16000", out)
        self.assertEqual(out[1:3], ["--output_limit", "16000"])

    def test_unset_adds_nothing(self):
        out = self.run_block(None)
        self.assertEqual(out[0], "output limit: OUTPUT_LIMITS entry")
        self.assertNotIn("--output_limit", out)


if __name__ == "__main__":
    unittest.main()
