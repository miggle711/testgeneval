# Each request carries an explicit, recorded sampling seed (fyp-paper#5 and
# testgeneval#118): a stable hash of the run's base seed, the instance, the
# setting, the sample count and the request index. It controls sampling; it
# does not promise identical regeneration (vLLM's batching and GPU arithmetic).

import json
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from datasets import Dataset

from inference.api import run_api


class RequestSeedTest(unittest.TestCase):
    def test_stable_and_in_range(self):
        a = run_api.request_seed(0, "django__django-1-1", "full", 5)
        self.assertEqual(a, run_api.request_seed(0, "django__django-1-1", "full", 5))
        self.assertTrue(0 <= a < 2 ** 31)

    def test_differs_by_instance_base_and_sample_count(self):
        seeds = {run_api.request_seed(0, "a", "full", 5), run_api.request_seed(0, "b", "full", 5),
                 run_api.request_seed(1, "a", "full", 5)}
        self.assertEqual(len(seeds), 3)

    def test_staged_one_plus_four_does_not_repeat_the_first_draw(self):
        self.assertNotEqual(run_api.request_seed(0, "a", "full", 1),
                            run_api.request_seed(0, "a", "full", 4))

    def test_off_when_no_base(self):
        self.assertIsNone(run_api.request_seed(None, "a", "full", 5))


def fake_response(n):
    usage = NS(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    return NS(choices=[NS(message=NS(content="```python\npass\n```"), finish_reason="stop")] * n,
              usage=usage, model="served-model", system_fingerprint="fp-1"), 0.0


class InferenceRecordsSeedTest(unittest.TestCase):
    def run_inference(self, seed_base):
        seen = []

        def fake_call_chat(*a, n=1, seed=None, **kw):
            seen.append(seed)
            return fake_response(n)

        out = os.path.join(tempfile.mkdtemp(), "preds.jsonl")
        rows = [{"id": "a", "instance_id": "a", "preds_prompts": {"full": "p"}}]
        with patch.object(run_api, "call_chat", fake_call_chat), \
                patch.dict(os.environ, {"LOCAL_MODEL_BASE_URL": "http://x/v1/"}):
            run_api.openai_inference(Dataset.from_list(rows), "Qwen/Qwen3-4B-Instruct-2507", out,
                                     {"temperature": 0.8}, set(), None, 5, lambda t, f: t,
                                     "sys", "sys", False, skip_completion=True, seed_base=seed_base)
        row = json.loads(open(out).readline())
        return seen, row

    def test_seed_is_sent_and_recorded_with_the_sampling_settings(self):
        seen, row = self.run_inference(7)
        expected = run_api.request_seed(7, "a", "full", 5)
        self.assertEqual(seen, [expected])
        self.assertEqual(row["sampling"]["full"], {
            "seed": expected, "temperature": 0.8, "top_p": 0.95,
            "max_tokens": run_api.OUTPUT_LIMITS["Qwen/Qwen3-4B-Instruct-2507"], "n": 5,
            "served_model": "served-model", "system_fingerprint": "fp-1"})

    def test_no_seed_sends_none_and_records_it(self):
        seen, row = self.run_inference(None)
        self.assertEqual(seen, [None])
        self.assertIsNone(row["sampling"]["full"]["seed"])


class CallChatSendsSeedTest(unittest.TestCase):
    def capture(self, **kw):
        sent = {}

        class Client:
            def __init__(self, *a, **k):
                self.chat = NS(completions=NS(create=self.create))

            def create(self, **kwargs):
                sent.update(kwargs)
                return fake_response(1)[0]

        with patch.object(run_api.openai, "OpenAI", Client), \
                patch.object(run_api, "_thread_local", NS()):
            run_api.call_chat("m", "hi", 0.8, 0.95, 10, "sys", **kw)
        return sent

    def test_seed_reaches_the_request(self):
        self.assertEqual(self.capture(seed=123)["seed"], 123)

    def test_no_seed_leaves_it_out(self):
        self.assertNotIn("seed", self.capture())


if __name__ == "__main__":
    unittest.main()
