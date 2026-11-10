# Regression test for testgeneval#44: prompts were truncated to the model's
# published context (MODEL_LIMITS), but the vLLM server was started with a
# smaller --max-model-len (32768 by default), so every longer prompt was
# rejected on each retry and the instance was lost. The budget now uses the
# smaller of the two.

import unittest

from inference.api.run_api import EPSILON, MODEL_LIMITS, OUTPUT_LIMITS, prompt_token_budget

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


class PromptTokenBudgetTest(unittest.TestCase):
    def test_server_limit_below_published_context_wins(self):
        self.assertGreater(MODEL_LIMITS[MODEL], 32_768)
        self.assertEqual(
            prompt_token_budget(MODEL, 32_768),
            32_768 - OUTPUT_LIMITS[MODEL] - EPSILON,
        )

    def test_published_context_used_without_a_server_limit(self):
        self.assertEqual(
            prompt_token_budget(MODEL),
            MODEL_LIMITS[MODEL] - OUTPUT_LIMITS[MODEL] - EPSILON,
        )

    def test_server_limit_above_published_context_is_ignored(self):
        self.assertEqual(
            prompt_token_budget(MODEL, MODEL_LIMITS[MODEL] * 2),
            prompt_token_budget(MODEL),
        )

    def test_no_room_for_a_prompt_is_an_error(self):
        with self.assertRaises(ValueError):
            prompt_token_budget(MODEL, OUTPUT_LIMITS[MODEL])


if __name__ == "__main__":
    unittest.main()
