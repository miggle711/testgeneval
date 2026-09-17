# Real regression protection for call_chat's model-specific branching
# (testgeneval#63): gpt-5's real API shape differs from every other
# model this project calls -- max_completion_tokens instead of
# max_tokens, no temperature/top_p at all. Nothing else in this repo's
# test suite exercises this branch directly.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from inference.api import run_api


def _fake_response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="```test```"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, completion_tokens_details=None
        ),
    )


class TestCallChatModelBranching(unittest.TestCase):
    def _call_chat_and_capture_kwargs(self, model_name_or_path):
        mock_create = MagicMock(return_value=_fake_response())
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
        )
        with patch.object(run_api, "openai") as mock_openai, patch.object(
            run_api, "_thread_local", SimpleNamespace()
        ):
            mock_openai.OpenAI.return_value = fake_client
            run_api.call_chat(
                model_name_or_path=model_name_or_path,
                inputs="hello",
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
                system_message="system",
            )
        self.assertEqual(mock_create.call_count, 1)
        return mock_create.call_args.kwargs

    def test_gpt5_uses_max_completion_tokens_no_temperature_no_top_p(self):
        kwargs = self._call_chat_and_capture_kwargs("gpt-5")
        self.assertEqual(kwargs["max_completion_tokens"], 100)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)

    def test_non_gpt5_uses_max_tokens_temperature_top_p(self):
        kwargs = self._call_chat_and_capture_kwargs("openai/gpt-oss-20b")
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertEqual(kwargs["temperature"], 0.8)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertNotIn("max_completion_tokens", kwargs)


if __name__ == "__main__":
    unittest.main()
