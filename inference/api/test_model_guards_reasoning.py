# Two things a model new to this code must not slip past (fyp-paper#7 §2):
# - its output cap and context must be set, not defaulted: an unlisted model
#   used to get a 4,096-token cap and a 32,000-token context silently;
# - reasoning it writes into its reply must not be taken as the answer: the
#   test file is extracted from the first code fence, so a fence inside the
#   reasoning would have been used instead of the final one.

import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from datasets import Dataset

from inference.api import run_api
from inference.configs.config_utils import postprocess_python_output

KNOWN = "Qwen/Qwen3-4B-Instruct-2507"
NEW = "google/gemma-4-12B-it"
SLURM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3_run_inference.slurm")

GOOD = "```python\ndef test_good():\n    assert True\n```"
BAD = "```python\ndef test_from_reasoning():\n    pass\n```"


def extract(reply):
    answer, status = run_api.split_reasoning(reply)
    return postprocess_python_output(answer, True), status


class LimitGuardsTest(unittest.TestCase):
    def test_an_unlisted_model_needs_an_explicit_output_limit(self):
        with self.assertRaisesRegex(ValueError, "no OUTPUT_LIMITS entry"):
            run_api.output_limit_for(NEW)
        self.assertEqual(run_api.output_limit_for(NEW, 8_000), 8_000)

    def test_a_listed_model_keeps_its_entry(self):
        self.assertEqual(run_api.output_limit_for(KNOWN), run_api.OUTPUT_LIMITS[KNOWN])

    def test_an_unlisted_model_needs_the_serving_context(self):
        with self.assertRaisesRegex(ValueError, "no MODEL_LIMITS entry"):
            run_api.prompt_token_budget(NEW, None, 8_000)
        self.assertEqual(run_api.prompt_token_budget(NEW, 65_536, 8_000),
                         65_536 - 8_000 - run_api.EPSILON)

    def test_a_listed_model_still_takes_the_smaller_of_entry_and_server(self):
        self.assertEqual(run_api.prompt_token_budget(KNOWN, 32_768),
                         32_768 - run_api.OUTPUT_LIMITS[KNOWN] - run_api.EPSILON)


class SplitReasoningTest(unittest.TestCase):
    def test_reasoning_containing_code_is_not_the_answer(self):
        for opening, closing in run_api.REASONING_BLOCKS:
            with self.subTest(opening=opening):
                pred, status = extract(f"{opening}let me try\n{BAD}\nok{closing}\n{GOOD}")
                self.assertIn("test_good", pred)
                self.assertEqual(status, "reasoning_stripped")

    def test_a_closing_marker_alone_ends_the_reasoning(self):
        # The chat template can open the block in the prompt, so the reply
        # holds only the closing marker.
        pred, status = extract(f"thinking about it\n{BAD}".replace("```", "~~~") + f"</think>\n{GOOD}")
        self.assertIn("test_good", pred)
        self.assertEqual(status, "reasoning_stripped")

    def test_unfinished_reasoning_gives_no_answer(self):
        for opening, _closing in run_api.REASONING_BLOCKS:
            with self.subTest(opening=opening):
                answer, status = run_api.split_reasoning(f"{opening}still going\n{BAD}")
                self.assertEqual((answer, status), ("", "reasoning_unfinished"))
                self.assertEqual(postprocess_python_output(answer, True), "compilation error")

    def test_a_plain_answer_is_unchanged(self):
        self.assertEqual(run_api.split_reasoning(GOOD), (GOOD, None))

    def test_an_empty_reply_stays_empty(self):
        self.assertEqual(run_api.split_reasoning(""), ("", None))

    def test_tag_text_inside_the_test_file_is_left_alone(self):
        reply = ('```python\ndef test_tags():\n    assert "</think>" in render()\n'
                 '    assert "<channel|>" != "<think>"\n```')
        self.assertEqual(run_api.split_reasoning(reply), (reply, None))


class InferenceRecordsReasoningTest(unittest.TestCase):
    def run_inference(self, contents, chat_template_kwargs=None, reasoning_attr=None, env=None):
        sent = []

        def fake_call_chat(model, prompt, temperature, top_p, max_tokens, *a, n=1, seed=None,
                           chat_template_kwargs=None, **kw):
            sent.append(chat_template_kwargs)
            choices = [NS(message=NS(content=c, **({reasoning_attr: "r"} if reasoning_attr else {})),
                          finish_reason="stop") for c in contents]
            usage = NS(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
            return NS(choices=choices, usage=usage, model="m", system_fingerprint=None), 0.0

        out = os.path.join(tempfile.mkdtemp(), "preds.jsonl")
        rows = [{"id": "a", "instance_id": "a", "preds_prompts": {"full": "p"}}]
        with patch.object(run_api, "call_chat", fake_call_chat), \
                patch.dict(os.environ, {"LOCAL_MODEL_BASE_URL": "http://x/v1/",
                                        "SERVER_MAX_MODEL_LEN": "65536", **(env or {})}):
            run_api.openai_inference(Dataset.from_list(rows), NEW, out, {"temperature": 0.8}, set(),
                                     None, len(contents), lambda t, f: postprocess_python_output(t, f),
                                     "sys", "sys", False, skip_completion=True, seed_base=0,
                                     output_limit=8_000, chat_template_kwargs=chat_template_kwargs)
        return sent, json.loads(open(out).readline())

    def test_settings_are_sent_and_recorded(self):
        kwargs = {"enable_thinking": False}
        sent, row = self.run_inference([GOOD], chat_template_kwargs=kwargs,
                                       env={"REASONING_PARSER": "qwen3"})
        self.assertEqual(sent, [kwargs])
        sampling = row["sampling"]["full"]
        self.assertEqual((sampling["chat_template_kwargs"], sampling["reasoning_parser"],
                          sampling["max_model_len"], sampling["max_tokens"]),
                         (kwargs, "qwen3", 65_536, 8_000))

    def test_reasoning_in_the_reply_is_stripped_and_the_raw_reply_kept(self):
        reply = f"<think>hmm\n{BAD}</think>\n{GOOD}"
        _, row = self.run_inference([reply])
        self.assertIn("test_good", row["preds"]["full"][0])
        self.assertEqual(row["raw"]["full"], [reply])
        self.assertEqual(row["usage"]["full"][0]["reasoning_in_content"], "reasoning_stripped")
        self.assertFalse(row["usage"]["full"][0]["reasoning_separated"])

    def test_a_reply_ending_inside_its_reasoning_has_no_answer(self):
        _, row = self.run_inference([f"<|channel>thought\n{BAD}", GOOD])
        self.assertEqual(row["preds"]["full"][0], "compilation error")
        self.assertTrue(row["usage"]["full"][0]["no_final_answer"])
        self.assertIn("test_good", row["preds"]["full"][1])
        self.assertNotIn("no_final_answer", row["usage"]["full"][1])

    def test_reasoning_the_server_separated_is_recorded(self):
        _, row = self.run_inference([GOOD], reasoning_attr="reasoning_content")
        self.assertTrue(row["usage"]["full"][0]["reasoning_separated"])
        self.assertNotIn("reasoning_in_content", row["usage"]["full"][0])

    def test_without_settings_nothing_is_sent(self):
        sent, row = self.run_inference([GOOD])
        self.assertEqual(sent, [None])
        self.assertIsNone(row["sampling"]["full"]["chat_template_kwargs"])


class CallChatSendsKwargsTest(unittest.TestCase):
    def test_kwargs_go_in_extra_body(self):
        seen = {}

        class FakeCompletions:
            def create(self, **kwargs):
                seen.update(kwargs)
                usage = NS(prompt_tokens=1, completion_tokens=1)
                return NS(usage=usage, choices=[])

        client = NS(chat=NS(completions=FakeCompletions()))
        with patch.object(run_api, "_key_rotator", None), \
                patch.object(run_api._thread_local, "client", client, create=True), \
                patch.object(run_api, "calc_cost", return_value=0.0, create=True):
            try:
                run_api.call_chat(NEW, "p", 0.8, 0.95, 100, "sys",
                                  chat_template_kwargs={"enable_thinking": False})
            except Exception:
                pass  # only the request matters here
        self.assertEqual(seen.get("extra_body"), {"chat_template_kwargs": {"enable_thinking": False}})


class SlurmBlocksTest(unittest.TestCase):
    def block(self, start, end):
        text = open(SLURM).read()
        i = text.index(start)
        j = text.index(end, i)
        return text[i:text.index("\n", j) + 1]

    def run_block(self, block, array, env):
        script = f"{array}=()\n" + block + f'printf "%s\\n" "${{{array}[@]}}"\n'
        clean = {k: v for k, v in os.environ.items() if k not in ("REASONING_PARSER", "CHAT_TEMPLATE_KWARGS")}
        return subprocess.run(["bash", "-c", script], env={**clean, **env}, capture_output=True,
                              text=True, check=True).stdout.split("\n")

    def test_reasoning_parser(self):
        block = self.block('REASONING_PARSER="${REASONING_PARSER:-}"', 'echo "reasoning parser:')
        out = self.run_block(block, "VLLM_SERVE_ARGS", {"REASONING_PARSER": "qwen3"})
        self.assertIn("--reasoning-parser", out)
        self.assertIn("qwen3", out)
        self.assertNotIn("--reasoning-parser", self.run_block(block, "VLLM_SERVE_ARGS", {}))

    def test_chat_template_kwargs(self):
        block = self.block('CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-}"', 'echo "chat template kwargs:')
        value = '{"enable_thinking": false}'
        out = self.run_block(block, "EXTRA_ARGS", {"CHAT_TEMPLATE_KWARGS": value})
        self.assertIn("--chat_template_kwargs", out)
        self.assertIn(value, out)
        self.assertNotIn("--chat_template_kwargs", self.run_block(block, "EXTRA_ARGS", {}))


if __name__ == "__main__":
    unittest.main()
