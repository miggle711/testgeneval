"""A log directory holds one run. Logs are named by id, model and setting,
so a second arm, sample count or mutation setting evaluated into the same
directory used to be skipped as already done, or to replace the first
run's logs, with nothing in the output to show it."""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

import run_evaluation
from swebench_docker.constants import (
    KEY_ID,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTIONS,
    UNFILTERED_TESTS_FAILED,
)

ID = "django__django-16707-16204"
TASK = {
    "repo": "django/django", "version": "5.0", "base_commit": "x", KEY_ID: ID,
    KEY_INSTANCE_ID: "django__django-16707", "preds_context": {}, "test_patch": "",
    "test_file": "", "code_file": "", "patch": "",
}
# Both arms of one model at one temperature record the same model name.
MODEL = "Qwen3-4B-Instruct-2507_t=0.8"


def prediction(arm, k):
    return {
        KEY_ID: ID, KEY_INSTANCE_ID: "django__django-16707", KEY_MODEL: MODEL,
        KEY_PREDICTIONS: {"full": [f"{arm} sample {i}" for i in range(k)]},
    }


async def fake_container(task_instance, namespace, log_dir, setting, ind, **kwargs):
    name = f"{task_instance[KEY_ID]}.{task_instance[KEY_MODEL]}.{setting}.eval.log"
    with open(os.path.join(log_dir, name), "w") as f:
        for sample in task_instance[KEY_PREDICTIONS][setting]:
            f.write(f"evaluated: {sample}\n{UNFILTERED_TESTS_FAILED}\n")


class TestLogDirHoldsOneRun(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.log_dir = d.name
        self.log = os.path.join(self.log_dir, f"{ID}.{MODEL}.full.eval.log")

    def evaluate(self, arm, k, predictions_path="preds.jsonl", skip_mutation=False, scope=None):
        env = {"MUTATION_SCOPE": scope} if scope else {}
        with patch("run_evaluation.get_eval_refs", return_value={ID: TASK}), \
             patch("run_evaluation.validate_predictions"), \
             patch("run_evaluation.get_instances", return_value=[prediction(arm, k)]), \
             patch("run_evaluation.get_test_directives", return_value=[]), \
             patch("run_evaluation.run_apptainer_evaluation", side_effect=fake_container), \
             patch.dict(os.environ, env):
            asyncio.run(run_evaluation.main(
                predictions_path, "tasks", "kdjain", self.log_dir,
                skip_existing=True, skip_mutation=skip_mutation, backend="apptainer",
            ))

    def first_line(self):
        with open(self.log) as f:
            return f.readline().strip()

    def test_a_different_run_into_the_same_directory_is_refused(self):
        second_runs = {
            "other arm, same sample count": dict(arm="kg_only", k=1),
            "other arm, more samples": dict(arm="kg_only", k=5),
            "same predictions, mutation now skipped": dict(arm="instruct", k=1, skip_mutation=True),
            "same predictions, other mutation scope": dict(arm="instruct", k=1, scope="function"),
        }
        for name, second in second_runs.items():
            with self.subTest(name):
                for leftover in os.listdir(self.log_dir):
                    os.remove(os.path.join(self.log_dir, leftover))
                self.evaluate("instruct", 1)
                with self.assertRaisesRegex(ValueError, "separate --log_dir"):
                    self.evaluate(**second)
                self.assertEqual(self.first_line(), "evaluated: instruct sample 0")
                self.assertEqual(
                    [n for n in os.listdir(self.log_dir) if ".partial" in n], []
                )

    def test_the_same_predictions_from_another_file_share_the_directory(self):
        # A shard, or a resubmit from a copied file: same samples, new path.
        self.evaluate("instruct", 1, predictions_path="results/instruct/run.jsonl")
        self.evaluate("instruct", 1, predictions_path="shards/shard-0.jsonl")
        self.assertEqual(self.first_line(), "evaluated: instruct sample 0")


if __name__ == "__main__":
    unittest.main()
