# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Regression test for testgeneval#86: run_evaluation.py's call into
run_apptainer_evaluation/run_docker_evaluation must pass the real,
configured timeout -- not silently drop it into the unrelated `ind`
slot via positional args and fall back to run_apptainer_evaluation's
own hardcoded default of 60.

This bug class went undetected for the entire lifetime of the
apptainer backend (present since #52) because nothing asserted what
value actually reached the eval function's timeout parameter.
"""

import asyncio
import unittest
from unittest.mock import patch

import run_evaluation
from swebench_docker.constants import KEY_ID, KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTIONS

CONFIGURED_TIMEOUT = 3600
DEFAULT_TIMEOUT = 60  # run_apptainer_evaluation/run_docker_evaluation's own default


class TestRunEvaluationTimeoutWiring(unittest.TestCase):
    def _run_main_with_mocks(self, backend):
        task = {
            "repo": "django/django",
            "version": "1.0",
            "base_commit": "abc123",
            KEY_ID: "django__django-1-1",
            KEY_INSTANCE_ID: "django__django-1",
            "preds_context": {},
            "test_patch": "",
            "test_file": "",
            "code_file": "",
            "patch": "",
        }
        prediction = {
            KEY_ID: "django__django-1-1",
            KEY_INSTANCE_ID: "django__django-1",
            KEY_MODEL: "some-model",
            KEY_PREDICTIONS: {"full": "print(1)"},
        }

        received_calls = []

        async def fake_eval_fn(*args, **kwargs):
            received_calls.append((args, kwargs))
            return None

        with patch("run_evaluation.get_eval_refs", return_value={task[KEY_ID]: task}), \
             patch("run_evaluation.validate_predictions", return_value=None), \
             patch("run_evaluation.get_instances", return_value=[prediction]), \
             patch("run_evaluation.get_test_directives", return_value=[]), \
             patch("run_evaluation.MAP_REPO_TO_TEST_FRAMEWORK", {"django/django": "pytest"}), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.isdir", return_value=True), \
             patch("os.chmod", return_value=None), \
             patch("run_evaluation.run_apptainer_evaluation", side_effect=fake_eval_fn), \
             patch("run_evaluation.run_docker_evaluation", side_effect=fake_eval_fn):
            asyncio.run(
                run_evaluation.main(
                    predictions_path="fake.jsonl",
                    swe_bench_tasks="fake_tasks.json",
                    namespace="fake-namespace",
                    log_dir="/fake/log/dir",
                    timeout=CONFIGURED_TIMEOUT,
                    backend=backend,
                )
            )

        return received_calls

    def test_apptainer_backend_passes_configured_timeout(self):
        calls = self._run_main_with_mocks(backend="apptainer")
        self.assertEqual(len(calls), 1)
        _, kwargs = calls[0]
        self.assertEqual(
            kwargs.get("timeout"),
            CONFIGURED_TIMEOUT,
            "configured timeout must reach the eval function as a keyword arg, "
            "not silently fall back to the eval function's own default",
        )
        self.assertNotEqual(kwargs.get("timeout"), DEFAULT_TIMEOUT)

    def test_docker_backend_passes_configured_timeout(self):
        calls = self._run_main_with_mocks(backend="docker")
        self.assertEqual(len(calls), 1)
        _, kwargs = calls[0]
        self.assertEqual(kwargs.get("timeout"), CONFIGURED_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
