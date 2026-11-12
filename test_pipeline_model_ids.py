"""The pipeline must accept model ids beyond its historical API shortlist."""

import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock


class TestPipelineModelIds(unittest.TestCase):
    def test_model_ids_reach_the_existing_backend_and_evaluation_paths(self):
        for model, generation in [
            ("Qwen/Qwen3-4B-Instruct-2507", "inference.huggingface.run_huggingface"),
            ("/models/custom-checkpoint", "inference.huggingface.run_huggingface"),
            ("gpt-4o-2024-05-13", "inference.api.run_api"),
            ("baseline", None),
        ]:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as results:
                argv = ["run_pipeline.py", "--model", model, "--results_dir", results,
                        "--dataset_name_or_path", "kjain14/testgeneval", "--skip_completion"]
                with mock.patch.object(sys, "argv", argv), mock.patch("subprocess.run") as run:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        runpy.run_path(str(Path(__file__).with_name("run_pipeline.py")), run_name="__main__")
                commands = [call.args[0] for call in run.call_args_list]
                if generation:
                    self.assertEqual(commands[0][:3], ["python", "-m", generation])
                    index = commands[0].index("--model_name_or_path")
                    self.assertEqual(commands[0][index + 1], model)
                    self.assertIn("--skip_completion", commands[0])
                    self.assertEqual(commands[1][:2], ["python", "run_evaluation.py"])
                    self.assertEqual(commands[2][:2], ["python", "generate_report.py"])
                else:
                    self.assertEqual(len(commands), 2)
                    self.assertEqual(commands[0][:2], ["python", "run_evaluation_baseline.py"])
                    self.assertEqual(commands[1][:2], ["python", "generate_report_baseline.py"])


if __name__ == "__main__":
    unittest.main()
