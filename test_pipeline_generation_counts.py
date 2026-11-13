"""Pipeline requests and downstream filenames must describe the same samples."""

import argparse
import ast
import contextlib
import io
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("run_pipeline.py")


class PipelineGenerationCountTest(unittest.TestCase):
    def run_pipeline(self, results, *options, runner=None):
        argv = [str(SCRIPT), "--model", "gpt-4o-2024-05-13", "--results_dir", results,
                "--dataset_name_or_path", "kjain14/testgeneval", *options]
        with patch.object(sys, "argv", argv), patch("subprocess.run", side_effect=runner) as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                runpy.run_path(str(SCRIPT), run_name="__main__")
        return run.call_args_list

    def test_active_setting_count_matches_the_evaluated_prediction_filename(self):
        for options, expected in [
            (["--skip_completion", "--num_samples_full", "1", "--num_samples_completion", "5"], 1),
            (["--skip_completion", "--num_samples_full", "5", "--num_samples_completion", "1"], 5),
            (["--skip_full", "--num_samples_full", "1", "--num_samples_completion", "5"], 5),
            (["--num_samples_full", "5", "--num_samples_completion", "5"], 5),
        ]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as results:
                calls = self.run_pipeline(results, *options)
                generation, evaluation, report = [c.args[0] for c in calls]
                self.assertEqual(generation[generation.index("--num_samples") + 1], str(expected))
                prediction = evaluation[evaluation.index("--predictions_path") + 1]
                self.assertTrue(prediction.endswith(f"__k{expected}__test.jsonl"), prediction)
                self.assertEqual(report[report.index("--predictions_path") + 1], prediction)

    def test_unequal_active_api_counts_are_rejected_before_starting_any_stage(self):
        with tempfile.TemporaryDirectory() as results, patch("subprocess.run") as run:
            with self.assertRaises(SystemExit) as error:
                self.run_pipeline(results, "--num_samples_full", "1", "--num_samples_completion", "5",
                                  runner=lambda *a, **k: self.fail("started generation for ambiguous counts"))
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
            self.assertEqual(list(Path(results).iterdir()), [])

    def test_huggingface_command_uses_its_real_parser_flags(self):
        # Load the real parser without importing GPU/model dependencies.
        source = SCRIPT.parent / "inference/huggingface/run_huggingface.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as results:
            argv = [str(SCRIPT), "--model", "meta-llama/CodeLlama-7b-Instruct-hf",
                    "--results_dir", results, "--dataset_name_or_path", "kjain14/testgeneval",
                    "--skip_completion", "--num_samples_full", "5"]
            with patch.object(sys, "argv", argv), patch("subprocess.run") as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(SCRIPT), run_name="__main__")
            command = run.call_args_list[0].args[0]
            with patch.object(sys, "argv", ["run_huggingface.py", *command[3:]]):
                args = namespace["parse_args"]()
            self.assertEqual(args.num_samples_generation, 5)
            from inference.prediction_paths import prediction_stem
            generated = Path(args.output_dir) / (prediction_stem(
                args.model_name_or_path, args.dataset_name_or_path, args.temperature,
                args.num_samples_generation, args.context_size) + ".jsonl")
            evaluation = run.call_args_list[1].args[0]
            self.assertEqual(str(generated), evaluation[evaluation.index("--predictions_path") + 1])

    def test_huggingface_kg_only_cannot_silently_use_an_instruct_prompt(self):
        with tempfile.TemporaryDirectory() as results:
            with self.assertRaises(SystemExit) as error:
                self.run_pipeline(results, "--model", "meta-llama/CodeLlama-7b-Instruct-hf",
                                  "--prompt_config", "kg_only", "--skip_completion",
                                  runner=lambda *a, **k: self.fail("started unsupported KG generation"))
            self.assertEqual(error.exception.code, 2)

    def test_stage_failure_stops_the_pipeline_before_later_stages(self):
        for fail_at in range(3):
            with self.subTest(stage=fail_at), tempfile.TemporaryDirectory() as results:
                observed = []

                def runner(command, **kwargs):
                    observed.append(command)
                    if len(observed) == fail_at + 1:
                        if kwargs.get("check"):
                            raise subprocess.CalledProcessError(7, command)
                        return subprocess.CompletedProcess(command, 7)
                    return subprocess.CompletedProcess(command, 0)

                with self.assertRaises(subprocess.CalledProcessError):
                    self.run_pipeline(results, "--skip_completion", runner=runner)
                self.assertEqual(len(observed), fail_at + 1)


if __name__ == "__main__":
    unittest.main()
