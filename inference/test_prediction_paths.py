"""Shared generation/consumer names must remain flat and preserve run identity."""

from pathlib import Path
import unittest

from inference.prediction_paths import prediction_stem


class PredictionPathsTest(unittest.TestCase):
    def test_hub_model_and_dataset_do_not_create_subdirectories(self):
        name = prediction_stem("meta-llama/CodeLlama-7b-Instruct-hf", "kjain14/testgeneval", 0.8, 5)
        self.assertEqual(Path(name).name, name)
        self.assertEqual(name, "CodeLlama-7b-Instruct-hf__testgeneval__0.8__k5__test")

    def test_local_dataset_optional_split_and_context_limit_are_preserved(self):
        self.assertEqual(
            prediction_stem("model", "/scratch/dataset/", 0.2, 1, 4096, split="validation"),
            "model__dataset__0.2__k1__validation__4096",
        )


if __name__ == "__main__":
    unittest.main()
