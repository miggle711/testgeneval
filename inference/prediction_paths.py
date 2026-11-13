"""Prediction filenames shared by generation and pipeline consumers."""

from pathlib import Path


def prediction_stem(model_name, dataset_name_or_path, temperature, num_samples, context_size=-1, split="test"):
    stem = (f"{Path(model_name).name}__{Path(dataset_name_or_path).name}__"
            f"{temperature}__k{num_samples}__{split}")
    if context_size != -1:
        stem += f"__{context_size}"
    return stem
