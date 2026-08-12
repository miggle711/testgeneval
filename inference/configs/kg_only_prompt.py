# Copyright (c) Meta Platforms, Inc. and affiliates.

import json

from datasets import Dataset, DatasetDict
from inference.configs.config_utils import get_first_method_partial_python


class KGOnlyPrompt:
    """KG-only test-generation prompt: reads pre-computed prompt text
    (built from pycodekg's TestContextExtractor + LLMSerializer output,
    in a separate repo/environment) instead of code_src/test_src.

    Targets TestGenEval's `full` setting only -- generate a complete test
    file from scratch, given only the seed function's KG-derived
    structural context (its own source, callers/callees/siblings,
    existing tests already linked to it) instead of the whole code file.
    No test content of any kind is shown to either arm under `full`, so
    this is a fair comparison to instruct's `full`-setting prompt on the
    test side; the completion settings (first/last/extra) are out of
    scope for this arm (see miggle711/pycodekg's
    docs/EXPERIMENT_PLAN.md for the scope decision).
    """

    SYSTEM_MESSAGE = (
        "You are an expert Python software testing assistant. Your job "
        "is to generate a complete test file for the given code, using "
        "structural context about the function under test (no full "
        "source file is provided -- work from the function's own source "
        "and its callers/callees/related tests)."
    )

    def __init__(self, prompts_path: str = "kg_prompts.json"):
        with open(prompts_path) as f:
            self._prompts_by_id = json.load(f)

    @property
    def system_message(self):
        return self.SYSTEM_MESSAGE

    @property
    def system_message_full(self):
        return self.SYSTEM_MESSAGE

    def postprocess_output(self, text, is_full):
        text = text.replace("```python", "```")
        if "```" not in text:
            return "compilation error"
        text_cleaned = text.split("```")[1].split("```")[0]
        return (
            text_cleaned if is_full else get_first_method_partial_python(text_cleaned)
        )

    def add_prompts_to_dataset(self, dataset, no_import=False, tokenizer=None):
        test_data = dataset["test"]

        new_arr = []
        missing = []
        for new_data in test_data:
            row_id = new_data["id"]
            pre_computed = self._prompts_by_id.get(row_id)
            if pre_computed is None:
                missing.append(row_id)
                continue

            new_data["preds_prompts"] = {
                "full": pre_computed["prompt"],
            }
            new_arr.append(new_data)

        if missing:
            raise ValueError(
                f"{len(missing)} row(s) have no pre-computed KG prompt "
                f"(run build_kg_prompts.py first): {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )

        final_dataset = DatasetDict({"test": Dataset.from_list(new_arr)})
        return final_dataset
