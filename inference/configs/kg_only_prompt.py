# Copyright (c) Meta Platforms, Inc. and affiliates.

import json

from datasets import Dataset, DatasetDict
from inference.configs.config_utils import postprocess_python_output
from inference.configs import shared_task


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
    scope for this arm.
    """

    # Shared with instruct (inference/configs/shared_task.py, fyp-paper#5
    # decision B): only the context differs between the arms.
    SYSTEM_MESSAGE = shared_task.SYSTEM_MESSAGE

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
        return postprocess_python_output(text, is_full)

    def add_prompts_to_dataset(self, dataset, no_import=False, tokenizer=None):
        test_data = dataset["test"]

        new_arr = []
        missing = []
        wrong_schema = []
        for new_data in test_data:
            row_id = new_data["id"]
            pre_computed = self._prompts_by_id.get(row_id)
            if pre_computed is None:
                missing.append(row_id)
                continue
            if "context" not in pre_computed:
                wrong_schema.append(row_id)
                continue
            targets = shared_task.target_names(
                pre_computed.get("target_functions", []),
                pre_computed.get("target_classes", []),
            )
            new_data["preds_prompts"] = {
                "full": pre_computed["context"].rstrip() + "\n\n" + shared_task.task_text(targets),
            }
            new_arr.append(new_data)

        if missing:
            raise ValueError(
                f"{len(missing)} row(s) have no pre-computed KG prompt "
                f"(run build_kg_prompts.py first): {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        if wrong_schema:
            raise ValueError(
                f"{len(wrong_schema)} row(s) have a kg_prompts.json entry "
                f"with no 'context' key -- built by an older pycodekg, whose "
                f"entries carried a full 'prompt' with kg_only's own task "
                f"wording. Rebuild kg_prompts.json with the current "
                f"build_kg_prompts.py, so both arms share one task text: "
                f"{wrong_schema[:5]}"
                + (" ..." if len(wrong_schema) > 5 else "")
            )

        final_dataset = DatasetDict({"test": Dataset.from_list(new_arr)})
        return final_dataset
