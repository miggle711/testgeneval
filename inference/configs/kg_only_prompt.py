# Copyright (c) Meta Platforms, Inc. and affiliates.

import json

from datasets import Dataset, DatasetDict
from inference.configs.config_utils import get_first_method_partial_python


class KGOnlyPrompt:
    """KG-only completion prompt: reads pre-computed prompt text (built from
    pycodekg's TestContextExtractor + LLMSerializer output, in a separate
    repo/environment) instead of code_src/test_src.

    Deliberately different from InstructPrompt's additive design (whole
    file + test fragment) -- this arm gets NO whole-file context at all,
    only the seed function's own source plus structural context
    (callers/callees/siblings/existing tests). Answers "does KG-only,
    surgical context work as a substitute for the whole file," not "does
    KG context help on top of it."

    'full' setting is not supported (KG-only full-file generation is
    separate, deferred work -- pycodekg-side issue tracking that) --
    add_prompts_to_dataset raises if a row has no pre-computed prompt.
    """

    SYSTEM_MESSAGE = (
        "You are an expert Python software testing assistant. Your job is "
        "to complete the next test given structural context about the "
        "function under test (no full source file is provided -- work "
        "from the function's own source and its callers/callees/related "
        "tests)."
    )

    def __init__(self, prompts_path: str = "kg_prompts.json"):
        with open(prompts_path) as f:
            self._prompts_by_id = json.load(f)

    @property
    def system_message(self):
        return self.SYSTEM_MESSAGE

    @property
    def system_message_full(self):
        # Read unconditionally by run_api.py's inference_args construction
        # regardless of --skip_full, so this can't raise -- the real guard
        # is postprocess_output(is_full=True) and add_prompts_to_dataset
        # never producing a 'full' key, so this value is built but never
        # actually sent to the model when --skip_full is passed.
        return self.SYSTEM_MESSAGE

    def postprocess_output(self, text, is_full):
        if is_full:
            raise NotImplementedError(
                "KGOnlyPrompt has no 'full' setting -- pass --skip_full."
            )
        text = text.replace("```python", "```")
        if "```" not in text:
            return "compilation error"
        text_cleaned = text.split("```")[1].split("```")[0]
        return get_first_method_partial_python(text_cleaned)

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
                "first": pre_computed["first"],
                "last": pre_computed["last"],
                "extra": pre_computed["extra"],
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
