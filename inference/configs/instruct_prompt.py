# Copyright (c) Meta Platforms, Inc. and affiliates.

import json

from datasets import Dataset, DatasetDict
from inference.configs.config_utils import postprocess_python_output
from inference.configs import shared_task


class InstructPrompt:
    def __init__(self, kg_prompts_path: str = None):
        """
        Args:
            kg_prompts_path: Path to the same kg_prompts.json the kg_only
                arm reads (built by pycodekg's build_kg_prompts.py). Used
                only for its target_functions/target_classes fields (both
                lists -- a patch can change more than one function), to
                name the same changed function(s) to this arm that
                kg_only's own patch-seeded subgraph already focuses on --
                see miggle711/pycodekg#125, miggle711/testgeneval#6. If
                not given, prompts fall back to their original, unfocused
                wording (no crash -- lets instruct run standalone without
                a pycodekg-side build having happened first).
        """
        self._target_by_id = {}
        if kg_prompts_path:
            with open(kg_prompts_path) as f:
                kg_prompts = json.load(f)
            # Keys coerced to str (JSON keys always are, dataset ids
            # might not be) so a type mismatch can't cause silent misses.
            self._target_by_id = {
                str(row_id): (
                    entry.get("target_functions", []),
                    entry.get("target_classes", []),
                )
                for row_id, entry in kg_prompts.items()
            }

        self.SYSTEM_MESSAGE = "You are an expert Python software testing assistant. Your job is to complete the next test given a code file an some existing test context."
        # The full setting's system message and task wording are shared
        # with kg_only (inference/configs/shared_task.py, fyp-paper#5
        # decision B); only the context above the task differs.
        self.SYSTEM_MESSAGE_FULL = shared_task.SYSTEM_MESSAGE

        self.PROMPT_FULL = """Below is a code file:
```python
{code_src}
```

The code file is called: {code_filename}

Here are some examples of how to import {code_filename}, (you should use these as reference)

```python
{imports}
```

{task}"""

        self.PROMPT_FULL_NO_IMPORT = """Below is a code file:
```python
{code_src}
```

The code file is called: {code_filename}

{task}"""

        self.PROMPT_COMPLETION = """Below is a code file:
```python
{code_src}
```

And the current unit test file
```python
{test_src}
```

Your job write the Python code the next test in the file. Ideally your next test should improve coverage of the 
existing unit test file for the code file.

Only output the next unit test, preserve indentation and formatting. Do not output anything else. Format like this:

```python
Next unit test Python code
```
"""

    @property
    def system_message(self):
        return self.SYSTEM_MESSAGE

    @property
    def system_message_full(self):
        return self.SYSTEM_MESSAGE_FULL

    def postprocess_output(self, text, is_full):
        return postprocess_python_output(text, is_full)

    def add_prompts_to_dataset(self, dataset, no_import=False, tokenizer=None):
        test_data = dataset["test"]

        new_arr = []
        target_lookup_misses = 0
        for new_data in test_data:
            code_src = new_data["code_src"]

            target_functions, target_classes = self._target_by_id.get(
                str(new_data["id"]), ([], [])
            )
            if self._target_by_id and not target_functions:
                target_lookup_misses += 1
            # Named per instance's changed function(s), from kg_prompts.json's
            # target_functions/target_classes -- the same targets kg_only's
            # prompt names, so both arms answer the same task.
            task = shared_task.task_text(
                shared_task.target_names(target_functions, target_classes)
            )

            full_context = self.PROMPT_FULL.format(
                code_src=code_src,
                code_filename=new_data["code_file"],
                imports="\n".join(new_data["local_imports"]),
                task=task,
            )
            full_context_no_import = self.PROMPT_FULL_NO_IMPORT.format(
                code_src=code_src, code_filename=new_data["code_file"], task=task
            )
            first_context = self.PROMPT_COMPLETION.format(
                code_src=code_src, test_src=new_data["preds_context"]["preamble"]
            )
            last_context = self.PROMPT_COMPLETION.format(
                code_src=code_src, test_src=new_data["preds_context"]["last_minus_one"]
            )
            extra_context = self.PROMPT_COMPLETION.format(
                code_src=code_src, test_src=new_data["preds_context"]["last"]
            )

            new_data["preds_prompts"] = {
                "full": full_context_no_import if no_import else full_context,
                "first": first_context,
                "last": last_context,
                "extra": extra_context,
            }
            new_arr.append(new_data)

        # 100% miss against a non-empty kg_prompts.json signals a
        # systemic problem (e.g. id type mismatch), not routine misses.
        if self._target_by_id and len(new_arr) > 0 and target_lookup_misses == len(new_arr):
            print(
                f"WARNING: InstructPrompt found 0/{len(new_arr)} rows in "
                f"kg_prompts.json despite it having {len(self._target_by_id)} "
                f"entries -- check that dataset row ids match kg_prompts.json's "
                f"keys."
            )

        final_dataset = DatasetDict({"test": Dataset.from_list(new_arr)})

        return final_dataset
