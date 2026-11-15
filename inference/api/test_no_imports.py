# NO_IMPORTS=1 in m3_run_inference.slurm passes --no_imports, which drops
# the reference test file's imports (local_imports) from instruct's full
# prompt (fyp-paper#7). Unset keeps them, as before.

import os
import re
import subprocess
import unittest

from datasets import Dataset, DatasetDict

from inference.configs.instruct_prompt import InstructPrompt

SLURM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3_run_inference.slurm")
MARKER = "examples of how to import"


def one_row():
    return DatasetDict({"test": Dataset.from_list([{
        "id": "a", "code_src": "def f():\n    return 1\n", "code_file": "pkg/mod.py",
        "local_imports": ["from pkg.mod import f"],
        "preds_context": {"preamble": "", "last_minus_one": "", "last": ""},
    }])})


class InstructPromptImportsTest(unittest.TestCase):
    def full_prompt(self, no_import):
        out = InstructPrompt().add_prompts_to_dataset(one_row(), no_import=no_import)
        return out["test"][0]["preds_prompts"]["full"]

    def test_default_keeps_the_import_examples(self):
        p = self.full_prompt(False)
        self.assertIn(MARKER, p)
        self.assertIn("from pkg.mod import f", p)

    def test_no_imports_drops_them_and_keeps_the_rest(self):
        with_imports, without = self.full_prompt(False), self.full_prompt(True)
        self.assertNotIn(MARKER, without)
        self.assertNotIn("from pkg.mod import f", without)
        self.assertIn("def f():", without)
        self.assertIn("pkg/mod.py", without)
        # Only the import block differs: removing it from the default prompt
        # gives exactly the NO_IMPORTS prompt (on main, and with #103's layout).
        block = re.compile(r"Here are some examples of how to import .*?```python\n.*?\n```\n\n", re.S)
        self.assertEqual(len(block.findall(with_imports)), 1)
        self.assertEqual(block.sub("", with_imports), without)


class SlurmPassesNoImportsTest(unittest.TestCase):
    """The slurm script's NO_IMPORTS block, run on its own."""

    def block(self):
        text = open(SLURM).read()
        start = text.index('NO_IMPORTS="${NO_IMPORTS:-0}"')
        end = text.index('echo "import examples in instruct:', start)
        return text[start:text.index("\n", end) + 1]

    def run_block(self, value):
        script = "EXTRA_ARGS=()\n" + self.block() + 'printf "%s\\n" "${EXTRA_ARGS[@]}"\n'
        env = {k: v for k, v in os.environ.items() if k != "NO_IMPORTS"}
        if value is not None:
            env["NO_IMPORTS"] = value
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    def test_set(self):
        r = self.run_block("1")
        self.assertEqual(r.returncode, 0)
        self.assertIn("--no_imports", r.stdout.split("\n"))
        self.assertIn("dropped (NO_IMPORTS=1)", r.stdout)

    def test_unset_keeps_them(self):
        r = self.run_block(None)
        self.assertEqual(r.returncode, 0)
        self.assertNotIn("--no_imports", r.stdout.split("\n"))
        self.assertIn("kept (NO_IMPORTS=0)", r.stdout)

    def test_another_value_stops_the_job(self):
        r = self.run_block("yes")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("NO_IMPORTS must be 0 or 1", r.stderr)


if __name__ == "__main__":
    unittest.main()
