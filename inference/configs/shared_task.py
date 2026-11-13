"""The instructions both arms share (fyp-paper#5, decision B).

instruct and kg_only differ only in the context they show: the whole code
file for instruct, the KG-derived context for kg_only (pycodekg's
build_kg_prompts.py writes it to kg_prompts.json as `context`). Everything
the model is told to do -- the system message, which function to test, the
coverage and error-case wording, and the output format -- comes from here,
so the two arms can not drift apart. The submitted paper noted that the
earlier, different wording alone could explain part of the mutation gain
(§VI-A).
"""

from typing import List, Optional, Sequence

SYSTEM_MESSAGE = (
    "You are an expert Python software testing assistant. Your job is to "
    "generate a complete unit test file for code that was recently "
    "changed, using the context provided."
)

_RULES = """Each unit test must be a function starting with test_. Include all your test imports and setup before your first test. Do not run the tests in the file, just output a series of tests. Do not include a main method to run the tests.

Only output the unit test Python file in this format:

```python
Unit test Python code (file level)
```
"""


def target_names(
    target_functions: Sequence[str], target_classes: Sequence[Optional[str]]
) -> List[str]:
    """`Class.function` (or `function`) per target, from kg_prompts.json's
    parallel target_functions/target_classes lists."""
    classes = list(target_classes) + [None] * (len(target_functions) - len(target_classes))
    return [f"{cls}.{fn}" if cls else fn for fn, cls in zip(target_functions, classes) if fn]


def task_text(targets: Sequence[str]) -> str:
    """The task paragraph, rules and output format, placed after the
    context in both arms. With no target (a row kg_prompts.json does not
    cover), the same text refers to the code above instead."""
    if len(targets) == 1:
        what = f"tests `{targets[0]}`, which was recently modified. Aim for high coverage of it"
    elif targets:
        names = ", ".join(f"`{t}`" for t in targets)
        what = f"tests {names}, which were recently modified. Aim for high coverage of them"
    else:
        what = ("tests the code above, which was recently modified. "
                "Aim for high coverage of it")
    return (
        f"Your job is to output a unit test file that {what}, including error "
        "cases and boundary conditions. The tests must import and call the "
        "code under test directly.\n\n" + _RULES
    )
