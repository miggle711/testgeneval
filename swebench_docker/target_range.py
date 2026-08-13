# Copyright (c) Meta Platforms, Inc. and affiliates.

import ast
import re
from typing import List, Optional, Tuple

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _changed_line_ranges(patch: str, code_file: str) -> List[Tuple[int, int]]:
    """Line ranges (post-patch, 1-indexed, inclusive) touched by hunks
    against code_file in a unified diff.
    """
    lines = patch.split("\n")
    if patch.endswith("\n"):
        lines = lines[:-1]

    ranges = []
    in_target_file = False
    cur_line = None

    for line in lines:
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            # Exact match, not substring: avoids "models.py" matching
            # "+++ b/test_models.py" in a multi-file patch.
            path = line[len("+++ "):]
            if path.startswith(("a/", "b/")):
                path = path[2:]
            in_target_file = path == code_file
            continue

        if not in_target_file:
            continue

        header = _HUNK_HEADER_RE.match(line)
        if header:
            cur_line = int(header.group(1))
            continue

        if cur_line is None:
            continue

        if line.startswith("+"):
            ranges.append((cur_line, cur_line))
            cur_line += 1
        elif line.startswith("-"):
            # Anchor a pure deletion to the post-patch line immediately
            # after it, so a hunk that only removes code still resolves
            # to the enclosing function instead of being dropped.
            ranges.append((cur_line, cur_line))
        else:
            cur_line += 1

    return ranges


def resolve_target_line_range(
    post_patch_source: str, patch: str, code_file: str
) -> Optional[Tuple[int, int]]:
    """Smallest set of function def line ranges (1-indexed, inclusive)
    covering the patch's changed lines in code_file, merged into one
    (min start, max end) range. None if no changed line falls inside a
    function (e.g. a module-level-only change).
    """
    changed_ranges = _changed_line_ranges(patch, code_file)
    if not changed_ranges:
        return None

    try:
        tree = ast.parse(post_patch_source)
    except SyntaxError:
        return None

    func_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                continue
            func_ranges.append((node.lineno, end_lineno))

    matched = []
    for changed_start, changed_end in changed_ranges:
        best = None
        for func_start, func_end in func_ranges:
            if func_start <= changed_start and changed_end <= func_end:
                if best is None or (func_end - func_start) < (best[1] - best[0]):
                    best = (func_start, func_end)
        if best:
            matched.append(best)

    if not matched:
        return None

    return (min(r[0] for r in matched), max(r[1] for r in matched))
