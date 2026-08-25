# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Audits kg_prompts.json + the TestGenEval dataset for two known risks:

1. Multi-function patches: counts how many instances touch more than
   one distinct function, and (as a sanity check on target_range.py's
   current behavior) how much of a gap there would be if ranges were
   naively merged into one (min start, max end) span instead of kept
   separate. A non-zero gap here means the instance would have had
   untouched code folded into its metric under the old merge behavior.

2. Missing target_functions/target_classes coverage in kg_prompts.json:
   instruct_prompt.py falls back to unfocused wording per-row with no
   log line unless *every* row is missing it, so a partial gap can break
   the "same target function for both arms" guarantee silently.

Usage:
    python3 scripts/audit_kg_prompts.py --kg_prompts_path kg_prompts.json \
        --dataset kjain14/testgeneval
"""

import argparse
import ast
import json
import sys

from datasets import load_dataset

sys.path.insert(0, ".")
from swebench_docker.target_range import (  # noqa: E402
    _changed_line_ranges,
    resolve_target_line_range,
)


def audit_multi_function_patches(dataset):
    multi_hunk_instances = []
    multi_function_instances = []
    gap_instances = []

    for row in dataset:
        instance_id = row["id"]
        patch = row.get("patch", "")
        code_file = row.get("code_file", "")
        post_patch_source = row.get("code_src", "")
        if not patch or not code_file or not post_patch_source:
            continue

        changed_ranges = _changed_line_ranges(patch, code_file)
        if len(changed_ranges) <= 1:
            continue
        multi_hunk_instances.append(instance_id)

        matched = resolve_target_line_range(post_patch_source, patch, code_file)
        if not matched or len(matched) <= 1:
            continue
        multi_function_instances.append(instance_id)

        merged_start = min(r[0] for r in matched)
        merged_end = max(r[1] for r in matched)
        merged_span = merged_end - merged_start + 1
        reported_span = sum(r[1] - r[0] + 1 for r in matched)
        gap = merged_span - reported_span
        if gap > 0:
            gap_instances.append((instance_id, gap, merged_span, reported_span))

    return multi_hunk_instances, multi_function_instances, gap_instances


def audit_target_function_coverage(dataset, kg_prompts_path):
    with open(kg_prompts_path) as f:
        kg_prompts = json.load(f)

    dataset_ids = {str(row["id"]) for row in dataset}
    kg_ids = set(kg_prompts.keys())

    missing_from_kg_prompts = sorted(dataset_ids - kg_ids)

    missing_prompt_key = []
    missing_target_functions = []
    for row_id, entry in kg_prompts.items():
        if row_id not in dataset_ids:
            continue
        if "prompt" not in entry:
            missing_prompt_key.append(row_id)
        target_functions = entry.get("target_functions", [])
        target_classes = entry.get("target_classes", [])
        if not target_functions and not target_classes:
            missing_target_functions.append(row_id)

    return (
        len(dataset_ids),
        len(kg_ids),
        missing_from_kg_prompts,
        missing_prompt_key,
        missing_target_functions,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_prompts_path", required=True)
    parser.add_argument("--dataset", default="kjain14/testgeneval")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)

    print("=" * 70)
    print("1. Multi-function patch audit")
    print("=" * 70)
    multi_hunk, multi_func, gaps = audit_multi_function_patches(dataset)
    print(f"Total instances: {len(dataset)}")
    print(f"Multi-hunk patches (>1 changed range): {len(multi_hunk)}")
    print(f"  ...of which touch >1 distinct function: {len(multi_func)}")
    print(f"  ...of which would have a gap if ranges were naively merged: {len(gaps)}")
    if gaps:
        gaps_sorted = sorted(gaps, key=lambda x: -x[1])
        print("\nTop 10 by gap size (lines of untouched code a naive merge would fold in):")
        for instance_id, gap, merged_span, touched_span in gaps_sorted[:10]:
            print(f"  {instance_id}: gap={gap} lines (merged={merged_span}, actually touched={touched_span})")

    print()
    print("=" * 70)
    print("2. kg_prompts.json target_functions/target_classes coverage audit")
    print("=" * 70)
    (
        n_dataset,
        n_kg,
        missing_from_kg,
        missing_prompt_key,
        missing_target,
    ) = audit_target_function_coverage(dataset, args.kg_prompts_path)
    print(f"Dataset instances: {n_dataset}")
    print(f"kg_prompts.json entries: {n_kg}")
    print(f"Dataset instances missing from kg_prompts.json entirely: {len(missing_from_kg)}")
    if missing_from_kg[:10]:
        print(f"  sample: {missing_from_kg[:10]}")
    print(f"kg_prompts.json entries missing 'prompt' key (would hard-fail kg_only): {len(missing_prompt_key)}")
    if missing_prompt_key[:10]:
        print(f"  sample: {missing_prompt_key[:10]}")
    print(f"kg_prompts.json entries with EMPTY target_functions AND target_classes")
    print(f"  (instruct silently reverts to unfocused wording for these): {len(missing_target)}")
    if missing_target[:10]:
        print(f"  sample: {missing_target[:10]}")

    if missing_target:
        pct = 100 * len(missing_target) / n_kg if n_kg else 0
        print(f"\n  {pct:.1f}% of kg_prompts.json rows would silently lose target-function")
        print(f"  framing on the instruct side while kg_only still has full structural")
        print(f"  context for the same rows. This breaks 'same target function for both")
        print(f"  arms' for these instances with no warning in the run logs.")


if __name__ == "__main__":
    main()
