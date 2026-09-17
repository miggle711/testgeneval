# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Merges a SKIP_MUTATION=1 any_pass_at_1 backfill run's real .eval.log
files into a copy of the original (mutation-included) evaluation's logs,
producing one combined log directory generate_report.py can read.

Real problem this solves (see CLAUDE.md's any_pass_at_1 gotcha,
testgeneval#76): get_logs_eval reads every metric for one instance
(coverage, mutation, any_tests_passed) from the same single .eval.log
file, and generate_report.py only reads from one --log_dir at a time.
A cheap backfill run (SKIP_MUTATION=1, to avoid re-paying the real
20-40min/instance mutation cost) has to write to a NEW log dir, since
--skip_existing only checks file existence, not content -- so the
backfill produces real any_tests_passed data with no mutation data,
in a directory disjoint from the original's real mutation/coverage
data. Neither directory alone gives the full picture.

This script never modifies either source directory -- only reads both,
writes a new, third directory with one real combined log per instance.
Safe to re-run.

Usage:
    python scripts/merge_any_pass_logs.py \
      --original-log-dir /path/to/original/logs \
      --backfill-log-dir /path/to/anypass1_logs/some_model \
      --out-dir /path/to/merged_logs
"""

import argparse
import glob
import os
import sys

# Inlined rather than imported from swebench_docker.constants -- this
# script runs standalone (same convention as merge_and_validate.py),
# no package install/PYTHONPATH setup required. Real, stable marker
# strings, confirmed against swebench_docker/constants.py.
TESTS_CONFIG = ">>>>> Tests config"
ANY_TESTS_PASSED = ">>>>> Any Test Passed"
ANY_TESTS_FAILED = ">>>>> No Tests Passed"


def _has_any_pass_marker(config_block: str) -> bool:
    return ANY_TESTS_PASSED in config_block or ANY_TESTS_FAILED in config_block


def _setting_name(config_block: str) -> str:
    """First token of a real "Tests config" block, e.g. "full" -- the
    same real setting name swebench_utils.py's own parser extracts via
    `config.split("\n")[0].split()[0]`.
    """
    first_line = config_block.split("\n", 1)[0]
    tokens = first_line.split()
    return tokens[0] if tokens else ""


def _inject_any_pass_marker(original_content: str, backfill_content: str) -> str:
    """Returns original_content with each of its real "Tests config"
    blocks getting the matching backfill block's any_tests_passed
    marker injected, if the original block doesn't already have one and
    the backfill block does.

    Matches blocks positionally (Nth "Tests config" block in original
    <-> Nth in backfill), then confirms the real setting names actually
    match before injecting -- same block *count* doesn't guarantee the
    same *order*: run_evaluation.py iterates a task instance's real
    predictions dict (`for setting in task_instance[KEY_PREDICTIONS]`),
    so block order follows dict-insertion order in each run's own
    predictions JSON, which two separately-built files aren't
    guaranteed to preserve even for the same real settings. Confirmed
    real via review: same count + different order would otherwise
    silently inject the wrong setting's marker.
    """
    original_blocks = original_content.split(TESTS_CONFIG)
    backfill_blocks = backfill_content.split(TESTS_CONFIG)

    if len(original_blocks) != len(backfill_blocks):
        # Real, genuine mismatch (e.g. one run crashed partway through
        # a setting the other completed) -- don't guess at alignment,
        # return the original unchanged and let the caller report this
        # instance as unmerged rather than silently injecting into the
        # wrong block.
        return original_content

    merged_blocks = [original_blocks[0]]  # preamble before the first marker
    for orig_block, backfill_block in zip(original_blocks[1:], backfill_blocks[1:]):
        if _setting_name(orig_block) != _setting_name(backfill_block):
            # Same block count, different setting order -- same real
            # risk as a count mismatch. Return the original unchanged;
            # the caller reports this instance as unmerged rather than
            # injecting into a block for the wrong setting.
            return original_content
        if not _has_any_pass_marker(orig_block) and _has_any_pass_marker(backfill_block):
            marker = ANY_TESTS_PASSED if ANY_TESTS_PASSED in backfill_block else ANY_TESTS_FAILED
            orig_block = orig_block.rstrip("\n") + f"\n{marker}\n"
        merged_blocks.append(orig_block)

    return TESTS_CONFIG.join(merged_blocks)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--original-log-dir", required=True,
        help="Directory of the original evaluation's real .eval.log files (has mutation/coverage, no any_tests_passed).",
    )
    parser.add_argument(
        "--backfill-log-dir", required=True,
        help="Directory of the SKIP_MUTATION=1 backfill run's real .eval.log files (has any_tests_passed, no mutation).",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to write merged logs into.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    original_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(args.original_log_dir, "*.eval.log"))
    }
    backfill_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(args.backfill_log_dir, "*.eval.log"))
    }

    if not original_files:
        sys.exit(f"No .eval.log files found in {args.original_log_dir}")
    if not backfill_files:
        sys.exit(f"No .eval.log files found in {args.backfill_log_dir}")

    merged_count = 0
    original_only = []
    backfill_only = []
    mismatched_blocks = []

    for filename, original_path in original_files.items():
        if filename not in backfill_files:
            original_only.append(filename)
            continue
        with open(original_path) as f:
            original_content = f.read()
        with open(backfill_files[filename]) as f:
            backfill_content = f.read()

        merged_content = _inject_any_pass_marker(original_content, backfill_content)
        if merged_content == original_content and TESTS_CONFIG in backfill_content:
            # Real signal a mismatch path (block count OR setting
            # order/name) returned unchanged content -- worth
            # surfacing, not silently treating as "already had the
            # marker". Re-derive both mismatch conditions rather than
            # having _inject_any_pass_marker report its own reason, to
            # keep that function's return type a plain str.
            orig_blocks = original_content.split(TESTS_CONFIG)
            back_blocks = backfill_content.split(TESTS_CONFIG)
            count_mismatch = len(orig_blocks) != len(back_blocks)
            setting_mismatch = not count_mismatch and any(
                _setting_name(o) != _setting_name(b)
                for o, b in zip(orig_blocks[1:], back_blocks[1:])
            )
            if count_mismatch or setting_mismatch:
                mismatched_blocks.append(filename)

        with open(os.path.join(args.out_dir, filename), "w") as f:
            f.write(merged_content)
        merged_count += 1

    for filename in backfill_files:
        if filename not in original_files:
            backfill_only.append(filename)

    print(f"{merged_count} real logs merged -> {args.out_dir}")
    if original_only:
        print(
            f"  {len(original_only)} file(s) only in the original dir "
            f"(no backfill counterpart, copied as-is with no any_tests_passed): "
            f"{', '.join(original_only[:5])}{', ...' if len(original_only) > 5 else ''}"
        )
        for filename in original_only:
            with open(original_files[filename]) as f:
                content = f.read()
            with open(os.path.join(args.out_dir, filename), "w") as f:
                f.write(content)
    if backfill_only:
        print(
            f"  {len(backfill_only)} file(s) only in the backfill dir "
            f"(no original counterpart, skipped, no real mutation/coverage data to merge with): "
            f"{', '.join(backfill_only[:5])}{', ...' if len(backfill_only) > 5 else ''}"
        )
    if mismatched_blocks:
        print(
            f"  {len(mismatched_blocks)} file(s) had a real config-block count mismatch "
            f"between original and backfill (one run likely crashed partway through a "
            f"setting the other completed) -- written unmerged (original content only), "
            f"no any_tests_passed injected: {', '.join(mismatched_blocks[:5])}"
            f"{', ...' if len(mismatched_blocks) > 5 else ''}"
        )


if __name__ == "__main__":
    main()
