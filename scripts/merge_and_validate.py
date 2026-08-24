# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Merges a sharded inference run's output files into one, then validates
the result. Automates the manual cat + wc -l + JSON-validity check that
was done by hand throughout this project's M3 runs -- see
results/RUN_LOG.md for the real bugs this caught (a permission conflict
between two accounts writing the same file, a stale pre-fix kg_prompts.json
run mistaken for a current one).

Only reads the shard files and writes a new merged file; never deletes or
modifies the shards themselves, so it's safe to re-run.
"""

import argparse
import glob
import json
import os
import sys


def find_shards(output_dir: str, model_nickname: str, dataset: str, temperature, num_shards: int):
    pattern = os.path.join(
        output_dir,
        f"{model_nickname}__{dataset}__{temperature}__test__shard-*__num_shards-{num_shards}.jsonl",
    )
    shards = sorted(glob.glob(pattern))
    return shards


def merge(shards: list[str], merged_path: str) -> int:
    """Concatenates shard files in order. Returns the number of lines
    written. Plain concatenation is safe here since each line is a
    self-contained JSON object -- no header/footer to worry about (same
    approach documented in docs/GUIDE.md's sharded-inference section).
    """
    line_count = 0
    with open(merged_path, "w") as out:
        for shard_path in shards:
            with open(shard_path) as shard_file:
                for line in shard_file:
                    out.write(line)
                    line_count += 1
    return line_count


def validate(merged_path: str) -> tuple[int, int]:
    """Returns (total_lines, unique_ids). A mismatch between the two
    means either a corrupted/interleaved write (concurrent writes under
    MAX_CONCURRENCY without proper locking) or a genuine duplicate
    instance across shards -- both worth a human looking at, not
    something this script silently papers over.
    """
    ids = set()
    total = 0
    with open(merged_path) as f:
        for line in f:
            total += 1
            data = json.loads(line)  # raises on any malformed line
            ids.add(data["id"])
    return total, len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory containing the shard files, e.g. results/instruct",
    )
    parser.add_argument(
        "--model_nickname", type=str, required=True,
        help="Model name as it appears in the shard filenames (e.g. Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument("--dataset", type=str, default="testgeneval")
    parser.add_argument("--temperature", type=str, default="0")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument(
        "--expected_total", type=int, default=None,
        help="If given, warn (not fail) when the merged line count differs. "
             "Context-overflow losses are expected for some models, see "
             "results/RUN_LOG.md, so this is informational, not a hard check.",
    )
    args = parser.parse_args()

    shards = find_shards(
        args.output_dir, args.model_nickname, args.dataset, args.temperature, args.num_shards
    )
    if len(shards) != args.num_shards:
        print(
            f"Found {len(shards)}/{args.num_shards} shard files, "
            f"not merging until all shards are present. Missing shards "
            f"usually mean a job is still running or was never submitted.",
            file=sys.stderr,
        )
        sys.exit(1)

    merged_path = os.path.join(
        args.output_dir,
        f"{args.model_nickname}__{args.dataset}__{args.temperature}__test.jsonl",
    )
    print(f"Merging {len(shards)} shards into {merged_path}")
    for s in shards:
        print(f"  {s}")

    merge(shards, merged_path)
    total, unique = validate(merged_path)

    print(f"\nMerged file: {total} lines, {unique} unique ids")
    if total != unique:
        print(
            f"WARNING: {total - unique} duplicate id(s) found -- either "
            f"corrupted/interleaved concurrent writes or a genuine repeat "
            f"across shards. Investigate before trusting this file.",
            file=sys.stderr,
        )
    if args.expected_total is not None and total != args.expected_total:
        print(
            f"NOTE: expected {args.expected_total} but got {total} "
            f"({args.expected_total - total} missing). Check for context-"
            f"overflow losses in the shard jobs' own slurm-*.out logs "
            f"(grep -c 'Failed, skipping') before assuming this is wrong.",
            file=sys.stderr,
        )

    print(f"\nWrote {merged_path}")


if __name__ == "__main__":
    main()
