# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Split a predictions .jsonl into N shards for parallel evaluation
across several m3_run_evaluation.slurm jobs.

Each shard is a valid predictions file on its own. run_evaluation.py's
own --skip_existing plus the per-instance .eval.log naming means shards
writing to the same --log_dir do not collide (one file per instance).

Sharding is round-robin within each repo's own group of instances, not
plain round-robin by line. A real 100-instance/4-shard dry run
(2026-09-13) found plain round-robin (i % num_shards straight down the
file) let one shard draw 7 of the sample's sympy instances against
another's 3, even though the source file was already randomly
shuffled -- repo membership correlates strongly with real per-instance
runtime (mutation testing scales with file size: real observed range
in that run was 82s to ~4.8h, with sympy/matplotlib instances
dominating the slow end), so an uneven repo split directly produced a
3.5x real wall-clock spread across the 4 shards (1h56m to 6h55m) for
an equal 25-instances-per-shard split. Grouping by repo first and
round-robining each group across shards guarantees no shard can draw a
disproportionate share of any one repo. This narrows the real variance
but does not eliminate it -- runtime still varies a lot between
individual instances of the same repo, not just between repos.

Usage:
    python scripts/shard_predictions.py \
        --predictions results/instruct/<model>__testgeneval__0.8__test.jsonl \
        --num-shards 4 \
        --out-dir /path/to/shards

    # then submit one job per shard:
    for i in 0 1 2 3; do
      PREDICTIONS_PATH=/path/to/shards/shard-$i.jsonl \
        LOG_DIR=/scratch/.../eval_logs \
        sbatch m3_run_evaluation.slurm
    done
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="Predictions .jsonl to split.")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of shards.")
    parser.add_argument("--out-dir", required=True, help="Directory to write shard-<i>.jsonl files into.")
    parser.add_argument("--prefix", default="shard", help="Shard filename prefix (default: shard).")
    args = parser.parse_args()

    if args.num_shards < 1:
        sys.exit("--num-shards must be >= 1")
    if not os.path.exists(args.predictions):
        sys.exit(f"predictions file not found: {args.predictions}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Read all non-empty lines, validating each is JSON so a shard never
    # contains a half-written line. Group by repo (derived the same way
    # as elsewhere in this project: instance_id with its trailing
    # "-<pr_number>" stripped) so repo-correlated runtime doesn't let
    # one shard draw a disproportionate share of a slow repo's
    # instances (see the module docstring for the real, measured case
    # that motivated this).
    by_repo = defaultdict(list)
    total = 0
    with open(args.predictions) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"line {n} of {args.predictions} is not valid JSON: {e}")
            instance_id = rec.get("instance_id", "")
            repo = instance_id.rsplit("-", 1)[0] if "-" in instance_id else instance_id
            by_repo[repo].append(line)
            total += 1

    if total == 0:
        sys.exit(f"no rows in {args.predictions}")

    shards = [[] for _ in range(args.num_shards)]
    for repo_lines in by_repo.values():
        for i, line in enumerate(repo_lines):
            shards[i % args.num_shards].append(line)

    for i, shard_lines in enumerate(shards):
        path = os.path.join(args.out_dir, f"{args.prefix}-{i}.jsonl")
        with open(path, "w") as f:
            for line in shard_lines:
                f.write(line + "\n")
        print(f"  {path}: {len(shard_lines)} rows")

    print(f"{total} rows ({len(by_repo)} repos) -> {args.num_shards} shards in {args.out_dir}")


if __name__ == "__main__":
    main()
