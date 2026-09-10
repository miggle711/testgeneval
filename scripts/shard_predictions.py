# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Split a predictions .jsonl into N shards for parallel evaluation
across several m3_run_evaluation.slurm jobs.

Each shard is a valid predictions file on its own. run_evaluation.py's
own --skip_existing plus the per-instance .eval.log naming means shards
writing to the same --log_dir do not collide (one file per instance).

Sharding is round-robin by line, not by repo, so each shard gets a
roughly even mix of fast (requests, flask) and slow (sympy, matplotlib)
instances rather than one shard drawing all the slow ones.

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
    # contains a half-written line.
    lines = []
    with open(args.predictions) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"line {n} of {args.predictions} is not valid JSON: {e}")
            lines.append(line)

    if not lines:
        sys.exit(f"no rows in {args.predictions}")

    shards = [[] for _ in range(args.num_shards)]
    for i, line in enumerate(lines):
        shards[i % args.num_shards].append(line)

    for i, shard_lines in enumerate(shards):
        path = os.path.join(args.out_dir, f"{args.prefix}-{i}.jsonl")
        with open(path, "w") as f:
            for line in shard_lines:
                f.write(line + "\n")
        print(f"  {path}: {len(shard_lines)} rows")

    print(f"{len(lines)} rows -> {args.num_shards} shards in {args.out_dir}")


if __name__ == "__main__":
    main()
