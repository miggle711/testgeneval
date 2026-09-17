# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Aggregate per-job input/output token counts logged by run_api.py's
calc_cost() out of one or more m3_run_inference.slurm job logs, for RQ4's
kg_only-vs-instruct prompt token count comparison.

run_api.py logs one line per API call: "input_tokens=X, output_tokens=Y,
cost=Z" (inference/api/run_api.py:202-204). It doesn't log which
TestGenEval instance the call was for, and calls run concurrently under a
ThreadPoolExecutor, so log line order can't be used to recover a true
per-instance mapping. This script therefore reports per-job aggregates
(one row per slurm-*.out file: request count, total/mean input and output
tokens), not per-instance token counts. That's enough to compare kg_only
vs instruct prompt size directly, just not to join token counts back to
individual instance IDs.

Usage:
    python scripts/aggregate_token_counts.py \
        --logs "slurm-*.out" \
        --labels job_labels.csv \
        --out token_counts.csv

--labels is optional: a CSV mapping job_id,model,arm so the output can be
grouped/compared directly, e.g.:
    job_id,model,arm
    59749639,gpt-oss-20b,instruct
    59749640,gpt-oss-20b,kg_only
Without --labels, output still includes job_id per row, just unlabeled.
"""

import argparse
import csv
import glob
import re
import sys

TOKEN_LINE = re.compile(r"input_tokens=(\d+), output_tokens=(\d+), cost=([\d.]+)")
JOB_ID_FROM_FILENAME = re.compile(r"slurm-(?:eval-)?(\d+)\.out$")


def parse_log(path):
    input_tokens = []
    output_tokens = []
    with open(path, errors="replace") as f:
        for line in f:
            m = TOKEN_LINE.search(line)
            if m:
                input_tokens.append(int(m.group(1)))
                output_tokens.append(int(m.group(2)))
    return input_tokens, output_tokens


def load_labels(path):
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["job_id"]] = (row["model"], row["arm"])
    return labels


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--logs",
        required=True,
        help="Glob pattern matching slurm-*.out files, e.g. 'slurm-*.out' or '/path/to/logs/*.out'.",
    )
    parser.add_argument(
        "--labels", help="Optional CSV with columns job_id,model,arm to label each job's row."
    )
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.logs))
    if not files:
        sys.exit(f"No files matched: {args.logs}")

    labels = load_labels(args.labels) if args.labels else {}

    rows = []
    for path in files:
        m = JOB_ID_FROM_FILENAME.search(path)
        job_id = m.group(1) if m else path

        input_tokens, output_tokens = parse_log(path)
        if not input_tokens:
            continue  # no API calls logged in this file (e.g. an eval log, or a failed/empty job)

        model, arm = labels.get(job_id, ("", ""))
        n = len(input_tokens)
        rows.append(
            {
                "job_id": job_id,
                "model": model,
                "arm": arm,
                "file": path,
                "num_requests": n,
                "total_input_tokens": sum(input_tokens),
                "total_output_tokens": sum(output_tokens),
                "mean_input_tokens": round(sum(input_tokens) / n, 1),
                "mean_output_tokens": round(sum(output_tokens) / n, 1),
            }
        )

    if not rows:
        sys.exit("No input_tokens=/output_tokens= lines found in any matched file.")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} job(s) -> {args.out}")
    unlabeled = [r["job_id"] for r in rows if not r["model"]]
    if unlabeled:
        print(f"  {len(unlabeled)} unlabeled (no --labels match): {', '.join(unlabeled)}")


if __name__ == "__main__":
    main()
