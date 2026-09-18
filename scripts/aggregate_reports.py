# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Aggregate per-model/arm `*_summary.json` files (produced by
generate_report.py) across everyone's own M3 scratch space into one real
comparison table, for RQ3's kg_only-vs-instruct results.

Real gap this fills: #56's team handoff asks each person to comment their
own final numbers on their own sub-issue (#57/#58/#59/#62), but nothing
pulls those into one place -- so a real cross-model comparison currently
means manually copying numbers out of several different GitHub comments.
generate_report.py's summary.json (get_model_eval_summary in
swebench_docker/swebench_utils.py) doesn't embed which model or arm it's
for, only whatever metric keys the real run produced (full_pass_at_k,
full_av_coverage/full_av_function_coverage, full_av_mutation_score/
full_av_function_mutation_score, total_predictions, ...), so this script
needs an external --labels mapping to know which file is which, same
reasoning as aggregate_token_counts.py's --labels for job_id.

Usage:
    python scripts/aggregate_reports.py \
        --reports "/fs04/scratch2/al49/*/reports/*/*_summary.json" \
        --labels report_labels.csv \
        --out results/rq3_summary.csv

--labels is a CSV mapping file,model,arm, e.g.:
    file,model,arm
    /fs04/scratch2/al49/wlee0060/reports/gptoss20b_instruct/gpt-oss-20b_t=0.8_summary.json,gpt-oss-20b,instruct
    /fs04/scratch2/al49/wlee0060/reports/gptoss20b_kgonly/gpt-oss-20b_t=0.8_summary.json,gpt-oss-20b,kg_only
`file` must match a matched --reports path exactly (absolute path, as
glob resolves it). Without --labels, output still includes every real
metric per file, just unlabeled by model/arm.
"""

import argparse
import csv
import glob
import json
import sys


def load_labels(path):
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["file"]] = (row["model"], row["arm"])
    return labels


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reports",
        required=True,
        help="Glob pattern matching *_summary.json files, e.g. "
        "'/fs04/scratch2/al49/*/reports/*/*_summary.json'.",
    )
    parser.add_argument(
        "--labels", help="Optional CSV with columns file,model,arm to label each row."
    )
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.reports))
    if not files:
        sys.exit(f"No files matched: {args.reports}")

    labels = load_labels(args.labels) if args.labels else {}

    rows = []
    for path in files:
        with open(path) as f:
            summary = json.load(f)

        model, arm = labels.get(path, ("", ""))
        row = {"file": path, "model": model, "arm": arm}
        row.update(summary)
        rows.append(row)

    # Real summary.json files can have slightly different metric keys
    # (e.g. a repo-scoped run vs. a full-dataset run) -- union the keys
    # across every row rather than assuming they all match, so a CSV
    # write doesn't silently drop a real metric some rows have and
    # others don't.
    fieldnames = ["file", "model", "arm"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} report(s) -> {args.out}")
    unlabeled = [r["file"] for r in rows if not r["model"]]
    if unlabeled:
        print(f"  {len(unlabeled)} unlabeled (no --labels match):")
        for f in unlabeled:
            print(f"    {f}")


if __name__ == "__main__":
    main()
