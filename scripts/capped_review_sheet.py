"""Review sheet for the capped-completion audit's categories (fyp-paper#5).

capped_completion_audit.py sorts each capped sample into loop, enumeration,
progressing or unclear by heuristic. Before those categories inform a
decision, a person checks a sample of them: this writes a seeded sample from
every (arm, category) cell, with each completion's tail and blank fields for
the reviewer's label and reason.

    python3 scripts/capped_review_sheet.py --audit-csv audit.csv \\
        --per-cell 5 --seed 20261114 --out review \\
        instruct:<predictions.jsonl> kg_only:<predictions.jsonl>

writes review.md (one section per sample: instance, category, where the full
output is, length, sampling seed, the last 60 lines) and review.csv (the same
samples, with empty `label` and `reason` columns to fill in). When the tail is
not enough to tell new tests from repetition, print the whole completion:

    python3 scripts/capped_review_sheet.py --show instruct:<predictions.jsonl> <id> <sample>
"""

import argparse
import csv
import json
import random
import sys

TAIL = 60
CATEGORIES = ("progressing", "enumeration", "loop", "unclear")


def load_predictions(path):
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def seed_of(row, sample, setting="full"):
    s = (row.get("sampling") or {}).get(setting)
    if isinstance(s, dict) and s.get("seed") is not None:
        return str(s["seed"])
    return "none recorded (no per-request seed)"


def build(audit_csv, runs, per_cell, seed):
    audit = list(csv.DictReader(open(audit_csv)))
    preds = {arm: (path, load_predictions(path)) for arm, path in runs}
    rng = random.Random(seed)
    picked = []
    for arm in sorted(preds):
        for cat in CATEGORIES:
            cell = sorted((r for r in audit if r["arm"] == arm and r["category"] == cat),
                          key=lambda r: (r["id"], int(r["sample"])))
            picked.extend(rng.sample(cell, min(per_cell, len(cell))))
    items = []
    for r in picked:
        path, rows = preds[r["arm"]]
        row = rows.get(r["id"])
        if row is None:
            continue
        s = int(r["sample"])
        text = (row["preds"]["full"][s] or "")
        items.append({"arm": r["arm"], "id": r["id"], "sample": s, "category": r["category"],
                      "file": path, "chars": len(text), "completion_tokens": r.get("completion_tokens", ""),
                      "seed": seed_of(row, s), "tail": "\n".join(text.splitlines()[-TAIL:])})
    return items


def write(items, out):
    with open(out + ".csv", "w", newline="") as f:
        cols = ["arm", "id", "sample", "category", "file", "chars", "completion_tokens", "seed",
                "label", "reason"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({**{c: it.get(c, "") for c in cols}, "label": "", "reason": ""} for it in items)
    with open(out + ".md", "w") as f:
        f.write("# Capped completions: category review\n\n"
                "Label each sample in the CSV as `new tests`, `repetition/enumeration`, `loop` or "
                "`unclear`, with a short reason. Use `--show` for the full completion when the tail "
                "is not enough.\n\n")
        for n, it in enumerate(items, 1):
            f.write(f"## {n}. {it['arm']} / {it['category']}: {it['id']} sample {it['sample']}\n\n"
                    f"- full output: `{it['file']}`, id `{it['id']}`, sample {it['sample']}\n"
                    f"- length: {it['chars']} characters, {it['completion_tokens']} completion tokens"
                    f" (the request's total)\n- sampling seed: {it['seed']}\n\n"
                    f"Last {TAIL} lines:\n\n```python\n{it['tail']}\n```\n\n")


def show(spec, instance, sample):
    _, path = spec.split(":", 1)
    row = load_predictions(path)[instance]
    print(row["preds"]["full"][int(sample)])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--show"]:
        show(*argv[1:4])
        return None
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="ARM:PREDICTIONS.jsonl")
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--out", required=True, help="output stem: writes <stem>.md and <stem>.csv")
    args = ap.parse_args(argv)
    items = build(args.audit_csv, [s.split(":", 1) for s in args.runs], args.per_cell, args.seed)
    write(items, args.out)
    print(f"{len(items)} samples -> {args.out}.md, {args.out}.csv")
    return items


if __name__ == "__main__":
    main()
