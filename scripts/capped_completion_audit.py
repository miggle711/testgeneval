"""Audit completions that stopped at the output cap, per arm (#47, fyp-paper#5).

For a model whose completions often hit max_tokens (Qwen3-4B at 8,000), the
question is whether the capped ones are repetition loops, where more tokens
would not help, or long test files still being written, where they might.
This reads prediction files and classifies every sample by the same rules in
each arm. It assigns no score: a capped sample can still contain usable
tests, and evaluation decides its outcome.

Per sample (the row's usage entry gives finish_reason):

  capped     finish_reason == "length"
  stopped    any other finish_reason

For capped samples:

  loop         the tail repeats: among the last 60 non-blank lines, at most
               a quarter are distinct, or one block of 1-8 lines repeats at
               least 4 times in a row at the end
  progressing  not a loop, and the longest parseable prefix holds at least
               one complete test function (def test_...)
  unclear      neither: more tokens might or might not help. These are the
               cases for a small fixed 8k against 16k comparison

Also counted: complete tests in the longest parseable prefix (capped
samples), and stopped samples whose text does not parse (#47's separate
finding, which a larger cap would not change).

Usage (standard library only):

    python3 scripts/capped_completion_audit.py --out audit.csv \\
        instruct:<results>/instruct/Qwen3-4B-Instruct-2507__testgeneval__0.8__k1__test.jsonl \\
        kg_only:<results>/kg_only/Qwen3-4B-Instruct-2507__testgeneval__0.8__k1__test.jsonl
"""

import argparse
import ast
import csv
import json
import re
import statistics
from collections import Counter

TAIL_LINES = 60
DISTINCT_SHARE = 0.25
MAX_BLOCK = 8
MIN_REPEATS = 4


def is_loop(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    tail = lines[-TAIL_LINES:]
    if len(tail) >= 20 and len(set(tail)) <= DISTINCT_SHARE * len(tail):
        return True
    for size in range(1, MAX_BLOCK + 1):
        block = lines[-size:]
        if len(block) < size:
            break
        repeats, end = 1, len(lines) - size
        while end - size >= 0 and lines[end - size:end] == block:
            repeats += 1
            end -= size
        if repeats >= MIN_REPEATS:
            return True
    return False


def complete_tests(text):
    """Test functions (module level or in a class) in the longest prefix of
    whole lines that parses. Returns (count, prefix parses at full length)."""
    lines = text.splitlines()
    for cut in range(len(lines), 0, -1):
        try:
            tree = ast.parse("\n".join(lines[:cut]))
        except (SyntaxError, ValueError):
            continue
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                count += 1
        return count, cut == len(lines)
    return 0, False


def parses(text):
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        return False


def classify(text, finish_reason):
    if finish_reason != "length":
        return {"capped": False, "category": "stopped", "parses": parses(text)}
    if is_loop(text):
        category = "loop"
    else:
        tests, _ = complete_tests(text)
        category = "progressing" if tests else "unclear"
    tests, whole = complete_tests(text)
    return {"capped": True, "category": category, "parses": whole, "complete_tests": tests}


def audit_file(arm, path, setting="full"):
    rows, misaligned = [], 0
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        preds = (r.get("preds") or {}).get(setting) or []
        usage = (r.get("usage") or {}).get(setting) or []
        if len(usage) != len(preds):
            misaligned += 1
            continue
        for i, (text, u) in enumerate(zip(preds, usage)):
            c = classify(text or "", (u or {}).get("finish_reason"))
            rows.append({"arm": arm, "id": r["id"], "sample": i,
                         "finish_reason": (u or {}).get("finish_reason"),
                         "completion_tokens": (u or {}).get("completion_tokens"),
                         "chars": len(text or ""), **c})
    return rows, misaligned


def summarize(rows):
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        capped = [r for r in rs if r["capped"]]
        stopped = [r for r in rs if not r["capped"]]
        cats = Counter(r["category"] for r in capped)
        tests = [r["complete_tests"] for r in capped]
        out[arm] = {
            "samples": len(rs),
            "capped": len(capped),
            "capped_share": round(len(capped) / len(rs), 4) if rs else None,
            "capped_by_category": dict(cats),
            "capped_complete_tests_median": statistics.median(tests) if tests else None,
            "capped_with_no_complete_test": sum(t == 0 for t in tests),
            "stopped": len(stopped),
            "stopped_not_parsing": sum(not r["parses"] for r in stopped),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="ARM:PREDICTIONS.jsonl")
    ap.add_argument("--setting", default="full")
    ap.add_argument("--out", required=True, help="per-sample CSV")
    args = ap.parse_args(argv)
    rows, misaligned = [], {}
    for spec in args.runs:
        arm, path = spec.split(":", 1)
        r, m = audit_file(arm, path, args.setting)
        rows.extend(r)
        misaligned[arm] = m
    with open(args.out, "w", newline="") as f:
        cols = ["arm", "id", "sample", "finish_reason", "completion_tokens", "chars",
                "capped", "category", "parses", "complete_tests"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    summary = summarize(rows)
    for arm, m in misaligned.items():
        summary.setdefault(arm, {})["rows_skipped_usage_not_per_sample"] = m
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
