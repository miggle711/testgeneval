"""Find samples evaluated after a mutation timeout in the same instance (#145).

Before #146, a timed-out `cosmic-ray exec` could leave the code file mutated
(python images) or keep mutating it (conda images) while the instance's later
samples ran, so those samples' tests, coverage and mutation may have run
against altered code. This reads .eval.log files and lists every sample that
ran after a `MutationTimeout` earlier in the same log.

For each such sample it also checks the mutant count against the instance's
first sample: `MutationNum` (whole-file runs, cr-report's total jobs) or
`FunctionMutationTotal` (function-scope runs). The count comes from parsing
the code file, so a different count is direct evidence that the sample saw
altered code. The same count does not show the sample was unaffected: many
mutations leave the number of mutation sites unchanged.

    python3 scripts/mutation_timeout_exposure.py --out exposure.csv <log dir>...

Prints a summary per directory (one directory is usually one model and arm)
and writes one CSV row per exposed sample. Read-only.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

CONFIG = ">>>>> Tests config"
OUTCOMES = (
    (">>>>> All Tests Passed", "passed"),
    (">>>>> Some Tests Failed", "failed"),
    (">>>>> Tests Timed Out", "tests_timed_out"),
    (">>>>> Tests Errored", "errored"),
)
FIELDS = ("MutationNum", "FunctionMutationTotal", "MutationLOG", "FunctionMutationLOG",
          "CoverageLOG", "FunctionCoverageLOG")


def _value(block, field):
    m = re.search(r"\n" + field + r": ([\d.]+)", block)
    return float(m.group(1)) if m else None


def parse_log(text):
    """One dict per sample block, in order."""
    samples = []
    for i, block in enumerate(text.split(CONFIG)[1:]):
        outcome = next((name for marker, name in OUTCOMES if marker in block), "")
        row = {"sample": i, "outcome": outcome, "mutation_timeout": "MutationTimeout" in block}
        row.update({f: _value(block, f) for f in FIELDS})
        samples.append(row)
    return samples


def count_of(sample):
    """The sample's mutant count: whole-file if logged, else function scope."""
    if sample["MutationNum"] is not None:
        return ("MutationNum", sample["MutationNum"])
    if sample["FunctionMutationTotal"] is not None:
        return ("FunctionMutationTotal", sample["FunctionMutationTotal"])
    return (None, None)


def exposed_samples(samples):
    """Samples after the first one that logged a MutationTimeout, each with
    whether its mutant count differs from the first sample that logged one."""
    first_timeout = next((s["sample"] for s in samples if s["mutation_timeout"]), None)
    if first_timeout is None:
        return []
    reference = next((count_of(s) for s in samples if count_of(s)[0] is not None), (None, None))
    out = []
    for s in samples[first_timeout + 1:]:
        kind, n = count_of(s)
        changed = None
        if kind is not None and kind == reference[0]:
            changed = n != reference[1]
        out.append({**s, "after_timeout_in_sample": first_timeout, "count_kind": kind or "",
                    "count": n, "reference_count": reference[1] if kind == reference[0] else None,
                    "count_changed": changed})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True, help="CSV, one row per exposed sample")
    args = ap.parse_args(argv)
    rows = []
    summary = defaultdict(lambda: defaultdict(int))
    for root in args.dirs:
        for d, _, files in os.walk(root):
            for f in sorted(files):
                if not f.endswith(".eval.log"):
                    continue
                with open(os.path.join(d, f), errors="replace") as fh:
                    samples = parse_log(fh.read())
                if not samples:
                    continue
                s = summary[d]
                s["logs"] += 1
                s["samples"] += len(samples)
                s["logs_with_a_timeout"] += any(x["mutation_timeout"] for x in samples)
                for e in exposed_samples(samples):
                    s["exposed_samples"] += 1
                    s["exposed_with_a_count"] += e["count"] is not None
                    s["exposed_count_changed"] += bool(e["count_changed"])
                    s["exposed_passed"] += e["outcome"] == "passed"
                    rows.append({"dir": d, "log": f, "instance": f.split(".")[0], **e})
    cols = ["dir", "log", "instance", "sample", "after_timeout_in_sample", "outcome", "mutation_timeout",
            "count_kind", "count", "reference_count", "count_changed", *FIELDS]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    keys = ["logs", "samples", "logs_with_a_timeout", "exposed_samples", "exposed_passed",
            "exposed_with_a_count", "exposed_count_changed"]
    print("\t".join(["dir"] + keys))
    total = defaultdict(int)
    for d in sorted(summary):
        print("\t".join([d] + [str(summary[d][k]) for k in keys]))
        for k in keys:
            total[k] += summary[d][k]
    print("\t".join(["TOTAL"] + [str(total[k]) for k in keys]))
    print(f"{len(rows)} exposed samples -> {args.out}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
