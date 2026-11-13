# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Usable target-function mutation score from eval logs (fyp-paper#7, section 4).

Scores every sample of every instance, including the ones that never reached
mutation testing, so a lower pass rate counts against an arm:

  score     the filtered suite passed when run together and target-function
            mutation finished: the sample's FunctionMutationLOG
  zero      no test passed alone, the filtered suite failed when run together,
            or it hung past the test time limit: 0
  unknown   mutation was cut short or killed, the harness errored, or the
            sample is missing from the log: not a result (see variants below)

An instance counts only if its target function has at least one mutant. Old
logs can show that only when some sample in either arm reached mutation, so
instances where none did are listed apart ("undetermined").

Usage (one RUN:ARM:DIR per run; ARM is instruct or kg_only; DIR is searched
for *.full.eval.log files, including in sub-directories):
    python3 scripts/usable_score.py --out usable_score.csv --k 5 \
        Scout:instruct:<eval_logs>/<scout instruct run> \
        Scout:kg_only:<eval_logs>/<scout kg_only run>

Re-run the analysis from the CSV, anywhere:
    python3 scripts/usable_score.py --from-csv usable_score.csv --k 5

The intervals are bootstrap intervals over instances with a fixed seed. They
are not corrected for the five ways of handling unknown samples shown side by
side. This is a description of logs, not the confirmatory analysis.
"""

import argparse
import csv
import glob
import os
import random
import re
import statistics
from collections import Counter, defaultdict

CONFIG = ">>>>> Tests config"
PASSED = ">>>>> All Tests Passed"
FAILED = ">>>>> Some Tests Failed"
ERRORED = ">>>>> Tests Errored"
TIMED_OUT = ">>>>> Tests Timed Out"
UNFILTERED_PASSED = ">>>>> Unfiltered Tests Passed"
UNFILTERED_FAILED = ">>>>> Unfiltered Tests Failed"
STARTED = "Running mutation testing"
NO_RANGE = "FunctionMutationFAIL: no target function range"

ARMS = ("instruct", "kg_only")
FIELDS = ["run", "arm", "instance", "repo", "sample", "outcome", "passed",
          "unfiltered_passed", "func_score", "func_done", "func_total"]


def num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def classify(block):
    """(outcome, func_score, func_done, func_total) for one sample's block:
    everything the harness logged from its "full" config line to the next."""
    passed = PASSED in block
    ended = UNFILTERED_PASSED in block or UNFILTERED_FAILED in block
    score = num(r"FunctionMutationLOG: ([\d.]+)%", block)
    done = num(r"FunctionMutationNum: (\d+)", block)
    total = num(r"FunctionMutationTotal: (\d+)", block)
    out = lambda name: (name, score, done, total)

    if not passed:
        # No test passed alone: the harness logs a zero time and the failure
        # markers without running anything (each write ends " \n", and may
        # start with a prefix).
        if re.search(r"TestsTime: 0\.0 *\n[^\n]*" + re.escape(FAILED), block):
            return out("zero_no_test_passed")
        if TIMED_OUT in block:
            return out("zero_test_timeout")
        if ERRORED in block:
            return out("unknown_harness_error")
        if FAILED in block or ended:
            return out("zero_suite_failed")
        return out("unknown_cut_off")

    if STARTED not in block:
        return out("unknown_no_mutation" if ended else "unknown_cut_off")
    if NO_RANGE in block:
        return out("target_unresolved")
    if "FunctionMutationFAIL" in block:
        return out("unknown_function_score_failed")
    if total is not None:  # logs written with testgeneval#102
        if total == 0:
            return out("no_target_mutants")
        if score is not None and done == total:
            return out("score")
        return out("unknown_mutation_incomplete")
    if "MutationTimeout" in block:
        return out("unknown_mutation_incomplete")
    if score is not None:
        return out("score")
    if num(r"\nMutationLOG: ([\d.]+)%", block) is not None:
        return out("no_target_mutants")
    if "\nMutationFAIL" in block:
        return out("unknown_mutation_failed")
    return out("unknown_mutation_killed")


def read_log(path):
    with open(path, errors="replace") as f:
        content = f.read()
    return [b for b in content.split(CONFIG)[1:]
            if b.split("\n", 1)[0].strip().startswith("full")]


def collect(specs, k):
    rows, notes = [], []
    for spec in specs:
        run, arm, root = spec.split(":", 2)
        if arm not in ARMS:
            raise SystemExit(f"{spec}: arm must be one of {ARMS}")
        by_instance = {}
        paths = sorted(glob.glob(os.path.join(root, "**", "*.full.eval.log"), recursive=True))
        for path in paths:
            instance = os.path.basename(path).partition(".")[0]
            try:
                blocks = read_log(path)
            except OSError:
                continue
            # A repeated log for one instance: keep the fuller one.
            if instance not in by_instance or len(blocks) > len(by_instance[instance]):
                by_instance[instance] = blocks
        notes.append(f"{run} {arm}: {len(paths)} logs, {len(by_instance)} instances ({root})")
        for instance, blocks in by_instance.items():
            base = {"run": run, "arm": arm, "instance": instance,
                    "repo": instance.rsplit("-", 2)[0]}
            for i, block in enumerate(blocks[:k]):
                outcome, score, done, total = classify(block)
                rows.append({**base, "sample": i, "outcome": outcome,
                             "passed": int(PASSED in block),
                             "unfiltered_passed": int(UNFILTERED_PASSED in block),
                             "func_score": score, "func_done": done, "func_total": total})
            for i in range(len(blocks), k):
                rows.append({**base, "sample": i, "outcome": "unknown_missing", "passed": 0,
                             "unfiltered_passed": 0, "func_score": None,
                             "func_done": None, "func_total": None})
    return rows, notes


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in ("func_score", "func_done", "func_total"):
            r[c] = float(r[c]) if r[c] not in ("", None) else None
        for c in ("sample", "passed", "unfiltered_passed"):
            r[c] = int(r[c])
    return rows


def boot_ci(values, n=10000, seed=0):
    """95% interval of the mean, resampling instances."""
    rnd = random.Random(seed)
    size = len(values)
    means = sorted(sum(rnd.choices(values, k=size)) / size for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def flag_short_runs(rows):
    """An old log has no FunctionMutationTotal, so a sample that stopped
    early without a timeout marker looks finished. Take the most target
    mutants any sample of the instance finished as the total."""
    most = defaultdict(float)
    for r in rows:
        if r["func_done"]:
            key = (r["run"], r["instance"])
            most[key] = max(most[key], r["func_total"] or r["func_done"])
    flagged = 0
    for r in rows:
        if (r["outcome"] == "score" and r["func_total"] is None
                and r["func_done"] is not None
                and r["func_done"] < most[(r["run"], r["instance"])]):
            r["outcome"] = "unknown_mutation_incomplete"
            flagged += 1
    return flagged


def value(r, variant):
    """A sample's usable score under a variant, or None to leave it out."""
    o = r["outcome"]
    if o == "score":
        return r["func_score"]
    if o.startswith("zero"):
        return 0.0
    if variant == "unknown_as_zero":
        return 0.0
    if variant == "unknown_as_partial" and r["func_score"] is not None:
        return r["func_score"]
    return None


def paired(per_instance, variant, k, strict=False):
    """Per-instance (instruct, kg_only) mean scores for instances usable in
    both arms."""
    pairs = []
    for arms in per_instance.values():
        means = {}
        for arm in ARMS:
            vals = [v for v in (value(r, variant) for r in arms.get(arm, [])) if v is not None]
            if not vals or (strict and len(vals) < k):
                break
            means[arm] = statistics.mean(vals)
        else:
            pairs.append((means["instruct"], means["kg_only"]))
    return pairs


def show(label, pairs):
    if len(pairs) < 2:
        print(f"   {label:46} too few instances ({len(pairs)})")
        return
    diffs = [kg - ins for ins, kg in pairs]
    lo, hi = boot_ci(diffs)
    print(f"   {label:46} {len(pairs):5}  {statistics.mean(p[0] for p in pairs):8.1f} "
          f"{statistics.mean(p[1] for p in pairs):8.1f}  {statistics.mean(diffs):+6.1f} "
          f"[{lo:+.1f}, {hi:+.1f}]")


def analyse(rows, k):
    flagged = flag_short_runs(rows)
    print(f"\nSamples re-marked incomplete (fewer target mutants than another sample "
          f"of the same instance): {flagged}")
    for run in sorted({r["run"] for r in rows}):
        sel = [r for r in rows if r["run"] == run]
        print(f"\n######## {run}")

        print("\n1. Sample outcomes (share of the arm's samples)")
        counts = {arm: Counter(r["outcome"] for r in sel if r["arm"] == arm) for arm in ARMS}
        print(f"   {'outcome':32} {'instruct':>9} {'kg_only':>9}")
        for o in sorted(set(counts["instruct"]) | set(counts["kg_only"])):
            cells = [f"{counts[a][o] / max(1, sum(counts[a].values())):9.1%}" for a in ARMS]
            print(f"   {o:32} {cells[0]} {cells[1]}")
        print(f"   {'samples':32} {sum(counts['instruct'].values()):9} "
              f"{sum(counts['kg_only'].values()):9}")
        for arm in ARMS:
            rs = [r for r in sel if r["arm"] == arm and r["outcome"] != "unknown_missing"]
            if rs:
                print(f"   {arm}: filtered suite passed {sum(r['passed'] for r in rs) / len(rs):.1%}, "
                      f"every test passed {sum(r['unfiltered_passed'] for r in rs) / len(rs):.1%} "
                      f"(of {len(rs)} logged samples)")

        per = defaultdict(lambda: defaultdict(list))
        for r in sel:
            per[r["instance"]][r["arm"]].append(r)
        both = {i: a for i, a in per.items() if all(arm in a for arm in ARMS)}
        status = {}
        for inst, arms in both.items():
            rs = [r for arm in ARMS for r in arms[arm]]
            if any((r["func_total"] or r["func_done"] or 0) > 0 for r in rs):
                status[inst] = "eligible"
            elif any(r["outcome"] == "target_unresolved" for r in rs):
                status[inst] = "target unresolved"
            elif any(r["outcome"] == "no_target_mutants" for r in rs):
                status[inst] = "no target mutants"
            else:
                status[inst] = "undetermined (no sample reached mutation)"
        print(f"\n2. Instances: {len(per)} in either arm, {len(both)} in both")
        for s, n in Counter(status.values()).most_common():
            print(f"   {n:5}  {s}")

        eligible = {i: a for i, a in both.items() if status[i] == "eligible"}
        with_undetermined = {i: a for i, a in both.items()
                             if status[i] in ("eligible", "undetermined (no sample reached mutation)")}
        print("\n3. Usable target-function mutation, kg_only minus instruct "
              "(95% interval, by instance)")
        print(f"   {'how unknown samples are handled':46} {'inst.':>5}  {'instruct':>8} "
              f"{'kg_only':>8}  {'diff':>6}")
        show("left out (mean of the known samples)", paired(eligible, "known", k))
        show(f"instances with all {k} samples known, both arms",
             paired(eligible, "known", k, strict=True))
        show("counted as 0", paired(eligible, "unknown_as_zero", k))
        show("partial score where there is one", paired(eligible, "unknown_as_partial", k))
        show("left out, plus undetermined instances",
             paired(with_undetermined, "known", k))

        cond = []
        for arms in eligible.values():
            m = {}
            for arm in ARMS:
                vals = [r["func_score"] for r in arms[arm] if r["outcome"] == "score"]
                if vals:
                    m[arm] = statistics.mean(vals)
            if len(m) == 2:
                cond.append((m["instruct"], m["kg_only"]))
        print("\n4. For comparison: passing, finished samples only (the paper's conditional score)")
        show("instances both arms pass", cond)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="RUN:ARM:DIR")
    ap.add_argument("--out", help="write the per-sample CSV here")
    ap.add_argument("--from-csv", help="analyse a CSV written earlier instead of reading logs")
    ap.add_argument("--k", type=int, default=5, help="samples per instance (default 5)")
    args = ap.parse_args()
    if args.from_csv:
        rows = load(args.from_csv)
    else:
        if not args.runs or not args.out:
            ap.error("give RUN:ARM:DIR runs and --out, or --from-csv")
        rows, notes = collect(args.runs, args.k)
        print("\n".join(notes))
        if not rows:
            raise SystemExit("No full-setting samples found.")
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.out} ({len(rows)} samples)")
    analyse(rows, args.k)


if __name__ == "__main__":
    main()
