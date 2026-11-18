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

An instance counts only if its target function has at least one mutant
(fyp-paper#7, eligibility rule 3). The harness counts the target's mutants
once per instance, before any sample runs, and logs `TargetMutantCount` ahead
of the first sample. Where either arm's log has that line, it alone decides:
the count comes from the patched source and the mutation configuration, which
are the same in both arms, so one arm's count is enough. A count that failed,
or that differs between the arms, leaves the instance undetermined, whatever
its samples logged. The samples' FunctionMutationTotal is only compared with
the count, and every difference is reported.

Logs written before the count existed can show a mutant only when some
sample in either arm reached mutation, so instances where none did are listed
apart ("undetermined").

Usage (one RUN:ARM:DIR per run; ARM is instruct or kg_only; DIR is searched
for *.full.eval.log files, including in sub-directories):
    python3 scripts/usable_score.py --out usable_score.csv --k 5 \
        Scout:instruct:<eval_logs>/<scout instruct run> \
        Scout:kg_only:<eval_logs>/<scout kg_only run>

Re-run the analysis from the CSV, anywhere:
    python3 scripts/usable_score.py --from-csv usable_score.csv --k 5

Instances generation skipped (testgeneval#134 writes them, with a reason, to
<predictions stem>.skipped.tsv) are read with --skipped RUN:ARM:FILE, once per
file. An instance whose prompt the server refused as too long in either arm
(reason prompt_too_long) is excluded for that run in both arms (fyp-paper#7,
rule 4) and counted, unless that arm has a log for it after all (a resubmit
generated it). Without --skipped, such an instance has a log in one arm only
and is left out of the paired analysis without being counted.

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
# The per-instance count, written before the first sample. Each log line is
# "<prefix> <message> \n" and these messages start with a newline, so they
# begin a line of their own.
COUNT = re.compile(r"^TargetMutantCount: (\d+)\s*$", re.M)
COUNT_FAIL = re.compile(r"^TargetMutantCountFAIL: (.*)$", re.M)
COUNT_NO_RANGE = "no target function range"

NO_SAMPLE_REACHED = "undetermined (no sample reached mutation)"
COUNT_FAILED = "undetermined (the count failed)"
COUNTS_DIFFER = "undetermined (the arms' counts differ)"
UNDETERMINED = (NO_SAMPLE_REACHED, COUNT_FAILED, COUNTS_DIFFER)

ARMS = ("instruct", "kg_only")
FIELDS = ["run", "arm", "instance", "repo", "sample", "outcome", "passed",
          "unfiltered_passed", "func_score", "func_done", "func_total",
          "count_outcome", "target_count"]


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


def read_count(head):
    """(count_outcome, target_count) from the part of a log before its first
    sample: ("counted", n), ("unresolved", None) when the target function
    could not be found in the patch, ("failed", None) when the count could
    not be taken, or ("", None) for a log with no count line."""
    counts = {float(n) for n in COUNT.findall(head)}
    fails = [reason.strip() for reason in COUNT_FAIL.findall(head)]
    if len(counts) > 1 or (counts and fails):
        return "failed", None
    if counts:
        return "counted", counts.pop()
    if fails:
        unresolved = all(reason.startswith(COUNT_NO_RANGE) for reason in fails)
        return ("unresolved" if unresolved else "failed"), None
    return "", None


def read_log(path):
    """The log's "full" sample blocks, and its per-instance count."""
    with open(path, errors="replace") as f:
        parts = f.read().split(CONFIG)
    blocks = [b for b in parts[1:] if b.split("\n", 1)[0].strip().startswith("full")]
    return blocks, read_count(parts[0])


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
                blocks, count = read_log(path)
            except OSError:
                continue
            # A repeated log for one instance: keep the fuller one.
            if instance not in by_instance or len(blocks) > len(by_instance[instance][0]):
                by_instance[instance] = (blocks, count)
        notes.append(f"{run} {arm}: {len(paths)} logs, {len(by_instance)} instances ({root})")
        for instance, (blocks, count) in by_instance.items():
            base = {"run": run, "arm": arm, "instance": instance,
                    "repo": instance.rsplit("-", 2)[0],
                    "count_outcome": count[0], "target_count": count[1]}
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
        # A CSV written before the count existed has no count columns.
        r["count_outcome"] = r.get("count_outcome") or ""
        for c in ("func_score", "func_done", "func_total", "target_count"):
            r[c] = float(r[c]) if r.get(c) not in ("", None) else None
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


def eligibility(rs):
    """Whether an instance counts, from its samples' rows in both arms."""
    lines = {(r["count_outcome"], r["target_count"]) for r in rs if r["count_outcome"]}
    if lines:
        counts = {n for outcome, n in lines if outcome == "counted"}
        if len(counts) > 1:
            return COUNTS_DIFFER
        if counts:
            return "eligible" if counts.pop() > 0 else "no target mutants"
        if any(outcome == "unresolved" for outcome, _ in lines):
            return "target unresolved"
        return COUNT_FAILED
    # No count line in either arm's log: all there is to go on is the samples.
    if any((r["func_total"] or r["func_done"] or 0) > 0 for r in rs):
        return "eligible"
    if any(r["outcome"] == "target_unresolved" for r in rs):
        return "target unresolved"
    if any(r["outcome"] == "no_target_mutants" for r in rs):
        return "no target mutants"
    return NO_SAMPLE_REACHED


def names(instances, limit=10):
    listed = ", ".join(sorted(instances)[:limit])
    return listed + (f", and {len(instances) - limit} more" if len(instances) > limit else "")


def count_checks(both, status):
    """Report lines comparing the per-instance count with what the samples
    logged. Empty when no log has a count line."""
    counted = {i for i, arms in both.items()
               if any(r["count_outcome"] for arm in ARMS for r in arms[arm])}
    if not counted:
        return []
    lines = [f"   Decided by the per-instance count: {len(counted)}; "
             f"by the samples (no count line in either arm's log): {len(both) - len(counted)}"]
    wrong = [(i, r) for i in counted for arm in ARMS for r in both[i][arm]
             if r["count_outcome"] == "counted" and r["func_total"] is not None
             and r["func_total"] != r["target_count"]]
    lines.append(f"   Samples whose FunctionMutationTotal differs from their log's count: "
                 f"{len(wrong)}" + (f", in {names({i for i, _ in wrong})}" if wrong else ""))
    apart = {i for i in counted
             if len({(r["count_outcome"], r["target_count"])
                     for arm in ARMS for r in both[i][arm] if r["count_outcome"]}) > 1}
    lines.append(f"   Instances whose two arms log a different count line: {len(apart)}"
                 + (f": {names(apart)}" if apart else ""))
    failed = {i for i in counted if status[i] == COUNT_FAILED}
    if failed:
        with_total = {i for i in failed
                      if any(r["func_total"] for arm in ARMS for r in both[i][arm])}
        lines.append(f"   Instances whose count failed: {len(failed)}, of which "
                     f"{len(with_total)} have a sample that logs a total above 0: {names(failed)}")
    return lines


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


TOO_LONG = "prompt_too_long"
RULE4 = "excluded: prompt too long for the model (rule 4)"


def read_skipped(specs):
    """{(run, arm): {id: reason}} from RUN:ARM:FILE specs, the last row per id."""
    out = defaultdict(dict)
    for spec in specs:
        run, arm, path = spec.split(":", 2)
        if arm not in ARMS:
            raise SystemExit(f"{spec}: arm must be one of {ARMS}")
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                out[(run, arm)][r["id"]] = r["reason"]
    return out


def rule4_excluded(run, per, skipped):
    """Instances of a run excluded under rule 4: refused as too long in an arm
    that has no log for them."""
    out = set()
    for arm in ARMS:
        for i, reason in skipped.get((run, arm), {}).items():
            if reason == TOO_LONG and arm not in per.get(i, {}):
                out.add(i)
    return out


def analyse(rows, k, skipped=None):
    skipped = skipped or {}
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
        too_long = rule4_excluded(run, per, skipped)
        both = {i: a for i, a in per.items()
                if all(arm in a for arm in ARMS) and i not in too_long}
        status = {inst: eligibility([r for arm in ARMS for r in arms[arm]])
                  for inst, arms in both.items()}
        print(f"\n2. Instances: {len(per | dict.fromkeys(too_long))} in either arm or skipped "
              f"for being too long, {len(both)} in both arms and not excluded")
        if too_long:
            print(f"   {len(too_long):5}  {RULE4}")
        for s, n in Counter(status.values()).most_common():
            print(f"   {n:5}  {s}")
        for arm in ARMS:
            failed = {i for i, why in skipped.get((run, arm), {}).items()
                      if why != TOO_LONG and arm not in per.get(i, {})}
            if failed:
                print(f"   {arm}: {len(failed)} instance(s) without a prediction for another "
                      f"reason (request_failed; unknown, not excluded): {names(failed)}")
        for line in count_checks(both, status):
            print(line)

        eligible = {i: a for i, a in both.items() if status[i] == "eligible"}
        with_undetermined = {i: a for i, a in both.items()
                             if status[i] == "eligible" or status[i] in UNDETERMINED}
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
    ap.add_argument("--skipped", action="append", default=[], metavar="RUN:ARM:FILE",
                    help="a <predictions stem>.skipped.tsv from generation; repeat per file")
    args = ap.parse_args()
    skipped = read_skipped(args.skipped)
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
    analyse(rows, args.k, skipped)


if __name__ == "__main__":
    main()
