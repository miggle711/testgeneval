"""The confirmatory analysis of the rerun (fyp-paper#7, sections 3 to 6).

Written and frozen before any confirmatory result; its md5 goes in the freeze
record (section 8). It reads the per-sample rows usable_score.py writes (one
CSV per model, from that model's two arms' evaluation logs) and does only what
the protocol fixes in advance. usable_score.py remains the descriptive tool;
the classification of each sample is its `classify`, reused here unchanged.

Inputs are checked before anything is computed: each CSV holds one run, and
exactly one row per sample index 0..k-1 for every instance and arm it has.
A sample missing from an evaluation log is not an absent row:
usable_score.py writes it as an unknown_missing row, so it is kept, scored
unknown and counted. An instance with no log at all in one arm is kept as
attrition too. A duplicate, extra or absent row can only come from a
malformed CSV (a different k, or files concatenated), and stops the run; the
development list matches the md5 recorded on #7; the breadth subset has no
development ids and its weights file covers every repository in it with the
subset's own sample sizes. Any failure stops the run.

Models have one of two roles, each its own family for the Holm adjustment:
main-study models (`--model`, run on every instance) and breadth models
(`--breadth-model`, run on the breadth subset only).

Per model, on instances present in both arms:
  1. Exclusions, counted by reason: rule 4 (prompt too long in either arm,
     from the skipped files), the target not resolved or with no mutants
     (rule 2/3, from the per-instance count), undetermined.
  2. Sample scores (section 4). A sample whose FunctionMutationTotal differs
     from its instance's TargetMutantCount is unknown (a different mutant set;
     testgeneval#133). Primary: an instance with any unknown sample in either
     arm is left out for that model; sensitivity: unknown scored 0; and a
     second sensitivity scoring a test-command timeout as unknown, not 0.
  3. Populations: the confirmatory set is every eligible instance outside the
     151 development instances (`--dev-ids`); all eligible instances are
     reported in addition, labelled as including the development set.
  4. Statistics (section 6), on per-instance differences d = kg_only minus
     instruct of the mean over the k samples:
       - Wilcoxon signed-rank, two-sided; zero differences dropped (Wilcoxon's
         method); tied |d| get average ranks. Up to 50 non-zero differences
         the p is exact: the sign-permutation distribution over the observed
         ranks, ties included. Above 50, the normal approximation with tie
         and continuity corrections. It tests whether the differences are
         symmetric about 0, not the mean gain;
       - matched-pairs rank-biserial r = (R+ - R-) / (R+ + R-) on the same ranks;
       - mean d (the estimate of the gain) with a 95% percentile bootstrap
         interval, resampling instances (10,000 resamples, seed 0), reported
         separately from the Wilcoxon p;
       - filtered pass@5 (it measures the generation-plus-filtering
         pipeline), per instance and arm: true if any sample is known to
         pass, false if all k are known not to pass, otherwise unknown. One
         known pass settles it, so an unknown sibling does not discard it.
         Instances known in both arms: exact two-sided McNemar on the
         discordant ones, and a paired bootstrap interval of the difference;
       - pass@1 (the mean over the k samples' pass indicators): complete
         case, since a mean with an unknown term is not determined; the
         difference of means with a bootstrap interval, descriptive;
       - Holm adjustment, alpha 0.05, within four families: {main-study,
         breadth} models x {primary, filtered pass@k}, on the confirmatory
         population only. Sensitivity scorings and the population including
         the development set are descriptive and not adjusted. The headline
         claim rests on the main-study primary family alone; the breadth
         families are secondary and are not another route to it.
  5. Across main-study models: each one's mean d on the instances eligible
     and usable for all of them (descriptive).
  6. Breadth subset (secondary; `--breadth-ids`, `--breadth-weights`): per model, the
     weighted mean of d over the subset's usable instances,
     sum(w d) / sum(w), with a stratified bootstrap resampling within each
     repository, the strata the subset was drawn in. A repository with one
     analysed instance contributes no resampling variance; how many there are
     is reported, as a limitation of the interval. Also the subset's
     attrition per id (analysed, unknown, excluded by reason, missing an
     arm's log, absent). The
     weights correct for sampling only, not for exclusions or unknown
     samples.

"Confirmatory" names the prospectively specified rerun: these instances are
outside the 151 development instances, but historical results on the whole
benchmark (the original submission's runs) were inspected before the
protocol was written, so they are not an untouched holdout. The script keeps
the identifier; its report states the definition.

Robustness analyses, all descriptive and never Holm-adjusted (section 7 of
the output; none is a route to the headline claim):
  a. Missing-data bounds, conditional on established eligibility (not
     benchmark-wide): every confirmatory instance whose eligibility the
     evaluations establish, including one with no log in an arm (all k
     samples unknown there). Expected instances come from a manifest
     (`--expected-ids` for main-study models, the breadth subset for breadth
     models), so an instance with no log in either arm is not invisible: its
     eligibility is unresolved and it is listed with the other unresolved
     ones. Known scores, and test-command
     timeouts scored 0, are kept; only unknown samples vary, over the
     score's range 0-100. Reported: the mean d with unknowns at 0 in
     kg_only and 100 in instruct (lower bound) and the reverse (upper), the
     mean d with both arms' unknowns at the same value, and for each assumed
     instruct value the kg_only value at which the estimated mean d crosses
     zero (a statement about the estimate, not about significance).
     Instances whose eligibility is unresolved are listed separately.
  b. Filtered pass@k bounds on the same population: each arm's rate with
     unresolved instances counted as failures and as passes, and the
     difference's lower and upper bounds. The McNemar test above covers only
     instances resolved in both arms, which success can select; its
     population is labelled so.
  c. Attrition per repository: each instance's fate, and every sample's
     outcome per arm. Test-command timeouts are listed on their own: scored
     0 in the primary analysis, so not attrition there.
  d. Per repository: eligible and analysed instances, the completion rate,
     and the mean d with its bootstrap interval. No p-values.
  e. Leave one repository out: the primary analysis without each repository
     in turn (unadjusted; shows whether one repository drives the result).
  f. Prompt length (`--predictions`, predictions written since
     testgeneval#154 record the server's prompt_tokens once per request): the
     rank correlation between the arms' prompt-token difference and d, and
     mean d by tertile of that difference, with the number of pairs and the
     instances missing a token count; the correlation is undefined when the
     difference is constant. An association only: it cannot separate useful
     structure from context selection, difficulty or repository effects.

Standard library only, so it runs in any environment; usable_score.py from the
same folder.

Usage:
    python3 scripts/confirmatory_analysis.py --k 5 --dev-ids pilot_dev_ids.txt \\
        --model Scout:scout_usable_score.csv --skipped Scout:instruct:<file> ... \\
        [--breadth-model Gemma:gemma_usable_score.csv ...] \\
        [--breadth-ids breadth_ids.txt --breadth-weights breadth_ids.weights.tsv] \\
        [--predictions Scout:instruct:<predictions.jsonl> ...] [--json results.json]
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usable_score as us  # noqa: E402

ARMS = us.ARMS
ALPHA = 0.05
N_BOOT = 10_000
SEED = 0
EXACT_MAX_N = 50
DEV_MD5 = "0093dd887fa0d3a221e847fe82af476c"   # pilot_dev_ids.txt, recorded on #7, 12 Nov
DEV_N = 151
ROLES = ("main", "breadth")
SCORE_MAX = 100.0          # FunctionMutationLOG is a percentage
GRID = (0.0, 25.0, 50.0, 75.0, 100.0)
CONFIRMATORY_LABEL = ("confirmatory: prospectively specified rerun, excluding the 151 development "
                      "instances; historical whole-benchmark results were inspected before the "
                      "protocol, so not an untouched holdout")


# ---------------------------------------------------------------- statistics

def rank_abs(values):
    """Average ranks of |values| (1-based), ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def wilcoxon(d):
    """(n non-zero, W+, W-, two-sided p, rank-biserial r) for differences d."""
    nz = [x for x in d if x != 0]
    n = len(nz)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "p": 1.0, "r": 0.0, "method": "none"}
    ranks = rank_abs(nz)
    w_plus = sum(r for r, x in zip(ranks, nz) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, nz) if x < 0)
    total = w_plus + w_minus
    r_rb = (w_plus - w_minus) / total
    ties = len(set(abs(x) for x in nz)) < n
    if n <= EXACT_MAX_N:
        # Sign-permutation distribution of W+ given the observed ranks. Average
        # ranks are multiples of 1/2, so work in doubled ranks (integers) and
        # count, for every attainable sum, how many of the 2^n sign
        # assignments give it.
        doubled = [int(round(2 * r)) for r in ranks]
        counts = [1] + [0] * sum(doubled)
        for r2 in doubled:
            for t in range(len(counts) - 1, r2 - 1, -1):
                counts[t] += counts[t - r2]
        centre = sum(doubled)            # twice the null mean, doubled: 2 * (sum/2)
        obs = abs(2 * int(round(2 * w_plus)) - centre)
        extreme = sum(c for t, c in enumerate(counts) if abs(2 * t - centre) >= obs)
        p = min(1.0, extreme / 2 ** n)
        method = "exact_with_ties" if ties else "exact"
    else:
        mean = n * (n + 1) / 4
        tie_counts = Counter(abs(x) for x in nz).values()
        var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in tie_counts) / 48
        z = (abs(w_plus - mean) - 0.5) / math.sqrt(var) if var > 0 else 0.0
        p = min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2)))
        method = "normal"
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "p": p, "r": r_rb, "method": method}


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p from discordant counts b (kg_only only) and
    c (instruct only): the binomial test of b against n = b + c at 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm-adjusted p-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def boot_mean_ci(values, n=N_BOOT, seed=SEED):
    if len(values) < 2:
        return None
    lo, hi = us.boot_ci(values, n=n, seed=seed)
    return [lo, hi]


def weighted_mean(pairs):
    """pairs: (weight, value)."""
    w = sum(p[0] for p in pairs)
    return sum(p[0] * p[1] for p in pairs) / w if w else None


def stratified_boot_ci(items, n=N_BOOT, seed=SEED):
    """items: (stratum, weight, value). Percentile interval of the weighted
    mean, resampling within each stratum as drawn. A stratum with a single
    item is resampled as itself and adds no variance."""
    by = defaultdict(list)
    for stratum, w, v in items:
        by[stratum].append((w, v))
    if len(items) < 2:
        return None
    rnd = random.Random(seed)
    groups = [by[k] for k in sorted(by)]
    stats = []
    for _ in range(n):
        sample = []
        for group in groups:
            sample.extend(rnd.choices(group, k=len(group)))
        stats.append(weighted_mean(sample))
    stats.sort()
    return [stats[int(0.025 * n)], stats[int(0.975 * n) - 1]]


# ------------------------------------------------------------ sample scoring

def apply_count_rule(rows):
    """A sample whose total differs from its instance's count is unknown
    (fyp-paper#7 section 4; testgeneval#133). Returns how many changed."""
    changed = 0
    for r in rows:
        if (r["func_total"] is not None and r["target_count"] is not None
                and r["count_outcome"] == "counted" and r["func_total"] != r["target_count"]
                and not r["outcome"].startswith("unknown")):
            r["outcome"] = "unknown_count_mismatch"
            changed += 1
    return changed


def sample_value(r, variant):
    """primary: as usable_score's `known`; timeout_unknown: a test-command
    timeout is unknown instead of 0; unknown_as_zero: every unknown is 0."""
    if variant == "timeout_unknown" and r["outcome"] == "zero_test_timeout":
        return None
    return us.value(r, "unknown_as_zero" if variant == "unknown_as_zero" else "known")


def instance_means(arms, variant, k):
    """{arm: mean over k samples} if every sample in both arms has a value."""
    out = {}
    for arm in ARMS:
        rs = sorted(arms.get(arm, []), key=lambda r: r["sample"])
        vals = [sample_value(r, variant) for r in rs]
        if len(vals) < k or any(v is None for v in vals[:k]):
            return None
        out[arm] = statistics.mean(vals[:k])
    return out


def test_known(r):
    return r["passed"] == 1 or r["outcome"].startswith("zero")


def filtered_pass(rs):
    """1 if any sample is known to pass, 0 if every one is known not to,
    None (unknown) otherwise."""
    if any(r["passed"] == 1 for r in rs):
        return 1
    return 0 if all(test_known(r) for r in rs) else None


def pass_share(rs):
    """Share of samples passing, if every sample's test result is known."""
    return statistics.mean(r["passed"] for r in rs) if all(test_known(r) for r in rs) else None


# ----------------------------------------------------------------- validation

def repo_of(instance_id):
    """django__django-12091-15824 -> django/django (the weights file's form)."""
    return instance_id.rsplit("-", 2)[0].replace("__", "/", 1)


def check_rows(rows, k, source):
    """One run per CSV, and exactly one row per sample index 0..k-1 for every
    instance and arm present. Returns the run name the CSV holds."""
    runs = {r["run"] for r in rows}
    if len(runs) != 1:
        raise SystemExit(f"{source}: expected one run, found {sorted(runs)}")
    problems = []
    by = defaultdict(list)
    for r in rows:
        if r["arm"] not in ARMS:
            problems.append(f"{r['instance']}: unknown arm {r['arm']!r}")
        by[(r["instance"], r["arm"])].append(r["sample"])
    for (i, arm), samples in sorted(by.items()):
        if sorted(samples) != list(range(k)):
            seen = Counter(samples)
            dup = sorted(x for x, n in seen.items() if n > 1)
            extra = sorted(x for x in seen if not 0 <= x < k)
            absent = sorted(set(range(k)) - set(seen))
            problems.append(f"{i} {arm}: duplicate {dup}, outside 0..{k - 1} {extra}, absent {absent}")
    if problems:
        raise SystemExit(f"{source}: malformed CSV, {len(problems)} problem(s) (a sample missing from a log "
                         f"is an unknown_missing row, never an absent one), e.g.\n  " + "\n  ".join(problems[:10]))
    return runs.pop()


def read_dev_ids(path, expected_md5=DEV_MD5):
    with open(path, "rb") as f:
        data = f.read()
    got = hashlib.md5(data).hexdigest()
    if got != expected_md5:
        raise SystemExit(f"{path}: md5 {got}, not the development set recorded on #7 ({expected_md5})")
    ids = {l.strip() for l in data.decode().splitlines() if l.strip()}
    if len(ids) != DEV_N:
        raise SystemExit(f"{path}: {len(ids)} ids, expected {DEV_N}")
    return ids


def read_breadth(ids_path, weights_path, dev_ids):
    """The subset and its per-repository weights, checked against each other."""
    with open(ids_path) as f:
        subset = [l.strip() for l in f if l.strip()]
    if len(set(subset)) != len(subset):
        raise SystemExit(f"{ids_path}: duplicate ids")
    overlap = set(subset) & dev_ids
    if overlap:
        raise SystemExit(f"{ids_path}: {len(overlap)} development ids, e.g. {sorted(overlap)[:3]}")
    weights, sizes = {}, {}
    with open(weights_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["repo", "population", "sample", "weight"]:
            raise SystemExit(f"{weights_path}: unexpected header {header}")
        for n, line in enumerate(f, 2):
            parts = line.rstrip("\n").split("\t")
            try:
                repo, pop, k, w = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
            except (IndexError, ValueError):
                raise SystemExit(f"{weights_path}:{n}: cannot parse {line!r}")
            if (repo in weights or not 0 < k <= pop or not math.isfinite(w) or w <= 0
                    or abs(w - pop / k) > 1e-5 * max(1.0, w)):
                raise SystemExit(f"{weights_path}:{n}: invalid or repeated row {line!r}")
            weights[repo], sizes[repo] = w, k
    drawn = Counter(repo_of(i) for i in subset)
    if dict(drawn) != sizes:
        bad = sorted(r for r in set(drawn) | set(sizes) if drawn.get(r) != sizes.get(r))
        raise SystemExit(f"breadth ids and weights disagree on sample sizes for {bad}")
    return subset, weights


# ------------------------------------------------------------------- analysis

def analyse_model(model, rows, skipped, dev_ids, k, role="main", expected=None):
    changed = apply_count_rule(rows)
    flagged = us.flag_short_runs(rows)
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["instance"]][r["arm"]].append(r)
    too_long = us.rule4_excluded(model, per, skipped)
    both = {i: a for i, a in per.items() if all(arm in a for arm in ARMS) and i not in too_long}
    one_arm = {i: next(arm for arm in ARMS if arm not in a)
               for i, a in per.items() if i not in both and i not in too_long}
    missing_samples = sum(r["outcome"] == "unknown_missing" for r in rows)
    status = {i: us.eligibility([r for arm in ARMS for r in a[arm]]) for i, a in both.items()}
    eligible = {i: a for i, a in both.items() if status[i] == "eligible"}

    result = {"model": model, "role": role, "k": k,
              "samples_count_mismatch_unknown": changed, "samples_flagged_incomplete": flagged,
              "samples_missing_from_log": missing_samples,
              "missing_one_arm_log": {arm: sorted(i for i, a in one_arm.items() if a == arm) for arm in ARMS},
              "exclusions": {"rule4_prompt_too_long": len(too_long),
                             **{s: n for s, n in Counter(status.values()).items() if s != "eligible"}},
              "eligible": len(eligible), "populations": {}}
    diffs = {}
    for pop_name, ids in (("confirmatory", [i for i in eligible if i not in dev_ids]),
                          ("all_including_development", list(eligible))):
        pop = {}
        for variant in ("primary", "unknown_as_zero", "timeout_unknown"):
            d, unknown = {}, 0
            for i in ids:
                m = instance_means(eligible[i], variant, k)
                if m is None:
                    unknown += 1
                    continue
                d[i] = m["kg_only"] - m["instruct"]
            vals = [d[i] for i in sorted(d)]
            w = wilcoxon(vals)
            pop[variant] = {"instances": len(vals), "dropped_unknown": unknown,
                            "mean_d": statistics.mean(vals) if vals else None,
                            "ci95_mean_d": boot_mean_ci(vals), "wilcoxon": w}
            if variant == "primary":
                diffs[pop_name] = d
        # filtered pass@k (any known pass) and pass@1 (complete case)
        b = c = both_pass = neither = unknown5 = unknown1 = 0
        p5 = {arm: [] for arm in ARMS}
        p1_d = []
        for i in ids:
            x, y = (filtered_pass(eligible[i][arm]) for arm in ARMS)
            if x is None or y is None:
                unknown5 += 1
            else:
                p5["instruct"].append(x); p5["kg_only"].append(y)
                b += (y == 1 and x == 0); c += (x == 1 and y == 0)
                both_pass += (x == 1 and y == 1); neither += (x == 0 and y == 0)
            s1, s2 = (pass_share(eligible[i][arm]) for arm in ARMS)
            if s1 is None or s2 is None:
                unknown1 += 1
            else:
                p1_d.append(s2 - s1)
        n5 = len(p5["instruct"])
        diff5 = [y - x for x, y in zip(p5["instruct"], p5["kg_only"])]
        pop["filtered_pass_at_k"] = {
            "instances": n5, "dropped_unknown": unknown5,
            "instruct": statistics.mean(p5["instruct"]) if n5 else None,
            "kg_only": statistics.mean(p5["kg_only"]) if n5 else None,
            "discordant": {"kg_only_only": b, "instruct_only": c}, "concordant": {"both": both_pass, "neither": neither},
            "mcnemar_p": mcnemar_exact(b, c), "ci95_difference": boot_mean_ci(diff5)}
        pop["pass_at_1_mean_d"] = {"instances": len(p1_d), "dropped_unknown": unknown1, "mean_d": statistics.mean(p1_d) if p1_d else None,
                                   "ci95": boot_mean_ci(p1_d)}
        result["populations"][pop_name] = pop
    fate = {i: "missing: no log in either arm" for i in (expected or ()) if i not in per}
    fate.update({i: f"missing: no {arm} log" for i, arm in one_arm.items()})
    fate.update({i: "excluded: rule4_prompt_too_long" for i in too_long})
    fate.update({i: f"excluded: {s}" for i, s in status.items() if s != "eligible"})
    fate.update({i: ("analysed" if i in diffs["all_including_development"] else "unknown")
                 for i in eligible})
    conf_pass = result["populations"]["confirmatory"]["filtered_pass_at_k"]
    conf_pass["population"] = "instances with filtered pass@k resolved in both arms"
    pop, unresolved = bounds_population(per, both, status, too_long, dev_ids, expected)
    conf_eligible = [i for i in eligible if i not in dev_ids]
    result["robustness"] = {
        "bounds_population": {
            "definition": ("conditional on established eligibility: confirmatory instances whose eligibility the "
                           "evaluations establish; not benchmark-wide"),
            "manifest": None if expected is None else len(expected),
            "instances": len(pop), "eligibility_unresolved": dict(sorted(unresolved.items())),
            # outside the bounds altogether: their number and where they fall
            "eligibility_unresolved_by_repo": dict(sorted(Counter(repo_of(i) for i in unresolved).items()))},
        "mean_d_bounds": mean_d_bounds(pop, k),
        "filtered_pass_at_k_bounds": pass_bounds(pop),
        "attrition_by_repo": attrition_by_repo(per, fate, dev_ids, expected),
        "by_repo": by_repo(diffs["confirmatory"], conf_eligible),
        "leave_one_repo_out": leave_one_repo_out(diffs["confirmatory"]),
    }
    return result, diffs["confirmatory"], fate


# ---------------------------------------------------------------- robustness

def arm_known(rs, k):
    """(sum of known primary scores, number of unknown samples) for an arm's
    rows; an arm with no log has k unknown samples."""
    vals = [us.value(r, "known") for r in rs]
    return sum(v for v in vals if v is not None), (k if not rs else sum(v is None for v in vals))


NO_LOG = "undetermined (no log in either arm)"


def bounds_population(per, both, status, too_long, dev_ids, expected=None):
    """Confirmatory instances whose eligibility the evaluations establish (both
    arms, or the one arm with a log), and those whose eligibility is
    unresolved, including expected instances with no log in either arm."""
    pop, unresolved = {}, {}
    for i in sorted(set(expected or ()) - set(per) - set(too_long) - set(dev_ids)):
        unresolved[i] = NO_LOG
    for i, arms in per.items():
        if i in dev_ids or i in too_long:
            continue
        st = status[i] if i in both else us.eligibility([r for a in ARMS for r in arms.get(a, [])])
        if st == "eligible":
            pop[i] = arms
        elif st.startswith("undetermined"):
            unresolved[i] = st
    return pop, unresolved


def mean_d_bounds(pop, k):
    """Mean d over a fixed population as a function of the value assumed for
    unknown samples in each arm: base + (U_kg a_kg - U_in a_in) / (k N)."""
    n = len(pop)
    if not n:
        return None
    base = u_kg = u_in = 0.0
    with_unknown = 0
    for arms in pop.values():
        s_kg, n_kg = arm_known(arms.get("kg_only", []), k)
        s_in, n_in = arm_known(arms.get("instruct", []), k)
        base += (s_kg - s_in) / k
        u_kg += n_kg; u_in += n_in
        with_unknown += (n_kg + n_in) > 0
    base /= n
    D = lambda a_kg, a_in: base + (u_kg * a_kg - u_in * a_in) / (k * n)
    tipping = []
    for a_in in GRID:
        if u_kg == 0:
            a = None
            note = "no unknown kg_only samples: mean d is " + ("positive" if D(0, a_in) > 0 else "not positive")
        else:
            a = (u_in * a_in - k * n * base) / u_kg
            if a < 0:
                note, a = "mean d stays positive for every kg_only value in 0-100", None
            elif a > SCORE_MAX:
                note, a = "mean d stays at or below zero for every kg_only value in 0-100", None
            else:
                note = "mean d is positive when unknown kg_only samples score above this"
        tipping.append({"instruct_unknowns_at": a_in, "kg_only_value_where_mean_d_is_zero": a, "note": note})
    return {"instances": n, "instances_with_unknown_samples": with_unknown,
            "unknown_samples": {"kg_only": int(u_kg), "instruct": int(u_in)},
            "lower_bound": D(0.0, SCORE_MAX), "upper_bound": D(SCORE_MAX, 0.0),
            "same_value_in_both_arms": {str(a): D(a, a) for a in GRID},
            "tipping": tipping}


def pass_bounds(pop):
    """Filtered pass@k rates over a fixed population, unresolved counted as
    failures (low) and passes (high), and the difference's bounds."""
    n = len(pop)
    if not n:
        return None
    status = {arm: [filtered_pass(arms[arm]) if arms.get(arm) else None for arms in pop.values()]
              for arm in ARMS}
    rate = {arm: {"low": sum(x == 1 for x in v) / n, "high": sum(x != 0 for x in v) / n,
                  "unresolved": sum(x is None for x in v)} for arm, v in status.items()}
    return {"instances": n,
            "resolved_in_both_arms": sum(x is not None and y is not None
                                         for x, y in zip(status["instruct"], status["kg_only"])),
            "rate": rate,
            "difference_low": rate["kg_only"]["low"] - rate["instruct"]["high"],
            "difference_high": rate["kg_only"]["high"] - rate["instruct"]["low"]}


def attrition_by_repo(per, fate, dev_ids, expected=None):
    """Per repository: each confirmatory instance's fate, and its samples'
    outcomes per arm (test-command timeouts are their own outcome)."""
    out = defaultdict(lambda: {"instances": Counter(), "samples": {arm: Counter() for arm in ARMS}})
    for i in sorted(set(per) | set(expected or ())):
        if i in dev_ids:
            continue
        arms = per.get(i, {})
        repo = repo_of(i)
        out[repo]["instances"][fate.get(i, "absent")] += 1
        for arm in ARMS:
            for r in arms.get(arm, []):
                out[repo]["samples"][arm][r["outcome"]] += 1
    return {repo: {"instances": dict(v["instances"]),
                   "samples": {arm: dict(c) for arm, c in v["samples"].items()}}
            for repo, v in sorted(out.items())}


def by_repo(d, eligible_ids):
    """Per repository, descriptive: eligible, analysed, completion, mean d."""
    groups = defaultdict(list)
    for i in sorted(d):
        groups[repo_of(i)].append(d[i])
    elig = Counter(repo_of(i) for i in eligible_ids)
    return {repo: {"eligible": elig[repo], "analysed": len(groups[repo]),
                   "completion": len(groups[repo]) / elig[repo],
                   "mean_d": statistics.mean(groups[repo]) if groups[repo] else None,
                   "ci95_mean_d": boot_mean_ci(groups[repo])}
            for repo in sorted(elig)}


def leave_one_repo_out(d):
    out = {}
    for repo in sorted({repo_of(i) for i in d}):
        vals = [d[i] for i in sorted(d) if repo_of(i) != repo]
        w = wilcoxon(vals)
        out[repo] = {"instances": len(vals), "mean_d": statistics.mean(vals) if vals else None,
                     "ci95_mean_d": boot_mean_ci(vals), "wilcoxon_p_unadjusted": w["p"], "rank_biserial_r": w["r"]}
    return out


def spearman(x, y):
    rx, ry = rank_plain(x), rank_plain(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def rank_plain(values):
    """Average ranks (1-based) of the values themselves, ties sharing."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def length_diagnostic(d, tokens):
    """tokens: {arm: {id: prompt_tokens}}. Association of the kg_only minus
    instruct prompt-token difference with d; descriptive only."""
    ids = [i for i in sorted(d) if all(tokens.get(arm, {}).get(i) is not None for arm in ARMS)]
    missing = {arm: sum(tokens.get(arm, {}).get(i) is None for i in d) for arm in ARMS}
    if len(ids) < 3:
        return {"pairs": len(ids), "analysed_instances": len(d), "missing_token_counts": missing,
                "note": "too few instances with prompt_tokens in both arms "
                "(predictions before testgeneval#154 do not record them)"}
    diff = [tokens["kg_only"][i] - tokens["instruct"][i] for i in ids]
    vals = [d[i] for i in ids]
    # Tertiles cut at values, so tied differences always share a group (a
    # cut by position would split ties arbitrarily); empty groups are dropped.
    srt = sorted(diff)
    cuts = (srt[(len(srt) - 1) // 3], srt[2 * (len(srt) - 1) // 3])
    groups = defaultdict(list)
    for t, x in enumerate(diff):
        groups[0 if x <= cuts[0] else 1 if x <= cuts[1] else 2].append(t)
    rho = spearman(diff, vals)
    return {"pairs": len(ids), "analysed_instances": len(d), "missing_token_counts": missing,
            "mean_prompt_tokens": {arm: statistics.mean(tokens[arm][i] for i in ids) for arm in ARMS},
            "spearman_token_difference_vs_d": rho if rho is not None else "undefined (constant token difference or d)",
            "tertiles": [{"token_difference_range": [min(diff[t] for t in g), max(diff[t] for t in g)],
                          "instances": len(g), "mean_d": statistics.mean(vals[t] for t in g)}
                         for _, g in sorted(groups.items())],
            "note": ("association only: cannot separate useful structure from context selection, "
                     "difficulty or repository effects")}


def read_prompt_tokens(specs):
    """{(model, arm): {id: prompt_tokens}} from NAME:ARM:predictions.jsonl."""
    out = {}
    for spec in specs:
        model, arm, path = spec.split(":", 2)
        if arm not in ARMS:
            raise SystemExit(f"{spec}: arm must be one of {ARMS}")
        got = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in got:
                    raise SystemExit(f"{path}: id {row['id']} appears twice")
                got[row["id"]] = (row.get("sampling") or {}).get("full", {}).get("prompt_tokens")
        out[(model, arm)] = got
    return out


def holm_across(results):
    """Holm within each family: role x outcome, confirmatory population."""
    for role in ROLES:
        members = [r for r in results if r["role"] == role]
        for outcome, getter in (("primary", lambda p: p["primary"]["wilcoxon"]["p"]),
                                ("filtered_pass_at_k", lambda p: p["filtered_pass_at_k"]["mcnemar_p"])):
            ps = [getter(r["populations"]["confirmatory"]) for r in members]
            for r, adj in zip(members, holm(ps)):
                r["populations"]["confirmatory"].setdefault("holm", {})[outcome] = {
                    "family": f"{role} models, {outcome}", "family_size": len(members),
                    "p_adjusted": adj, "significant": adj < ALPHA}


def across_models(diffs):
    common = set.intersection(*(set(d) for d in diffs.values())) if diffs else set()
    return {"common_instances": len(common),
            "mean_d": {m: (statistics.mean(d[i] for i in common) if common else None)
                       for m, d in diffs.items()}}


def breadth(diffs, fates, subset, weights):
    """Weighted estimate over the breadth subset, per model, with attrition:
    each selected id is analysed, excluded (by reason), unknown, or absent
    (no row in either arm, e.g. not generated)."""
    out = {}
    for model, d in diffs.items():
        attrition = Counter(fates[model].get(i, "absent") for i in subset)
        # read_breadth has checked every subset repository has a weight
        items = [(repo_of(i), weights[repo_of(i)], d[i]) for i in subset if i in d]
        per_stratum = Counter(st for st, _, _ in items)
        out[model] = {"selected": len(subset), "attrition": dict(attrition),
                      "analysed": len(items),
                      "strata_analysed": len(per_stratum),
                      "strata_with_one_instance": sorted(st for st, c in per_stratum.items() if c == 1),
                      "strata_with_none": sorted(set(weights) - set(per_stratum)),
                      "weighted_mean_d": weighted_mean([(w, v) for _, w, v in items]) if items else None,
                      "ci95_stratified": stratified_boot_ci(items),
                      "unweighted_mean_d": statistics.mean(v for _, _, v in items) if items else None}
    return out


def report(results, cross, breadth_out):
    for r in results:
        print(f"\n######## {r['model']}  ({r['role']} model, k = {r['k']})")
        print(f"  eligible instances: {r['eligible']}; exclusions: {r['exclusions']}")
        print(f"  samples made unknown by the count rule: {r['samples_count_mismatch_unknown']}; "
              f"re-marked incomplete: {r['samples_flagged_incomplete']}; "
              f"missing from their log: {r['samples_missing_from_log']}")
        for arm, ids in r["missing_one_arm_log"].items():
            if ids:
                print(f"  no {arm} log ({len(ids)}): {', '.join(ids[:10])}{' ...' if len(ids) > 10 else ''}")
        for pop_name, pop in r["populations"].items():
            label = f"  ({CONFIRMATORY_LABEL})" if pop_name == "confirmatory" else "  (descriptive, includes the development set)"
            print(f"  [{pop_name}]{label}")
            for variant in ("primary", "unknown_as_zero", "timeout_unknown"):
                v = pop[variant]
                w = v["wilcoxon"]
                print(f"    {variant:16} n={v['instances']:4} (dropped {v['dropped_unknown']}) "
                      f"mean d={v['mean_d'] if v['mean_d'] is None else round(v['mean_d'], 3)} "
                      f"bootstrap CI={v['ci95_mean_d']} | Wilcoxon signed-rank p={w['p']:.4g} "
                      f"({w['method']}) rank-biserial r={w['r']:.3f}")
            f = pop["filtered_pass_at_k"]
            print(f"    filtered pass@k n={f['instances']} (dropped {f['dropped_unknown']}) instruct={f['instruct']} kg_only={f['kg_only']} "
                  f"McNemar p={f['mcnemar_p']:.4g} discordant={f['discordant']}")
            if "holm" in pop:
                print(f"    Holm across models: {pop['holm']}")
        rb = r["robustness"]
        print("  [robustness, descriptive, not adjusted]")
        b = rb["mean_d_bounds"]
        if b:
            print(f"    missing-data bounds, conditional on established eligibility "
                  f"(manifest: {rb['bounds_population']['manifest'] or 'none given, instances with no log are invisible'})")
            print(f"    on {b['instances']} instances ({b['instances_with_unknown_samples']} with "
                  f"unknown samples; unknown samples {b['unknown_samples']}): mean d in "
                  f"[{b['lower_bound']:.3f}, {b['upper_bound']:.3f}]")
            print(f"    same value in both arms: " + ", ".join(f"{a}->{v:.3f}" for a, v in b["same_value_in_both_arms"].items()))
            for t in b["tipping"]:
                v = t["kg_only_value_where_mean_d_is_zero"]
                print(f"    instruct unknowns at {t['instruct_unknowns_at']:5.1f}: "
                      f"{'kg_only unknowns at %.1f' % v if v is not None else ''} {t['note']}")
        pb = rb["filtered_pass_at_k_bounds"]
        if pb:
            print(f"    filtered pass@k bounds on {pb['instances']} instances ({pb['resolved_in_both_arms']} resolved "
                  f"in both): difference in [{pb['difference_low']:.3f}, {pb['difference_high']:.3f}]; {pb['rate']}")
        if rb["bounds_population"]["eligibility_unresolved"]:
            print(f"    eligibility unresolved, outside the bounds ({len(rb['bounds_population']['eligibility_unresolved'])}): "
                  f"by repository {rb['bounds_population']['eligibility_unresolved_by_repo']}")
        for repo, v in rb["by_repo"].items():
            loo = rb["leave_one_repo_out"].get(repo)
            print(f"    {repo:28} analysed {v['analysed']:4}/{v['eligible']:4} mean d "
                  f"{'-' if v['mean_d'] is None else round(v['mean_d'], 3)}  | without it: "
                  + ("-" if not loo or loo["mean_d"] is None else
                     f"n={loo['instances']} mean d={loo['mean_d']:.3f} p={loo['wilcoxon_p_unadjusted']:.3g} (unadjusted)"))
        if r.get("prompt_length"):
            print(f"    prompt length: {r['prompt_length']}")
    print(f"\nAcross main-study models (descriptive): {cross}")
    if breadth_out:
        print(f"\nBreadth subset (secondary): {breadth_out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dev-ids", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="NAME:CSV of a main-study model (usable_score.py --out); NAME must match the CSV's run")
    ap.add_argument("--breadth-model", action="append", default=[], help="NAME:CSV of a breadth-only model")
    ap.add_argument("--skipped", action="append", default=[], help="NAME:ARM:FILE")
    ap.add_argument("--expected-ids", help="every instance id the main-study runs were asked to do (the run "
                    "manifest, e.g. the dataset's 1,210 ids); without it an instance with no log in either "
                    "arm is invisible to the bounds")
    ap.add_argument("--predictions", action="append", default=[],
                    help="NAME:ARM:predictions.jsonl, for the prompt-length diagnostic")
    ap.add_argument("--breadth-ids")
    ap.add_argument("--breadth-weights")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    if bool(args.breadth_ids) != bool(args.breadth_weights):
        ap.error("--breadth-ids and --breadth-weights go together")
    if args.breadth_model and not args.breadth_ids:
        ap.error("--breadth-model needs --breadth-ids and --breadth-weights")

    dev_ids = read_dev_ids(args.dev_ids)
    breadth_in = read_breadth(args.breadth_ids, args.breadth_weights, dev_ids) if args.breadth_ids else None
    skipped = us.read_skipped(args.skipped)
    tokens = read_prompt_tokens(args.predictions)
    expected_main = None
    if args.expected_ids:
        with open(args.expected_ids) as f:
            expected_main = {l.strip() for l in f if l.strip()}
    results, diffs, fates = [], {}, {}
    for role, specs in (("main", args.model), ("breadth", args.breadth_model)):
        for spec in specs:
            name, path = spec.split(":", 1)
            if name in diffs:
                raise SystemExit(f"model {name} given twice")
            rows = us.load(path)
            run = check_rows(rows, args.k, path)
            if run != name:
                raise SystemExit(f"{path} holds run {run!r}, not {name!r}")
            expected = expected_main if role == "main" else set(breadth_in[0])
            res, d, fate = analyse_model(name, rows, skipped, dev_ids, args.k, role, expected)
            if any(m == name for m, _ in tokens):
                res["prompt_length"] = length_diagnostic(d, {arm: tokens.get((name, arm), {}) for arm in ARMS})
            results.append(res); diffs[name] = d; fates[name] = fate
    holm_across(results)
    cross = across_models({m: d for m, d in diffs.items() if m in args_names(args.model)})
    breadth_out = breadth(diffs, fates, *breadth_in) if breadth_in else None
    report(results, cross, breadth_out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"models": results, "across_main_models": cross, "breadth": breadth_out}, f, indent=2)


def args_names(specs):
    return {spec.split(":", 1)[0] for spec in specs}


if __name__ == "__main__":
    main()
