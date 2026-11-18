"""The confirmatory analysis of the rerun (fyp-paper#7, sections 3 to 6).

Written and frozen before any confirmatory result; its md5 goes in the freeze
record (section 8). It reads the per-sample rows usable_score.py writes (one
CSV per model, from that model's two arms' evaluation logs) and does only what
the protocol fixes in advance. usable_score.py remains the descriptive tool;
the classification of each sample is its `classify`, reused here unchanged.

Per model (RUN in the CSV), on instances present in both arms:
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
         method); tied |d| get average ranks; exact p when there are at most
         25 non-zero differences and no ties, otherwise the normal
         approximation with tie and continuity corrections;
       - matched-pairs rank-biserial r = (R+ - R-) / (R+ + R-) on the same ranks;
       - mean d with a 95% percentile bootstrap interval, resampling
         instances with all their samples (10,000 resamples, seed 0);
       - filtered pass@5 (any of the k samples passed after filtering; it
         measures the generation-plus-filtering pipeline): exact two-sided
         McNemar on discordant instances, and a paired bootstrap interval of
         the difference in proportions;
       - Holm adjustment across models, separately per outcome; alpha 0.05.
  5. Across models: each model's mean d on the instances eligible and usable
     for every model compared (descriptive).
  6. Breadth subset (`--breadth-ids`, `--breadth-weights`): per model, the
     weighted mean of d over the subset's usable instances,
     sum(w d) / sum(w), with a stratified bootstrap (resampling within each
     repository; repositories with fewer than 5 usable instances are pooled
     into one stratum for resampling only, their weights unchanged), and the
     subset's attrition (selected, excluded, unknown, analysed). The weights
     correct for sampling only, not for exclusions or unknown samples.

Standard library only, so it runs in any environment; usable_score.py from the
same folder.

Usage:
    python3 scripts/confirmatory_analysis.py --k 5 --dev-ids pilot_dev_ids.txt \\
        --model Scout:scout_usable_score.csv --skipped Scout:instruct:<file> ... \\
        [--breadth-ids breadth_ids.txt --breadth-weights breadth_ids.weights.tsv] \\
        [--json results.json]
"""

import argparse
import itertools
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
EXACT_MAX_N = 25
SMALL_STRATUM = 5


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
    if n <= EXACT_MAX_N and not ties:
        # Exact null distribution of W+ over all 2^n sign assignments (ranks 1..n).
        counts = [1] + [0] * (n * (n + 1) // 2)
        for k in range(1, n + 1):
            for s in range(len(counts) - 1, k - 1, -1):
                counts[s] += counts[s - k]
        total_cases = 2 ** n
        w = min(w_plus, w_minus)
        tail = sum(counts[s] for s in range(0, int(w) + 1)) / total_cases
        p = min(1.0, 2 * tail)
        method = "exact"
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
    mean, resampling within strata; strata with fewer than SMALL_STRATUM
    items are pooled into one stratum for resampling only."""
    by = defaultdict(list)
    for stratum, w, v in items:
        by[stratum].append((w, v))
    pooled = defaultdict(list)
    for stratum, group in by.items():
        pooled["__small__" if len(group) < SMALL_STRATUM else stratum].extend(group)
    if sum(len(g) for g in pooled.values()) < 2:
        return None
    rnd = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = []
        for group in pooled.values():
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


def pass_at_k(arms, k):
    """{arm: (any passed, share passed)} if every sample's test run is known."""
    out = {}
    for arm in ARMS:
        rs = sorted(arms.get(arm, []), key=lambda r: r["sample"])[:k]
        if len(rs) < k or not all(test_known(r) for r in rs):
            return None
        out[arm] = (int(any(r["passed"] for r in rs)), statistics.mean(r["passed"] for r in rs))
    return out


# ------------------------------------------------------------------- analysis

def analyse_model(model, rows, skipped, dev_ids, k):
    changed = apply_count_rule(rows)
    flagged = us.flag_short_runs(rows)
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["instance"]][r["arm"]].append(r)
    too_long = us.rule4_excluded(model, per, skipped)
    both = {i: a for i, a in per.items() if all(arm in a for arm in ARMS) and i not in too_long}
    status = {i: us.eligibility([r for arm in ARMS for r in a[arm]]) for i, a in both.items()}
    eligible = {i: a for i, a in both.items() if status[i] == "eligible"}
    repo = {i: next(iter(a["instruct"]))["repo"] for i, a in eligible.items()}

    result = {"model": model, "k": k,
              "samples_count_mismatch_unknown": changed, "samples_flagged_incomplete": flagged,
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
        # filtered pass@5 and pass@1 (mean over samples)
        b = c = both_pass = neither = 0
        p5 = {arm: [] for arm in ARMS}
        p1_d = []
        for i in ids:
            pk = pass_at_k(eligible[i], k)
            if pk is None:
                continue
            x, y = pk["instruct"][0], pk["kg_only"][0]
            p5["instruct"].append(x); p5["kg_only"].append(y)
            b += (y == 1 and x == 0); c += (x == 1 and y == 0)
            both_pass += (x == 1 and y == 1); neither += (x == 0 and y == 0)
            p1_d.append(pk["kg_only"][1] - pk["instruct"][1])
        n5 = len(p5["instruct"])
        diff5 = [y - x for x, y in zip(p5["instruct"], p5["kg_only"])]
        pop["filtered_pass_at_k"] = {
            "instances": n5,
            "instruct": statistics.mean(p5["instruct"]) if n5 else None,
            "kg_only": statistics.mean(p5["kg_only"]) if n5 else None,
            "discordant": {"kg_only_only": b, "instruct_only": c}, "concordant": {"both": both_pass, "neither": neither},
            "mcnemar_p": mcnemar_exact(b, c), "ci95_difference": boot_mean_ci(diff5)}
        pop["pass_at_1_mean_d"] = {"instances": len(p1_d), "mean_d": statistics.mean(p1_d) if p1_d else None,
                                   "ci95": boot_mean_ci(p1_d)}
        result["populations"][pop_name] = pop
    fate = {i: "excluded: rule4_prompt_too_long" for i in too_long}
    fate.update({i: f"excluded: {s}" for i, s in status.items() if s != "eligible"})
    fate.update({i: ("analysed" if i in diffs["all_including_development"] else "unknown")
                 for i in eligible})
    return result, diffs["confirmatory"], repo, fate


def holm_across(results):
    for outcome, getter in (("primary", lambda p: p["primary"]["wilcoxon"]["p"]),
                            ("filtered_pass_at_k", lambda p: p["filtered_pass_at_k"]["mcnemar_p"])):
        ps = [getter(r["populations"]["confirmatory"]) for r in results]
        for r, adj in zip(results, holm(ps)):
            r["populations"]["confirmatory"].setdefault("holm", {})[outcome] = {
                "p_adjusted": adj, "significant": adj < ALPHA}


def across_models(diffs):
    common = set.intersection(*(set(d) for d in diffs.values())) if diffs else set()
    return {"common_instances": len(common),
            "mean_d": {m: (statistics.mean(d[i] for i in common) if common else None)
                       for m, d in diffs.items()}}


def breadth(diffs, repos, fates, subset, weights):
    """Weighted estimate over the breadth subset, per model, with attrition:
    each selected id is analysed, excluded (by reason), unknown, or absent
    (no row in either arm, e.g. not generated)."""
    out = {}
    for model, d in diffs.items():
        attrition = Counter(fates[model].get(i, "absent") for i in subset)
        items, no_weight = [], 0
        for i in subset:
            if i not in d:
                continue
            w = weights.get(repos[model][i])
            if w is None:
                no_weight += 1
                continue
            items.append((repos[model][i], w, d[i]))
        out[model] = {"selected": len(subset), "attrition": dict(attrition), "no_weight": no_weight,
                      "analysed": len(items),
                      "weighted_mean_d": weighted_mean([(w, v) for _, w, v in items]) if items else None,
                      "ci95_stratified": stratified_boot_ci(items),
                      "unweighted_mean_d": statistics.mean(v for _, _, v in items) if items else None}
    return out


def report(results, cross, breadth_out):
    for r in results:
        print(f"\n######## {r['model']}  (k = {r['k']})")
        print(f"  eligible instances: {r['eligible']}; exclusions: {r['exclusions']}")
        print(f"  samples made unknown by the count rule: {r['samples_count_mismatch_unknown']}; "
              f"re-marked incomplete: {r['samples_flagged_incomplete']}")
        for pop_name, pop in r["populations"].items():
            print(f"  [{pop_name}]")
            for variant in ("primary", "unknown_as_zero", "timeout_unknown"):
                v = pop[variant]
                w = v["wilcoxon"]
                print(f"    {variant:16} n={v['instances']:4} (dropped {v['dropped_unknown']}) "
                      f"mean d={v['mean_d'] if v['mean_d'] is None else round(v['mean_d'], 3)} "
                      f"CI={v['ci95_mean_d']}  Wilcoxon p={w['p']:.4g} ({w['method']}) r={w['r']:.3f}")
            f = pop["filtered_pass_at_k"]
            print(f"    filtered pass@k n={f['instances']} instruct={f['instruct']} kg_only={f['kg_only']} "
                  f"McNemar p={f['mcnemar_p']:.4g} discordant={f['discordant']}")
            if "holm" in pop:
                print(f"    Holm across models: {pop['holm']}")
    print(f"\nAcross models (descriptive): {cross}")
    if breadth_out:
        print(f"\nBreadth subset: {breadth_out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dev-ids", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME:CSV (usable_score.py --out)")
    ap.add_argument("--skipped", action="append", default=[], help="NAME:ARM:FILE")
    ap.add_argument("--breadth-ids")
    ap.add_argument("--breadth-weights")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    dev_ids = {l.strip() for l in open(args.dev_ids) if l.strip()}
    skipped = us.read_skipped(args.skipped)
    results, diffs, repos, fates = [], {}, {}, {}
    for spec in args.model:
        name, path = spec.split(":", 1)
        rows = us.load(path)
        for r in rows:
            r["run"] = name
        res, d, repo, fate = analyse_model(name, rows, skipped, dev_ids, args.k)
        results.append(res); diffs[name] = d; repos[name] = repo; fates[name] = fate
    holm_across(results)
    cross = across_models(diffs)
    breadth_out = None
    if args.breadth_ids and args.breadth_weights:
        subset = [l.strip() for l in open(args.breadth_ids) if l.strip()]
        weights = {}
        with open(args.breadth_weights) as f:
            next(f)
            for line in f:
                repo, _pop, _k, w = line.rstrip("\n").split("\t")
                weights[repo] = float(w)
        breadth_out = breadth(diffs, repos, fates, subset, weights)
    report(results, cross, breadth_out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"models": results, "across_models": cross, "breadth": breadth_out}, f, indent=2)


if __name__ == "__main__":
    main()
