"""The confirmatory analysis of the rerun (fyp-paper#7, sections 3 to 6).

Written and frozen before any confirmatory result; its md5 goes in the freeze
record (section 8). It reads the per-sample rows usable_score.py writes (one
CSV per model, from that model's two arms' evaluation logs) and does only what
the protocol fixes in advance. usable_score.py remains the descriptive tool;
the classification of each sample is its `classify`, reused here unchanged.

Inputs are checked before anything is computed: each CSV holds one run, and
exactly one row per sample index 0..k-1 for every instance and arm it has
(usable_score.py writes missing samples as unknown_missing rows); the
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
       - filtered pass@5 (any of the k samples passed after filtering; it
         measures the generation-plus-filtering pipeline): exact two-sided
         McNemar on discordant instances, and a paired bootstrap interval of
         the difference in proportions;
       - Holm adjustment, alpha 0.05, within four families: {main-study,
         breadth} models x {primary, filtered pass@k}, on the confirmatory
         population only. Sensitivity scorings and the population including
         the development set are descriptive and not adjusted.
  5. Across main-study models: each one's mean d on the instances eligible
     and usable for all of them (descriptive).
  6. Breadth subset (`--breadth-ids`, `--breadth-weights`): per model, the
     weighted mean of d over the subset's usable instances,
     sum(w d) / sum(w), with a stratified bootstrap resampling within each
     repository, the strata the subset was drawn in. A repository with one
     analysed instance contributes no resampling variance; how many there are
     is reported, as a limitation of the interval. Also the subset's
     attrition per id (analysed, unknown, excluded by reason, absent). The
     weights correct for sampling only, not for exclusions or unknown
     samples.

Standard library only, so it runs in any environment; usable_score.py from the
same folder.

Usage:
    python3 scripts/confirmatory_analysis.py --k 5 --dev-ids pilot_dev_ids.txt \\
        --model Scout:scout_usable_score.csv --skipped Scout:instruct:<file> ... \\
        [--breadth-model Gemma:gemma_usable_score.csv ...] \\
        [--breadth-ids breadth_ids.txt --breadth-weights breadth_ids.weights.tsv] \\
        [--json results.json]
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


def pass_at_k(arms, k):
    """{arm: (any passed, share passed)} if every sample's test run is known."""
    out = {}
    for arm in ARMS:
        rs = sorted(arms.get(arm, []), key=lambda r: r["sample"])[:k]
        if len(rs) < k or not all(test_known(r) for r in rs):
            return None
        out[arm] = (int(any(r["passed"] for r in rs)), statistics.mean(r["passed"] for r in rs))
    return out


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
            problems.append(f"{i} {arm}: samples {sorted(samples)}, expected 0..{k - 1} once each")
    if problems:
        raise SystemExit(f"{source}: {len(problems)} problem(s), e.g.\n  " + "\n  ".join(problems[:10]))
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

def analyse_model(model, rows, skipped, dev_ids, k, role="main"):
    changed = apply_count_rule(rows)
    flagged = us.flag_short_runs(rows)
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["instance"]][r["arm"]].append(r)
    too_long = us.rule4_excluded(model, per, skipped)
    both = {i: a for i, a in per.items() if all(arm in a for arm in ARMS) and i not in too_long}
    status = {i: us.eligibility([r for arm in ARMS for r in a[arm]]) for i, a in both.items()}
    eligible = {i: a for i, a in both.items() if status[i] == "eligible"}

    result = {"model": model, "role": role, "k": k,
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
    return result, diffs["confirmatory"], fate


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
              f"re-marked incomplete: {r['samples_flagged_incomplete']}")
        for pop_name, pop in r["populations"].items():
            label = "" if pop_name == "confirmatory" else "  (descriptive, includes the development set)"
            print(f"  [{pop_name}]{label}")
            for variant in ("primary", "unknown_as_zero", "timeout_unknown"):
                v = pop[variant]
                w = v["wilcoxon"]
                print(f"    {variant:16} n={v['instances']:4} (dropped {v['dropped_unknown']}) "
                      f"mean d={v['mean_d'] if v['mean_d'] is None else round(v['mean_d'], 3)} "
                      f"bootstrap CI={v['ci95_mean_d']} | Wilcoxon signed-rank p={w['p']:.4g} "
                      f"({w['method']}) rank-biserial r={w['r']:.3f}")
            f = pop["filtered_pass_at_k"]
            print(f"    filtered pass@k n={f['instances']} instruct={f['instruct']} kg_only={f['kg_only']} "
                  f"McNemar p={f['mcnemar_p']:.4g} discordant={f['discordant']}")
            if "holm" in pop:
                print(f"    Holm across models: {pop['holm']}")
    print(f"\nAcross main-study models (descriptive): {cross}")
    if breadth_out:
        print(f"\nBreadth subset: {breadth_out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dev-ids", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="NAME:CSV of a main-study model (usable_score.py --out); NAME must match the CSV's run")
    ap.add_argument("--breadth-model", action="append", default=[], help="NAME:CSV of a breadth-only model")
    ap.add_argument("--skipped", action="append", default=[], help="NAME:ARM:FILE")
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
            res, d, fate = analyse_model(name, rows, skipped, dev_ids, args.k, role)
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
