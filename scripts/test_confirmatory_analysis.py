"""Tests for confirmatory_analysis.py: python3 -m unittest scripts.test_confirmatory_analysis"""

import contextlib
import hashlib
import io
import itertools
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirmatory_analysis as ca  # noqa: E402


def brute_wilcoxon_p(d):
    """Two-sided exact p by enumerating every sign assignment of the ranks."""
    nz = [x for x in d if x != 0]
    ranks = ca.rank_abs(nz)
    w_obs = sum(r for r, x in zip(ranks, nz) if x > 0)
    mean = sum(ranks) / 2
    hits = total = 0
    for signs in itertools.product((0, 1), repeat=len(nz)):
        w = sum(r for r, s in zip(ranks, signs) if s)
        hits += abs(w - mean) >= abs(w_obs - mean) - 1e-12
        total += 1
    return hits / total


class StatisticsTest(unittest.TestCase):
    def test_rank_abs_averages_ties(self):
        self.assertEqual(ca.rank_abs([0.5, -0.1, 0.1, 0.3]), [4, 1.5, 1.5, 3])

    def test_wilcoxon_exact_matches_enumeration(self):
        for d in ([0.1, 0.2, 0.3, 0.4, 0.5], [-0.1, 0.2, 0.3, -0.45, 0.5, 0.6],
                  [0.05, -0.15, 0.25, 0.35, -0.55, 0.65, 0.75, 0.85]):
            with self.subTest(d=d):
                w = ca.wilcoxon(d)
                self.assertEqual(w["method"], "exact")
                self.assertAlmostEqual(w["p"], brute_wilcoxon_p(d))

    def test_wilcoxon_all_positive_five(self):
        w = ca.wilcoxon([1, 2, 3, 4, 5])
        self.assertAlmostEqual(w["p"], 2 / 32)    # scipy.stats.wilcoxon gives 0.0625
        self.assertEqual(w["r"], 1.0)

    def test_wilcoxon_drops_zero_differences(self):
        self.assertEqual(ca.wilcoxon([0, 0, 1, 2, 3, 4, 5]), ca.wilcoxon([1, 2, 3, 4, 5]))
        self.assertEqual(ca.wilcoxon([0, 0])["p"], 1.0)

    def test_wilcoxon_rank_biserial_sign_and_symmetry(self):
        d = [0.3, -0.1, 0.2, 0.5, -0.4, 0.6]
        a, b = ca.wilcoxon(d), ca.wilcoxon([-x for x in d])
        self.assertAlmostEqual(a["r"], -b["r"])
        self.assertAlmostEqual(a["p"], b["p"])
        self.assertAlmostEqual(a["r"], (a["w_plus"] - a["w_minus"]) / 21)

    def test_wilcoxon_small_with_ties_is_exact(self):
        # Tied |d| in a small sample: the exact sign-permutation p over the
        # average ranks, matching full enumeration.
        for d in ([0.2, -0.2, 0.4, 0.4, 0.6, -0.1, 0.8, 0.3, 0.3, 0.5, -0.7, 0.9, 0.2, 1.0, 0.45, 0.35],
                  [0.1, 0.1, 0.1, -0.1, 0.2], [0.5, 0.5, -0.5, 0.5, 0.5, 0.5]):
            with self.subTest(d=d):
                w = ca.wilcoxon(d)
                self.assertEqual(w["method"], "exact_with_ties")
                self.assertAlmostEqual(w["p"], brute_wilcoxon_p(d))

    def test_wilcoxon_large_n_uses_normal_close_to_exact(self):
        d = [(i % 7 - 2) / 10 + i / 1000 for i in range(60)]
        w = ca.wilcoxon(d)
        self.assertEqual(w["method"], "normal")
        nz = [x for x in d if x != 0]
        saved = ca.EXACT_MAX_N
        ca.EXACT_MAX_N = 100
        try:
            exact = ca.wilcoxon(d)["p"]
        finally:
            ca.EXACT_MAX_N = saved
        self.assertLess(abs(w["p"] - exact), 0.01)

    def test_mcnemar_exact(self):
        self.assertAlmostEqual(ca.mcnemar_exact(0, 5), 2 / 32)
        self.assertAlmostEqual(ca.mcnemar_exact(5, 0), 2 / 32)
        self.assertEqual(ca.mcnemar_exact(3, 3), 1.0)
        self.assertEqual(ca.mcnemar_exact(0, 0), 1.0)
        # b=2, c=8: 2 * P(X <= 2 | n=10) = 2 * 56/1024
        self.assertAlmostEqual(ca.mcnemar_exact(2, 8), 2 * 56 / 1024)

    def test_holm(self):
        adj = ca.holm([0.01, 0.04, 0.03])
        for got, want in zip(adj, [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(got, want)
        self.assertEqual(ca.holm([0.5, 0.9]), [1.0, 1.0])

    def test_stratified_bootstrap_keeps_strata_and_singletons_add_no_variance(self):
        # Two strata, each internally constant: every replicate keeps the
        # composition, so the interval collapses to the estimate. Pooling
        # would have mixed them and produced a spread.
        items = [("a", 1.0, 0.0)] * 3 + [("b", 1.0, 1.0)] * 3 + [("c", 2.0, 5.0)]
        est = ca.weighted_mean([(w, v) for _, w, v in items])
        lo, hi = ca.stratified_boot_ci(items, n=500)
        self.assertAlmostEqual(lo, est)
        self.assertAlmostEqual(hi, est)

    def test_stratified_bootstrap_contains_estimate_and_is_reproducible(self):
        items = [("a", 2.0, v / 10) for v in range(10)] + [("b", 1.0, 0.5)] * 3 + [("c", 4.0, -0.2)] * 2
        ci = ca.stratified_boot_ci(items, n=2000)
        est = ca.weighted_mean([(w, v) for _, w, v in items])
        self.assertLessEqual(ci[0], est)
        self.assertGreaterEqual(ci[1], est)
        self.assertEqual(ci, ca.stratified_boot_ci(items, n=2000))

    def test_weighted_mean(self):
        self.assertAlmostEqual(ca.weighted_mean([(1, 0.0), (3, 1.0)]), 0.75)


def iid(name):
    return f"django__django-{name}-1"


def row(instance, arm, sample, outcome="score", score=50.0, passed=1, count=4, total=4, run="M"):
    return {"run": run, "arm": arm, "instance": instance, "repo": instance.rsplit("-", 2)[0],
            "sample": sample, "outcome": outcome, "passed": passed, "unfiltered_passed": passed,
            "func_score": score if outcome == "score" else None,
            "func_done": total, "func_total": total, "count_outcome": "counted", "target_count": count}


def instance(name, kg_scores, in_scores, over=None):
    """Five samples per arm with the given scores; `over` maps (arm, sample)
    to keyword overrides for that one row."""
    rows = []
    for arm, scores in (("kg_only", kg_scores), ("instruct", in_scores)):
        for s, sc in enumerate(scores):
            kw = {"score": sc}
            kw.update((over or {}).get((arm, s), {}))
            rows.append(row(iid(name), arm, s, **kw))
    return rows


DEV = {iid("dev0")}


def build():
    rows = []
    for n in range(8):   # kg_only better by 10 points on each
        rows += instance(f"good{n}", [60] * 5, [50] * 5)
    rows += instance("dev0", [0] * 5, [100] * 5)   # development: kept out of confirmatory
    rows += instance("mismatch", [60] * 5, [50] * 5, {("kg_only", 2): {"total": 3}})
    rows += instance("timeout", [60] * 5, [50] * 5,
                     {("instruct", 0): {"outcome": "zero_test_timeout", "passed": 0}})
    rows += instance("harness", [60] * 5, [50] * 5,
                     {("kg_only", 4): {"outcome": "unknown_harness_error", "passed": 0}})
    rows += instance("nomut", [0] * 5, [0] * 5,
                     {(a, s): {"count": 0, "total": 0, "outcome": "no_target_mutants"}
                      for a in ("kg_only", "instruct") for s in range(5)})
    rows += [r for r in instance("long", [60] * 5, [50] * 5) if r["arm"] == "kg_only"]
    rows += [r for r in instance("nolog", [60] * 5, [50] * 5) if r["arm"] == "instruct"]   # kg_only log missing
    skipped = {("M", "instruct"): {iid("long"): "prompt_too_long"}}
    return rows, skipped


def relabel(rows, skipped, name):
    return [dict(r, run=name) for r in rows], {(name, "instruct"): skipped[("M", "instruct")]}


class AnalyseModelTest(unittest.TestCase):
    def test_rules(self):
        rows, skipped = build()
        res, d, fate = ca.analyse_model("M", rows, skipped, DEV, 5)
        self.assertEqual(res["samples_count_mismatch_unknown"], 1)
        self.assertEqual(res["exclusions"]["rule4_prompt_too_long"], 1)
        self.assertEqual(res["exclusions"]["no target mutants"], 1)
        self.assertEqual(fate[iid("long")], "excluded: rule4_prompt_too_long")
        self.assertEqual(fate[iid("nomut")], "excluded: no target mutants")
        self.assertEqual(fate[iid("mismatch")], "unknown")
        self.assertEqual(fate[iid("harness")], "unknown")

        conf = res["populations"]["confirmatory"]
        # primary: 8 good + timeout (scored 0 in one instruct sample); mismatch and harness unknown
        self.assertEqual(conf["primary"]["instances"], 9)
        self.assertEqual(conf["primary"]["dropped_unknown"], 2)
        self.assertNotIn(iid("dev0"), d)
        self.assertAlmostEqual(d[iid("timeout")], 60 - 40)
        # unknown as zero: all 11 confirmatory eligible instances, harness counted as 0
        self.assertEqual(conf["unknown_as_zero"]["instances"], 11)
        # timeout as unknown: the timeout instance drops out instead
        self.assertEqual(conf["timeout_unknown"]["instances"], 8)
        self.assertEqual(conf["timeout_unknown"]["dropped_unknown"], 3)

        allp = res["populations"]["all_including_development"]
        self.assertEqual(allp["primary"]["instances"], 10)
        self.assertGreater(conf["primary"]["mean_d"], allp["primary"]["mean_d"])

    def test_pass_at_k(self):
        rows, skipped = build()
        res, *_ = ca.analyse_model("M", rows, skipped, DEV, 5)
        conf = res["populations"]["confirmatory"]
        f = conf["filtered_pass_at_k"]
        # the harness error leaves one sample's test result unknown, but its
        # four siblings are known passes, so filtered pass@5 is known
        self.assertEqual(f["instances"], 11)
        self.assertEqual(f["dropped_unknown"], 0)
        self.assertEqual(f["discordant"], {"kg_only_only": 0, "instruct_only": 0})
        self.assertEqual(f["mcnemar_p"], 1.0)
        # pass@1 is complete case: that instance is dropped from it
        self.assertEqual(conf["pass_at_1_mean_d"]["instances"], 10)
        self.assertEqual(conf["pass_at_1_mean_d"]["dropped_unknown"], 1)

    def test_filtered_pass_three_way(self):
        p = lambda **kw: row(iid("x"), "kg_only", 0, **kw)
        ok, fail, unk = p(), p(outcome="zero_suite_failed", passed=0), p(outcome="unknown_harness_error", passed=0)
        self.assertEqual(ca.filtered_pass([fail, unk, ok]), 1)      # a known pass settles it
        self.assertIsNone(ca.filtered_pass([fail, fail, unk]))      # could still have passed
        self.assertEqual(ca.filtered_pass([fail] * 5), 0)
        self.assertIsNone(ca.pass_share([ok, unk]))
        self.assertEqual(ca.pass_share([ok, fail]), 0.5)

    def test_mcnemar_counts_discordant_instances(self):
        rows = []
        fail = {"outcome": "zero_suite_failed", "passed": 0}
        for n in range(6):   # kg_only passes, instruct never does
            rows += instance(f"kg{n}", [60] * 5, [0] * 5, {("instruct", s): fail for s in range(5)})
        rows += instance("inonly", [0] * 5, [50] * 5, {("kg_only", s): fail for s in range(5)})
        res, *_ = ca.analyse_model("M", rows, {}, DEV, 5)
        f = res["populations"]["confirmatory"]["filtered_pass_at_k"]
        self.assertEqual(f["discordant"], {"kg_only_only": 6, "instruct_only": 1})
        self.assertAlmostEqual(f["mcnemar_p"], 2 * (1 + 7) / 2 ** 7)

    def test_missing_arm_log_is_attrition(self):
        rows, skipped = build()
        res, _, fate = ca.analyse_model("M", rows, skipped, DEV, 5)
        self.assertEqual(res["missing_one_arm_log"], {"instruct": [], "kg_only": [iid("nolog")]})
        self.assertEqual(fate[iid("nolog")], "missing: no kg_only log")
        self.assertEqual(res["samples_missing_from_log"], 0)
        rows[0]["outcome"] = "unknown_missing"
        res, *_ = ca.analyse_model("M", rows, skipped, DEV, 5)
        self.assertEqual(res["samples_missing_from_log"], 1)

    def test_holm_families_by_role(self):
        rows, skipped = build()
        results, diffs = [], {}
        for name, role in (("A", "main"), ("B", "main"), ("C", "breadth")):
            res, d, _ = ca.analyse_model(name, *relabel(rows, skipped, name), DEV, 5, role)
            results.append(res); diffs[name] = d
        ca.holm_across(results)
        hs = [r["populations"]["confirmatory"]["holm"]["primary"] for r in results]
        self.assertEqual([h["family_size"] for h in hs], [2, 2, 1])
        self.assertEqual(hs[2]["family"], "breadth models, primary")
        raw = results[2]["populations"]["confirmatory"]["primary"]["wilcoxon"]["p"]
        self.assertAlmostEqual(hs[2]["p_adjusted"], raw)   # a family of one is unadjusted
        self.assertNotIn("holm", results[0]["populations"]["all_including_development"])
        self.assertEqual(ca.across_models({m: diffs[m] for m in "AB"})["common_instances"], 9)

    def test_breadth_attrition_and_strata(self):
        rows, skipped = build()
        res, d, fate = ca.analyse_model("M", rows, skipped, DEV, 5)
        subset = [iid(n) for n in ("good0", "good1", "mismatch", "long", "nolog", "neverrun")]
        out = ca.breadth({"M": d}, {"M": fate}, subset, {"django/django": 2.5, "pallets/flask": 1.0})["M"]
        self.assertEqual(out["analysed"], 2)
        self.assertEqual(out["attrition"], {"analysed": 2, "unknown": 1,
                                            "excluded: rule4_prompt_too_long": 1,
                                            "missing: no kg_only log": 1, "absent": 1})
        self.assertEqual(out["strata_with_none"], ["pallets/flask"])
        self.assertAlmostEqual(out["weighted_mean_d"], 10.0)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_repo_of(self):
        self.assertEqual(ca.repo_of("scikit-learn__scikit-learn-10198-16700"), "scikit-learn/scikit-learn")
        self.assertEqual(ca.repo_of("pytest-dev__pytest-5103-1"), "pytest-dev/pytest")

    def test_check_rows_accepts_complete_and_returns_run(self):
        rows, _ = build()
        self.assertEqual(ca.check_rows(rows, 5, "x"), "M")

    def test_check_rows_rejects_duplicate_extra_missing_and_mixed(self):
        rows, _ = build()
        cases = {"duplicate": rows + [dict(rows[0])],
                 "extra": rows + [dict(rows[0], sample=5)],
                 "missing": rows[1:],
                 "mixed run": rows[:-1] + [dict(rows[-1], run="Other")],
                 "bad arm": rows[:-1] + [dict(rows[-1], arm="kg")]}
        for label, rs in cases.items():
            with self.subTest(label), self.assertRaises(SystemExit):
                ca.check_rows(rs, 5, "x")

    def test_dev_ids_md5_and_size(self):
        ids = "".join(f"django__django-{i}-1\n" for i in range(151))
        path = self.write("dev.txt", ids)
        md5 = hashlib.md5(ids.encode()).hexdigest()
        self.assertEqual(len(ca.read_dev_ids(path, md5)), 151)
        with self.assertRaises(SystemExit):
            ca.read_dev_ids(path)                      # not the recorded list
        rest = ids.split("\n", 1)[1]
        short = self.write("short.txt", rest)
        with self.assertRaises(SystemExit):
            ca.read_dev_ids(short, hashlib.md5(rest.encode()).hexdigest())

    def breadth_files(self, ids, weights):
        return self.write("b.txt", "".join(i + "\n" for i in ids)), self.write(
            "b.weights.tsv", "repo\tpopulation\tsample\tweight\n" + "".join(l + "\n" for l in weights))

    def test_breadth_accepts_consistent_files(self):
        ids = [iid("a"), iid("b"), "pallets__flask-7-1"]
        subset, w = ca.read_breadth(*self.breadth_files(
            ids, ["django/django\t10\t2\t5.000000", "pallets/flask\t3\t1\t3.000000"]), DEV)
        self.assertEqual(subset, ids)
        self.assertEqual(w, {"django/django": 5.0, "pallets/flask": 3.0})

    def test_breadth_rejects(self):
        good = ["django/django\t10\t2\t5.000000"]
        ids = [iid("a"), iid("b")]
        cases = {"development id": (ids[:1] + [iid("dev0")], good),
                 "duplicate id": (ids + ids[:1], ["django/django\t10\t3\t3.333333"]),
                 "repo without weight": (ids + ["pallets__flask-7-1"], good),
                 "sample size disagrees": (ids, ["django/django\t10\t3\t3.333333"]),
                 "weight not population/sample": (ids, ["django/django\t10\t2\t4.000000"]),
                 "non-positive weight": (ids, ["django/django\t10\t2\t-5"]),
                 "nan weight": (ids, ["django/django\t10\t2\tnan"]),
                 "unparseable": (ids, ["django/django\tten\t2\t5"]),
                 "repeated repo": (ids, good + good)}
        for label, (i, w) in cases.items():
            with self.subTest(label), self.assertRaises(SystemExit):
                ca.read_breadth(*self.breadth_files(i, w), DEV)


class RobustnessTest(unittest.TestCase):
    def setUp(self):
        rows, skipped = build()
        self.res, self.d, self.fate = ca.analyse_model("M", rows, skipped, DEV, 5)
        self.rb = self.res["robustness"]

    def test_bounds_population_includes_a_missing_arm_and_excludes_the_rest(self):
        # 8 good + mismatch + timeout + harness + nolog (kg_only log missing);
        # not dev0, long (rule 4) or nomut (no target mutants)
        self.assertEqual(self.rb["bounds_population"]["instances"], 12)
        self.assertEqual(self.rb["bounds_population"]["eligibility_unresolved"], {})

    def test_mean_d_bounds_by_hand(self):
        b = self.rb["mean_d_bounds"]
        # base: good 8 x 10, timeout +20 (its 0 is known), mismatch and harness
        # (240 - 250) / 5 = -2 each, nolog (0 - 250) / 5 = -50 -> 46 / 12
        self.assertEqual(b["unknown_samples"], {"kg_only": 7, "instruct": 0})
        self.assertEqual(b["instances_with_unknown_samples"], 3)
        self.assertAlmostEqual(b["lower_bound"], 46 / 12)
        self.assertAlmostEqual(b["upper_bound"], 46 / 12 + 7 * 100 / (5 * 12))
        self.assertAlmostEqual(b["same_value_in_both_arms"]["50.0"], 46 / 12 + 7 * 50 / 60)
        self.assertTrue(all(t["kg_only_value_where_mean_d_is_zero"] is None and "positive" in t["note"]
                            for t in b["tipping"]))

    def test_manifest_makes_no_log_instances_visible(self):
        rows, skipped = build()
        expected = {r["instance"] for r in rows} | {iid("ghost"), iid("dev0")}
        res, _, fate = ca.analyse_model("M", rows, skipped, DEV, 5, expected=expected)
        bp = res["robustness"]["bounds_population"]
        self.assertEqual(bp["instances"], 12)                       # unchanged: eligibility not established
        self.assertEqual(bp["eligibility_unresolved"], {iid("ghost"): ca.NO_LOG})
        self.assertEqual(bp["eligibility_unresolved_by_repo"], {"django/django": 1})
        self.assertEqual(bp["manifest"], len(expected))
        self.assertEqual(fate[iid("ghost")], "missing: no log in either arm")
        self.assertEqual(res["robustness"]["attrition_by_repo"]["django/django"]["instances"]
                         ["missing: no log in either arm"], 1)     # dev0 has logs; ghost is the one
        self.assertIn("conditional on established eligibility", bp["definition"])

    def test_tipping_value_makes_mean_d_zero(self):
        rows = instance("tip", [50] * 5, [55] * 5, {("kg_only", 4): {"outcome": "unknown_harness_error", "passed": 0}})
        per = {iid("tip"): {"kg_only": [r for r in rows if r["arm"] == "kg_only"],
                            "instruct": [r for r in rows if r["arm"] == "instruct"]}}
        b = ca.mean_d_bounds(per, 5)
        t0 = b["tipping"][0]   # instruct unknowns at 0 (there are none)
        self.assertAlmostEqual(t0["kg_only_value_where_mean_d_is_zero"], 75.0)   # (4*50 + 75)/5 = 55

    def test_bounds_equal_the_primary_mean_with_no_unknowns(self):
        rows = []
        for n in range(4):
            rows += instance(f"c{n}", [60 + n] * 5, [50] * 5)
        res, d, _ = ca.analyse_model("M", rows, {}, DEV, 5)
        b = res["robustness"]["mean_d_bounds"]
        prim = res["populations"]["confirmatory"]["primary"]["mean_d"]
        self.assertAlmostEqual(b["lower_bound"], prim)
        self.assertAlmostEqual(b["upper_bound"], prim)

    def test_pass_bounds(self):
        pb = self.rb["filtered_pass_at_k_bounds"]
        self.assertEqual(pb["instances"], 12)
        self.assertEqual(pb["resolved_in_both_arms"], 11)          # nolog's kg_only is unresolved
        self.assertAlmostEqual(pb["rate"]["kg_only"]["low"], 11 / 12)
        self.assertAlmostEqual(pb["rate"]["kg_only"]["high"], 1.0)
        self.assertAlmostEqual(pb["difference_low"], 11 / 12 - 1)
        self.assertAlmostEqual(pb["difference_high"], 0.0)
        self.assertEqual(self.res["populations"]["confirmatory"]["filtered_pass_at_k"]["population"],
                         "instances with filtered pass@k resolved in both arms")

    def test_attrition_lists_timeouts_and_fates(self):
        a = self.rb["attrition_by_repo"]["django/django"]
        self.assertEqual(a["samples"]["instruct"]["zero_test_timeout"], 1)
        self.assertEqual(a["samples"]["kg_only"]["unknown_harness_error"], 1)
        self.assertEqual(a["instances"]["missing: no kg_only log"], 1)
        self.assertEqual(a["instances"]["analysed"], 9)
        self.assertNotIn("dev0", str(a))

    def test_by_repo_and_leave_one_out(self):
        rows = []
        for n in range(5):
            rows += instance(f"a{n}", [70] * 5, [50] * 5)
        for n in range(3):
            rows += [dict(r, instance=f"pallets__flask-b{n}-1", repo="pallets__flask")
                     for r in instance(f"b{n}", [40] * 5, [50] * 5)]
        res, *_ = ca.analyse_model("M", rows, {}, DEV, 5)
        br, loo = res["robustness"]["by_repo"], res["robustness"]["leave_one_repo_out"]
        self.assertAlmostEqual(br["django/django"]["mean_d"], 20)
        self.assertAlmostEqual(br["pallets/flask"]["mean_d"], -10)
        self.assertEqual(br["pallets/flask"]["completion"], 1.0)
        self.assertEqual(loo["django/django"]["instances"], 3)
        self.assertAlmostEqual(loo["django/django"]["mean_d"], -10)
        self.assertAlmostEqual(loo["pallets/flask"]["mean_d"], 20)


class ReportTest(unittest.TestCase):
    def test_report_prints_with_one_repository_and_missing_pieces(self):
        rows, skipped = build()          # one repository: leaving it out leaves nothing
        res, d, _ = ca.analyse_model("M", rows, skipped, DEV, 5)
        res["prompt_length"] = ca.length_diagnostic(d, {"instruct": {}, "kg_only": {}})
        ca.holm_across([res])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ca.report([res], ca.across_models({"M": d}), None)
        self.assertIn("prospectively specified rerun", buf.getvalue())
        self.assertIn("without it: -", buf.getvalue())


class PromptLengthTest(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(ca.spearman([1, 2, 3, 4], [10, 20, 30, 45]), 1.0)
        self.assertAlmostEqual(ca.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_diagnostic(self):
        d = {f"i{n}": float(n) for n in range(9)}
        tokens = {"instruct": {f"i{n}": 1000 for n in range(9)},
                  "kg_only": {f"i{n}": 1000 + 100 * n for n in range(9)}}
        out = ca.length_diagnostic(d, tokens)
        self.assertEqual(out["pairs"], 9)
        self.assertEqual(out["missing_token_counts"], {"instruct": 0, "kg_only": 0})
        self.assertAlmostEqual(out["spearman_token_difference_vs_d"], 1.0)
        self.assertEqual([t["mean_d"] for t in out["tertiles"]], [1.0, 4.0, 7.0])

    def test_tied_differences_share_a_tertile(self):
        d = {f"i{n}": float(n) for n in range(6)}
        tokens = {"instruct": {f"i{n}": 1000 for n in range(6)},
                  "kg_only": {f"i{n}": 1600 for n in range(6)}}
        out = ca.length_diagnostic(d, tokens)
        self.assertEqual(len(out["tertiles"]), 1)
        self.assertEqual(out["tertiles"][0]["instances"], 6)
        self.assertTrue(out["spearman_token_difference_vs_d"].startswith("undefined"))
        # two distinct values: never a group mixing them, never one split
        tokens["kg_only"].update({"i4": 2000, "i5": 2000})
        groups = ca.length_diagnostic(d, tokens)["tertiles"]
        self.assertEqual(sorted(g["instances"] for g in groups), [2, 4])
        self.assertTrue(all(g["token_difference_range"][0] == g["token_difference_range"][1] for g in groups))

    def test_diagnostic_counts_missing_tokens(self):
        d = {f"i{n}": float(n) for n in range(5)}
        tokens = {"instruct": {f"i{n}": 1000 for n in range(5)},
                  "kg_only": {f"i{n}": 1000 + n for n in range(4)}}     # i4 has no kg_only count
        out = ca.length_diagnostic(d, tokens)
        self.assertEqual((out["pairs"], out["analysed_instances"]), (4, 5))
        self.assertEqual(out["missing_token_counts"], {"instruct": 0, "kg_only": 1})

    def test_diagnostic_without_tokens(self):
        out = ca.length_diagnostic({"a": 1.0}, {"instruct": {}, "kg_only": {}})
        self.assertIn("testgeneval#154", out["note"])

    def test_read_prompt_tokens(self):
        dirname = tempfile.mkdtemp()
        try:
            path = os.path.join(dirname, "p.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"id": "a", "sampling": {"full": {"prompt_tokens": 12}}}) + "\n")
                f.write(json.dumps({"id": "b"}) + "\n")                       # before #154
            self.assertEqual(ca.read_prompt_tokens([f"M:kg_only:{path}"]), {("M", "kg_only"): {"a": 12, "b": None}})
            with open(path, "a") as f:
                f.write(json.dumps({"id": "a"}) + "\n")
            with self.assertRaises(SystemExit):
                ca.read_prompt_tokens([f"M:kg_only:{path}"])
        finally:
            shutil.rmtree(dirname)


if __name__ == "__main__":
    unittest.main()
