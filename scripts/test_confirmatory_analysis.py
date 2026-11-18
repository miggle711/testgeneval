"""Tests for confirmatory_analysis.py: python3 -m unittest scripts.test_confirmatory_analysis"""

import itertools
import math
import os
import sys
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

    def test_wilcoxon_normal_with_ties_close_to_enumeration(self):
        # Tied |d| force the normal approximation; with n = 16 it should be
        # near the exact permutation p computed on the same average ranks.
        d = [0.2, -0.2, 0.4, 0.4, 0.6, -0.1, 0.8, 0.3, 0.3, 0.5, -0.7, 0.9, 0.2, 1.0, 0.45, 0.35]
        w = ca.wilcoxon(d)
        self.assertEqual(w["method"], "normal")
        self.assertLess(abs(w["p"] - brute_wilcoxon_p(d)), 0.01)

    def test_wilcoxon_large_n_uses_normal(self):
        d = [(i % 7 - 2) / 10 + i / 1000 for i in range(40)]
        self.assertEqual(ca.wilcoxon(d)["method"], "normal")

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

    def test_stratified_bootstrap_contains_estimate_and_is_reproducible(self):
        items = [("a", 2.0, v / 10) for v in range(10)] + [("b", 1.0, 0.5)] * 3 + [("c", 4.0, -0.2)] * 2
        ci = ca.stratified_boot_ci(items, n=2000)
        est = ca.weighted_mean([(w, v) for _, w, v in items])
        self.assertLessEqual(ci[0], est)
        self.assertGreaterEqual(ci[1], est)
        self.assertEqual(ci, ca.stratified_boot_ci(items, n=2000))

    def test_weighted_mean(self):
        self.assertAlmostEqual(ca.weighted_mean([(1, 0.0), (3, 1.0)]), 0.75)


def row(instance, arm, sample, outcome="score", score=50.0, passed=1, count=4, total=4, repo="django/django"):
    return {"run": "M", "arm": arm, "instance": instance, "repo": repo, "sample": sample,
            "outcome": outcome, "passed": passed, "unfiltered_passed": passed,
            "func_score": score if outcome == "score" else None,
            "func_done": total, "func_total": total, "count_outcome": "counted", "target_count": count}


def instance(iid, kg_scores, in_scores, over=None, repo="django/django"):
    """Five samples per arm with the given scores; `over` maps (arm, sample)
    to keyword overrides for that one row."""
    rows = []
    for arm, scores in (("kg_only", kg_scores), ("instruct", in_scores)):
        for s, sc in enumerate(scores):
            kw = {"score": sc, "repo": repo}
            kw.update((over or {}).get((arm, s), {}))
            rows.append(row(iid, arm, s, **kw))
    return rows


class AnalyseModelTest(unittest.TestCase):
    def build(self):
        rows = []
        for n in range(8):   # kg_only better by 10 points on each
            rows += instance(f"good-{n}", [60] * 5, [50] * 5)
        rows += instance("dev-0", [0] * 5, [100] * 5)   # development: kept out of confirmatory
        rows += instance("mismatch", [60] * 5, [50] * 5, {("kg_only", 2): {"total": 3}})
        rows += instance("timeout", [60] * 5, [50] * 5,
                         {("instruct", 0): {"outcome": "zero_test_timeout", "passed": 0}})
        rows += instance("harness", [60] * 5, [50] * 5,
                         {("kg_only", 4): {"outcome": "unknown_harness_error", "passed": 0}})
        rows += instance("nomut", [0] * 5, [0] * 5,
                         {(a, s): {"count": 0, "total": 0, "outcome": "no_target_mutants"}
                            for a in ("kg_only", "instruct") for s in range(5)})
        rows += [r for r in instance("long", [60] * 5, [50] * 5) if r["arm"] == "kg_only"]
        skipped = {("M", "instruct"): {"long": "prompt_too_long"}}
        return rows, skipped

    def test_rules(self):
        rows, skipped = self.build()
        res, d, repo, fate = ca.analyse_model("M", rows, skipped, {"dev-0"}, 5)
        self.assertEqual(res["samples_count_mismatch_unknown"], 1)
        self.assertEqual(res["exclusions"]["rule4_prompt_too_long"], 1)
        self.assertEqual(res["exclusions"]["no target mutants"], 1)
        self.assertEqual(fate["long"], "excluded: rule4_prompt_too_long")
        self.assertEqual(fate["nomut"], "excluded: no target mutants")
        self.assertEqual(fate["mismatch"], "unknown")
        self.assertEqual(fate["harness"], "unknown")

        conf = res["populations"]["confirmatory"]
        # primary: 8 good + timeout (scored 0 in one instruct sample); mismatch and harness unknown
        self.assertEqual(conf["primary"]["instances"], 9)
        self.assertEqual(conf["primary"]["dropped_unknown"], 2)
        self.assertNotIn("dev-0", d)
        self.assertAlmostEqual(d["timeout"], 60 - 40)
        # unknown as zero: all 11 confirmatory eligible instances, harness counted as 0
        self.assertEqual(conf["unknown_as_zero"]["instances"], 11)
        # timeout as unknown: the timeout instance drops out instead
        self.assertEqual(conf["timeout_unknown"]["instances"], 8)
        self.assertEqual(conf["timeout_unknown"]["dropped_unknown"], 3)

        allp = res["populations"]["all_including_development"]
        self.assertEqual(allp["primary"]["instances"], 10)
        self.assertGreater(conf["primary"]["mean_d"], allp["primary"]["mean_d"])

    def test_pass_at_k(self):
        rows, skipped = self.build()
        res, *_ = ca.analyse_model("M", rows, skipped, {"dev-0"}, 5)
        f = res["populations"]["confirmatory"]["filtered_pass_at_k"]
        # harness error leaves that sample's test outcome unknown; every other one passed
        self.assertEqual(f["instances"], 10)
        self.assertEqual(f["discordant"], {"kg_only_only": 0, "instruct_only": 0})
        self.assertEqual(f["mcnemar_p"], 1.0)

    def test_holm_and_breadth(self):
        rows, skipped = self.build()
        results, diffs, repos, fates = [], {}, {}, {}
        for name in ("A", "B"):
            rs = [dict(r, run=name) for r in rows]
            sk = {(name, "instruct"): skipped[("M", "instruct")]}
            res, d, repo, fate = ca.analyse_model(name, rs, sk, {"dev-0"}, 5)
            results.append(res); diffs[name] = d; repos[name] = repo; fates[name] = fate
        ca.holm_across(results)
        h = results[0]["populations"]["confirmatory"]["holm"]["primary"]
        self.assertGreaterEqual(h["p_adjusted"], results[0]["populations"]["confirmatory"]["primary"]["wilcoxon"]["p"])
        self.assertEqual(ca.across_models(diffs)["common_instances"], 9)

        subset = ["good-0", "good-1", "mismatch", "long", "never-run"]
        out = ca.breadth(diffs, repos, fates, subset, {"django/django": 2.5})["A"]
        self.assertEqual(out["analysed"], 2)
        self.assertEqual(out["attrition"], {"analysed": 2, "unknown": 1,
                                            "excluded: rule4_prompt_too_long": 1, "absent": 1})
        self.assertAlmostEqual(out["weighted_mean_d"], 10.0)


if __name__ == "__main__":
    unittest.main()
