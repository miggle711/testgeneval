"""Cross-check confirmatory_analysis.py's tests against scipy, on random data.

A one-off validation, not part of the analysis: scipy is not a dependency of
the analysis. Run from the repository root in any environment with scipy
(results on 18 Nov with scipy 1.15.3 are on fyp-paper#7):
    python3 scripts/check_confirmatory_vs_scipy.py
"""
import random, sys, time
import numpy as np
from scipy import stats
sys.path.insert(0, "scripts")
import confirmatory_analysis as ca

def data(rnd, n, tied):
    shift = rnd.uniform(-0.3, 0.6)
    return [round(rnd.gauss(shift, 1) * 5) / 5 if tied else rnd.gauss(shift, 1) for _ in range(n)]

def run(label, gen, ref, trials):
    rnd = random.Random(1)
    worst, done, t0 = 0.0, 0, time.time()
    for _ in range(trials):
        d = gen(rnd)
        ours = ca.wilcoxon(d)
        if ours["n"] == 0:
            continue
        worst = max(worst, abs(ours["p"] - ref(np.array(d)))); done += 1
    print(f"{label:44} cases {done:4}  max |ours - scipy| = {worst:.1e}  ({time.time() - t0:.0f}s)", flush=True)

print(f"scipy {stats.__name__} {__import__('scipy').__version__}", flush=True)
run("Wilcoxon exact, no ties, n 5-50", lambda r: data(r, r.randint(5, 50), False),
    lambda a: stats.wilcoxon(a, zero_method="wilcox", method="exact").pvalue, 300)
run("Wilcoxon exact, ties, n 5-16 (full enumeration)", lambda r: data(r, r.randint(5, 16), True),
    lambda a: stats.wilcoxon(a, zero_method="wilcox", method=stats.PermutationMethod(n_resamples=np.inf)).pvalue, 150)
run("Wilcoxon normal approx, ties, n 60-500", lambda r: data(r, r.randint(60, 500), True),
    lambda a: stats.wilcoxon(a, zero_method="wilcox", correction=True, method="asymptotic").pvalue, 300)
run("Wilcoxon normal approx, no ties, n 60-500", lambda r: data(r, r.randint(60, 500), False),
    lambda a: stats.wilcoxon(a, zero_method="wilcox", correction=True, method="asymptotic").pvalue, 300)
rnd, worst = random.Random(2), 0.0
for _ in range(500):
    b, c = rnd.randint(0, 80), rnd.randint(0, 80)
    if b + c:
        worst = max(worst, abs(ca.mcnemar_exact(b, c) - stats.binomtest(b, b + c, 0.5).pvalue))
print(f"{'McNemar exact (binomial test), b,c 0-80':44} cases  500  max |ours - scipy| = {worst:.1e}", flush=True)
