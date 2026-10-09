"""Tests of the conformal e-detector guarantees. Each test can fail if the implementation or the theory is wrong. Run: pytest -q tests"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats
from scipy.integrate import quad
from ztids.edetector import bet, run_edetector, run_edetector_flows, separable_delay_law, threshold_for, OnlineConformal

Q = "Quarantined"


def test_bet_is_a_density():
    assert abs(quad(bet, 0, 1, limit=200)[0] - 1.0) < 1e-3


def test_pvalues_are_uniform_and_independent_under_exchangeability():
    rng = np.random.RandomState(0); P = []
    for i in range(400):
        x = rng.pareto(1.5, 80); _, _, _, p = run_edetector(x, warm=10, c=1e300, cap_factor=1e6, freeze=False, seed=i); P.append(p[10:])
    P = np.array(P); assert stats.kstest(P.ravel(), "uniform").pvalue > 0.01
    lag = np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in P]); assert abs(lag) < 0.03


def test_pvalues_uniform_with_ties():
    rng = np.random.RandomState(1); P = []
    for i in range(400):
        x = rng.poisson(1.0, 80).astype(float); P.append(run_edetector(x, warm=10, c=1e300, cap_factor=1e6, freeze=False, seed=i)[3][10:])
    assert stats.kstest(np.array(P).ravel(), "uniform").pvalue > 0.01


def test_sr_statistic_has_expectation_t():
    """E[R_T] = T. Tested with the finite-variance bet eps = 0.9 (the default mixture contains eps < 1/2, whose infinite variance makes Monte Carlo means of R_T converge far too slowly)."""
    rng = np.random.RandomState(2); T = 20; v = []
    for i in range(6000):
        _, _, logR, _ = run_edetector(rng.beta(2, 8, 10 + T), warm=10, c=1e300, cap_factor=1e6, freeze=False, seed=i, eps=(0.9,)); v.append(math.exp(logR[-1]))
    se = np.std(v) / math.sqrt(len(v)); assert abs(np.mean(v) - T) < 4 * se, (np.mean(v), se)


def test_horizon_false_quarantine_bound():
    rng = np.random.RandomState(3); T, a = 150, 0.1; c = threshold_for(T, a); n = 400; fa = 0
    for i in range(n): fa += (run_edetector(rng.beta(2, 8, 20 + T), warm=20, c=c, seed=i)[1][20:] == Q).any()
    assert fa / n <= a + 3 * math.sqrt(a * (1 - a) / n)


def test_freezing_and_cap_do_not_change_the_first_alarm():
    rng = np.random.RandomState(4)
    for i in range(30):
        x = np.r_[rng.beta(2, 8, 40), rng.beta(6, 2, 60)]
        a = np.where(run_edetector(x, warm=10, c=500, seed=i, freeze=True, cap_factor=2)[1] == Q)[0]; b = np.where(run_edetector(x, warm=10, c=500, seed=i, freeze=False, cap_factor=1e6)[1] == Q)[0]
        assert (len(a) == 0) == (len(b) == 0) and (len(a) == 0 or a[0] == b[0])


def test_delay_law_dominates_the_detector():
    rng = np.random.RandomState(5); s, c = 100, 1e3; law = separable_delay_law(s, c, n_mc=5000, kmax=200); sim = []
    for i in range(300):
        x = np.r_[rng.beta(2, 8, s) * 0.5, 0.6 + 0.4 * rng.beta(5, 2, 100)]; q = np.where(run_edetector(x, warm=50, c=c, seed=i)[1][s:] == Q)[0]; sim.append(q[0] + 1 if len(q) else 101)
    for q in (50, 90): assert np.percentile(sim, q) <= np.percentile(law, q) + 1


def test_mode_conditional_ranks_restore_validity_under_persistent_modes():
    rng = np.random.RandomState(6); T, a = 250, 0.1; c = threshold_for(T, a); n = 300; plain = cond = 0
    for i in range(n):
        st = 0; ctx = []; x = []
        for t in range(50 + T):
            if rng.rand() > 0.9: st = 1 - st
            ctx.append(st); x.append(rng.normal(0.0 if st == 0 else 3.0, 1.0))
        plain += (run_edetector(x, warm=50, c=c, seed=i)[1][50:] == Q).any(); cond += (run_edetector(x, warm=50, c=c, seed=i, context=ctx)[1][50:] == Q).any()
    assert cond / n <= a + 3 * math.sqrt(a * (1 - a) / n) and plain / n > cond / n


def test_flow_level_variant_controls_false_quarantine_for_iid_flows():
    rng = np.random.RandomState(7); T, a = 80, 0.1; c = threshold_for(T, a); n = 120; fa = 0
    for i in range(n): fa += (run_edetector_flows(rng.beta(1.2, 10, (20 + T, 10)), warm=20, c=c, seed=i)[1][20:] == Q).any()
    assert fa / n <= a + 3 * math.sqrt(a * (1 - a) / n)


def test_online_conformal_counts_ties():
    oc = OnlineConformal([1.0, 2.0, 2.0, 3.0]); p = oc.pvalue(2.0, 0.0, add=False); assert abs(p - (1 + 0.0 * 3) / 5) < 1e-12
