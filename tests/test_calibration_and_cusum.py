import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ztids.calibrate import conformal_threshold, window_ok
from ztids.trust import design_horizon, run_rule


def test_split_conformal_false_alarm_rate_is_at_most_alpha():
    rng = np.random.RandomState(0); a = 0.05; n = 400; rates = []
    for _ in range(300):
        w = rng.normal(size=n); t = conformal_threshold(w, a); rates.append((rng.normal(size=4000) > t).mean())
    assert np.mean(rates) <= a + 0.004 and abs(np.std(rates) - math.sqrt(a * (1 - a) / (n + 2))) < 0.01


def test_window_size_rule():
    assert window_ok(100, 0.05) and not window_ok(10, 0.05)


def test_cusum_per_round_tail_holds_and_unbounded_horizon_claim_fails():
    rng = np.random.RandomState(1); m, k, N, mu = 20, 0.05, 3000, 0.5; h = math.log(100) / (8 * m * k); theta = 8 * m * k
    S = np.zeros(N); first = np.full(N, np.inf); above = 0
    for t in range(1, 1501):
        S = np.minimum(2 * h, np.maximum(0, S + rng.binomial(m, mu, N) / m - mu - k)); first[(S >= h) & np.isinf(first)] = t
        if t > 100: above += (S >= h).mean()
    assert above / 1400 <= math.exp(-theta * h) + 0.003            # per-round tail bound holds
    assert (first <= 1500).mean() > 5 * math.exp(-theta * h)       # the probability of an alarm over a long horizon is far above the per-round bound


def test_horizon_design_controls_false_alarms_over_the_horizon():
    rng = np.random.RandomState(2); m, k, N, mu, T, a = 20, 0.05, 2000, 0.5, 250, 0.05; h = design_horizon(mu, mu, k, m, a, T)
    S = np.zeros(N); fq = np.zeros(N, bool)
    for _ in range(T):
        S = np.minimum(2 * h, np.maximum(0, S + rng.binomial(m, mu, N) / m - mu - k)); fq |= S >= h
    assert fq.mean() <= a
