"""
Conformal e-detector for device trust (anytime-valid, distribution-free).

Per device and round t the evidence x_t in [0,1] (mean severity-weighted harm of the m flows of the round, Eq. (evidence) of the paper) is turned into
  1. an ONLINE SMOOTHED CONFORMAL p-value against the device's own history (commissioning rounds + every earlier round):
        p_t = ( #{i<=t : x_i > x_t} + theta_t * #{i<=t : x_i = x_t} ) / t ,   theta_t ~ U(0,1) i.i.d.
     If the benign rounds of the device are exchangeable with its commissioning rounds, p_1, p_2, ... are i.i.d. U(0,1) whatever the score
     distribution is (Vovk, Nouretdinov & Gammerman 2003; Vovk, Gammerman & Shafer 2005, Thm 8.2).
  2. a betting function f with integral 1 on (0,1) (here an equal-weight mixture of f_eps(p) = eps * p^(eps-1)), e_t = f(p_t), E[e_t | past] = 1;
  3. the Shiryaev-Roberts e-detector  R_t = (1 + R_{t-1}) e_t,  R_0 = 0  (Shin, Ramdas & Rinaldo 2024), capped at cap_factor*c for bounded recovery.
Guarantees (proved in the paper, Section III-E; checked numerically by tests/test_edetector.py and run_edetector.py):
  (a) R_t - t is a martingale, so the expected time to a false quarantine at threshold c is >= c (average run length);
  (b) R_t is a nonnegative submartingale, so P(false quarantine within T rounds) <= T / c   (Doob's maximal inequality);
  (c) if the first K attack rounds all score above all but j of the earlier s rounds, the device is quarantined within K*(s, j, c) rounds, where K* is the
      smallest K with  sum_{k<=K} log f((j+k)/(s+k)) >= log c   (deterministic; `delay_bound`).
The cap does not change the first passage time of R_t above c (the capped and uncapped paths coincide until then), so (a)-(b) hold for the capped detector.
"""
from __future__ import annotations
import math
from bisect import bisect_left, bisect_right, insort
import numpy as np

H, D, Q = "Healthy", "Degraded", "Quarantined"
EPS_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def bet(p, eps=EPS_GRID):
    """Mixture betting function f(p) = mean_j eps_j p^(eps_j - 1): nonnegative, decreasing, integrates to 1 on (0,1), hence a valid e-value map."""
    p = max(float(p), 1e-12)
    return float(np.mean([e * p ** (e - 1.0) for e in eps]))


def log_bet(p, eps=EPS_GRID):
    return math.log(bet(p, eps))


def bet_array(p, eps=EPS_GRID):
    """Vectorised `bet` for an array of p-values."""
    p = np.maximum(np.asarray(p, float), 1e-12)
    return np.mean([e * p ** (e - 1.0) for e in eps], axis=0)


class OnlineConformal:
    """Growing reference set with O(log t) rank queries. `ref` is the commissioning window (assumed clean and exchangeable with later benign rounds)."""

    def __init__(self, ref=()):
        self.s = sorted(float(v) for v in ref)

    def __len__(self):
        return len(self.s)

    def pvalue(self, x, theta, add=True):
        """Smoothed conformal p-value of x against the current reference set (x counts as its own tie); x is then added to the reference unless add=False."""
        x = float(x)
        lo, hi = bisect_left(self.s, x), bisect_right(self.s, x)
        g, e, n = len(self.s) - hi, (hi - lo) + 1, len(self.s) + 1
        if add: insort(self.s, x)
        return (g + theta * e) / n


def run_edetector(E, ref_scores=None, warm=10, c=1e3, cap_factor=2.0, eps=EPS_GRID, seed=0, healthy_frac=0.25, release_frac=0.5, freeze=True, context=None):
    """E: (rounds,) or (rounds, >=1) evidence (first column = harm x_t). The first `warm` rounds are the commissioning window (clean by assumption).
    ref_scores: optional pooled reference scores (fleet variant), prepended to the device's own commissioning rounds.
    State machine on L_t = max(log R_t, 0) with threshold log c:  Quarantined when L_t >= log c; released when L_t <= release_frac*log c;
    Healthy when L_t <= healthy_frac*log c; Degraded otherwise. Trust T_t = max(0, 1 - L_t / log c).
    freeze=True: rounds observed while the device is Quarantined are not added to the reference, so a sustained attack cannot erase its own evidence. Before the
    first quarantine nothing is frozen, so the first-alarm time (the object of the guarantees) is identical with and without freezing.
    context: optional integer operating-mode label per round, known before the round is observed (Mondrian variant). Ranks are then taken inside the mode, which keeps
    the p-values i.i.d. uniform whenever scores are exchangeable within each mode and independent across modes, even if the mode sequence itself is persistent.
    Returns (T, state, logR, p), each of length rounds."""
    x = np.asarray(E, float); x = x if x.ndim == 1 else x[:, 0]
    rng = np.random.RandomState(seed); n = len(x)
    ref = list(x[:warm]) if ref_scores is None else list(map(float, ref_scores)) + list(x[:warm])
    if context is None: ocs = {0: OnlineConformal(ref)}; ctx = np.zeros(n, int)
    else:
        ctx = np.asarray(context, int); pool = [] if ref_scores is None else list(map(float, ref_scores))
        ocs = {k: OnlineConformal(pool + [x[i] for i in range(warm) if ctx[i] == k]) for k in set(ctx.tolist())}
    logc, logcap = math.log(c), math.log(c * cap_factor)
    T = np.ones(n); st = np.empty(n, dtype=object); p = np.full(n, np.nan); logR = np.full(n, -np.inf)
    L, state = -math.inf, H                                     # L = log R_t, R_0 = 0
    for t in range(n):
        if t >= warm:
            p[t] = ocs[ctx[t]].pvalue(x[t], rng.rand(), add=not (freeze and state == Q))
            prev = math.log1p(math.exp(L)) if L > -700 else 0.0   # log(1 + R_{t-1}); equals 0 when R_{t-1} = 0
            L = min(logcap, prev + log_bet(p[t], eps)); logR[t] = L
        Lp = max(L, 0.0); T[t] = max(0.0, 1.0 - Lp / logc)
        if state == Q:
            if Lp <= release_frac * logc: state = D if Lp > healthy_frac * logc else H
        else:
            state = Q if Lp >= logc else (H if Lp <= healthy_frac * logc else D)
        st[t] = state
    return T, st, logR, p


def threshold_for(horizon, alpha):
    """Smallest threshold c certifying P(false quarantine within `horizon` rounds) <= alpha (Doob: P <= horizon / c)."""
    return float(horizon) / float(alpha)


def delay_bound(s, j, c, eps=EPS_GRID, kmax=10000):
    """Deterministic detection-delay bound K*(s, j, c): smallest K with sum_{k<=K} log f((j+k)/(s+k)) >= log c, or None if not reached within kmax."""
    acc, lc = 0.0, math.log(c)
    for k in range(1, kmax + 1):
        acc += log_bet(min(1.0, (j + k) / (s + k)), eps)
        if acc >= lc: return k
    return None


def separable_delay_law(s, c, n_mc=20000, eps=EPS_GRID, kmax=400, seed=0):
    """Exact delay law of the change-point product e_{s+1}...e_{s+K} under SEPARABLE, EXCHANGEABLE attacks (attack scores exceed all s earlier scores and are
    exchangeable among themselves). Then p_{s+k} = (r_k + theta_k)/(s+k) with r_k uniform on {0..k-1}, independent over k (Renyi), i.e. p_{s+k} ~ U(0, k/(s+k))
    independently. The SR statistic is at least this product, so its alarm time is stochastically no larger. Returns an array of first-passage rounds
    (kmax+1 where log c is not reached). Depends on (s, c, eps) only: no score distribution enters."""
    rng = np.random.RandomState(seed); lc = math.log(c)
    ks = np.arange(1, kmax + 1); a = ks / (s + ks)                                  # support upper end of p_{s+k}
    u = rng.rand(n_mc, kmax) * a[None, :]
    u = np.maximum(u, 1e-12)
    f = np.mean([e * u ** (e - 1.0) for e in eps], axis=0)                          # mixture bet, vectorised
    cum = np.cumsum(np.log(f), axis=1)
    hit = cum >= lc
    first = np.where(hit.any(1), hit.argmax(1) + 1, kmax + 1)
    return first


def expected_logwealth(s, K, eps=0.5):
    """Exact E[log prod_{k<=K} f_eps(p_{s+k})] for the single-eps bet under the separable model: K (log eps + 1 - eps) + (1 - eps) log C(s+K, K)."""
    return K * (math.log(eps) + 1 - eps) + (1 - eps) * (math.lgamma(s + K + 1) - math.lgamma(K + 1) - math.lgamma(s + 1))


def run_edetector_flows(Fh, warm=10, c=1e3, cap_factor=2.0, eps=EPS_GRID, seed=0, healthy_frac=0.25, release_frac=0.5, freeze=True, ref_scores=None):
    """Flow-level variant. Fh: (rounds, m) per-flow harm. Flows are processed sequentially against a growing reference that starts with the flows of the `warm`
    commissioning rounds; the round e-value is the mean of the m flow e-values, a valid conditional e-value if the flows of the device are exchangeable.
    Resolution is m times finer than the round-level variant, but within-round dependence (bursts) is not tolerated: see the correlated-flow stress test."""
    Fh = np.asarray(Fh, float); n, m = Fh.shape; rng = np.random.RandomState(seed)
    ref = list(Fh[:warm].ravel()) if ref_scores is None else list(map(float, ref_scores)) + list(Fh[:warm].ravel())
    oc = OnlineConformal(ref); logc, logcap = math.log(c), math.log(c * cap_factor)
    T = np.ones(n); st = np.empty(n, dtype=object); p = np.full((n, m), np.nan); logR = np.full(n, -np.inf)
    L, state = -math.inf, H
    for t in range(n):
        if t >= warm:
            add = not (freeze and state == Q); th = rng.rand(m)
            for j in range(m): p[t, j] = oc.pvalue(Fh[t, j], th[j], add=add)
            prev = math.log1p(math.exp(L)) if L > -700 else 0.0
            L = min(logcap, prev + math.log(float(bet_array(p[t], eps).mean()))); logR[t] = L
        Lp = max(L, 0.0); T[t] = max(0.0, 1.0 - Lp / logc)
        if state == Q:
            if Lp <= release_frac * logc: state = D if Lp > healthy_frac * logc else H
        else:
            state = Q if Lp >= logc else (H if Lp <= healthy_frac * logc else D)
        st[t] = state
    return T, st, logR, p
