"""
Class-aware, baseline-relative sequential trust (CUSUM) with provable false-quarantine and detection-delay bounds.

Per device and round t, m flows are classified; the round evidence is the mean severity-weighted harm
        x_t = (1/m) sum_j  w^T p_j  in [0,1]            (w = severity policy table, w_Normal = 0)
A reference level mu0 is estimated from a commissioning window (fleet median of per-device means, or the device's own
mean for the per-device variant). The score is the one-sided CUSUM, capped at cap*h so that recovery time is bounded:
        S_t = min(cap*h, max(0, S_{t-1} + x_t - mu0 - k)),     T_t = clip(1 - S_t/h, 0, 1).
Guarantees that hold (flows within a round treated as m_eff independent, bounded in [0,1]):
  Hoeffding's lemma: E exp(theta (x - E x)) <= exp(theta^2 / (8 m_eff)). A benign device with E x = mu_b has increment drift -delta,
  delta = k + mu0 - mu_b > 0; with theta* = 8 m_eff delta, the reflected walk satisfies for EVERY round t (Kingman-type bound, the cap only lowers S)
        P(S_t >= h) <= exp(-8 m_eff delta h)                       (per-round tail)
        P(S_t >= h for some t <= T) <= T exp(-8 m_eff delta h)      (horizon T, union bound over the start of the excursion).
  The probability of at least one alarm grows with the horizon and tends to one when T -> infinity (the walk is reflected at zero), so a design
  that certifies only the per-round tail does not control false quarantine over a horizon: use `design_horizon`, or the conformal e-detector (edetector.py),
  whose guarantees are time-uniform.
  A compromised device with E x = mu_a > mu0 + k, drift d = mu_a - mu0 - k: Wald's identity with overshoot <= 1 gives
        E[time to quarantine] <= (h + 1) / d.
  Minimum detectable attack fraction for rounds with a fraction f of malicious flows (mean harm mu_att):
        f* = (mu0 + k - mu_b) / (mu_att - mu_b).
Also contains the two reference rules that are compared against: the original confidence-only rule and a Beta reputation.
"""
import numpy as np

SEVERITY = np.array([0.0, 0.6, 0.4, 0.8, 1.0])      # Normal, DoS, Probe, R2L, U2R (policy table, configurable)
H, D, Q = "Healthy", "Degraded", "Quarantined"


def to_evidence(P, flag=None, w_u=0.5):
    """Per-flow evidence rows [harm, p_normal, confidence] from posteriors (and an optional novelty flag, see ztids.novelty)."""
    from .novelty import evidence_rows
    return evidence_rows(P, flag, w_u)


def gen_evidence(Pn, Pa, scenario, rng, rounds=60, m=20, onset=15, end=30, attack_cls=None, p_burst=0.3, frac=0.1, rep=1):
    """Replay real classifier posteriors as a device stream. Returns per-round [h, g, c] and the attacked-flow fraction.
    Pn: evidence rows [harm, p_normal, conf] of benign flows; Pa: dict key -> evidence rows of that attack pool; rep>1 duplicates flows (correlated)."""
    cls = attack_cls if attack_cls is not None else int(rng.choice([k for k in Pa if isinstance(k, (int, np.integer))]))
    out = np.zeros((rounds, 3)); fr = np.zeros(rounds)
    for t in range(rounds):
        if scenario == "benign": a = 0.0
        elif scenario == "compromised": a = 1.0 if t >= onset else 0.0
        elif scenario == "recovery": a = 1.0 if onset <= t < end else 0.0
        elif scenario == "intermittent": a = 1.0 if (t >= onset and rng.rand() < p_burst) else 0.0
        elif scenario == "stealth": a = frac if t >= onset else 0.0
        else: raise ValueError(scenario)
        na = rng.binomial(m, a); fr[t] = na / m
        rows = [np.repeat(Pn[rng.randint(0, len(Pn), -(-(m - na) // rep))], rep, axis=0)[: m - na]]
        if na: rows.append(np.repeat(Pa[cls][rng.randint(0, len(Pa[cls]), -(-na // rep))], rep, axis=0)[:na])
        out[t] = np.vstack(rows).mean(0)
    return out, fr


def run_rule(E, rule, mu0=None, k=0.05, h=1.0, cap=2.0, warm=10, lam=0.9, kappa=3.0, th_q=0.35, th_r=0.6, th_h=0.75):
    """E: (rounds,3). Returns (T[rounds], state[rounds]) for one device."""
    T = np.zeros(len(E)); S = np.empty(len(E), dtype=object); st = D
    if rule == "cusum":
        s = 0.0
        for t in range(len(E)):
            if t >= warm: s = min(cap * h, max(0.0, s + E[t, 0] - mu0 - k))
            T[t] = max(0.0, 1 - s / h)
            if st == Q:
                if s <= 0.5 * h: st = D if s > 0.25 * h else H
            else: st = Q if s >= h else (H if s <= 0.25 * h else D)
            S[t] = st
    elif rule == "confidence":                          # original manuscript rule: T <- clip(T + (max softmax - 0.5) * 0.5)
        tr = 0.5
        for t in range(len(E)):
            tr = float(np.clip(tr + (E[t, 2] - 0.5) * 0.5, 0, 1)); T[t] = tr
            if st == Q:
                if tr >= th_r: st = H if tr >= th_h else D
            else: st = Q if tr < th_q else (H if tr >= th_h else D)
            S[t] = st
    elif rule == "beta":                                # Beta reputation with forgetting (fixed defaults, not tuned)
        a = b = 1.0
        for t in range(len(E)):
            a = lam * a + E[t, 1]; b = lam * b + kappa * E[t, 0]; tr = a / (a + b); T[t] = tr
            if st == Q:
                if tr >= th_r: st = H if tr >= th_h else D
            else: st = Q if tr < th_q else (H if tr >= th_h else D)
            S[t] = st
    else:
        raise ValueError(rule)
    return T, S


def design(mu_b, mu0, k, m_eff, alpha):
    """Smallest h with per-round tail P(S_t >= h) <= alpha for a device whose mean evidence is mu_b (NOT a bound over a horizon)."""
    delta = k + mu0 - mu_b
    return np.log(1 / alpha) / (8 * m_eff * delta) if delta > 0 else np.inf


def design_horizon(mu_b, mu0, k, m_eff, alpha, T):
    """Smallest h with P(at least one alarm within T rounds) <= alpha (union bound): h = ln(T/alpha) / (8 m_eff delta)."""
    delta = k + mu0 - mu_b
    return np.log(T / alpha) / (8 * m_eff * delta) if delta > 0 else np.inf
