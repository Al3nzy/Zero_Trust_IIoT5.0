"""
Device-stream simulator that replays REAL classifier outputs as per-device flow streams.

A flow is a row [harm, p_normal, confidence] (harm = max(severity-weighted posterior, w_u * novelty flag)). A device emits `m` flows per round. Benign flows are
drawn from the benign pool of the device's service group; attack flows from the pool of one attack class (or of unseen / seen sub-types). Benign behaviour modes:
  iid     flows i.i.d. from the device's own pool (the setting of the guarantees),
  regime  two-state Markov chain between the device's pool and another group's pool (persistent, temporally dependent traffic),
  drift   the share of the other group's pool grows linearly over the horizon (non-stationary traffic),
  rep>1   every sampled flow is repeated `rep` times inside a round (correlated flows).
This is a replay simulation: temporal structure is injected by construction, it is not measured from real device traces.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class Pools:
    def __init__(self, EV, yt, svc, novel, n_groups=3):
        top = pd.Series(svc[yt == 0]).value_counts().index[:n_groups].tolist()
        gid = {s: i for i, s in enumerate(top)}
        self.group_names = top + ["other"]
        grp = np.array([gid.get(s, n_groups) for s in svc])
        self.benign = {g: EV[(yt == 0) & (grp == g)] for g in range(n_groups + 1)}
        self.benign_all = EV[yt == 0]
        self.attack = {k: EV[yt == k] for k in (1, 2, 3, 4)}
        self.attack["novel"] = EV[(yt != 0) & novel]; self.attack["seen"] = EV[(yt != 0) & ~novel]
        self.n_groups = n_groups + 1


def attack_fraction(scn, t, rng, onset, end, p_burst=0.3, frac=0.1):
    if scn == "benign": return 0.0
    if scn == "compromised": return 1.0 if t >= onset else 0.0
    if scn == "recovery": return 1.0 if onset <= t < end else 0.0
    if scn == "intermittent": return 1.0 if (t >= onset and rng.rand() < p_burst) else 0.0
    if scn == "stealth": return frac if t >= onset else 0.0
    raise ValueError(scn)


def make_device(P: Pools, group, scn, rng, rounds, m=20, onset=60, end=120, attack="any", frac=0.1, p_burst=0.3, mode="iid", rep=1, alt_group=None,
                persist=0.9, drift_max=0.5):
    """Returns F (rounds, m, 3) flow evidence, the per-round attacked fraction and the per-round operating-mode label (regime state; 0 for iid and drift)."""
    own = P.benign[group] if group is not None and group >= 0 else P.benign_all
    alt = P.benign[alt_group] if alt_group is not None else P.benign_all
    if attack == "any": attack = int(rng.choice([1, 2, 3, 4]))
    F = np.zeros((rounds, m, 3)); fr = np.zeros(rounds); state = 0; ctx = np.zeros(rounds, int)
    for t in range(rounds):
        a = attack_fraction(scn, t, rng, onset, end, p_burst, frac); na = rng.binomial(m, a); fr[t] = na / m; nb = m - na
        if mode == "regime":
            if rng.rand() > persist: state = 1 - state
            pool = alt if state == 1 else own; ctx[t] = state
        elif mode == "drift":
            pool = alt if rng.rand() < drift_max * t / max(rounds - 1, 1) else own
        else:
            pool = own
        rows = np.repeat(pool[rng.randint(0, len(pool), -(-nb // rep))], rep, axis=0)[:nb] if nb else np.zeros((0, 3))
        if na:
            ap = P.attack[attack]; rows = np.vstack([rows, np.repeat(ap[rng.randint(0, len(ap), -(-na // rep))], rep, axis=0)[:na]])
        F[t] = rows[rng.permutation(m)] if len(rows) == m else rows
    return F, fr, ctx


def make_fleet(P: Pools, n, scn, seed, rounds, m=20, hetero=True, **kw):
    """n devices; heterogeneous fleets assign each device a service group (and another group for regime/drift modes)."""
    rng = np.random.RandomState(seed); fleet = []; ctxs = []
    for _ in range(n):
        g = int(rng.randint(0, P.n_groups)) if hetero else -1
        alt = int(rng.choice([x for x in range(P.n_groups) if x != g])) if hetero else None
        F, _, ctx = make_device(P, g, scn, rng, rounds, m, alt_group=alt, **kw); fleet.append(F); ctxs.append(ctx)
    return fleet, ctxs
