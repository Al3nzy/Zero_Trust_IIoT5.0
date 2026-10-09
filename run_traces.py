#!/usr/bin/env python3
"""Evaluate device trust on YOUR time-ordered flow records (real traces).  Input: CSV with one row per flow and the columns
    device   device identifier (e.g. source IP)           time   sortable timestamp           harm   per-flow harm in [0,1] from any detector
    label    optional, 0 benign / 1 attack (used only to score the result)
Harm can be exported from this repository (ztids.trust.to_evidence: severity-weighted posterior, plus the novelty weight) or from any other detector.
Flows of a device are cut into rounds of `--m` consecutive flows. The first `--warm` rounds of each device are its commissioning window.
Rules: CUSUM (horizon design, device baseline) and the conformal e-detector (device reference, c = horizon/alpha).  A device is 'compromised' if label is given and any flow is an attack;
its onset is the first round with an attack flow.  Output: per-device table (csv) and a summary (json) with false-quarantine rate on benign devices and delay on compromised ones.
    python run_traces.py traces.csv --m 20 --warm 50 --alpha 0.05 --out results_v3/traces
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ztids.edetector import run_edetector, threshold_for
from ztids.trust import run_rule, design_horizon

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--m", type=int, default=20); ap.add_argument("--warm", type=int, default=50); ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--kappa", type=float, default=0.05); ap.add_argument("--max-horizon", type=int, default=250); ap.add_argument("--out", default="results_v3/traces"); a = ap.parse_args()
df = pd.read_csv(a.csv); need = {"device", "time", "harm"}; assert need <= set(df.columns), f"missing columns {need - set(df.columns)}"
has_lab = "label" in df.columns; df = df.sort_values(["device", "time"]); rows = []
for dev, g in df.groupby("device", sort=False):
    n = len(g) // a.m
    if n < a.warm + 10: continue
    h = g["harm"].values[: n * a.m].reshape(n, a.m); x = h.mean(1); lab = g["label"].values[: n * a.m].reshape(n, a.m).max(1) if has_lab else np.zeros(n); onset = int(np.argmax(lab > 0)) if (lab > 0).any() else None
    T = min(n - a.warm, a.max_horizon); c = threshold_for(T, a.alpha); st_e = run_edetector(x, warm=a.warm, c=c, seed=0)[1]
    E = np.c_[x, 1 - x, np.ones(n)]; hh = design_horizon(float(x[:a.warm].mean()), float(x[:a.warm].mean()), a.kappa, a.m, a.alpha, T); st_c = run_rule(E, "cusum", mu0=float(x[:a.warm].mean()), k=a.kappa, h=hh, warm=a.warm)[1]
    first = lambda st, lo: (int(np.argmax(np.asarray(st[lo:]) == "Quarantined")) + lo if (np.asarray(st[lo:]) == "Quarantined").any() else None)
    for name, st in (("e-detector", st_e), ("cusum", st_c)):
        f = first(st, a.warm); rows.append(dict(device=dev, rule=name, rounds=n, compromised=onset is not None, onset_round=onset, first_alarm_round=f,
                                                  false_quarantine=bool(onset is None and f is not None), detected=bool(onset is not None and f is not None and f >= onset),
                                                  delay=(f - onset + 1) if (onset is not None and f is not None and f >= onset) else None))
R = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True); R.to_csv(os.path.join(a.out, "per_device.csv"), index=False); S = {}
for rule, g in R.groupby("rule"):
    b, c_ = g[~g.compromised], g[g.compromised]
    S[rule] = dict(benign_devices=int(len(b)), false_quarantine_rate=float(b.false_quarantine.mean()) if len(b) else None, compromised_devices=int(len(c_)),
                   detection_rate=float(c_.detected.mean()) if len(c_) else None, median_delay_rounds=float(c_.delay.dropna().median()) if c_.delay.notna().any() else None)
json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1); print(json.dumps(S, indent=1)); print("per-device table:", os.path.join(a.out, "per_device.csv"))
