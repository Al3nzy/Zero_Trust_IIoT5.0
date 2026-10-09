#!/usr/bin/env python3
"""Stage `sensitivity`: sensitivity of the round-level conformal e-detector to commissioning length W, false-alarm level alpha (c = 250/alpha), flows per round m and the betting function.
Heterogeneous fleet, device reference, iid benign traffic. Output: results_v3/sensitivity.json"""
import os, sys, json, math, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
from ztids import config as C
from ztids import evidence as EV_
from ztids.devices import Pools, make_fleet
from ztids.edetector import run_edetector, EPS_GRID

ND = 40 if C.QUICK else 150; HOR = 250; Q = "Quarantined"
E_ = EV_.load(0); P = Pools(E_["EVf"], E_["yt"], E_["svc"], E_["novel"])


def cell(W, alpha, m, eps):
    rounds, onset = W + HOR, W + 50; c = HOR / alpha; res = {}
    for scn, kw in (("benign", {}), ("compromised", {}), ("stealth", dict(frac=0.3))):
        fleet, _ = make_fleet(P, ND, scn, 17, rounds, m, True, onset=onset, **kw); S = [run_edetector(F.mean(1)[:, 0], warm=W, c=c, eps=eps, seed=i)[1] for i, F in enumerate(fleet)]
        if scn == "benign": res["false_quarantine"] = float(np.mean([(s[W:] == Q).any() for s in S]))
        else:
            dl = [np.where(s[onset:] == Q)[0][0] + 1 if (s[onset:] == Q).any() else np.nan for s in S]; res[f"detect_{scn}"] = float(np.mean(~np.isnan(dl)))
            if scn == "compromised": res["median_delay"] = None if np.all(np.isnan(dl)) else float(np.nanmedian(dl))
    return dict(W=W, alpha=alpha, m=m, eps=list(eps), c=c, **res)


rows = []; t0 = time.time()
for W in (10, 20, 50, 100):
    for alpha in (0.1, 0.05, 0.01): rows.append(cell(W, alpha, 20, EPS_GRID)); print(rows[-1], flush=True)
for m in (10, 50): rows.append(cell(50, 0.05, m, EPS_GRID)); print(rows[-1], flush=True)
for eps in ((0.1,), (0.5,), (0.9,)): rows.append(cell(50, 0.05, 20, eps)); print(rows[-1], flush=True)
json.dump(dict(n_devices=ND, horizon=HOR, rows=rows), open(os.path.join(C.RES, "sensitivity.json"), "w"), indent=1)
print(f"sensitivity done in {time.time() - t0:.0f}s")
