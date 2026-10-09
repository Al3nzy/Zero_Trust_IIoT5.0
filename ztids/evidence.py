"""Loads the primary detector's test posteriors and Normal-profile flags (stage `core`) and builds the per-flow evidence rows [harm, p_normal, confidence]."""
import os
import numpy as np
from . import config as C
from .trust import to_evidence

W_U = 0.5            # severity of a flow flagged only by the novelty detector ('unknown' behaviour)
KIND = "mahalanobis"  # Normal-profile detector used for fusion and for the novelty flag in the evidence


def load(seed=0, kind=KIND, w_u=W_U):
    src = os.path.join(C.RES, f"core_s{seed}.npz")
    if not os.path.exists(src): raise SystemExit(f"{src} missing: run the `core` stage first")
    z = np.load(src, allow_pickle=True)            # own artefacts; string columns are stored as object arrays
    pt, yt = z["pt"].astype(float), z["yt"]
    seen = set(z["sub_fit"]) | set(z["sub_val"])                      # sub-types present in the original training records
    novel = (yt != 0) & ~np.isin(z["sub"], list(seen))
    flag = z[f"sc_{kind}_te"] > float(z[f"thr_{kind}"])
    return dict(pt=pt, yt=yt, svc=z["svc"], sub=z["sub"], novel=novel, flag=flag, EVc=to_evidence(pt), EVf=to_evidence(pt, flag, w_u), z=z)


def attack_pools(EV, yt, novel):
    p = {k: EV[yt == k] for k in (1, 2, 3, 4)}
    p["novel"] = EV[(yt != 0) & novel]; p["seen"] = EV[(yt != 0) & ~novel]
    return p
