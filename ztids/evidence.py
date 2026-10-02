"""Loads the proposed detector's test posteriors (+ optional novelty flag) and builds the per-flow trust evidence pools."""
import os
import numpy as np
from . import config as C
from .data import make_split, CLASSES
from .trust import to_evidence

W_U = 0.5            # severity policy for flows flagged only by the novelty detector ('unknown' behaviour)
FPR = 0.02           # novelty-detector false-alarm target on validation Normal flows


def load(w_u=W_U, fpr=FPR, seed=0):
    src = os.path.join(C.RES, C.PROPOSED + "_probs.npz")
    if not os.path.exists(src): raise SystemExit("proposed-model posteriors missing; run the detector stage first")
    z = np.load(src); pt, yt = z["pt"], z["yt"]
    fit, _, test_df, _ = make_split("official", seed)
    assert len(test_df) == len(yt) and (test_df["cls"].map({c: i for i, c in enumerate(CLASSES)}).values == yt).all(), "test order mismatch"
    novel = ~test_df["sub"].isin(set(fit["sub"])).values
    nv = os.path.join(C.RES, f"novelty_s{seed}_f{fpr}.npz"); flag = np.load(nv)["flag_te"] if os.path.exists(nv) else None
    EVc = to_evidence(pt); EVf = to_evidence(pt, flag, w_u) if flag is not None else EVc
    return dict(pt=pt, yt=yt, flag=flag, EVc=EVc, EVf=EVf, svc=test_df["service"].values, novel=novel, test_df=test_df, has_novelty=flag is not None)


def attack_pools(EV, yt, novel):
    p = {k: EV[yt == k] for k in (1, 2, 3, 4)}
    p["novel"] = EV[(yt != 0) & novel]; p["seen"] = EV[(yt != 0) & ~novel]
    return p
