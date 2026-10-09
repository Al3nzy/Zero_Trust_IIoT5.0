#!/usr/bin/env python3
"""Stage `core`: audited NSL-KDD protocol with the primary detector (LightGBM) and the three Normal-profile detectors, for many seeds.
Per seed it stores everything the downstream stages need in results_v3/core_s<seed>.npz (test/validation posteriors, anomaly scores and validation
thresholds, labels, sub-types, services) so that no later stage retrains anything. LightGBM runs deterministically on one thread.
    python run_core.py                 # seeds 0..9 (ZTIDS_SEEDS=0,1,2 to change)
"""
import os, sys, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
from ztids import config as C
from ztids.data import make_split, Preprocessor, balance_train, CLASS_ID, CLASSES
from ztids.novelty import NormalProfile, KINDS
from ztids.evalutil import metrics
from ztids.core import lgbm

SEEDS = [int(s) for s in C.CORE_SEEDS.split(",") if s != ""]
FPR = float(os.environ.get("ZTIDS_NOVELTY_FPR", 0.02))


def run_seed(s):
    path = os.path.join(C.RES, f"core_s{s}.npz")
    if os.path.exists(path): return
    t0 = time.time(); print(f"[core] seed {s}: audited split, feature selection and training (about 1 minute per seed, silent while it runs; please wait)...", flush=True)
    fit, val, test, audit = make_split("official", s)
    pp = Preprocessor(k=25, seed=s, sel_mode="ovr").fit(fit)
    yf, yv, yt = (d["cls"].map(CLASS_ID).values for d in (fit, val, test))
    Xf, Xv, Xt = pp.transform(fit), pp.transform(val), pp.transform(test)
    Xb, yb = balance_train(Xf, yf, C.NPC, s)
    t1 = time.time(); m = lgbm(s).fit(Xb, yb); pt, pv = m.predict_proba(Xt), m.predict_proba(Xv); t_fit = time.time() - t1
    Af, Av, At = pp.transform_all(fit), pp.transform_all(val), pp.transform_all(test)
    out = dict(pt=pt.astype(np.float32), pv=pv.astype(np.float32), yt=yt, yv=yv, yf=yf, svc=test["service"].values.astype(str), sub=test["sub"].values.astype(str),
               sub_val=val["sub"].values.astype(str), sub_fit=fit["sub"].values.astype(str), svc_val=val["service"].values.astype(str))
    for kind in KINDS:
        npf = NormalProfile(fpr=FPR, kind=kind, seed=s).fit(Af[yf == 0], Av[yv == 0])
        out[f"sc_{kind}_te"] = npf.score(At).astype(np.float32); out[f"sc_{kind}_val"] = npf.score(Av).astype(np.float32); out[f"thr_{kind}"] = np.float64(npf.thr_)
    np.savez_compressed(path, **out)
    r = metrics(yt, pt)
    json.dump(dict(seed=s, test=r, audit={k: v for k, v in audit.items()}, features=list(pp.names_), lgbm_fit_s=round(t_fit, 2), total_s=round(time.time() - t0, 1)),
              open(os.path.join(C.RES, f"core_s{s}.json"), "w"))
    print(f"[core] seed {s}: acc={r['acc']:.4f} macroF1={r['macro_f1']:.4f} U2R={r['per_class']['U2R']['r']:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    print(f"core stage | seeds={SEEDS} results -> {C.RES}", flush=True)
    for s in SEEDS: run_seed(s)
