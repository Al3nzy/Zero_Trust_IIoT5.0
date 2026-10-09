#!/usr/bin/env python3
"""Stage `latency`: cost of one monitoring round (m flows of one device) on ONE CPU thread. Run it on an idle machine; on an edge board it gives the edge numbers.
Measures: LightGBM scoring of m flows, Mahalanobis scoring, e-detector update (round level and flow level) for reference sizes N, signed ledger append, and their sum.
Output: results_v3/latency.json (with the macros used by the manuscript). Timings are medians over repeated calls; p95 is reported as well."""
import os, sys, json, time, platform, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
from ztids import config as C
from ztids.data import make_split, Preprocessor, balance_train, CLASS_ID
from ztids.core import lgbm
from ztids.novelty import NormalProfile
from ztids.edetector import OnlineConformal, bet_array, run_edetector
from ztids.ledger import Ledger
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

M = 20; REP = 100 if C.QUICK else 400


def timeit(fn, rep=REP):
    t = []
    for _ in range(rep):
        a = time.perf_counter(); fn(); t.append((time.perf_counter() - a) * 1e3)
    return dict(median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)))


def cpu():
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"): return l.split(":", 1)[1].strip()
    except OSError: pass
    return platform.processor() or platform.machine()


fit, val, test, _ = make_split("official", 0); pp = Preprocessor(k=25, seed=0, sel_mode="ovr").fit(fit)
yf, yv = fit["cls"].map(CLASS_ID).values, val["cls"].map(CLASS_ID).values; Xb, yb = balance_train(pp.transform(fit), yf, C.NPC, 0); clf = lgbm(0).fit(Xb, yb)
Af, Av = pp.transform_all(fit), pp.transform_all(val); det = NormalProfile(fpr=0.02, kind="mahalanobis", seed=0).fit(Af[yf == 0], Av[yv == 0])
X = pp.transform(test.iloc[:M]); A = pp.transform_all(test.iloc[:M]); out = dict(cpu=cpu(), threads=1, flows_per_round=M, repetitions=REP)
out["lightgbm_batch_m"] = timeit(lambda: clf.predict_proba(X)); out["lightgbm_single_flow"] = timeit(lambda: clf.predict_proba(X[:1]))
out["mahalanobis_batch_m"] = timeit(lambda: det.score(A)); rng = np.random.RandomState(0)
for N in (100, 1000, 10000):
    oc = OnlineConformal(rng.beta(1, 12, N)); xs = rng.beta(1, 12, 100000); i = [0]
    def step():
        p = oc.pvalue(xs[i[0]], 0.5); i[0] += 1; return float(bet_array([p]).mean())
    out[f"edetector_round_N{N}"] = timeit(step, 2000)
    def step_flow():
        for _ in range(M):
            p = oc.pvalue(xs[i[0]], 0.5); i[0] += 1; bet_array([p])
    out[f"edetector_flow_N{N}"] = timeit(step_flow, 300)
key = Ed25519PrivateKey.generate(); L = Ledger(signer=key, anchor_every=50); out["ledger_append"] = timeit(lambda: L.append({"dev": "d1", "T": 0.93, "state": "Healthy"}), 1000)
oc = OnlineConformal(rng.beta(1, 12, 1000))
def e2e():
    clf.predict_proba(X); det.score(A); oc.pvalue(0.05, 0.5); L.append({"dev": "d1", "T": 0.93, "state": "Healthy"})
out["end_to_end_round_N1000"] = timeit(e2e, 300)
f = lambda d: f"{d['median_ms']:.2f}"
out["macros"] = dict(LatClf=f(out["lightgbm_batch_m"]), LatClfSingle=f(out["lightgbm_single_flow"]), LatMah=f(out["mahalanobis_batch_m"]), LatEdet=f(out["edetector_round_N1000"]), LatEdetBig=f(out["edetector_round_N10000"]),
                     LatEdetFlow=f(out["edetector_flow_N1000"]), LatLedger=f(out["ledger_append"]), LatEnd=f(out["end_to_end_round_N1000"]), LatEndP=f"{out['end_to_end_round_N1000']['p95_ms']:.2f}", LatCpu=out["cpu"].replace("&", "\\&"))
json.dump(out, open(os.path.join(C.RES, "latency.json"), "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict) and "median_ms" in v: print(f"{k:28s} median {v['median_ms']:.3f} ms  p95 {v['p95_ms']:.3f} ms")
print("cpu:", out["cpu"])
