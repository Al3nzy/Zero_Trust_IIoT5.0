"""
Leave-one-attack-class-out test of the novelty fusion on any corpus. For each held-out attack class k: the supervised model is trained WITHOUT class k
(removed from the fit partition before rebalancing); the Normal-profile detectors never see attacks. We measure how often class k is detected by
the classifier alone vs fused with each detector, with two threshold calibrations:
  strict : (1-fpr) quantile of anomaly scores on training-domain validation Normal flows;
  site   : same quantile on a commissioning window = random 10% of clean test-domain Normal flows (excluded from evaluation), mean of 10 windows.
"""
import numpy as np
from lightgbm import LGBMClassifier
from .generic import balance
from .novelty import NormalProfile


def _fuse(P, flag, nidx):
    pred = P.argmax(1).copy(); m = np.asarray(flag, bool) & (pred == nidx)
    if m.any():
        Q = P[m].copy(); Q[:, nidx] = -1.0; pred[m] = Q.argmax(1)
    return pred


def _eval(pred, yt, nidx, k, keep=None):
    keep = np.ones(len(yt), bool) if keep is None else keep
    y, p = yt[keep], pred[keep]; ben = y == nidx; held = y == k; other = (~ben) & (~held)
    return dict(det_heldout=float((p[held] != nidx).mean()) if held.any() else None, fpr=float((p[ben] != nidx).mean()),
                det_other=float((p[other] != nidx).mean()) if other.any() else None)


def run_loo(d, d_all, classes, normal_idx, seed, npc, kinds=("iforest", "mahalanobis"), top=5, min_test=30, fpr=0.02, frac=0.10, draws=10):
    """d: prepare_generic output (25 selected features, used by the classifier); d_all: same split with all standardised features (used by the detectors)."""
    yt, yf = d["yte"], d["yfit"]; assert (yt == d_all["yte"]).all() and (yf == d_all["yfit"]).all(), "row order differs between the two preparations"
    scores, thr = {}, {}
    for kind in kinds:
        npf = NormalProfile(fpr=fpr, kind=kind, seed=seed).fit(d_all["Xfit"][yf == normal_idx], d_all["Xval"][d_all["yval"] == normal_idx])
        scores[kind] = npf.score(d_all["Xte"]); thr[kind] = npf.thr_
    cnt = {c: int((yt == c).sum()) for c in range(len(classes)) if c != normal_idx}
    held = [c for c, n in sorted(cnt.items(), key=lambda kv: -kv[1]) if n >= min_test and (yf == c).sum() > 0][:top]
    nrm = np.where(yt == normal_idx)[0]; out = {}
    for k in held:
        m = yf != k
        Xb, yb = balance(d["Xfit"][m], yf[m], npc, seed, len(classes))
        gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=seed, verbose=-1).fit(Xb, yb)
        P = np.zeros((len(yt), len(classes))); P[:, gb.classes_] = gb.predict_proba(d["Xte"])
        r = dict(n_test=cnt[k], alone=_eval(P.argmax(1), yt, normal_idx, k), strict={}, site={})
        for kind in kinds:
            r["strict"][kind] = _eval(_fuse(P, scores[kind] > thr[kind], normal_idx), yt, normal_idx, k)
            rows = []
            for rep in range(draws):
                rng = np.random.RandomState(1000 * seed + rep); com = rng.choice(nrm, max(1, int(frac * len(nrm))), replace=False)
                keep = np.ones(len(yt), bool); keep[com] = False; flag = scores[kind] > np.quantile(scores[kind][com], 1 - fpr)
                rows.append(_eval(_fuse(P, flag, normal_idx), yt, normal_idx, k, keep))
            r["site"][kind] = {key: float(np.mean([x[key] for x in rows if x[key] is not None])) for key in rows[0]}
        out[str(classes[k])] = r
    return out
