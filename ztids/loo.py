"""
Leave-one-attack-class-out test of the novelty fusion on any corpus, at MATCHED false-alarm rates.
For each held-out attack class k the supervised model is trained WITHOUT k (removed before rebalancing); the Normal-profile detectors never see attacks.
Three scorers are compared: classifier (score = 1 - P(Normal)), detector (anomaly score) and fused (max of Normal-referenced ranks). Every scorer's threshold is the
(1-alpha) quantile of ITS OWN scores on reference Normal flows, so all scorers run at the same benign false-alarm rate and detections are directly comparable;
AUROC (held-out vs Normal) is threshold-free. Two reference sets: 'strict' = training-domain validation Normal flows, 'site' = commissioning window
(random 10% of clean test-domain Normal flows, excluded from evaluation, mean of 10 windows).
(An argmax-based 'detected = predicted non-Normal' comparison is meaningless when the classifier already flags a large share of benign flows.)
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from .generic import balance
from .novelty import NormalProfile
from .calibrate import conformal_threshold

ALPHAS = (0.02, 0.05)


def _rank(ref_sorted, x): return np.searchsorted(ref_sorted, x) / len(ref_sorted)


def _scorers(sc_c, sc_a, ref_c, ref_a):
    """-> dict name -> (scores for all flows, scores for the reference flows)"""
    rc, ra = np.sort(ref_c), np.sort(ref_a)
    return {"clf": (sc_c, ref_c), "det": (sc_a, ref_a), "fused": (np.maximum(_rank(rc, sc_c), _rank(ra, sc_a)), np.maximum(_rank(rc, ref_c), _rank(ra, ref_a)))}


def _eval(S, yt, nidx, k, keep):
    y = yt[keep]; ben, held = y == nidx, y == k; other = (~ben) & (~held); out = {"auroc": {}, "alpha": {str(a): {} for a in ALPHAS}}
    m = ben | held
    for name, (sc, ref) in S.items():
        s = sc[keep]; out["auroc"][name] = float(roc_auc_score(held[m].astype(int), s[m]))
        for a in ALPHAS:
            thr = conformal_threshold(ref, a)
            if not np.isfinite(thr): out["alpha"][str(a)][name] = dict(det_heldout=None, fpr=None, det_other=None); continue          # reference window too small for this alpha
            out["alpha"][str(a)][name] = dict(det_heldout=float((s[held] > thr).mean()), fpr=float((s[ben] > thr).mean()), det_other=float((s[other] > thr).mean()) if other.any() else None)
    return out


def run_loo(d, d_all, classes, normal_idx, seed, npc, kinds=("iforest", "mahalanobis"), top=5, min_test=30, fpr=0.02, frac=0.10, draws=10):
    """d: prepare_generic output (selected features; classifier); d_all: same split with all standardised features (detectors)."""
    yt, yf, yv = d["yte"], d["yfit"], d["yval"]; assert (yt == d_all["yte"]).all() and (yf == d_all["yfit"]).all() and (yv == d_all["yval"]).all(), "row order differs"
    det = {}
    for kind in kinds:
        npf = NormalProfile(fpr=fpr, kind=kind, seed=seed).fit(d_all["Xfit"][yf == normal_idx], d_all["Xval"][yv == normal_idx]); det[kind] = (npf.score(d_all["Xte"]), npf.score(d_all["Xval"]))
    cnt = {c: int((yt == c).sum()) for c in range(len(classes)) if c != normal_idx}
    held = [c for c, n in sorted(cnt.items(), key=lambda kv: -kv[1]) if n >= min_test and (yf == c).sum() > 0][:top]
    nrm = np.where(yt == normal_idx)[0]; vn = np.where(yv == normal_idx)[0]; out = {}
    for k in held:
        m = yf != k; Xb, yb = balance(d["Xfit"][m], yf[m], npc, seed, len(classes))
        gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=seed, verbose=-1).fit(Xb, yb)
        def pa(X):
            P = np.zeros((len(X), len(classes))); P[:, gb.classes_] = gb.predict_proba(X); return 1 - P[:, normal_idx]
        ct, cv = pa(d["Xte"]), pa(d["Xval"]); r = dict(n_test=cnt[k], strict={}, site={})
        for kind in kinds:
            at, av = det[kind]
            r["strict"][kind] = _eval(_scorers(ct, at, cv[vn], av[vn]), yt, normal_idx, k, np.ones(len(yt), bool))
            rows = []
            for rep in range(draws):
                rng = np.random.RandomState(1000 * seed + rep); com = rng.choice(nrm, max(1, int(frac * len(nrm))), replace=False); keep = np.ones(len(yt), bool); keep[com] = False
                rows.append(_eval(_scorers(ct, at, ct[com], at[com]), yt, normal_idx, k, keep))
            avg = lambda f: (float(np.mean([f(x) for x in rows if f(x) is not None])) if any(f(x) is not None for x in rows) else None)
            r["site"][kind] = {"auroc": {n: avg(lambda x: x["auroc"][n]) for n in ("clf", "det", "fused")},
                               "alpha": {str(a): {n: {key: avg(lambda x: x["alpha"][str(a)][n][key]) for key in ("det_heldout", "fpr", "det_other")} for n in ("clf", "det", "fused")} for a in ALPHAS}}
        out[str(classes[k])] = r
    return out
