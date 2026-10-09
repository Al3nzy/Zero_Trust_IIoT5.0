#!/usr/bin/env python3
"""Stage `site`: site-calibrated decisions and Normal-profile fusion on the audited NSL-KDD protocol (from the `core` posteriors, no retraining).
Rules, all evaluated on the same flows (the commissioning window is excluded; mean over `DRAWS` random windows per seed):
  argmax            classifier argmax
  site(alpha)       Normal iff 1-p0 <= conformal threshold of the window at level alpha                         (Eq. conformal / decision)
  fused-train       argmax OR detector flag, detector threshold = 98th percentile of training-domain validation Normal flows (no site data)
  fused-site(alpha) Algorithm 1: classifier threshold and detector threshold both calibrated on the window, each at alpha/2, so that the benign false-alarm
                    rate is <= alpha by the union bound; a flow flagged only by the detector gets the most probable attack class.
Output: results_v3/site.json (per seed) with seed-level mean, sd and 95% t-intervals; paired differences with flow-bootstrap intervals for seed 0."""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from ztids import config as C
from ztids.calibrate import conformal_threshold, window_ok
from ztids.novelty import fuse_labels

SEEDS = [int(s) for s in C.CORE_SEEDS.split(",")]
ALPHAS = (0.02, 0.05, 0.10); KINDS = ("mahalanobis", "iforest", "iforest+mahalanobis"); DRAWS = 20; FRAC = 0.10


def summ(pred, y, novel):
    att = y != 0; rec = lambda c: float((pred[y == c] == c).mean()) if (y == c).any() else float("nan")
    return dict(acc=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, labels=range(5), average="macro", zero_division=0)), mcc=float(matthews_corrcoef(y, pred)),
                fpr=float((pred[~att] != 0).mean()), attack_det=float((pred[att] != 0).mean()), det_unseen=float((pred[att & novel] != 0).mean()),
                det_seen=float((pred[att & ~novel] != 0).mean()), r2l=rec(3), u2r=rec(4))


def site_pred(P, sa, tau_c, tau_a=None):
    Q = P.copy(); Q[:, 0] = -1.0; att_lab = Q.argmax(1)
    pred = np.where((1 - P[:, 0]) > tau_c, att_lab, 0)
    if tau_a is not None: pred = np.where((pred == 0) & (sa > tau_a), att_lab, pred)
    return pred


def run_seed(s):
    z = np.load(os.path.join(C.RES, f"core_s{s}.npz"), allow_pickle=True); P, yt = z["pt"].astype(float), z["yt"]; seen = set(z["sub_fit"]) | set(z["sub_val"])
    novel = (yt != 0) & ~np.isin(z["sub"], list(seen)); nrm = np.where(yt == 0)[0]; pa = 1 - P[:, 0]; out = {}
    out["argmax"] = summ(P.argmax(1), yt, novel)
    for kind in KINDS:
        sa = z[f"sc_{kind}_te"].astype(float); flag = sa > float(z[f"thr_{kind}"]); out[f"fused-train|{kind}"] = summ(fuse_labels(P, flag), yt, novel)
    acc = {}
    for d in range(DRAWS):
        rng = np.random.RandomState(1000 * s + d); com = rng.choice(nrm, int(FRAC * len(nrm)), replace=False); keep = np.ones(len(yt), bool); keep[com] = False
        yk, nk = yt[keep], novel[keep]
        acc.setdefault("argmax|excl", []).append(summ(P.argmax(1)[keep], yk, nk))
        for a in ALPHAS:
            if not window_ok(len(com), a): continue
            tc = conformal_threshold(pa[com], a); acc.setdefault(f"site|{a}", []).append(summ(site_pred(P, None, tc)[keep], yk, nk))
            if window_ok(len(com), a / 2):
                for kind in KINDS:
                    sa = z[f"sc_{kind}_te"].astype(float); tc2, ta2 = conformal_threshold(pa[com], a / 2), conformal_threshold(sa[com], a / 2)
                    acc.setdefault(f"fused-site|{a}|{kind}", []).append(summ(site_pred(P, sa, tc2, ta2)[keep], yk, nk))
    for k, L in acc.items(): out[k] = {m: float(np.nanmean([r[m] for r in L])) for m in L[0]}
    return out


def tci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan"); return dict(mean=float(m), sd=float(v.std(ddof=1)) if n > 1 else 0.0, lo=float(m - h), hi=float(m + h), n=n)


if __name__ == "__main__":
    per = {s: run_seed(s) for s in SEEDS}; R = {"seeds": SEEDS, "per_seed": per, "summary": {}, "paired": {}}
    rules = list(per[SEEDS[0]].keys())
    for r in rules: R["summary"][r] = {m: tci([per[s][r][m] for s in SEEDS]) for m in per[SEEDS[0]][r]}
    for a, b in (("site|0.05", "argmax|excl"), ("fused-site|0.05|mahalanobis", "argmax|excl"), ("fused-site|0.05|mahalanobis", "site|0.05"), ("fused-train|mahalanobis", "argmax"), ("fused-site|0.1|mahalanobis", "site|0.1")):
        if a in per[SEEDS[0]] and b in per[SEEDS[0]]:
            R["paired"][f"{a} - {b}"] = {m: tci([per[s][a][m] - per[s][b][m] for s in SEEDS]) for m in ("acc", "macro_f1", "mcc", "fpr", "attack_det", "det_unseen")}
    # flow-bootstrap interval of the macro-F1 gain of fusion over argmax for seed 0 (test flows resampled; the classifier is fixed)
    z = np.load(os.path.join(C.RES, f"core_s{SEEDS[0]}.npz"), allow_pickle=True); P, yt = z["pt"].astype(float), z["yt"]
    flag = z["sc_mahalanobis_te"].astype(float) > float(z["thr_mahalanobis"]); p0, p1 = P.argmax(1), fuse_labels(P, flag); rng = np.random.RandomState(0); d = []
    for _ in range(1000):
        i = rng.randint(0, len(yt), len(yt)); d.append(f1_score(yt[i], p1[i], labels=range(5), average="macro", zero_division=0) - f1_score(yt[i], p0[i], labels=range(5), average="macro", zero_division=0))
    R["bootstrap_fusion_gain_macro_f1_seed0"] = dict(mean=float(np.mean(d)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), B=1000)
    # calibration check: measured benign false-alarm rate against the target (seed-level)
    R["calibration"] = {str(a): tci([per[s][f"site|{a}"]["fpr"] for s in SEEDS]) for a in ALPHAS}
    json.dump(R, open(os.path.join(C.RES, "site.json"), "w"), indent=1)
    for r in ("argmax", "site|0.05", "fused-train|mahalanobis", "fused-site|0.05|mahalanobis"):
        s_ = R["summary"][r]; print(f"{r:34s} acc {s_['acc']['mean']:.3f} mF1 {s_['macro_f1']['mean']:.3f}±{s_['macro_f1']['sd']:.3f} fpr {s_['fpr']['mean']:.3f} det_unseen {s_['det_unseen']['mean']:.3f}")
    print("bootstrap fusion gain:", R["bootstrap_fusion_gain_macro_f1_seed0"])
