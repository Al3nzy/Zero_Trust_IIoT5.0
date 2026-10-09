"""Stage 2: Normal-profile novelty detectors fused with every supervised model and with the fixed hybrid, for every seed.
All thresholds are set on VALIDATION Normal flows only. Reports detection of attack sub-types seen in training vs test-only sub-types,
plus threshold-free test AUROC of each detector (ranking quality under distribution shift)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from ztids import config as C
from ztids.data import prepare, make_split, CLASSES
from ztids.novelty import NormalProfile, fuse_labels, KINDS
from ztids.ensemble import HYBRID, probs as ens_probs
from ztids.evalutil import metrics
from ztids.calibrate import eval_site_decision, conformal_threshold
MODELS = {"proposed CNN-BiLSTM": "official_cnn_bilstm_{s}_-_-_-_selovr", "LightGBM (class-aware MI)": "official_lgbm_{s}_sel_selovr",
          "LightGBM (global MI)": "official_lgbm_{s}_sel", "MLP": "official_mlp_{s}_-_-_-_selovr", HYBRID["label"]: None}
FPRS = {"iforest": (0.02, 0.05), "mahalanobis": (0.02,), "iforest+mahalanobis": (0.02,)}


def summ(pred, yt, att, novel):
    m = metrics(yt, np.eye(5)[pred]); pc = m["per_class"]
    return dict(acc=m["acc"], macro_f1=m["macro_f1"], mcc=m["mcc"], fpr=float((pred[~att] != 0).mean()), det_seen=float((pred[att & ~novel] != 0).mean()),
                det_novel=float((pred[att & novel] != 0).mean()), exact_novel=float((pred[att & novel] == yt[att & novel]).mean()), r2l=pc["R2L"]["r"], u2r=pc["U2R"]["r"])


def comm_eval(P, sc, yt, att, novel, fpr, s, frac=0.10, draws=20):
    """Site calibration: the threshold is the (1-fpr) quantile of anomaly scores on a COMMISSIONING window = a random `frac` of clean test-domain
    Normal flows, which are then EXCLUDED from the evaluation. Averaged over `draws` random windows. Assumes a clean benign period at the site."""
    A, F, N = [], [], []; nrm = np.where(yt == 0)[0]
    for rep in range(draws):
        rng = np.random.RandomState(1000 * s + rep); com = rng.choice(nrm, int(frac * len(nrm)), replace=False)
        keep = np.ones(len(yt), bool); keep[com] = False; flag = sc > conformal_threshold(sc[com], fpr)
        a_, k_ = att[keep], novel[keep]
        A.append(summ(P.argmax(1)[keep], yt[keep], a_, k_)); F.append(summ(fuse_labels(P[keep], flag[keep]), yt[keep], a_, k_))
        N.append(dict(fpr=float(flag[keep][~a_].mean()), det_seen=float(flag[keep][a_ & ~k_].mean()), det_novel=float(flag[keep][a_ & k_].mean())))
    avg = lambda L: {k: float(np.mean([x[k] for x in L])) for k in L[0]}
    return dict(alone=avg(A), fused=avg(F), novelty_only=avg(N))


out = {"site_decision": {}, "detectors": {k: {} for k in KINDS}, "auc": {k: {} for k in KINDS}, "commissioning": {k: {} for k in KINDS}, "seeds": C.SEEDS}
for s in C.SEEDS:
    fit, val, test, _ = make_split("official", s); seen = set(fit["sub"]); novel = ~test["sub"].isin(seen).values
    yt = test["cls"].map({c: i for i, c in enumerate(CLASSES)}).values; att = yt != 0
    d = prepare("official", s, n_per_class=C.NPC, k=500, sel_mode="global")          # all standardised (clipped) features, original unbalanced records
    for kind in KINDS:
        for fpr in FPRS[kind]:
            npf = NormalProfile(fpr=fpr, kind=kind, seed=s).fit(d["Xfit"][d["yfit"] == 0], d["Xval"][d["yval"] == 0]); sc = npf.score(d["Xte"]); flag = sc > npf.thr_
            tag = f"novelty_s{s}_f{fpr}.npz" if kind == "iforest" else f"novelty_{kind}_s{s}_f{fpr}.npz"          # iforest keeps the original name (read by ztids.evidence)
            np.savez_compressed(os.path.join(C.RES, tag), flag_te=flag, score_te=sc.astype(np.float32), thr=npf.thr_)
            R = out["detectors"][kind].setdefault(str(fpr), {})
            R.setdefault("novelty only", {})[str(s)] = dict(fpr=float(flag[~att].mean()), det_seen=float(flag[att & ~novel].mean()), det_novel=float(flag[att & novel].mean()))
            if fpr == 0.02:
                mn = ~att | (att & novel); out["auc"][kind][str(s)] = dict(all=float(roc_auc_score(att.astype(int), sc)), novel=float(roc_auc_score(att[mn].astype(int), sc[mn])))
            for label, pat in MODELS.items():
                if pat is None: P, y2 = ens_probs(HYBRID["members"], s)
                else:
                    p = os.path.join(C.RES, pat.format(s=s) + "_probs.npz"); P = np.load(p)["pt"] if os.path.exists(p) else None; y2 = np.load(p)["yt"] if P is not None else None
                if P is None: continue
                assert (y2 == yt).all(), "test order mismatch"
                R.setdefault(label, {})[str(s)] = dict(alone=summ(P.argmax(1), yt, att, novel), fused=summ(fuse_labels(P, flag), yt, att, novel))
                if kind == "iforest" and fpr == 0.02: out["site_decision"].setdefault(label, {})[str(s)] = eval_site_decision(P, yt, 0, s)
                if fpr == 0.02:
                    ce = comm_eval(P, sc, yt, att, novel, fpr, s); out["commissioning"][kind].setdefault(label, {})[str(s)] = dict(alone=ce["alone"], fused=ce["fused"])
                    out["commissioning"][kind].setdefault("novelty only", {})[str(s)] = ce["novelty_only"]
            r0 = R["novelty only"][str(s)]
            print(f"seed {s} {kind:22s} fpr {fpr}: novelty-only det seen/novel = {r0['det_seen']:.3f}/{r0['det_novel']:.3f}, benign flag rate {r0['fpr']:.3f}", flush=True)
out["fprs"] = out["detectors"]["iforest"]                                            # default detector (a-priori choice), used by tables/trust
out["n_attack_test"] = int(att.sum()); out["n_attack_novel"] = int((att & novel).sum())
json.dump(out, open(os.path.join(C.RES, "novelty.json"), "w"), indent=1)
