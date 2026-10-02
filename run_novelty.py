"""Stage 2: Normal-profile novelty detector (isolation forest) fused with every supervised model, for every seed.
The anomaly threshold is set on VALIDATION Normal flows only. Reports detection of attack sub-types seen in training vs test-only sub-types."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ztids import config as C
from ztids.data import prepare, make_split, CLASSES
from ztids.novelty import NormalProfile, fuse_labels
from ztids.evalutil import metrics
MODELS = {"proposed CNN-BiLSTM": "official_cnn_bilstm_{s}_-_-_-_selovr", "LightGBM (class-aware MI)": "official_lgbm_{s}_sel_selovr",
          "LightGBM (global MI)": "official_lgbm_{s}_sel", "MLP": "official_mlp_{s}_-_-_-_selovr"}
FPRS = (0.02, 0.05)


def summ(pred, yt, att, novel):
    m = metrics(yt, np.eye(5)[pred]); pc = m["per_class"]
    return dict(acc=m["acc"], macro_f1=m["macro_f1"], mcc=m["mcc"], fpr=float((pred[~att] != 0).mean()), det_seen=float((pred[att & ~novel] != 0).mean()),
                det_novel=float((pred[att & novel] != 0).mean()), exact_novel=float((pred[att & novel] == yt[att & novel]).mean()), r2l=pc["R2L"]["r"], u2r=pc["U2R"]["r"])


out = {"fprs": {}, "seeds": C.SEEDS}
for s in C.SEEDS:
    fit, val, test, _ = make_split("official", s); seen = set(fit["sub"]); novel = ~test["sub"].isin(seen).values
    yt = test["cls"].map({c: i for i, c in enumerate(CLASSES)}).values; att = yt != 0
    d = prepare("official", s, n_per_class=C.NPC, k=500, sel_mode="global")          # all standardised (clipped) features, original unbalanced records
    for fpr in FPRS:
        npf = NormalProfile(fpr=fpr, seed=s).fit(d["Xfit"][d["yfit"] == 0], d["Xval"][d["yval"] == 0]); sc = npf.score(d["Xte"]); flag = sc > npf.thr_
        np.savez_compressed(os.path.join(C.RES, f"novelty_s{s}_f{fpr}.npz"), flag_te=flag, score_te=sc.astype(np.float32), thr=npf.thr_)
        R = out["fprs"].setdefault(str(fpr), {}); R.setdefault("novelty only", {})[str(s)] = dict(fpr=float(flag[~att].mean()), det_seen=float(flag[att & ~novel].mean()), det_novel=float(flag[att & novel].mean()))
        for label, pat in MODELS.items():
            p = os.path.join(C.RES, pat.format(s=s) + "_probs.npz")
            if not os.path.exists(p): continue
            z = np.load(p); assert (z["yt"] == yt).all(), "test order mismatch"; P = z["pt"]
            R.setdefault(label, {})[str(s)] = dict(alone=summ(P.argmax(1), yt, att, novel), fused=summ(fuse_labels(P, flag), yt, att, novel))
        print(f"seed {s} fpr {fpr}: novelty-only det seen/novel = {R['novelty only'][str(s)]['det_seen']:.3f}/{R['novelty only'][str(s)]['det_novel']:.3f}, benign flag rate {R['novelty only'][str(s)]['fpr']:.3f}", flush=True)
out["n_attack_test"] = int(att.sum()); out["n_attack_novel"] = int((att & novel).sum())
json.dump(out, open(os.path.join(C.RES, "novelty.json"), "w"), indent=1)
