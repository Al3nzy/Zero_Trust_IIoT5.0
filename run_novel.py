"""Stage 2: recall on attack sub-types seen in training vs sub-types that only exist in the official test set."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ztids import config as C
from ztids.data import make_split, CLASSES
fit, val, test, audit = make_split("official", 0)
seen = set(fit["sub"]); sub = test["sub"].values; yt_ref = test["cls"].map({c: i for i, c in enumerate(CLASSES)}).values
novel = ~pd.Series(sub).isin(seen).values
out = {"n_test": int(len(test)), "n_attack_test": int((yt_ref != 0).sum()), "n_attack_novel": int(((yt_ref != 0) & novel).sum()),
       "novel_subtypes": sorted(set(sub[novel & (yt_ref != 0)])), "models": {}}
cands = {"proposed CNN-BiLSTM": C.PROPOSED, "LightGBM (class-aware MI)": "official_lgbm_0_sel_selovr", "LightGBM (global MI)": "official_lgbm_0_sel",
         "CNN-BiLSTM + attention": C.ATTN}
for label, name in cands.items():
    p = os.path.join(C.RES, name + "_probs.npz")
    if not os.path.exists(p): continue
    z = np.load(p); pt, yt = z["pt"], z["yt"]; assert (yt == yt_ref).all(), "test order mismatch"
    pred = pt.argmax(1); att = yt != 0; r = {}
    for tag, m in (("seen", att & ~novel), ("novel", att & novel)):
        n = int(m.sum()); r[tag] = dict(n=n, exact_recall=float((pred[m] == yt[m]).mean()) if n else None, detection=float((pred[m] != 0).mean()) if n else None)
    per = []
    for s_ in sorted(set(sub[att])):
        m = att & (sub == s_)
        if m.sum() >= 20: per.append(dict(sub=s_, cls=CLASSES[int(yt[m][0])], n=int(m.sum()), seen_in_training=bool(s_ in seen),
                                          exact_recall=float((pred[m] == yt[m]).mean()), detection=float((pred[m] != 0).mean())))
    r["per_subtype"] = per; out["models"][label] = r
    print(label, {k: r[k] for k in ("seen", "novel")}, flush=True)
if not out["models"]: sys.exit("no model outputs found; run the detector stage first")
from ztids.data import load_nsl
tr_raw, te_raw = load_nsl(); diag = {}
for label, r in out["models"].items():
    cand = [x for x in r["per_subtype"] if x["seen_in_training"]]
    if not cand: continue
    w = max(cand, key=lambda x: x["n"] * (1 - x["exact_recall"]))
    if w["n"] * (1 - w["exact_recall"]) < 100: continue
    a, b = tr_raw[tr_raw["sub"] == w["sub"]], te_raw[te_raw["sub"] == w["sub"]]
    top = lambda df, c: {str(k): round(float(v), 2) for k, v in df[c].value_counts(normalize=True).head(3).items()}
    diag[label] = dict(sub=w["sub"], cls=w["cls"], missed=int(round(w["n"] * (1 - w["exact_recall"]))), exact_recall=w["exact_recall"], n_train=int(len(a)), n_test=int(len(b)),
                       train_service=top(a, "service"), test_service=top(b, "service"), train_flag=top(a, "flag"), test_flag=top(b, "flag"),
                       train_logged_in=float(a["logged_in"].mean()), test_logged_in=float(b["logged_in"].mean()))
out["shift_diagnostic"] = diag
json.dump(out, open(os.path.join(C.RES, "novel.json"), "w"), indent=1)
