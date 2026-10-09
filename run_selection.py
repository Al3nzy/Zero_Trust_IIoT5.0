#!/usr/bin/env python3
"""Stage `selection`: choose the Normal-profile detector from validation data only (no test access) and compare with the test ranking.
Rule: highest mean validation AUROC (attacks vs Normal of the training-domain validation split) over seeds; ties are not expected.
The rule is deliberately simple: it asks which detector ranks unseen validation attacks above benign flows. Output: results_v3/selection.json"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
from ztids import config as C
from ztids.novelty import KINDS

SEEDS = [int(s) for s in C.CORE_SEEDS.split(",")]
rows = {k: dict(val=[], test=[], test_unseen=[], test_seen=[], val_tpr=[], test_tpr_unseen=[]) for k in KINDS}
OP = 0.025   # operating false-alarm level of each scorer in Algorithm 1 (alpha/2 at alpha = 0.05)
for s in SEEDS:
    z = np.load(os.path.join(C.RES, f"core_s{s}.npz"), allow_pickle=True); yv, yt = z["yv"], z["yt"]; seen = set(z["sub_fit"]) | set(z["sub_val"]); nov = (yt != 0) & ~np.isin(z["sub"], list(seen))
    for k in KINDS:
        sv, st = z[f"sc_{k}_val"], z[f"sc_{k}_te"]
        thr = np.quantile(sv[yv == 0], 1 - OP); rows[k]["val_tpr"].append(float((sv[yv != 0] > thr).mean())); thrt = float(z[f"thr_{k}"]); rows[k]["test_tpr_unseen"].append(float((st[nov] > np.quantile(sv[yv == 0], 1 - OP)).mean()))
        rows[k]["val"].append(roc_auc_score(yv != 0, sv)); rows[k]["test"].append(roc_auc_score(yt != 0, st))
        m = (yt == 0) | nov; rows[k]["test_unseen"].append(roc_auc_score(yt[m] != 0, st[m])); m = (yt == 0) | ((yt != 0) & ~nov); rows[k]["test_seen"].append(roc_auc_score(yt[m] != 0, st[m]))
out = {k: {m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for m, v in d.items()} for k, d in rows.items()}
sel = max(KINDS, key=lambda k: out[k]["val"]["mean"]); sel_op = max(KINDS, key=lambda k: out[k]["val_tpr"]["mean"]); best_test = max(KINDS, key=lambda k: out[k]["test_unseen"]["mean"])
res = dict(seeds=SEEDS, auroc=out, operating_level=OP, selected_by_validation_auroc=sel, selected_by_validation_operating_point=sel_op, selected_by_validation=sel, best_on_test_unseen=best_test, agree=sel == best_test)
json.dump(res, open(os.path.join(C.RES, "selection.json"), "w"), indent=1)
for k in KINDS: print(f"{k:22s} val AUROC {out[k]['val']['mean']:.3f} | test {out[k]['test']['mean']:.3f} | test unseen types {out[k]['test_unseen']['mean']:.3f} | test seen {out[k]['test_seen']['mean']:.3f}")
for k in KINDS: print(f"{k:22s} validation detection at FPR {OP}: {out[k]['val_tpr']['mean']:.3f} | test unseen types at the same threshold: {out[k]['test_tpr_unseen']['mean']:.3f}")
print("selected by AUROC:", sel, "| by operating-point detection:", sel_op, "| best on unseen test types:", best_test)
