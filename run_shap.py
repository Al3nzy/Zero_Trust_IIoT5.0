"""Stage 7: SHAP attribution for the proposed CNN-BiLSTM (GradientExplainer, named features, class-stratified sample), exact TreeSHAP and
TreeSHAP *interaction* values on a LightGBM surrogate, rank agreement, and a deletion-faithfulness test."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
from ztids import config as C
MODEL = os.path.join(C.RES, C.PROPOSED + ".keras")
if not os.path.exists(MODEL): sys.exit("checkpoint missing; run the detector stage first")
import shap
from lightgbm import LGBMClassifier
from ztids.data import prepare, CLASSES
from ztids.models import load
d = prepare("official", 0, n_per_class=C.NPC, sel_mode="ovr"); names = d["audit"]["selected_features"]; model = load(MODEL)
rng = np.random.RandomState(0); N = C.SHAP_N
bg = d["Xtr"][rng.choice(len(d["Xtr"]), 200, replace=False)][..., None]
idx = np.concatenate([rng.choice(np.where(d["yte"] == c)[0], min(N // 5, int((d["yte"] == c).sum())), replace=False) for c in range(5)])
Xs, ys = d["Xte"][idx], d["yte"][idx]


def to_cls_first(sv):                                                  # -> (classes, n, features)
    if isinstance(sv, list): return np.stack([np.asarray(s).reshape(len(Xs), -1) for s in sv], 0)
    sv = np.asarray(sv); return np.moveaxis(sv.reshape(len(Xs), 25, -1), -1, 0)


t = time.time(); explainer_used = "GradientExplainer"
try:
    sv = to_cls_first(shap.GradientExplainer(model, bg).shap_values(Xs[..., None], nsamples=100))
except Exception as e:                                                  # fallback for TF/shap combinations where gradients are unsupported
    print("GradientExplainer failed (", repr(e)[:80], ") -> KernelExplainer on a reduced sample", flush=True)
    explainer_used = "KernelExplainer"; sub = np.arange(0, len(Xs), max(1, len(Xs) // 100)); Xs, ys = Xs[sub], ys[sub]
    f = lambda x: model.predict(x[..., None], verbose=0)
    sv = to_cls_first(shap.KernelExplainer(f, bg[:30, :, 0]).shap_values(Xs, nsamples=200))
print("NN attributions", sv.shape, explainer_used, round(time.time() - t, 1), "s", flush=True)
glob = np.abs(sv).mean((0, 1)); per_class = {CLASSES[c]: np.abs(sv[c][ys == c]).mean(0).tolist() for c in range(5)}
gb = LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31, n_jobs=-1, random_state=0, verbose=-1).fit(d["Xtr"], d["ytr"])
te = shap.TreeExplainer(gb); tsv = to_cls_first(te.shap_values(Xs)); tglob = np.abs(tsv).mean((0, 1)); rho = spearmanr(glob, tglob)
sub = np.arange(0, len(Xs), 5); inter = te.shap_interaction_values(Xs[sub])
inter = np.stack(inter, 0) if isinstance(inter, list) else np.moveaxis(np.asarray(inter), -1, 0)
I = np.abs(inter).mean((0, 1)); np.fill_diagonal(I, 0); top = np.dstack(np.unravel_index(np.argsort(-I, axis=None)[:20:2], I.shape))[0]
out = dict(names=names, explainer=explainer_used, n_samples=int(len(Xs)), nn_global=glob.tolist(), tree_global=tglob.tolist(), per_class=per_class,
           spearman_nn_vs_tree=float(rho.statistic), p=float(rho.pvalue), nn_top10=[names[i] for i in np.argsort(-glob)[:10]], tree_top10=[names[i] for i in np.argsort(-tglob)[:10]],
           top_interactions=[dict(a=names[i], b=names[j], v=float(I[i, j])) for i, j in top], interaction_matrix=I.tolist())
pp = model.predict(Xs[..., None], verbose=0); pc = pp.argmax(1); base = pp[np.arange(len(pp)), pc]
attr = np.stack([sv[pc[i], i] for i in range(len(Xs))]); tattr = np.stack([tsv[pc[i], i] for i in range(len(Xs))])
rg = np.random.RandomState(5); rand_orders = [rg.permutation(25) for _ in range(len(Xs))]


def drop(order_fn, k):                                                  # mask top-k features (set to training mean = 0 after z-scoring)
    Xm = Xs.copy()
    for i in range(len(Xm)): Xm[i, order_fn(i)[:k]] = 0.0
    q = model.predict(Xm[..., None], verbose=0); return float((base - q[np.arange(len(q)), pc]).mean())


out["deletion"] = {str(k): dict(nn_shap=drop(lambda i: np.argsort(-np.abs(attr[i])), k), tree_shap=drop(lambda i: np.argsort(-np.abs(tattr[i])), k),
                                random=drop(lambda i: rand_orders[i], k)) for k in (1, 3, 5, 10)}
print("rank agreement NN vs TreeSHAP: rho=%.3f p=%.3f" % (rho.statistic, rho.pvalue), "| deletion:", json.dumps(out["deletion"]), flush=True)
json.dump(out, open(os.path.join(C.RES, "shap.json"), "w"), indent=1)
