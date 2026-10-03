"""Stage 6: formally differentially private training (DP-SGD, Poisson subsampling, per-example clipping, RDP accountant).
Guarantee scope: record-level (eps, delta)-DP for adding/removing one ORIGINAL training record, for the released weights.
NOT covered (stated in the paper): the preprocessing (scaler, MI feature selection), the class weights (treated as public priors),
and the choice of hyper-parameters (clip norm, learning rate). Also reports a loss-threshold membership-inference AUC as an empirical check."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from ztids import config as C
from ztids.data import prepare
from ztids.dpsgd import train_dpsgd
from ztids.evalutil import metrics
ap = argparse.ArgumentParser(); ap.add_argument("--eps", nargs="*", type=float, default=[50.0, 25.0, 8.0, 4.0, 1.0]); a = ap.parse_args()
d = prepare("official", 0, n_per_class=C.NPC, sel_mode="ovr")
Xf, yf = d["Xfit"], d["yfit"]                                      # ORIGINAL training records only (no SMOTE / duplication)
if C.DP_N and C.DP_N < len(Xf):
    idx = np.random.RandomState(0).choice(len(Xf), C.DP_N, replace=False); Xf, yf = Xf[idx], yf[idx]
print(f"DP-SGD on N={len(Xf)} records, epochs={C.DP_EPOCHS}, batch={C.DP_BATCH}, delta=1/N", flush=True)
rng = np.random.RandomState(0); nm = min(4000, len(Xf)); mem = rng.choice(len(Xf), nm, replace=False); non = rng.choice(len(d["Xval"]), min(nm, len(d["Xval"])), replace=False)


def mia_auc(model):
    def loss(X, y):
        p = model.predict(X[..., None], verbose=0); return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    lm, ln = loss(Xf[mem], yf[mem]), loss(d["Xval"][non], d["yval"][non])
    return float(roc_auc_score(np.r_[np.ones(len(lm)), np.zeros(len(ln))], -np.r_[lm, ln]))     # lower loss => predicted member


results = []
for tgt in [None] + a.eps:
    name = f"dp_{'inf' if tgt is None else tgt}"; path = os.path.join(C.RES, name + ".json")
    if os.path.exists(path): results.append(json.load(open(path))); continue
    t0 = time.time()
    m, eps, sigma, steps, delta = train_dpsgd(Xf, yf, d["Xval"], d["yval"], target_eps=tgt, epochs=C.DP_EPOCHS, batch=C.DP_BATCH, seed=0, verbose=False)
    pt = m.predict(d["Xte"][..., None], verbose=0)
    res = dict(name=name, target_eps=tgt, eps=None if tgt is None else float(eps), sigma=float(sigma), steps=int(steps), delta=float(delta), epochs=C.DP_EPOCHS,
               batch=C.DP_BATCH, N=int(len(Xf)), clip=1.0, train_s=round(time.time() - t0, 1), params=int(m.count_params()),
               test=metrics(d["yte"], pt), mia_auc=mia_auc(m))
    json.dump(res, open(path, "w")); results.append(res)
    print(f"[done] {name} eps={res['eps']} sigma={sigma:.2f} acc={res['test']['acc']:.4f} mF1={res['test']['macro_f1']:.4f} MIA-AUC={res['mia_auc']:.3f} {res['train_s']}s", flush=True)
json.dump(results, open(os.path.join(C.RES, "dp_all.json"), "w"), indent=1)
