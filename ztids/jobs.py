"""Resumable experiment jobs. One job = one trained model + metrics + saved posteriors (finished jobs are skipped)."""
import json, os, re, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from . import config as C
from .data import prepare
from .evalutil import metrics

_CACHE = {}


def get_data(proto, seed, order="canonical", sel="global", k=25):
    key = (proto, seed, C.NPC, order, sel, k)
    if key not in _CACHE:
        _CACHE[key] = prepare(proto, seed, n_per_class=C.NPC, k=k, feature_order=order, sel_mode=sel)
    return _CACHE[key]


def _s(v):
    return "-" if v in (None, "") else re.sub(r"[^A-Za-z0-9_.\-]", "", str(v).replace(" ", ""))


def jid(j):
    """Canonical job id (also used by make_tables.py to find result files)."""
    base = "_".join(_s(j.get(k)) for k in ["proto", "model", "seed", "order", "laplace", "poison"])
    return base + (f"_sel{j['sel']}" if j.get("sel") else "")


def tree_id(j):
    return f"{j['proto']}_{j['model']}_{j['seed']}_{j['feats']}" + (f"_sel{j['sel']}" if j.get("sel") else "")


def poison(y, spec, rng):
    y = y.copy(); kind, rate = spec
    if kind == "random":                      # uniform random relabelling of a fraction of the training labels
        m = rng.rand(len(y)) < rate; y[m] = rng.randint(0, 5, m.sum())
    elif kind == "targeted":                  # attacker hides R2L/U2R by relabelling a fraction of them as Normal
        m = np.isin(y, [3, 4]) & (rng.rand(len(y)) < rate); y[m] = 0
    return y


def run(j):
    name = jid(j); path = os.path.join(C.RES, name + ".json")
    if os.path.exists(path): return json.load(open(path))
    from .models import BUILDERS, compile_model, fit_model, predict
    t0 = time.time(); d = get_data(j["proto"], j["seed"], j.get("order", "canonical"), j.get("sel", "global"))
    Xtr, ytr = d["Xtr"], d["ytr"]; rng = np.random.RandomState(j["seed"] + 101)
    if j.get("laplace"):                      # Laplace input-noise AUGMENTATION (scale 1/eps); no differential-privacy claim
        Xtr = Xtr + rng.laplace(0, 1.0 / j["laplace"], Xtr.shape).astype(np.float32)
    if j.get("poison"): ytr = poison(ytr, j["poison"], rng)
    m = compile_model(BUILDERS[j["model"]]())
    h = fit_model(m, Xtr, ytr, d["Xval"], d["yval"], epochs=j.get("epochs", C.EPOCHS), patience=3, seed=j["seed"])
    pt, pv = predict(m, d["Xte"]), predict(m, d["Xval"])
    res = dict(job={k: (list(v) if isinstance(v, tuple) else v) for k, v in j.items()}, id=name, epochs_run=len(h.history["loss"]),
               train_s=round(time.time() - t0, 1), params=int(m.count_params()), npc=C.NPC,
               test=metrics(d["yte"], pt), val=metrics(d["yval"], pv),
               audit={k: v for k, v in d["audit"].items() if k != "selected_features"}, features=d["audit"]["selected_features"])
    np.savez_compressed(os.path.join(C.RES, name + "_probs.npz"), pt=pt.astype(np.float32), yt=d["yte"], pv=pv.astype(np.float32), yv=d["yval"])
    if j.get("save_model"): m.save(os.path.join(C.RES, name + ".keras"))
    json.dump(res, open(path, "w"))
    print(f"[done] {name} acc={res['test']['acc']:.4f} mF1={res['test']['macro_f1']:.4f} ep={res['epochs_run']} {res['train_s']}s", flush=True)
    return res


def run_tree(j):
    """Classical baselines on the same balanced training partition (25 selected features, or all expanded features)."""
    name = tree_id(j); path = os.path.join(C.RES, name + ".json")
    if os.path.exists(path): return json.load(open(path))
    t0 = time.time(); allf = j["feats"] == "all"
    d = get_data(j["proto"], j["seed"], "canonical", "global" if allf else j.get("sel", "global"), k=500 if allf else 25)
    if j["model"] == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=j["seed"])
    else:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=j["seed"], verbose=-1)
    m.fit(d["Xtr"], d["ytr"]); t1 = time.time(); pt = m.predict_proba(d["Xte"]); lat = time.time() - t1
    res = dict(job=j, id=name, train_s=round(t1 - t0, 1), test=metrics(d["yte"], pt), predict_test_s=lat, npc=C.NPC)
    np.savez_compressed(os.path.join(C.RES, name + "_probs.npz"), pt=pt.astype(np.float32), yt=d["yte"])
    json.dump(res, open(path, "w"))
    print(f"[done] {name} acc={res['test']['acc']:.4f} mF1={res['test']['macro_f1']:.4f}", flush=True)
    return res
