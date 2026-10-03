"""
Leakage-free evaluation on additional corpora.  Examples
  python run_dataset.py unsw  --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --seeds 0 1 2
  python run_dataset.py edge  --train ML-EdgeIIoT-dataset.csv --seeds 0 1 2 --max-rows 120000
Presets set the label column and the identifier/leak-prone columns to drop; override with --label / --drop.
"""
import argparse, json, os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ztids.generic import prepare_generic, metrics_generic
PRESETS = {
    "unsw": dict(label="attack_cat", drop=["id", "label"]),
    "edge": dict(label="Attack_type", drop=["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                                          "http.file_data", "http.request.full_uri", "http.request.uri.query", "tcp.options", "tcp.payload",
                                          "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg", "icmp.transmission_time", "Attack_label"]),
    "cic": dict(label="Label", drop=["Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port", "Destination Port"]),
}
ap = argparse.ArgumentParser(); ap.add_argument("preset", choices=list(PRESETS)); ap.add_argument("--train", required=True); ap.add_argument("--test")
ap.add_argument("--label"); ap.add_argument("--drop", nargs="*"); ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
ap.add_argument("--npc", type=int, default=4000); ap.add_argument("--max-rows", type=int); ap.add_argument("--epochs", type=int, default=14)
ap.add_argument("--models", nargs="*", default=["cnn_bilstm", "lgbm", "rf", "mlp"]); ap.add_argument("--sel", default="ovr"); ap.add_argument("--no-loo", action="store_true"); ap.add_argument("--normal-label", default="Normal"); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_extra"))
a = ap.parse_args(); P = PRESETS[a.preset]; label = a.label or P["label"]; drop = a.drop if a.drop is not None else P["drop"]
os.makedirs(a.out, exist_ok=True)
tr = pd.read_csv(a.train, low_memory=False); tr.columns = [c.strip() for c in tr.columns]
te = pd.read_csv(a.test, low_memory=False) if a.test else None
if te is not None: te.columns = [c.strip() for c in te.columns]
drop = [c.strip() for c in drop if c.strip() in tr.columns]
for s in a.seeds:
    path = f"{a.out}/{a.preset}_seed{s}.json"
    if os.path.exists(path): continue
    t0 = time.time(); d = prepare_generic(tr, te, label, drop, s, a.npc, 25, a.sel, a.max_rows); C = d["classes"]; res = dict(preset=a.preset, seed=s, audit=d["audit"], models={})
    print(f"[{a.preset} seed {s}] fit={len(d['yfit'])} test={len(d['yte'])} classes={len(C)} overlap_after_split={d['audit']['test_overlap_after_split']}", flush=True)
    for name in a.models:
        t1 = time.time()
        if name == "lgbm":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=s, verbose=-1).fit(d["Xtr"], d["ytr"]); pt = m.predict_proba(d["Xte"])
        elif name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=s).fit(d["Xtr"], d["ytr"]); pt = m.predict_proba(d["Xte"])
        else:
            from ztids.models import BUILDERS, compile_model, fit_model, predict
            m = compile_model(BUILDERS[name](n_cls=len(C))); fit_model(m, d["Xtr"], d["ytr"], d["Xval"], d["yval"], epochs=a.epochs, patience=3, seed=s); pt = predict(m, d["Xte"])
        res["models"][name] = dict(test=metrics_generic(d["yte"], pt, C), train_s=round(time.time() - t1, 1))
        r = res["models"][name]["test"]; print(f"   {name:11s} acc={r['acc']:.4f} macroF1={r['macro_f1']:.4f} mcc={r['mcc']:.4f} ({res['models'][name]['train_s']}s)", flush=True)
    if not a.no_loo:                                      # leave-one-attack-class-out test of the novelty fusion (external confirmation)
        try:
            from ztids.loo import run_loo
            nl = [i for i, c in enumerate(C) if str(c).lower() == a.normal_label.lower() or str(c).lower() in ("normal", "benign")]
            if not nl: raise RuntimeError(f"no Normal/Benign class found among {C}; pass --normal-label")
            d_all = prepare_generic(tr, te, label, drop, s, a.npc, 500, "global", a.max_rows)
            res["loo"] = run_loo(d, d_all, C, nl[0], s, a.npc)
            for cname, r in res["loo"].items():
                print(f"   LOO {cname:18s} n={r['n_test']:6d} detected: alone {r['alone']['det_heldout']:.3f} | +iforest {r['strict']['iforest']['det_heldout']:.3f} | +mahalanobis {r['strict']['mahalanobis']['det_heldout']:.3f} "
                      f"(site-calibrated {r['site']['mahalanobis']['det_heldout']:.3f}) | benign FPR alone {r['alone']['fpr']:.3f} -> {r['strict']['mahalanobis']['fpr']:.3f}", flush=True)
        except Exception as e:
            print("   LOO stage failed:", repr(e)[:200], flush=True); res["loo_error"] = repr(e)
    json.dump(res, open(path, "w"), indent=1, default=str)
print("done ->", a.out)
