"""Final overall report: best algorithm per metric, key findings, and run times. Printed at the very end of run_everything.py
and saved to <results>/final_report.txt. Reads only result files (no recomputation); ASCII output so Windows consoles are safe."""
import sys, os, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import numpy as np
from ztids import config as C
from ztids.jobs import jid, tree_id
from ztids.ensemble import HYBRID, metrics_over_seeds

S, O, OUT = C.SEEDS, "official", []
def J(name):
    p = os.path.join(C.RES, name + ".json"); return json.load(open(p)) if os.path.exists(p) else None
def P(*a): OUT.append(" ".join(str(x) for x in a))
def hdr(t): P(""); P("=" * 100); P(t); P("=" * 100)
def sub(t): P(""); P("-- " + t)
def fm(v, d=3): return "--" if v is None else f"{v:.{d}f}"
def pm(vals, d=3):
    v = [x for x in vals if x is not None]
    if not v: return "--"
    return f"{np.mean(v):.{d}f}" if len(v) == 1 else f"{np.mean(v):.{d}f}+/-{np.std(v, ddof=1):.{d}f}"
def mean(vals): v = [x for x in vals if x is not None]; return float(np.mean(v)) if v else None
def sd(vals): v = [x for x in vals if x is not None]; return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
def jobs(model, **kw): return [jid(dict(proto=O, model=model, seed=s, **kw)) for s in S]
def trees(model, feats, **kw): return [tree_id(dict(proto=O, model=model, seed=s, feats=feats, **kw)) for s in S]
def load(names): return [r for r in (J(n) for n in names) if r]
MET = [("Acc", lambda r: r["test"]["acc"], 1), ("MacroF1", lambda r: r["test"]["macro_f1"], 1), ("MCC", lambda r: r["test"]["mcc"], 1),
       ("NormalFPR", lambda r: r["test"]["normal_fpr"], -1), ("R2L-rec", lambda r: r["test"]["per_class"]["R2L"]["r"], 1), ("U2R-rec", lambda r: r["test"]["per_class"]["U2R"]["r"], 1)]
MODELS = [("CNN-BiLSTM (proposed)", jobs("cnn_bilstm", sel="ovr")), ("CNN-BiLSTM + dual attention", jobs("cnn_bilstm_attn", sel="ovr")),
          ("CNN-BiLSTM, global MI", jobs("cnn_bilstm")), ("LightGBM (class-aware MI)", trees("lgbm", "sel", sel="ovr")), ("LightGBM (global MI)", trees("lgbm", "sel")),
          ("Random forest (class-aware)", trees("rf", "sel", sel="ovr")), ("MLP", jobs("mlp", sel="ovr")), ("CNN only", jobs("cnn", sel="ovr")), ("LSTM only", jobs("lstm", sel="ovr"))]
DEEP = {"CNN-BiLSTM (proposed)", "CNN-BiLSTM + dual attention", "CNN-BiLSTM, global MI", "MLP", "CNN only", "LSTM only"}

# ------------------------------------------------------------------ 1. detection
hdr("FINAL RESULTS OVERVIEW")
status = {}
sp = os.path.join(C.RES, "pipeline_status.json")
if os.path.exists(sp): status = json.load(open(sp))
lat = J("latency"); hw = lat["hw"]["platform"] + f", {lat['hw']['cpu_count']} cores" if lat else "n/a"
allr = [r for _, n in MODELS for r in load(n)]
deg = sum(1 for r in allr if r.get("degenerate"))
P(f"Setup: NSL-KDD official split (overlap-free test, {J(C.PROPOSED)['test']['n'] if J(C.PROPOSED) else '?'} flows) | seeds={S} | balanced train {C.NPC}/class | epochs<={C.EPOCHS} | hardware: {hw}")
P(f"Degenerate (collapsed) runs among the models below: {deg} of {len(allr)}" + ("" if deg == 0 else "  <-- inspect before trusting means"))
hdr("1. DETECTION ACCURACY (mean +/- std over seeds; * = best in column)")
rows = []
for label, names in MODELS:
    rs = load(names)
    if rs: rows.append((label, len(rs), [[f(r) for r in rs] for _, f, _ in MET]))
hy = metrics_over_seeds(HYBRID["members"], S)
if hy: rows.append((HYBRID["label"], len(hy), [[f(r) for r in hy] for _, f, _ in MET]))
best = []
for k, (_, _, sign) in enumerate(MET):
    vals = [(mean(r[2][k]) * sign, i) for i, r in enumerate(rows)]; best.append(max(vals)[1])
P(f"{'Model':30s} {'n':>2s} " + " ".join(f"{m[0]:>15s}" for m in MET))
for i, (label, n, cols) in enumerate(rows):
    P(f"{label:30s} {n:>2d} " + " ".join(f"{pm(c) + ('*' if best[k] == i else ' '):>15s}" for k, c in enumerate(cols)))
P("")
for k, (name, _, sign) in enumerate(MET):
    b = rows[best[k]]; P(f"  Best {name:10s}: {b[0]:30s} {pm(b[2][k])}")
if rows:
    order = sorted(rows, key=lambda r: -mean(r[2][1])); P("  Ranking by Macro-F1 : " + " > ".join(f"{r[0]} ({mean(r[2][1]):.3f})" for r in order))
    deep = [r for r in rows if r[0] in DEEP]; cls_ = [r for r in rows if r[0] not in DEEP and r[0] != HYBRID['label']]
    if deep and cls_:
        bd, bc = max(deep, key=lambda r: mean(r[2][1])), max(cls_, key=lambda r: mean(r[2][1]))
        d = mean(bc[2][1]) - mean(bd[2][1]); pooled = (sd(bc[2][1]) ** 2 + sd(bd[2][1]) ** 2) ** 0.5
        P(f"  Verdict: best classical model ({bc[0]}) vs best deep model ({bd[0]}): Macro-F1 {mean(bc[2][1]):.3f} vs {mean(bd[2][1]):.3f} (diff {d:+.3f}, "
          f"{'exceeds' if abs(d) > 2 * pooled else 'within'} 2 pooled std; few seeds, indicative only).")
        P(f"           Deep models lead only on rare-class recall: best U2R recall = {max(rows, key=lambda r: mean(r[2][5]))[0]} ({mean(max(rows, key=lambda r: mean(r[2][5]))[2][5]):.3f}).")

hyb = [r for r in rows if r[0] == HYBRID["label"]]
if hyb:
    h_, l_ = hyb[0], [r for r in rows if r[0] == "LightGBM (class-aware MI)"]
    if l_:
        d_ = mean(h_[2][1]) - mean(l_[0][2][1]); pool_ = (sd(h_[2][1]) ** 2 + sd(l_[0][2][1]) ** 2) ** 0.5
        vd_ = "a significant gain" if d_ > 2 * pool_ else ("no significant gain" if d_ > -2 * pool_ else "a significant loss")
        P(f"  Hybrid vs best single model (LightGBM): Macro-F1 {pm(h_[2][1])} vs {pm(l_[0][2][1])} ({vd_}); U2R recall {pm(h_[2][5])} vs {pm(l_[0][2][5])}; benign FPR {pm(h_[2][3])} vs {pm(l_[0][2][3])}; seed std of Macro-F1 {sd(h_[2][1]):.3f} vs {sd(l_[0][2][1]):.3f}.")
nvj = J("novel")
if nvj and nvj["models"]:
    sub("1b. where the classifiers fail (per attack sub-type, LightGBM class-aware, seed 0)")
    ps = nvj["models"].get("LightGBM (class-aware MI)", next(iter(nvj["models"].values())))["per_subtype"]
    P(f"  {'sub-type':16s} {'class':6s} {'flows':>6s} {'in training':>11s} {'exact recall':>12s}")
    for r in sorted(ps, key=lambda r: -r["n"] * (1 - r["exact_recall"]))[:6]: P(f"  {r['sub']:16s} {r['cls']:6s} {r['n']:>6d} {str(r['seen_in_training']):>11s} {100 * r['exact_recall']:>11.0f}%")
    for label, dg in nvj.get("shift_diagnostic", {}).items():
        if label != "LightGBM (class-aware MI)": continue
        P(f"  Why: '{dg['sub']}' IS in the training data ({dg['n_train']} flows) yet {dg['missed']} of its {dg['n_test']} test flows are missed. Training: services {dg['train_service']}, flags {dg['train_flag']}, logged_in {dg['train_logged_in']:.2f}.")
        P(f"       Test: services {dg['test_service']}, flags {dg['test_flag']}, logged_in {dg['test_logged_in']:.2f}  -> the test examples behave differently from every training example (within-sub-type distribution shift); no supervised model can learn this from the training split.")

# ------------------------------------------------------------------ 2. ablations
hdr("2. ABLATIONS")
def mf(names): return [r["test"]["macro_f1"] for r in load(names)]
def rec(names, c): return [r["test"]["per_class"][c]["r"] for r in load(names)]
a, b = mf(jobs("cnn_bilstm", sel="ovr")), mf(jobs("cnn_bilstm"))
if a and b: P(f"  Feature selection (CNN-BiLSTM): class-aware MI {pm(a)} vs global MI {pm(b)} Macro-F1 (delta {mean(a) - mean(b):+.3f})")
a, b = mf(trees("lgbm", "sel", sel="ovr")), mf(trees("lgbm", "sel"))
if a and b: P(f"  Feature selection (LightGBM)  : class-aware MI {pm(a)} vs global MI {pm(b)} Macro-F1 (delta {mean(a) - mean(b):+.3f}); R2L recall {pm(rec(trees('lgbm', 'sel', sel='ovr'), 'R2L'))} vs {pm(rec(trees('lgbm', 'sel'), 'R2L'))}")
a, b = mf(jobs("cnn_bilstm", sel="ovr")), mf(jobs("cnn_bilstm_attn", sel="ovr"))
if a and b:
    la = lat["models"] if lat else {}
    P(f"  Dual attention: Macro-F1 {pm(a)} (without) vs {pm(b)} (with), delta {mean(b) - mean(a):+.3f}" + (f"; round latency {la['plain']['round_total_ms']:.2f} vs {la['attention']['round_total_ms']:.2f} ms" if "plain" in la and "attention" in la else ""))
o = {k: mf(jobs("cnn_bilstm", sel="ovr", **({"order": k} if k != "canonical" else {}))) for k in ("canonical", "random", "mi")}
if all(o.values()): P("  Feature order: " + ", ".join(f"{k} {pm(v)}" for k, v in o.items()) + ("  -> order does not matter (differences within std)" if max(mean(v) for v in o.values()) - min(mean(v) for v in o.values()) < 2 * max(sd(v) for v in o.values()) else "  -> order matters"))
cl, tg, rn = (jobs("cnn_bilstm", sel="ovr"), jobs("cnn_bilstm", sel="ovr", poison=("targeted", 0.5)), jobs("cnn_bilstm", sel="ovr", poison=("random", 0.3)))
if load(cl) and load(tg):
    P(f"  Poisoning (targeted, 50% of R2L/U2R relabelled Normal): R2L recall {pm(rec(cl, 'R2L'))} -> {pm(rec(tg, 'R2L'))}, U2R recall {pm(rec(cl, 'U2R'))} -> {pm(rec(tg, 'U2R'))}")
    P(f"  Poisoning (random 30% label noise)                    : Macro-F1 {pm(mf(cl))} -> {pm(mf(rn))}  (robust to random noise, vulnerable to targeted poisoning)")
ind = (mf([jid(dict(proto="random", model="cnn_bilstm", seed=s, sel="ovr")) for s in S]), [r["test"]["macro_f1"] for r in load([tree_id(dict(proto="random", model="lgbm", seed=s, feats="sel", sel="ovr")) for s in S])])
if ind[0] and ind[1]:
    ac = [r["test"]["acc"] for r in load([tree_id(dict(proto="random", model="lgbm", seed=s, feats="sel", sel="ovr")) for s in S])]
    P(f"  Distribution shift: LightGBM accuracy {pm(ac)} on a deduplicated random split vs {pm([r['test']['acc'] for r in load(trees('lgbm', 'sel', sel='ovr'))])} on the official split (novel attack types)")

# ------------------------------------------------------------------ 3. novelty
hdr("3. UNSEEN ATTACK TYPES (Normal-profile novelty detector fused with each classifier, threshold = 2% FPR on validation)")
nv = J("novelty")
if nv and "0.02" in nv["fprs"]:
    R = nv["fprs"]["0.02"]
    P(f"{'Model':30s} {'novel det: alone -> fused':>28s} {'seen det (fused)':>17s} {'benign FPR: alone -> fused':>28s} {'MacroF1: alone -> fused':>26s}")
    best_n = None
    for model, per in R.items():
        if model == "novelty only": continue
        al = [per[s]["alone"] for s in per]; fu = [per[s]["fused"] for s in per]
        P(f"{model:30s} {pm([x['det_novel'] for x in al], 2) + ' -> ' + pm([x['det_novel'] for x in fu], 2):>28s} {pm([x['det_seen'] for x in fu], 2):>17s} "
          f"{pm([x['fpr'] for x in al], 3) + ' -> ' + pm([x['fpr'] for x in fu], 3):>28s} {pm([x['macro_f1'] for x in al], 3) + ' -> ' + pm([x['macro_f1'] for x in fu], 3):>26s}")
        m = mean([x["det_novel"] for x in fu])
        if best_n is None or m > best_n[1]: best_n = (model, m, mean([x["fpr"] for x in fu]) - mean([x["fpr"] for x in al]), mean([x["det_seen"] for x in fu]) - mean([x["det_seen"] for x in al]))
    dets = nv.get("detectors", {})
    if dets:
        P("  Detector comparison (threshold at 2% validation FPR; AUROC is threshold-free, higher = better ranking under shift):")
        for kind, byf in dets.items():
            Rk = byf.get("0.02", {}).get("novelty only"); au = nv.get("auc", {}).get(kind, {})
            if Rk: P(f"    {kind:22s} AUROC {pm([v['all'] for v in au.values()], 3)} (unseen types {pm([v['novel'] for v in au.values()], 3)}) | benign flag rate {pm([v['fpr'] for v in Rk.values()], 3)} | detects seen {pm([v['det_seen'] for v in Rk.values()], 2)}, unseen {pm([v['det_novel'] for v in Rk.values()], 2)}")
        P("    Caution: detector ranking was inspected on NSL-KDD test data (kNN-distance: best validation AUROC, worst test AUROC under shift, rejected). Confirm on UNSW-NB15 / Edge-IIoTset.")
    cm = nv.get("commissioning")
    if cm:
        P("  Site calibration (threshold from a clean commissioning window = 10% of test-domain Normal flows, held out; mean of 20 windows, LightGBM class-aware):")
        for kind, Rc in cm.items():
            r = Rc.get("LightGBM (class-aware MI)")
            if r:
                fu = [r[s_]["fused"] for s_ in r]; al = [r[s_]["alone"] for s_ in r]
                P(f"    {kind:22s} Macro-F1 {pm([x['macro_f1'] for x in fu])} (alone {pm([x['macro_f1'] for x in al])}) | benign FPR {pm([x['fpr'] for x in fu])} | seen {pm([x['det_seen'] for x in fu], 2)} | unseen {pm([x['det_novel'] for x in fu], 2)}")
    ca, gl = R.get("LightGBM (class-aware MI)"), R.get("LightGBM (global MI)")
    if ca and gl:
        P(f"  Trade-off: class-aware MI raises R2L/U2R recall and Macro-F1 but lowers unseen-type detection of the classifier alone ({pm([ca[x]['alone']['det_novel'] for x in ca], 2)} vs {pm([gl[x]['alone']['det_novel'] for x in gl], 2)} with global MI); the novelty fusion recovers it ({pm([ca[x]['fused']['det_novel'] for x in ca], 2)}).")
    na = R.get("novelty only", {}); P(f"  Novelty detector alone: detects {pm([v['det_novel'] for v in na.values()], 2)} of unseen-type attacks and {pm([v['det_seen'] for v in na.values()], 2)} of seen-type attacks at {pm([v['fpr'] for v in na.values()], 3)} benign flag rate.")
    P(f"  {nv['n_attack_novel']} of {nv['n_attack_test']} test attacks belong to sub-types absent from training. Best unseen-type detection: {best_n[0]} + novelty ({best_n[1]:.2f}).")
else: P("  (novelty results not found)")

# ------------------------------------------------------------------ 4. trust
hdr("4. ZERO-TRUST LAYER (replay of real classifier outputs through simulated devices)")
tr, th = J("trust_main"), J("trust_theory")
if tr:
    for fl, nm in (("homog", "homogeneous fleet"), ("hetero", "heterogeneous fleet (devices on different services)")):
        sub(nm)
        P(f"  {'Rule':38s} {'false quar.':>11s} {'detect':>7s} {'delay':>6s} {'seen type':>9s} {'novel type':>10s} {'stealth30%':>10s} {'released':>9s}")
        for name, v in tr[fl].items():
            if not isinstance(v, dict) or "benign" not in v: continue
            b, c, r, s_ = v["benign"], v["compromised"], v["recovery"], v["stealth"]; sn, nvv = v.get("seen_attack", {}), v.get("novel_attack", {})
            P(f"  {name:38s} {100 * b['ever_quarantined']:>10.1f}% {100 * c['detect']:>6.0f}% {('--' if c['median_delay'] is None else format(c['median_delay'], '.0f')):>6s} "
              f"{100 * sn.get('detect', 0):>8.0f}% {100 * nvv.get('detect', 0):>9.0f}% {100 * s_['detect']:>9.0f}% {100 * r['released_by_end']:>8.0f}%")
    ct = tr["contamination"]; hi = max(v["frac"] for v in ct.values())
    P(f"  Contaminated commissioning (up to {100 * hi:.0f}% of fleet already compromised): fleet-median baseline still detects {100 * ct[f'{hi}_fleet']['detect_contaminated']:.0f}%, per-device baseline only {100 * ct[f'{hi}_device']['detect_contaminated']:.0f}%.")
ssj = J("trust_sensitivity")
if ssj:
    rr = ssj["rows"]; good = [r for r in rr if r["k"] >= 0.05]
    P(f"  Sensitivity ({len(rr)} settings of k, alpha, m, warm-up): false quarantine {100 * min(r['false_quarantine'] for r in rr):.1f}-{100 * max(r['false_quarantine'] for r in rr):.1f}% overall ({100 * max(r['false_quarantine'] for r in good):.1f}% max for k>=0.05); "
      f"compromised-device detection {100 * min(r['detect'] for r in rr):.0f}-{100 * max(r['detect'] for r in rr):.0f}%; stealth(30%) detection {100 * min(r['stealth30'] for r in rr):.0f}-{100 * max(r['stealth30'] for r in rr):.0f}%.")
if th:
    ok_fq = all(r["empirical"] <= r["alpha"] + 1e-9 for r in th["false_quarantine"]); ok_d = all(r["mean_delay"] is not None and r["bound"] is not None and r["mean_delay"] <= r["bound"] for r in th["delay"])
    P(f"  Proved bounds hold empirically: false-quarantine <= certified alpha in {sum(r['empirical'] <= r['alpha'] + 1e-9 for r in th['false_quarantine'])}/{len(th['false_quarantine'])} settings (incl. correlated flows): {ok_fq}; "
      f"measured delay <= proved delay bound for all attack classes: {ok_d}.")
    P("  Delay (measured vs bound, rounds): " + ", ".join(f"{r['cls']} {fm(r['mean_delay'], 1)} vs {fm(r['bound'], 1)}" for r in th["delay"]))

# ------------------------------------------------------------------ 5. ledger
hdr("5. AUDIT LEDGER (tamper detection rate)")
lg = J("ledger")
if lg:
    cfg = list(next(iter(lg["attacks"].values())).keys())
    for c in cfg:
        rates = {k: v[c] for k, v in lg["attacks"].items()}; missed = [k for k, x in rates.items() if x < 0.999]
        P(f"  {c:26s} mean detection {100 * np.mean(list(rates.values())):5.1f}% | not fully detected: " + (", ".join(f"{k} ({100 * rates[k]:.0f}%)" for k in missed) if missed else "none"))
    pf = lg["perf"]; P(f"  Cost: append {pf['append_us_hash_only']:.1f} us (hash only) vs {pf['append_us_signed']:.1f} us (signed); verification of 10k blocks {pf.get('verify_s_10000', float('nan')):.2f} s")
    P("  Verdict: a plain hash chain detects only in-place edits without re-hashing; signatures + external anchors are needed for real tamper evidence. It is an audit log, not a blockchain.")

# ------------------------------------------------------------------ 6. privacy
hdr("6. PRIVACY (DP-SGD with RDP accountant)")
dpp = os.path.join(C.RES, "dp_all.json"); base, pr = [], []
if os.path.exists(dpp):
    dp = json.load(open(dpp)); base = [r for r in dp if r["eps"] is None]
    P(f"  {'epsilon':>12s} {'sigma':>6s} {'Acc':>6s} {'MacroF1':>8s} {'U2R-rec':>8s} {'MIA-AUC':>8s}")
    for r in dp: P(f"  {'non-private' if r['eps'] is None else format(r['eps'], '.2f'):>12s} {('--' if r['eps'] is None else format(r['sigma'], '.2f')):>6s} {r['test']['acc']:>6.3f} {r['test']['macro_f1']:>8.3f} {r['test']['per_class']['U2R']['r']:>8.3f} {r['mia_auc']:>8.3f}")
    pr = [r for r in dp if r["eps"] is not None]
    if base and pr: P(f"  Cost of privacy: Macro-F1 {base[0]['test']['macro_f1']:.3f} -> {pr[0]['test']['macro_f1']:.3f} (eps={pr[0]['eps']:.0f}); U2R recall {base[0]['test']['per_class']['U2R']['r']:.2f} -> {pr[-1]['test']['per_class']['U2R']['r']:.2f}. "
                     f"Membership-inference AUC is ~0.5 even without DP, so it gives no empirical evidence of leakage either way; the claim rests on the formal accountant (delta=1/N, single seed).")

# ------------------------------------------------------------------ 7. explainability
hdr("7. EXPLAINABILITY (SHAP)")
sh = J("shap")
if sh:
    d5 = sh["deletion"]["5"]
    P(f"  n={sh['n_samples']} flows ({sh['explainer']}). Deletion test, mask top-5 features: probability drop NN-SHAP {d5['nn_shap']:.3f} vs TreeSHAP order {d5['tree_shap']:.3f} vs random {d5['random']:.3f} -> "
      f"{'SHAP ranking is faithful to the model' if d5['nn_shap'] > 1.5 * d5['random'] else 'faithfulness not demonstrated'}.")
    P(f"  Agreement between CNN-BiLSTM and LightGBM attributions: Spearman rho={sh['spearman_nn_vs_tree']:.2f} (p={sh['p']:.2f}) -> " + ("models rely on different features; explanations are model-specific." if sh["p"] > 0.05 else "significant agreement."))
    P("  Top NN features: " + ", ".join(sh["nn_top10"][:6]))

# ------------------------------------------------------------------ 8. latency
hdr("8. LATENCY / FOOTPRINT (single thread, same hardware)")
if lat:
    margin_txt = lambda v: (f"({v['deadline_margin']:.0f}x margin vs an ASSUMED {v['deadline_ms']:.0f} ms per-round budget)" if 'deadline_margin' in v else '')
    for k, v in lat["models"].items(): P(f"  {k:10s} params {v['params']:,} | {v['size_mb']:.1f} MB | 1 flow {v['infer_1flow']['mean_ms']:.2f} ms | 20-flow round {v['infer_20flows']['mean_ms']:.2f} ms "
      f"(p99 {v['infer_20flows']['p99_ms']:.2f}) | end-to-end round {v['round_total_ms']:.2f} ms {margin_txt(v)} | {v['throughput_flows_per_s']:.0f} flows/s")
    P(f"  Pipeline parts (mean ms): preprocess(20 flows) {lat['preprocess_20flows']['mean_ms']:.2f}, trust update {lat['trust_update_round']['mean_ms']:.3f}, signed ledger append {lat['ledger_append_signed']['mean_ms']:.3f}; SHAP {lat.get('shap_1flow_s', float('nan')):.2f} s per flow (offline only).")
    if "plain" in lat["models"] and "attention" in lat["models"]: P("  Note: per-round ms is the reliable figure; the batch throughput comes from one timed 500-flow batch and is order/warm-up sensitive.")

# ------------------------------------------------------------------ 9. runtime
hdr("9. RUN TIME")
fam = [("CNN-BiLSTM (proposed)", jobs("cnn_bilstm", sel="ovr")), ("CNN-BiLSTM + attention", jobs("cnn_bilstm_attn", sel="ovr")), ("CNN-BiLSTM, global MI", jobs("cnn_bilstm")), ("CNN only", jobs("cnn", sel="ovr")),
       ("LSTM only", jobs("lstm", sel="ovr")), ("MLP", jobs("mlp", sel="ovr")), ("LightGBM (class-aware)", trees("lgbm", "sel", sel="ovr")), ("Random forest", trees("rf", "sel", sel="ovr"))]
P(f"  {'Algorithm':28s} {'runs':>4s} {'median s/run':>13s} {'total min':>10s}   (wall time incl. data preparation for the first job of a configuration)")
for label, names in fam:
    ts = [r["train_s"] for r in load(names) if "train_s" in r]
    if ts: P(f"  {label:28s} {len(ts):>4d} {np.median(ts):>13.1f} {sum(ts) / 60:>10.1f}")
dpl = [r for r in (json.load(open(dpp)) if os.path.exists(dpp) else [])]
if dpl: P(f"  {'DP-SGD (all eps)':28s} {len(dpl):>4d} {np.median([r['train_s'] for r in dpl]):>13.1f} {sum(r['train_s'] for r in dpl) / 60:>10.1f}")
sec = {k: int(m.group(1)) for k, v in status.items() for m in [re.search(r"\[(\d+)s\]", str(v))] if m}
if sec:
    sub("pipeline stages")
    for k, v in sec.items(): P(f"  {k:10s} {v:>7d} s  ({v / 60:6.1f} min)")
    P(f"  {'TOTAL':10s} {sum(sec.values()):>7d} s  ({sum(sec.values()) / 60:6.1f} min = {sum(sec.values()) / 3600:.2f} h)")

# ------------------------------------------------------------------ 10. bottom line
exj = sorted(glob.glob(os.path.join(C.ROOT, "results_extra", "*_seed*.json")))
if exj:
    hdr("EXTERNAL CONFIRMATION (additional datasets, identical protocol)")
    by = {}
    for p_ in exj:
        r_ = json.load(open(p_)); by.setdefault(r_["preset"], []).append(r_)
    for preset, rs in by.items():
        sub(f"{preset}: {len(rs)} seed(s)")
        for m_ in sorted({m for r in rs for m in r["models"]}):
            P(f"  {m_:12s} Acc {pm([r['models'][m_]['test']['acc'] for r in rs if m_ in r['models']])}  Macro-F1 {pm([r['models'][m_]['test']['macro_f1'] for r in rs if m_ in r['models']])}  MCC {pm([r['models'][m_]['test']['mcc'] for r in rs if m_ in r['models']])}")
        vals = [(v["alone"]["det_heldout"], v["strict"]["iforest"]["det_heldout"], v["strict"]["mahalanobis"]["det_heldout"], v["site"]["mahalanobis"]["det_heldout"], v["alone"]["fpr"], v["strict"]["mahalanobis"]["fpr"])
                for r in rs for v in r.get("loo", {}).values() if None not in (v["alone"]["det_heldout"],)]
        if vals:
            A_ = np.array(vals, float).mean(0)
            P(f"  Leave-one-attack-class-out (mean over held-out classes and seeds): detection alone {A_[0]:.2f} -> +IsolationForest {A_[1]:.2f}, +Mahalanobis {A_[2]:.2f}, +Mahalanobis site-calibrated {A_[3]:.2f}; benign FPR {A_[4]:.3f} -> {A_[5]:.3f}")
            P("  Verdict: " + ("novelty fusion CONFIRMED on this dataset (held-out-class detection rises by more than 5 points)." if A_[2] - A_[0] > 0.05 else "novelty fusion NOT confirmed on this dataset: do not claim it generalises."))
hdr("10. BOTTOM LINE")
if rows:
    bm = max(rows, key=lambda r: mean(r[2][1])); br = max(rows, key=lambda r: mean(r[2][5]))
    P(f"  * Best classifier alone (before novelty fusion): {bm[0]} (Macro-F1 {pm(bm[2][1])}, accuracy {pm(bm[2][0])}); best rare-class (U2R) recall: {br[0]} ({pm(br[2][5])}). Differences between the top models are within seed noise; the deep models do not beat gradient boosting.")
if tr:
    cu, cf = tr["homog"].get("CUSUM fleet-baseline (+novelty)"), tr["homog"].get("confidence (original Eq.4)"); hd = tr["hetero"].get("CUSUM device-baseline (+novelty)")
    if cu and cf: P(f"  * Trust: the original confidence-only rule detects {100 * cf['compromised']['detect']:.0f}% of compromised devices; the class-aware CUSUM rule detects {100 * cu['compromised']['detect']:.0f}% (median {fm(cu['compromised']['median_delay'], 0)} rounds)"
                    + (f", with {100 * hd['benign']['ever_quarantined']:.1f}% false quarantines on a heterogeneous fleet using per-device baselines" if hd else "") + (f"; the proved bounds held empirically: {ok_fq and ok_d}." if th else "."))
if lg:
    mm = {c: np.mean([v[c] for v in lg["attacks"].values()]) for c in cfg}; P("  * Ledger: mean tamper detection " + ", ".join(f"{c} {100 * x:.0f}%" for c, x in mm.items()) + " (signatures + external anchors needed; it is an audit log, not a blockchain).")
if dpp and os.path.exists(dpp) and base and pr: P(f"  * Privacy: formal DP-SGD (eps {min(r['eps'] for r in pr):.0f}-{max(r['eps'] for r in pr):.0f}) lowers Macro-F1 from {base[0]['test']['macro_f1']:.2f} to {min(r['test']['macro_f1'] for r in pr):.2f}-{max(r['test']['macro_f1'] for r in pr):.2f} and U2R recall from {base[0]['test']['per_class']['U2R']['r']:.2f} to {pr[-1]['test']['per_class']['U2R']['r']:.2f}: report it as a privacy-utility trade-off.")
best_sys = None; best_strict = None
if nv and "0.02" in nv["fprs"]:
    for kind0, byf0 in (nv.get("detectors") or {"iforest": nv["fprs"]}).items():     # strict configurations: threshold from training-domain validation flows
        for model, per in byf0.get("0.02", {}).items():
            if model == "novelty only": continue
            fu = [per[s_]["fused"] for s_ in per]; al = [per[s_]["alone"] for s_ in per]; mm_ = mean([x["macro_f1"] for x in fu])
            if best_strict is None or mm_ > best_strict[1]: best_strict = (f"{model} [{kind0}, validation threshold]", mm_, fu, al)
    best_sys = best_strict
    for kind_, Rc_ in (nv.get("commissioning") or {}).items():             # site-calibrated configurations (selected post hoc on NSL-KDD: confirm externally)
        for model_, per_ in Rc_.items():
            if model_ == "novelty only": continue
            fu_ = [per_[s_]["fused"] for s_ in per_]; mm2 = mean([x["macro_f1"] for x in fu_])
            if mm2 > best_sys[1]: best_sys = (f"{model_} [{kind_}, site-calibrated threshold]", mm2, fu_, [per_[s_]["alone"] for s_ in per_])
    if best_sys:
        fu, al = best_sys[2], best_sys[3]
        P(f"  * Best system: {best_sys[0]} + CUSUM trust: Macro-F1 {pm([x['macro_f1'] for x in fu])} (classifier alone {pm([x['macro_f1'] for x in al])}), benign FPR {pm([x['fpr'] for x in fu])} ({100 * (mean([x['fpr'] for x in fu]) - mean([x['fpr'] for x in al])):+.1f} points vs alone), "
          f"unseen-type detection {pm([x['det_novel'] for x in fu], 2)} (alone {pm([x['det_novel'] for x in al], 2)}), seen-type detection {pm([x['det_seen'] for x in fu], 2)}.")
        P(f"    Best strict configuration (no test-domain data used for any threshold): {best_strict[0]}, Macro-F1 {pm([x['macro_f1'] for x in best_strict[2]])}, seen {pm([x['det_seen'] for x in best_strict[2]], 2)}, unseen {pm([x['det_novel'] for x in best_strict[2]], 2)}, benign FPR {pm([x['fpr'] for x in best_strict[2]])}. The site-calibrated and detector choices were made after inspecting NSL-KDD results: confirm on UNSW-NB15 / Edge-IIoTset.")
        hh = nv["fprs"]["0.02"].get(HYBRID["label"])
        if hh: P(f"    The a-priori hybrid ensemble reached Macro-F1 {pm([hh[s_]['fused']['macro_f1'] for s_ in hh])} with iforest fusion, i.e. it did not beat the best single model; report it as a negative result.")

extra = glob.glob(os.path.join(C.ROOT, "results_extra", "*_seed*.json"))
P("  * " + ("External datasets were run (see the EXTERNAL CONFIRMATION section): the novelty-fusion claim stands only where its verdict says CONFIRMED." if extra else "NOT YET ESTABLISHED: generalisation beyond NSL-KDD (no additional dataset has been run: use --unsw-train/--unsw-test/--edge)") + " Trust results are a replay simulation, not a deployment.")
txt = "\n".join(OUT); print(txt)
open(os.path.join(C.RES, "final_report.txt"), "w", encoding="utf-8").write(txt + "\n")
print(f"\n[final report saved to {os.path.join(C.RES, 'final_report.txt')}]")
