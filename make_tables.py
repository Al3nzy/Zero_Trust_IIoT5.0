"""Stage 10: builds every manuscript table (LaTeX tabular + markdown summary) from the result files. Missing results show as '--'."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ztids import config as C
from ztids.jobs import jid, tree_id
from ztids.data import CLASSES
from ztids.ensemble import HYBRID, metrics_over_seeds

O, S = "official", C.SEEDS
def J(name):
    p = os.path.join(C.RES, name + ".json")
    try: return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    except Exception as e: print("WARNING: unreadable result file", p, repr(e)[:80]); return None
def esc(s): return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
def ms(vals, d=3, pct=False):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v: return "--"
    k = 100 if pct else 1; m = np.mean(v) * k
    return f"{m:.{d}f}" if len(v) == 1 else f"{m:.{d}f}$\\pm${np.std(v, ddof=1) * k:.{d}f}"
def fp(x, d=2, pct=False): return "--" if x is None else (f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}")
MET = [("Acc.", lambda r: r["test"]["acc"]), ("Macro-F1", lambda r: r["test"]["macro_f1"]), ("MCC", lambda r: r["test"]["mcc"]),
       ("Normal FPR", lambda r: r["test"]["normal_fpr"]), ("R2L rec.", lambda r: r["test"]["per_class"]["R2L"]["r"]), ("U2R rec.", lambda r: r["test"]["per_class"]["U2R"]["r"])]
def metric_row(label, names):
    rs = [J(n) for n in names]; rs = [r for r in rs if r]
    deg = sum(1 for r in rs if r["test"]["attack_detection_rate"] < 0.05 or r["test"]["normal_fpr"] > 0.5)      # collapsed runs are counted, never dropped
    return [esc(label)] + [ms([f(r) for r in rs]) for _, f in MET] + [str(len(rs)) + (f" ({deg} degenerate)" if deg else "")]
def metric_row_from(label, rs):
    return [esc(label)] + [ms([f(r) for r in rs]) for _, f in MET] + [str(len(rs))]
def jobs(model, **kw): return [jid(dict(proto=O, model=model, seed=s, **kw)) for s in S]
def trees(model, feats, proto=O, **kw): return [tree_id(dict(proto=proto, model=model, seed=s, feats=feats, **kw)) for s in S]
def tab(header, rows, spec=None):
    spec = spec or "l" + "c" * (len(header) - 1)
    return "\n".join([f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"] + [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"])
tex, md = [], []
def emit(label, title, header, rows, note=None, spec=None):
    tex.append(f"% ---- {label}: {title}\n" + tab(header, rows, spec) + "\n")
    md.append(f"### {label}: {title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + ("\n\n" + note if note else "") + "\n")

# ---------- protocol audit
pj = J(C.PROPOSED)
if pj:
    a = pj["audit"]; c = a["counts"]
    emit("tab:protocol", "Official-split data protocol after the audit", ["Class", "Fit (train)", "Validation", "Test"],
         [[k] + [str(c[p][k]) for p in ("fit", "val", "test")] for k in CLASSES] + [["Total"] + [str(sum(c[p].values())) for p in ("fit", "val", "test")]],
         note=f"Test records dropped because an identical feature vector exists in training: {a.get('test_exact_overlap_with_train')} of {a.get('test_total')}. "
              f"Exact overlap remaining between fit and test after the audit: {a.get('test_overlap_after_split')}. Balanced training size per class: {pj['npc']}.")
# ---------- main comparison
H1 = ["Model"] + [m for m, _ in MET] + ["Seeds"]
rows = [metric_row("CNN-BiLSTM + class-aware MI (proposed)", jobs("cnn_bilstm", sel="ovr")),
        metric_row("CNN-BiLSTM + dual attention", jobs("cnn_bilstm_attn", sel="ovr")),
        metric_row("CNN-BiLSTM + global MI (original selection)", jobs("cnn_bilstm")),
        metric_row("LightGBM (class-aware MI, 25 feat.)", trees("lgbm", "sel", sel="ovr")), metric_row("LightGBM (global MI, 25 feat.)", trees("lgbm", "sel")),
        metric_row("LightGBM (all features)", [tree_id(dict(proto=O, model="lgbm", seed=0, feats="all"))]),
        metric_row("Random forest (class-aware MI)", trees("rf", "sel", sel="ovr")), metric_row("MLP", jobs("mlp", sel="ovr")),
        metric_row("CNN only", jobs("cnn", sel="ovr")), metric_row("LSTM only", jobs("lstm", sel="ovr"))]
hy = metrics_over_seeds(HYBRID["members"], S)
if hy: rows.insert(3, metric_row_from(HYBRID["label"] + ", equal-weight posteriors", hy))
emit("tab:main", "Official overlap-free test set, mean $\\pm$ std over training seeds", H1, rows, spec="lccccccc")
# ---------- ablations
rows = [metric_row("canonical (family-adjacent) order", jobs("cnn_bilstm", sel="ovr")), metric_row("random feature order", jobs("cnn_bilstm", sel="ovr", order="random")),
        metric_row("descending-MI order", jobs("cnn_bilstm", sel="ovr", order="mi")), metric_row("Laplace input noise, scale 1/50 (augmentation)", [jid(dict(proto=O, model="cnn_bilstm", seed=0, sel="ovr", laplace=50))]),
        metric_row("Laplace input noise, scale 1/10 (augmentation)", [jid(dict(proto=O, model="cnn_bilstm", seed=0, sel="ovr", laplace=10))])]
emit("tab:ablation", "Sequence-order and noise-augmentation ablations", H1, rows)
rows = [metric_row("clean training labels", jobs("cnn_bilstm", sel="ovr")), metric_row("targeted: 50% of R2L/U2R relabelled Normal", jobs("cnn_bilstm", sel="ovr", poison=("targeted", 0.5))),
        metric_row("random label noise on 30% of samples", jobs("cnn_bilstm", sel="ovr", poison=("random", 0.3)))]
emit("tab:poison", "Label-poisoning experiments", H1, rows)
rows = [metric_row("official split (proposed)", jobs("cnn_bilstm", sel="ovr")),
        metric_row("deduplicated random split (proposed)", [jid(dict(proto="random", model="cnn_bilstm", seed=s, sel="ovr")) for s in S]),
        metric_row("deduplicated random split (LightGBM)", trees("lgbm", "sel", proto="random", sel="ovr"))]
emit("tab:indist", "Official versus deduplicated random split", H1, rows)
# ---------- novel attacks
nv = J("novel")
if nv:
    rows = [[esc(k), str(v["seen"]["n"]), fp(v["seen"]["exact_recall"], 1, True), fp(v["seen"]["detection"], 1, True), str(v["novel"]["n"]), fp(v["novel"]["exact_recall"], 1, True), fp(v["novel"]["detection"], 1, True)] for k, v in nv["models"].items()]
    emit("tab:novel", "Attack recall (\\%) on sub-types seen in training versus test-only sub-types", ["Model", "n seen", "Exact", "Detected", "n novel", "Exact", "Detected"], rows,
         note="Exact = correct 5-class label; Detected = any non-Normal label. Novel sub-types: " + ", ".join(nv["novel_subtypes"]))
# ---------- novelty-aware fusion
nvl = J("novelty")
if nvl:
    for fpr, Rf in nvl["fprs"].items():
        rows = []
        for model, per in Rf.items():
            if model == "novelty only":
                v = list(per.values()); rows.append(["Normal-profile detector alone", "--", ms([x["fpr"] for x in v]), ms([x["det_seen"] for x in v]), ms([x["det_novel"] for x in v]), "--", "--"])
            else:
                for mode in ("alone", "fused"):
                    v = [per[s][mode] for s in per]
                    rows.append([esc(model) + (" + novelty" if mode == "fused" else ""), ms([x["macro_f1"] for x in v]), ms([x["fpr"] for x in v]), ms([x["det_seen"] for x in v]), ms([x["det_novel"] for x in v]), ms([x["r2l"] for x in v]), ms([x["u2r"] for x in v])])
        emit(f"tab:novelty_fpr{fpr}", f"Seen vs novel attack detection, novelty threshold at {float(fpr) * 100:.0f}% validation FPR", ["Detector", "Macro-F1", "Benign FPR", "Det. seen", "Det. novel", "R2L rec.", "U2R rec."], rows,
             note=f"{nvl['n_attack_novel']} of {nvl['n_attack_test']} test attacks belong to sub-types absent from training. Threshold set on validation Normal flows only.")
# ---------- site-calibrated (commissioning-window) novelty fusion
if nvl and nvl.get("commissioning"):
    for kind, Rc in nvl["commissioning"].items():
        if not Rc: continue
        rows = []
        for model, per in Rc.items():
            if model == "novelty only":
                v = list(per.values()); rows.append(["Normal-profile detector alone", "--", ms([x["fpr"] for x in v]), ms([x["det_seen"] for x in v]), ms([x["det_novel"] for x in v]), "--", "--"])
            else:
                for mode in ("alone", "fused"):
                    v = [per[s][mode] for s in per]
                    rows.append([esc(model) + (" + novelty" if mode == "fused" else ""), ms([x["macro_f1"] for x in v]), ms([x["fpr"] for x in v]), ms([x["det_seen"] for x in v]), ms([x["det_novel"] for x in v]), ms([x["r2l"] for x in v]), ms([x["u2r"] for x in v])])
        emit("tab:novelty_comm_" + kind.replace("+", "_"), f"Site-calibrated novelty fusion ({kind}); threshold from a clean commissioning window", ["Detector", "Macro-F1", "Benign FPR", "Det. seen", "Det. novel", "R2L rec.", "U2R rec."], rows,
             note="Threshold = 98th percentile of anomaly scores on a random 10% of clean test-domain Normal flows (excluded from evaluation), mean of 20 windows. Requires a clean benign commissioning period at the site.")
# ---------- novelty detector comparison
if nvl and nvl.get("detectors"):
    rows = []
    for kind, byf in nvl["detectors"].items():
        Rk = byf.get("0.02", {}).get("novelty only"); au = nvl.get("auc", {}).get(kind, {})
        if Rk: rows.append([kind, ms([v["all"] for v in au.values()]), ms([v["novel"] for v in au.values()]), ms([v["fpr"] for v in Rk.values()]), ms([v["det_seen"] for v in Rk.values()]), ms([v["det_novel"] for v in Rk.values()])])
    emit("tab:detectors", "Normal-profile detectors (threshold at 2% validation FPR)", ["Detector", "AUROC all", "AUROC unseen types", "Benign flag rate", "Det. seen", "Det. unseen"], rows,
         note="AUROC is threshold-free. Exploratory (seed 0, not part of the pipeline): kNN-distance had the best validation AUROC (0.993) but the worst test AUROC (0.932; 0.890 for unseen types) and flagged 8.2% of test benign flows at a 2% validation threshold, so it was rejected. "
              "The detector ranking was inspected on NSL-KDD test data and must be confirmed on UNSW-NB15 / Edge-IIoTset.")
# ---------- site-calibrated classifier decision (NSL-KDD)
if nvl and nvl.get("site_decision"):
    rows = []
    for model, per in nvl["site_decision"].items():
        for key in ("argmax", "0.02", "0.05", "0.1"):
            v = [per[s][key] for s in per if key in per[s]]
            if v: rows.append([esc(model), "argmax" if key == "argmax" else f"site alpha={key}", ms([x["macro_f1"] for x in v]), ms([x["acc"] for x in v]), ms([x["mcc"] for x in v]), ms([x["fpr"] for x in v]), ms([x["attack_det"] for x in v])])
    emit("tab:site_decision", "Classifier decision: argmax vs site-calibrated Normal threshold (all targets shown)", ["Model", "Decision", "Macro-F1", "Acc.", "MCC", "Benign FPR", "Attack det."], rows,
         note="Threshold on P(attack) from a clean commissioning window (10% of test-domain Normal flows, excluded from evaluation); mean of 20 windows. No target is selected: choose from the application's false-alarm budget.")
# ---------- trust
tr = J("trust_main")
if tr:
    rows = []
    for fl in ("homog", "hetero"):
        for name, v in tr[fl].items():
            if not isinstance(v, dict) or "benign" not in v: continue
            b, c, r, s = v["benign"], v["compromised"], v["recovery"], v["stealth"]; sn, nv = v.get("seen_attack"), v.get("novel_attack")
            rows.append([fl, esc(name), fp(b["ever_quarantined"], 1, True), fp(c["detect"], 0, True), "--" if c["median_delay"] is None else f"{c['median_delay']:.0f}", fp(r["released_by_end"], 0, True), fp(s["detect"], 0, True),
                         "--" if sn is None else fp(sn["detect"], 0, True), "--" if nv is None else fp(nv["detect"], 0, True)])
    emit("tab:trust", "Trust-update rules on replayed real posteriors", ["Fleet", "Rule", "False quar. (\\%)", "Detect (\\%)", "Delay (rounds)", "Released (\\%)", "Stealth 30\\% (\\%)", "Seen sub-type (\\%)", "Novel sub-type (\\%)"], rows, spec="llccccccc",
         note=f"h={tr['setup']['h']:.3f}, k={tr['setup']['k']}, m={tr['setup']['m']} flows/round, {tr['setup']['n_devices']} devices per cell. Replay simulation, not a deployment.")
    rows = [[f"{v['frac']:.2f}", v["mode"], fp(v["detect_contaminated"], 0, True), fp(v["false_quarantine_benign"], 1, True)] for v in tr["contamination"].values()]
    emit("tab:contam", "Compromise during baseline commissioning", ["Contaminated fraction", "Baseline", "Detected (\\%)", "False quar. (\\%)"], rows)
th = J("trust_theory")
if th:
    rows = [[r["cls"], fp(r["mu_a"], 3), fp(r["drift"], 3), fp(r["bound"], 2), fp(r["mean_delay"], 2), fp(r["detect"], 2, True)] for r in th["delay"]]
    emit("tab:delay", "Detection-delay bound versus measurement", ["Attack", "$\\mu_a$", "Drift $d$", "Bound $(h+1)/d$", "Measured mean", "Detected (\\%)"], rows)
    rows = [[str(r["rep"]), f"{r['m_eff']:.0f}", fp(r["alpha"], 2), fp(r["empirical"], 3)] for r in th["false_quarantine"]]
    emit("tab:fqbound", "False-quarantine probability versus certified level", ["Flow repetition", "$m_{eff}$", "Certified $\\alpha$", "Empirical rate"], rows)
# ---------- trust sensitivity
ss = J("trust_sensitivity")
if ss:
    rows = [[f"{r['k']:.2f}", f"{r['alpha']:g}", str(r["m"]), str(r["warm"]), f"{r['h']:.2f}", fp(r["false_quarantine"], 1, True), fp(r["detect"], 0, True), "--" if r["median_delay"] is None else f"{r['median_delay']:.0f}", fp(r["stealth30"], 0, True)] for r in ss["rows"]]
    emit("tab:sensitivity", "Trust-engine sensitivity (per-device baseline, replayed evidence)", ["$k$", "$\\alpha$", "$m$", "Warm-up", "$h$", "False quar. (\\%)", "Detect (\\%)", "Delay", "Stealth 30\\% (\\%)"], rows, spec="cccccccc c".replace(" ", ""),
         note=f"{ss['n_devices']} devices per cell; h = ln(1/alpha) / (8 m k).")
# ---------- ledger
lg = J("ledger")
if lg:
    cf = list(next(iter(lg["attacks"].values())).keys()); rows = [[esc(k)] + [fp(v[c], 1, True) for c in cf] for k, v in lg["attacks"].items()]
    rows.append(["untampered chain accepted"] + [("yes" if lg["untampered_verifies"][c] else "no") for c in cf])
    emit("tab:ledger", "Tamper detection rate (\\%) of ledger variants", ["Attack"] + [esc(c) for c in cf], rows,
         note="append (us): " + ", ".join(f"{k}={v:.1f}" for k, v in lg["perf"].items() if k.startswith("append")) + " | verify (s): " + ", ".join(f"{k}={v:.2f}" for k, v in lg["perf"].items() if k.startswith("verify")))
# ---------- DP
dp = J("dp_all") if os.path.exists(os.path.join(C.RES, "dp_all.json")) else None
if dp:
    rows = [["non-private" if r["eps"] is None else f"{r['eps']:.2f}", "--" if r["eps"] is None else f"{r['sigma']:.2f}", ms([r["test"]["acc"]]), ms([r["test"]["macro_f1"]]), ms([r["test"]["mcc"]]), fp(r["test"]["per_class"]["U2R"]["r"], 3), fp(r["mia_auc"], 3)] for r in dp]
    emit("tab:dp", "DP-SGD (RDP accountant) privacy-utility trade-off", ["$\\varepsilon$", "$\\sigma$", "Acc.", "Macro-F1", "MCC", "U2R rec.", "MIA AUC"], rows,
         note=f"N={dp[0]['N']} records, delta=1/N, {dp[0]['epochs']} epochs, batch {dp[0]['batch']}, clip {dp[0]['clip']}. MIA AUC = loss-threshold membership inference (0.5 = chance).")
# ---------- latency
lt = J("latency")
if lt:
    rows = [[k, f"{v['params']:,}", fp(v["size_mb"], 1), fp(v["infer_1flow"]["mean_ms"], 2), fp(v["infer_20flows"]["mean_ms"], 2), fp(v["infer_20flows"]["p99_ms"], 2), fp(v["throughput_flows_per_s"], 0), fp(v["round_total_ms"], 2), (fp(v["deadline_margin"], 0) + "x") if "deadline_margin" in v else "--"] for k, v in lt["models"].items()]
    emit("tab:latency", "Per-round latency on identical hardware (1 thread)", ["Model", "Params", "MB", "1 flow (ms)", "20 flows (ms)", "p99 (ms)", "flows/s", "Round total (ms)", "Budget margin"], rows,
         note=f"Stages (mean ms): preprocess(20)={lt['preprocess_20flows']['mean_ms']:.2f}, trust update={lt['trust_update_round']['mean_ms']:.3f}, signed ledger append={lt['ledger_append_signed']['mean_ms']:.3f}; "
              f"SHAP per flow={lt.get('shap_1flow_s', float('nan')):.2f} s; peak RSS={'n/a' if lt['peak_rss_mb'] is None else format(lt['peak_rss_mb'], '.0f')} MB; hardware: {lt['hw']['platform']}, {lt['hw']['cpu_count']} cores.")
# ---------- extra datasets (fail-safe: an error here never prevents the other tables from being written)
try:
    ex = sorted(glob.glob(os.path.join(C.ROOT, "results_extra", "*_seed*.json"))); exr = []
    for p in ex:
        try: exr.append(json.load(open(p, encoding="utf-8")))
        except Exception as e: print("WARNING: unreadable result file", p, repr(e)[:80])
    if exr:
        bykey = {}
        for r in exr:
            for m, v in r["models"].items(): bykey.setdefault((r["preset"], m), []).append(v)
        rows = [[k[0], k[1], ms([x["test"]["acc"] for x in v]), ms([x["test"]["macro_f1"] for x in v]), ms([x["test"]["mcc"] for x in v]), str(len(v))] for k, v in sorted(bykey.items())]
        emit("tab:extra", "Additional corpora under the identical leakage-free protocol", ["Dataset", "Model", "Acc.", "Macro-F1", "MCC", "Seeds"], rows)
        rows = []
        for (preset, m), v in sorted(bykey.items()):
            for key in ("argmax", "0.02", "0.05", "0.1"):
                w = [x["site_decision"][key] for x in v if "site_decision" in x and key in x["site_decision"]]
                if w: rows.append([preset, m, "argmax" if key == "argmax" else f"site alpha={key}", ms([q["macro_f1"] for q in w]), ms([q["acc"] for q in w]), ms([q["fpr"] for q in w]), ms([q["attack_det"] for q in w])])
        if rows: emit("tab:extra_site", "Additional corpora: argmax vs site-calibrated Normal threshold", ["Dataset", "Model", "Decision", "Macro-F1", "Acc.", "Benign FPR", "Attack det."], rows)
        lrows = []
        for preset in sorted({r["preset"] for r in exr}):
            per_class = {}
            for r in exr:
                if r["preset"] == preset:
                    for cname, v in r.get("loo", {}).items():
                        if "auroc" in v.get("site", {}).get("mahalanobis", {}): per_class.setdefault(cname, []).append(v)
            for cname, vs in per_class.items():
                g = lambda f: ms([f(v) for v in vs]); st = lambda v: v["site"]["mahalanobis"]
                lrows.append([preset, esc(cname), str(vs[0]["n_test"]), g(lambda v: st(v)["auroc"]["clf"]), g(lambda v: st(v)["auroc"]["det"]), g(lambda v: st(v)["auroc"]["fused"]),
                              g(lambda v: st(v)["alpha"]["0.05"]["clf"]["det_heldout"]), g(lambda v: st(v)["alpha"]["0.05"]["det"]["det_heldout"]), g(lambda v: st(v)["alpha"]["0.05"]["fused"]["det_heldout"]), g(lambda v: st(v)["alpha"]["0.05"]["fused"]["fpr"])])
        if lrows: emit("tab:extra_loo", "Leave-one-attack-class-out at matched ~5% benign FPR (site-calibrated, Mahalanobis detector)", ["Dataset", "Held-out", "n", "AUROC clf", "AUROC det", "AUROC fused", "Det. clf", "Det. detector", "Det. fused", "FPR fused"], lrows, spec="llcccccccc")
except Exception:
    import traceback; traceback.print_exc(); print("WARNING: additional-dataset tables skipped because of the error above")
open(os.path.join(C.TAB, "tables.tex"), "w", encoding="utf-8").write("\n".join(tex)); open(os.path.join(C.TAB, "summary.md"), "w", encoding="utf-8").write("\n".join(md))
print(f"wrote {len(tex)} tables -> {C.TAB}/tables.tex and summary.md")
