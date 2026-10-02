"""Stage 11: manuscript figures (IEEE column width, colour + hatch/linestyle coding). Each figure is skipped if its inputs are missing."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, warnings; warnings.filterwarnings("ignore")
from ztids import config as C
from ztids.data import CLASSES, make_split
from ztids.trust import gen_evidence, run_rule
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "font.family": "serif"})
def J(n):
    p = os.path.join(C.RES, n + ".json"); return json.load(open(p)) if os.path.exists(p) else None
def save(fig, name): fig.savefig(os.path.join(C.FIG, name + ".pdf")); fig.savefig(os.path.join(C.FIG, name + ".png"), dpi=200); plt.close(fig); print("figure:", name, flush=True)
def guard(f):
    try: f()
    except Exception as e: print("skipped", f.__name__, "->", repr(e)[:120], flush=True)

def fig_trust_traj():
    from ztids import evidence as EV_
    E_ = EV_.load(); yt, svc, EV = E_["yt"], E_["svc"], E_["EVf"]; Pn = EV[(yt == 0) & (svc == "http")]; Pa = {k: EV[yt == k] for k in (1, 2, 3, 4)}; M, W, k, h = 20, 10, 0.05, 0.6
    rules = [("Confidence-only rule (original)", "confidence", "#7f7f7f", ":"), ("Beta reputation", "beta", "#1f77b4", "--"), ("Proposed CUSUM trust", "cusum", "#d62728", "-")]
    scn = [("(a) Benign", "benign"), ("(b) Compromised at t=15", "compromised"), ("(c) Compromised t=15-30, then repaired", "recovery"), ("(d) Intermittent bursts (p=0.3)", "intermittent")]
    fig, ax = plt.subplots(2, 2, figsize=(7.16, 3.9), sharex=True, sharey=True)
    for a, (title, sc) in zip(ax.ravel(), scn):
        E, fr = gen_evidence(Pn, Pa, sc, np.random.RandomState(11), 60, M, 15, 30, attack_cls=3); mu0 = E[:W, 0].mean()
        for name, rule, col, ls in rules:
            T, S = run_rule(E, rule, mu0=mu0, k=k, h=h, warm=W)
            if rule == "cusum":
                T = np.where(np.arange(60) < W, 1.0, T); q = np.array([s == "Quarantined" for s in S]); a.fill_between(range(60), 0, 1, where=q, color="#d62728", alpha=0.12, hatch="//", lw=0)
            a.plot(T, color=col, ls=ls, label=name)
        if sc == "intermittent": a.scatter(np.where(fr > 0.5)[0], np.full((fr > 0.5).sum(), 1.03), marker="v", s=8, color="k", clip_on=False)
        a.set_title(title, fontsize=7); a.set_ylim(-0.03, 1.08); a.grid(alpha=0.25, lw=0.4)
    for a in ax[1]: a.set_xlabel("Inference round t")
    for a in ax[:, 0]: a.set_ylabel("Trust score T")
    h_, l_ = ax[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7); fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "trust_trajectories")

def fig_trust_theory():
    th = J("trust_theory"); fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for rep, mk, ls in [(1, "o", "-"), (2, "s", "--"), (4, "^", ":")]:
        rows = [r for r in th["false_quarantine"] if r["rep"] == rep]; ax[0].plot([r["alpha"] for r in rows], [max(r["empirical"], 1e-3) for r in rows], marker=mk, ls=ls, ms=3, label=f"$m_{{eff}}={20 // rep}$")
    ax[0].plot([1e-2, 0.5], [1e-2, 0.5], "k-", lw=0.6, label="bound"); ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"Certified level $\alpha$")
    ax[0].set_ylabel("Empirical rate (0 plotted at $10^{-3}$)"); ax[0].legend(fontsize=5.5, frameon=False); ax[0].set_title("(a) False quarantine", fontsize=7)
    dl = [r for r in th["delay"] if r["bound"] is not None and r["mean_delay"] is not None]; x = np.arange(len(dl))
    ax[1].bar(x - 0.18, [r["mean_delay"] for r in dl], 0.36, label="measured mean", color="#d62728", hatch="//"); ax[1].bar(x + 0.18, [r["bound"] for r in dl], 0.36, label=r"bound $(h+1)/d$", color="#1f77b4", hatch="..")
    ax[1].set_xticks(x); ax[1].set_xticklabels([r["cls"] for r in dl]); ax[1].set_ylabel("Rounds to quarantine"); ax[1].legend(fontsize=5.5, frameon=False); ax[1].set_title("(b) Detection delay", fontsize=7)
    for c, mk, ls in [("DoS", "o", "-"), ("R2L", "s", "--")]:
        rows = [r for r in th["min_detectable"] if r["cls"] == c]
        ax[2].plot([r["f"] for r in rows], [r["detect"] for r in rows], marker=mk, ls=ls, ms=3, label=c)
        if rows[0]["fstar"]: ax[2].axvline(rows[0]["fstar"], color="k", ls=ls, lw=0.6)
    ax[2].set_xlabel("Attack fraction of flows $f$"); ax[2].set_ylabel("Detection probability"); ax[2].legend(fontsize=5.5, frameon=False); ax[2].set_title(r"(c) Threshold $f^{*}$ (vertical)", fontsize=7)
    for a in ax: a.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(); save(fig, "trust_theory")

def fig_cm():
    cm = np.array(J(C.PROPOSED)["test"]["cm"], float); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.4, 3.0)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5): ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=5.5, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASSES, rotation=30); ax.set_yticks(range(5)); ax.set_yticklabels(CLASSES); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); save(fig, "confusion_matrix")

def fig_recall():
    items = [("CNN-BiLSTM (proposed)", C.PROPOSED, "#d62728", "//"), ("+ dual attention", C.ATTN, "#ff7f0e", "\\\\"), ("CNN-BiLSTM, global MI", "official_cnn_bilstm_0_-_-_-", "#7f7f7f", ".."),
             ("LightGBM", "official_lgbm_0_sel_selovr", "#1f77b4", "xx"), ("MLP", "official_mlp_0_-_-_-_selovr", "#2ca02c", "--")]
    items = [(l, J(n), c, h) for l, n, c, h in items if J(n)]; w = 0.8 / len(items); fig, ax = plt.subplots(figsize=(7.16, 2.3))
    for i, (l, r, c, h) in enumerate(items): ax.bar(np.arange(5) + i * w, [r["test"]["per_class"][k]["r"] for k in CLASSES], w, label=l, color=c, hatch=h, edgecolor="k", lw=0.3)
    ax.set_xticks(np.arange(5) + 0.4 - w / 2); ax.set_xticklabels(CLASSES); ax.set_ylabel("Recall (seed 0)"); ax.legend(fontsize=6, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2)); ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout(); save(fig, "per_class_recall")

def fig_shap():
    s = J("shap"); o = np.argsort(-np.array(s["nn_global"]))[:15][::-1]; fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.8))
    ax[0].barh([s["names"][i] for i in o], [s["nn_global"][i] for i in o], color="#d62728", hatch="//", edgecolor="k", lw=0.3); ax[0].set_xlabel("mean |SHAP| (CNN-BiLSTM)"); ax[0].tick_params(axis="y", labelsize=5.5)
    ks = sorted(int(k) for k in s["deletion"]); 
    for key, ls, mk, lab in (("nn_shap", "-", "o", "CNN-BiLSTM SHAP order"), ("tree_shap", "--", "s", "TreeSHAP order"), ("random", ":", "^", "random order")):
        ax[1].plot(ks, [s["deletion"][str(k)][key] for k in ks], ls=ls, marker=mk, ms=3, label=lab)
    ax[1].set_xlabel("Features masked k"); ax[1].set_ylabel("Mean drop in predicted-class probability"); ax[1].legend(fontsize=6, frameon=False); ax[1].grid(alpha=0.25, lw=0.4)
    ax[1].set_title(f"n={s['n_samples']}, rank agreement rho={s['spearman_nn_vs_tree']:.2f}", fontsize=6.5); fig.tight_layout(); save(fig, "shap_faithfulness")

def fig_dp():
    dp = J("dp_all"); fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.2)); pr = [r for r in dp if r["eps"] is not None]; base = [r for r in dp if r["eps"] is None][0]
    ax[0].plot([r["eps"] for r in pr], [r["test"]["macro_f1"] for r in pr], "o-", ms=3); ax[0].axhline(base["test"]["macro_f1"], color="k", ls=":", lw=0.7, label="non-private"); ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\varepsilon$ ($\delta=1/N$)"); ax[0].set_ylabel("Macro-F1"); ax[0].legend(fontsize=6, frameon=False)
    ax[1].plot([r["eps"] for r in pr], [r["mia_auc"] for r in pr], "s--", ms=3); ax[1].axhline(base["mia_auc"], color="k", ls=":", lw=0.7); ax[1].axhline(0.5, color="gray", ls="-", lw=0.5); ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\varepsilon$"); ax[1].set_ylabel("Membership-inference AUC")
    for a in ax: a.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(); save(fig, "dp_tradeoff")

for f in (fig_trust_traj, fig_trust_theory, fig_cm, fig_recall, fig_shap, fig_dp): guard(f)
