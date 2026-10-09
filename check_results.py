#!/usr/bin/env python3
"""Compares the results of YOUR run with the reference values stored in expected_results.json (produced by the authors' run in the same pipeline).
Exact statistical equality is not expected in --quick mode (fewer seeds/devices); there the check only verifies that every file exists, the guarantees hold and the values are in a plausible range.
    python check_results.py            # full run: every reference value must match within its tolerance
    python check_results.py --quick    # quick run: structural and guarantee checks only"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
R = os.path.join(HERE, "results_v3_quick" if a.quick else "results_v3"); ok = True
def load(n):
    p = os.path.join(R, n)
    if not os.path.exists(p): raise FileNotFoundError(p)
    return json.load(open(p))
def chk(name, cond, detail=""):
    global ok; print(("PASS  " if cond else "FAIL  ") + name + (f"   {detail}" if detail else "")); ok &= bool(cond)
def get(d, path):
    for k in path.split("/"): d = d[int(k)] if isinstance(d, list) else d[k]
    return d
# --- guarantees (must hold in both modes)
try:
    th = load("theory.json"); chk("CUSUM per-round tail within the 1% design level", all(r["per_round_tail"] <= 0.012 for r in th["cusum"]["rows"]), str([round(r["per_round_tail"], 4) for r in th["cusum"]["rows"]]))
    r05 = th["cusum"]["rows"][0]; chk("CUSUM alarm probability over a long horizon exceeds the per-round level (reflected walk)", r05["within"]["250"] > 5 * 0.01, f"P(alarm<=250 rounds)={r05['within']['250']:.3f}")
    chk("CUSUM horizon design controls the 250-round probability at 5%", all(r["horizon_design"]["empirical"] <= 0.05 for r in th["cusum"]["rows"]))
    chk("e-detector false quarantine below alpha_T for every evidence family", all(r["empirical"] <= r["bound"] for r in th["null"]), f"max={max(r['empirical'] for r in th['null']):.4f}")
    chk("conformal p-values uniform (KS p>0.01) and uncorrelated", th["martingale"]["ks_pvalue"] > 0.01 and abs(th["martingale"]["lag1_autocorr"]) < 0.03)
    chk("bet integrates to one", abs(th["martingale"]["bet_integral"] - 1) < 1e-6); chk("ARL >= c", th["arl"]["mean_time_to_first_alarm_lower_estimate"] >= th["arl"]["c"])
    chk("delay law dominates the detector in every configuration", all(r["dominated"] for r in th["delay_check"]))
except FileNotFoundError as e: chk(f"theory.json present ({e})", False)
try:
    tr = load("trust_main.json"); H = tr["cells"]["hetero|iid"]; ed = "E-detector, round level, device reference"
    chk("e-detector (device reference) false quarantine <= 5% on heterogeneous fleets", H[ed]["benign"]["ever_quarantined"] <= 0.05, f"{H[ed]['benign']['ever_quarantined']:.3f}")
    chk("fleet-referenced rules violate their level on heterogeneous fleets", H["E-detector, round level, fleet reference"]["benign"]["ever_quarantined"] > 0.05)
    chk("mode-conditional ranks restore validity under persistent regimes", tr["cells"]["hetero|regime"]["E-detector, round level, device reference, mode-conditional"]["benign"]["ever_quarantined"] <= 0.05)
    chk("e-detector detects compromised devices", H[ed]["compromised"]["detect"] >= 0.99)
    chk("commissioning diagnostic flags heterogeneous fleets", tr["commissioning_diagnostic_share_flagged"]["hetero"] > 0.8 and tr["commissioning_diagnostic_share_flagged"]["homog"] < 0.15)
except FileNotFoundError as e: chk(f"trust_main.json present ({e})", False)
try:
    s = load("site.json"); sm = s["summary"]
    for al in ("0.05", "0.1"): chk(f"site-calibrated benign FPR close to target {al}", abs(sm[f"site|{al}"]["fpr"]["mean"] - float(al)) < 0.012, f"{sm[f'site|{al}']['fpr']['mean']:.3f}")
    chk("Algorithm 1 benign FPR <= alpha (union bound)", sm["fused-site|0.05|mahalanobis"]["fpr"]["mean"] <= 0.05)
except FileNotFoundError as e: chk(f"site.json present ({e})", False)
for n in ("selection.json", "sensitivity.json", "ledger.json") + (() if a.quick else ("inflation_nsl.json", "latency.json")):
    try: load(n); chk(f"{n} present", True)
    except FileNotFoundError as e: chk(f"{n} present", False)
# --- reference values (full run only)
if not a.quick:
    ref = json.load(open(os.path.join(HERE, "expected_results.json")))
    for fname, items in ref.items():
        try: d = load(fname)
        except FileNotFoundError: chk(f"{fname} present", False); continue
        for path, (val, tol) in items.items():
            try: got = get(d, path); chk(f"{fname}:{path}", abs(got - val) <= tol, f"got {got:.4f}, reference {val:.4f} (tol {tol})")
            except (KeyError, TypeError) as e: chk(f"{fname}:{path}", False, f"missing {e}")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED" + ("\nHint: result files are missing. Run `python run_everything.py" + (" --quick" if a.quick else "") + "` and let it finish (it prints PIPELINE SUMMARY at the end), then run this check again." if not os.path.exists(os.path.join(R, "site.json")) else "")); sys.exit(0 if ok else 1)
