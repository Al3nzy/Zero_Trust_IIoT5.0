#!/usr/bin/env python3
"""
MASTER SCRIPT (v3, LightGBM-based, no TensorFlow needed).  Run from the repository root.

    python verify_install.py                  # 10-second check of the environment and the e-detector
    python -m pytest -q tests                 # unit tests of the guarantees (about 10-30 seconds)
    python run_everything.py --quick          # whole pipeline on a small scale (about 10-15 minutes on one core)
    python run_everything.py                  # full run (about 1.5-2 hours on one core; stages are cached and resumable)
    python run_everything.py --only trust theory assets      # re-run selected stages
    python run_everything.py --skip inflation latency        # skip slow stages

Stages, in order:
  core       audited NSL-KDD protocol, LightGBM + three Normal-profile detectors, 10 seeds        -> results_v3/core_s*.npz|json
  site       argmax / site-calibrated / Algorithm 1 / fusion, paired statistics                    -> results_v3/site.json
  selection  validation-only choice of the Normal-profile detector                                -> results_v3/selection.json
  inflation  one-factor ablation ladder of evaluation flaws (3 seeds; slowest stage)               -> results_v3/inflation_nsl.json
  trust      device-trust study: 9 rules, stress modes, contamination, stealth sweep               -> results_v3/trust_main.json
  theory     numerical validation of the guarantees (CUSUM, e-detector, delay law)                -> results_v3/theory.json
  sensitivity  commissioning length, level, flows per round, betting function                      -> results_v3/sensitivity.json
  ledger     tamper-detection experiment of the audit ledger                                       -> results_v3/ledger.json
  latency    per-round cost on ONE CPU thread (run on an idle machine / on the edge board)         -> results_v3/latency.json
  assets     tables, figures and numbers.tex of the manuscript                                     -> paper_v3/
Deep-model, DP-SGD, SHAP and extra-dataset stages are in legacy/ (TensorFlow).  Inflation on UNSW-NB15 / Edge-IIoTset:  see README (python run_inflation.py generic ...).
"""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["core", "site", "selection", "inflation", "trust", "theory", "sensitivity", "ledger", "latency", "assets"]
SCRIPT = dict(core="run_core.py", site="run_site.py", selection="run_selection.py", inflation="run_inflation.py", trust="run_trust.py", theory="run_theory.py",
              sensitivity="run_sensitivity.py", ledger="run_ledger.py", latency="run_latency.py", assets=os.path.join("paper_v3", "make_paper_assets.py"))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--quick", action="store_true"); ap.add_argument("--only", nargs="*", choices=STAGES); ap.add_argument("--skip", nargs="*", default=[], choices=STAGES)
a = ap.parse_args(); env = dict(os.environ)
if a.quick: env["ZTIDS_QUICK"] = "1"
res = os.environ.get("ZTIDS_RES") or os.path.join(HERE, "results_v3_quick" if a.quick else "results_v3"); os.makedirs(res, exist_ok=True); env["ZTIDS_RES"] = res
EST = dict(core='about 1 min per seed', site='1-2 min', selection='seconds', inflation='about 4 min per seed (5 preprocessing fits each)', trust='about 2 min (quick) / 6 min (full)', theory='1-2 min',
           sensitivity='1-3 min', ledger='1-3 min', latency='about 1 min, run on an idle machine', assets='seconds')
todo = [s for s in (a.only or STAGES) if s not in a.skip and not (a.quick and s == "assets")]       # quick results never overwrite the manuscript tables
if a.quick: print("quick mode: the `assets` stage is skipped so that small-scale numbers cannot overwrite the manuscript tables")
spath = os.path.join(res, "pipeline_status.json"); status = json.load(open(spath)) if os.path.exists(spath) else {}; t_all = time.time()
for st in todo:
    t0 = time.time(); print(f"\n{'=' * 78}\n== stage: {st}\n{'=' * 78}", flush=True)
    cmd = [sys.executable, os.path.join(HERE, SCRIPT[st])] + (["nsl"] if st == "inflation" else [])
    print(f"expected time: {EST[st]}. Some steps print nothing for a minute; do not press Ctrl+C. If you do, run the same command again: finished stages are kept.", flush=True)
    try: rc = subprocess.run(cmd, env=env, cwd=HERE).returncode
    except KeyboardInterrupt:
        status[st] = "interrupted"; json.dump(status, open(spath, "w"), indent=1); print(f"\nInterrupted during stage {st}. Run the same command again to resume (finished stages and finished seeds are cached)."); sys.exit(130)
    status[st] = ("ok" if rc == 0 else f"FAILED (exit {rc})") + f" [{time.time() - t0:.0f}s]"; json.dump(status, open(spath, "w"), indent=1)
print(f"\n{'=' * 78}\nPIPELINE SUMMARY  (total {(time.time() - t_all) / 60:.1f} min)\n{'=' * 78}")
for k in todo: print(f"  {k:12s} {status.get(k, '-')}")
print(f"\nResults: {os.path.relpath(res, HERE)}/")
sys.exit(0 if all(str(status.get(k, "ok")).startswith("ok") for k in todo) else 1)
