#!/usr/bin/env python3
"""
MASTER SCRIPT. Run this file; it executes every experiment stage in order, never aborts on one failed stage, and writes a status report.

    python run_everything.py --quick          # smoke test (tiny data, ~15-30 min on a laptop CPU): verifies the whole pipeline
    python run_everything.py                  # full run (hours; much faster with a GPU for the detector/DP stages)
    python run_everything.py --only trust theory ledger tables figures      # re-run selected stages
    python run_everything.py --skip dp latency                              # skip slow stages
    python run_everything.py --unsw-train UNSW_NB15_training-set.csv --unsw-test UNSW_NB15_testing-set.csv \
                             --edge ML-EdgeIIoT-dataset.csv                 # also run the additional-dataset stage

Stages (in order): detector novelty novel trust theory sensitivity ledger dp shap latency extra tables figures report
(the final 'report' prints the overall results summary and run times at the very end of the log; `--only report` re-prints it from saved results)
Finished jobs are cached in results_v2/ (or results_v2_quick/), so an interrupted run resumes where it stopped.
"""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["detector", "novelty", "novel", "trust", "theory", "sensitivity", "ledger", "dp", "shap", "latency", "extra", "tables", "figures", "report"]
SCRIPT = dict(detector="run_detector.py", novelty="run_novelty.py", novel="run_novel.py", trust="run_trust.py", theory="run_theory.py", sensitivity="run_sensitivity.py", ledger="run_ledger.py", dp="run_dp.py",
              shap="run_shap.py", latency="run_latency.py", tables="make_tables.py", figures="make_figs.py", report="make_report.py")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--quick", action="store_true"); ap.add_argument("--only", nargs="*", choices=STAGES); ap.add_argument("--skip", nargs="*", default=[], choices=STAGES)
ap.add_argument("--seeds", nargs="*", type=int); ap.add_argument("--npc", type=int); ap.add_argument("--epochs", type=int); ap.add_argument("--threads", type=int)
ap.add_argument("--unsw-train"); ap.add_argument("--unsw-test"); ap.add_argument("--edge"); ap.add_argument("--edge-max-rows", type=int, default=120000)
a = ap.parse_args()
env = dict(os.environ)
if a.quick: env["ZTIDS_QUICK"] = "1"
if a.seeds: env["ZTIDS_SEEDS"] = ",".join(map(str, a.seeds))
if a.npc: env["ZTIDS_NPC"] = str(a.npc)
if a.epochs: env["ZTIDS_EPOCHS"] = str(a.epochs)
if a.threads is not None: env["ZTIDS_THREADS"] = str(a.threads)
res = os.path.join(HERE, "results_v2_quick" if a.quick else "results_v2"); os.makedirs(res, exist_ok=True)
todo = [s for s in (a.only or STAGES) if s not in a.skip]
spath = os.path.join(res, "pipeline_status.json")
status = json.load(open(spath)) if os.path.exists(spath) else {}      # merged, so a partial re-run never erases earlier stage timings
run_now = [s for s in todo if s != "report"]; t_all = time.time()
for st in run_now:
    t0 = time.time(); print(f"\n{'=' * 78}\n== stage: {st}\n{'=' * 78}", flush=True)
    if st == "extra":
        cmds = []
        if a.unsw_train and a.unsw_test: cmds.append(["unsw", "--train", a.unsw_train, "--test", a.unsw_test])
        if a.edge: cmds.append(["edge", "--train", a.edge, "--max-rows", str(a.edge_max_rows)])
        if not cmds: print("no --unsw-train/--unsw-test/--edge given: stage skipped"); status[st] = "skipped (no dataset paths given)"; continue
        rc = 0
        for c in cmds:
            extra = ["--seeds", *(map(str, a.seeds or ([0] if a.quick else [0, 1, 2])))] + (["--npc", "300", "--epochs", "2"] if a.quick else [])
            rc |= subprocess.run([sys.executable, os.path.join(HERE, "run_dataset.py"), *c, *extra], env=env, cwd=HERE).returncode
    else:
        rc = subprocess.run([sys.executable, os.path.join(HERE, SCRIPT[st])], env=env, cwd=HERE).returncode
    status[st] = ("ok" if rc == 0 else f"FAILED (exit {rc})") + f" [{time.time() - t0:.0f}s]"
    json.dump(status, open(os.path.join(res, "pipeline_status.json"), "w"), indent=1)
print(f"\n{'=' * 78}\nPIPELINE SUMMARY  (total {(time.time() - t_all) / 60:.1f} min)\n{'=' * 78}")
for k in run_now: print(f"  {k:12s} {status.get(k, '-')}")
tab, fig = ("tables_quick", "figures_quick") if a.quick else ("tables", "figures")
print(f"\nResults: {os.path.relpath(res, HERE)}/   Tables: {tab}/tables.tex + summary.md   Figures: {fig}/")
ok = all(str(status.get(k, "ok")).startswith(("ok", "skipped")) for k in run_now)
if "report" in todo:
    subprocess.run([sys.executable, os.path.join(HERE, "make_report.py")], env=env, cwd=HERE)      # final overall results + run times, printed last
sys.exit(0 if ok else 1)
