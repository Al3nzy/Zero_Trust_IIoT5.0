# Anytime-valid zero-trust intrusion detection for Industrial IoT

Code for *Anytime-Valid Zero-Trust Intrusion Detection for Industrial IoT: Site-Calibrated Decisions and Conformal Device Trust on Audited Benchmarks*.
The pipeline (v3) needs **no TensorFlow**: the primary classifier is a deterministic LightGBM model, the statistical layers are NumPy/SciPy.

What it contains
- an **audited evaluation protocol** and a one-factor **leakage-inflation ladder** (`run_inflation.py`);
- **site-calibrated decisions** with a union-bound false-alarm guarantee (Algorithm 1, `run_site.py`);
- a **conformal e-detector for device trust** (`ztids/edetector.py`) with ARL, horizon and delay-law guarantees, a flow-level variant and a mode-conditional (Mondrian) variant;
- a device-stream simulator on real classifier outputs (`ztids/devices.py`), nine trust rules, stress tests (`run_trust.py`), numerical validation of the theory (`run_theory.py`);
- a signed, anchored audit ledger (`ztids/ledger.py`), a latency stage, and `run_traces.py` to evaluate the rules on **your own time-ordered traces**.

## Test it in VS Code (about 20 minutes in total)
1. Open the folder in VS Code. Use Python 3.10 to 3.13.
2. Create the environment and install (terminal in VS Code):
   - Windows PowerShell: `py -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt`
   - macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
   Then select the interpreter (Ctrl+Shift+P, "Python: Select Interpreter", `.venv`).
3. Run the tasks from *Terminal > Run Task* (they are defined in `.vscode/tasks.json`), or the commands below:

| step | command | time | what it checks |
|---|---|---|---|
| 1 | `python verify_install.py` | 10 s | imports, shipped NSL-KDD files, deterministic LightGBM, e-detector bound and detection, ledger |
| 2 | `python -m pytest -q tests` | about 10-30 s | the guarantees as failing-capable tests (p-values uniform and independent, Doob bound, first-alarm equivalence under freezing, delay-law dominance, Mondrian validity, CUSUM per-round vs horizon, trace script) |
| 3 | `python run_everything.py --quick` | 10-15 min | the whole pipeline on a small scale (2 seeds, 20 devices, 1 inflation seed) |
| 4 | `python check_results.py --quick` | 1 s | guarantee checks on the quick results |
| 5 | `python run_everything.py` | about 2 h on one core, resumable | full results (10 seeds; inflation 3 seeds) |
| 6 | `python check_results.py` | 1 s | every headline number against `expected_results.json` |

All numbers of the manuscript are produced by `python paper_v3/make_paper_assets.py` (tables, figures, `numbers.tex`).
If a step fails, send the printed `FAIL` lines. A finished stage is cached in `results_v3/`, so a re-run resumes where it stopped.

## Stages (`python run_everything.py --only <stage> ...`)
`core` (LightGBM + 3 Normal-profile detectors, 10 seeds) -> `site` -> `selection` -> `inflation` -> `trust` -> `theory` -> `sensitivity` -> `ledger` -> `latency` -> `assets`.

## On your own data
- **Inflation on UNSW-NB15 / Edge-IIoTset** (the experiment that quantifies how much duplicates and leakage inflate your scores):
  `python run_inflation.py generic --name unsw --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --label attack_cat --drop id label`
  `python run_inflation.py generic --name edge --train ML-EdgeIIoT-dataset.csv --label Attack_type --drop <identifier columns> --max-rows 120000`
  (3 seeds by default; output `results_v3/inflation_<name>.json`).
- **Device trust on real traces**: export one row per flow with `device,time,harm[,label]` and run `python run_traces.py traces.csv --m 20 --warm 50 --alpha 0.05`
  (`examples/example_traces.csv` shows the format). It reports the false-quarantine rate of benign devices and the delay on compromised ones, for the CUSUM and the e-detector.
- **Edge timing**: run `python run_latency.py` on the gateway (single thread, idle machine).

## Layout
`ztids/` library; `run_*.py` stages; `tests/`; `paper_v3/` manuscript sources and asset generator; `legacy/` TensorFlow stages of the earlier design (CNN-BiLSTM, DP-SGD, SHAP, extra datasets; see `legacy/README.md`);
`results_v3/` results of this version; `results_v2/`, `results_extra/` earlier runs of the deep models and of UNSW-NB15 / Edge-IIoTset (read by the asset generator).

## Reproducibility notes
LightGBM runs with `deterministic=True`, `force_row_wise=True`, one thread; the numbers of a machine with the same library versions match the reference exactly. Other versions can move macro-F1 by a few thousandths
because the rare classes (U2R: 67 test flows) amplify tiny differences. Seed-to-seed standard deviation of macro-F1 is about 0.03, so differences below that are not meaningful.
The e-detector's guarantees need exchangeability of benign rounds with the commissioning rounds of the same device (or of the same operating mode); `run_trust.py` shows what happens when this fails.

License: MIT. Cite via `CITATION.cff`.
