# ztids: leakage-free adaptive zero-trust intrusion detection for IIoT

## Which file to run
**`run_everything.py`**. It is the only file you need to launch. It runs every experiment stage in order, keeps going if one stage fails,
caches finished jobs (an interrupted run resumes), and finally builds all manuscript tables and figures.

## 1. Set up (once)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # use tensorflow-cpu if you have no GPU
```
The NSL-KDD files are already in `data/`.

## 2. Check that everything works (smoke test, tiny data)
```bash
python run_everything.py --quick
```
Uses a few hundred samples per class and 2 epochs, writes to `results_v2_quick/`, `tables_quick/`, `figures_quick/`.
The numbers from this run are meaningless; it only proves every stage executes. It ends with a summary showing `ok` or `FAILED` per stage.

## 3. Full run
```bash
python run_everything.py
```
Results go to `results_v2/` (a NEW folder, so results cached by an older version of this code are never reused), tables to `tables/tables.tex` + `tables/summary.md`, figures to `figures/`.
Measured on a single CPU core: one CNN-BiLSTM job takes about 12 minutes, and the full run trains roughly 35 neural models
(plus DP-SGD), i.e. many hours on a 1-core machine. A multi-core CPU or a GPU shortens this a lot. DP-SGD is the slowest stage: use
`--skip dp` first, then run it alone later (`python run_everything.py --only dp`, optionally with `ZTIDS_DP_N=30000` to subsample).

Useful options
```bash
python run_everything.py --only trust theory ledger tables figures   # re-run only some stages (all stages: detector novelty novel trust theory sensitivity ledger dp shap latency extra tables figures report)
python run_everything.py --skip dp latency                          # skip slow stages
python run_everything.py --seeds 0 1 2 3 4                          # more seeds (seed 0 is always included)
python run_everything.py --threads 8                                # TensorFlow threads (default: automatic)
```
Run the `latency` stage alone on an otherwise idle machine (it forces 1 thread so plain and attention models are timed identically).

## 4. Additional datasets (UNSW-NB15, Edge-IIoTset)
Download the files yourself, then:
```bash
python run_everything.py --only extra --unsw-train UNSW_NB15_training-set.csv --unsw-test UNSW_NB15_testing-set.csv --edge ML-EdgeIIoT-dataset.csv
# or directly:
python run_dataset.py unsw --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --seeds 0 1 2
python run_dataset.py edge --train ML-EdgeIIoT-dataset.csv --seeds 0 1 2 --max-rows 120000
```
Results land in `results_extra/` and appear in `tables/tables.tex` (tab:extra) after `python run_everything.py --only tables`.
The presets in `run_dataset.py` assume the usual column names (`attack_cat` for UNSW-NB15, `Attack_type` for Edge-IIoTset). Check the first
printed line (rows, classes, overlap) and override with `--label` / `--drop` if your copy differs. This runner has only been tested on synthetic data
with the same structure, not on the real files, so inspect the first run.

## What each stage does (and which reviewer concern it answers)
| stage | script | output | answers |
|---|---|---|---|
| detector | run_detector.py | results/*.json | train/test leakage, MI-before-split, accuracy inconsistencies, baselines (MLP/CNN/LSTM/LightGBM/RF), attention ablation, order ablation, poisoning, multi-seed statistics |
| novelty | run_novelty.py | novelty.json, novelty_s*_f*.npz | Normal-profile (isolation-forest) novelty detector fused with each classifier; detection of attack sub-types never seen in training |
| novel | run_novel.py | novel.json | per-sub-type recall of the classifiers alone, seen vs test-only sub-types |
| trust | run_trust.py | trust_main.json | adaptive zero-trust validation: benign/compromised/recovery/intermittent/stealth, heterogeneous fleets, contaminated commissioning |
| sensitivity | run_sensitivity.py | trust_sensitivity.json | sensitivity of the trust engine to k, certified alpha, flows per round, commissioning window (Reviewer 6) |
| theory | run_theory.py | trust_theory.json | empirical check of the proved false-quarantine bound, delay bound, minimum detectable attack fraction |
| ledger | run_ledger.py | ledger.json | tamper-detection of hash chain vs signatures vs anchors (blockchain claims) |
| dp | run_dp.py | dp_*.json | formal DP-SGD with RDP accounting + membership-inference check (DP claims) |
| shap | run_shap.py | shap.json | real SHAP interaction values (TreeSHAP surrogate), named features, faithfulness test |
| latency | run_latency.py | latency.json | end-to-end latency, memory, throughput; plain vs attention on identical hardware |
| extra | run_dataset.py | results_extra/ | additional datasets |
| tables / figures | make_tables.py / make_figs.py | tables/, figures/ | every table and figure for the manuscript |

## Protocol (ztids/data.py), the rules that remove the leakage
- official KDDTrain+/KDDTest+ partitions are preserved; exact duplicates removed; test rows identical to a training row are removed (and counted);
- the validation set is carved from original training records before any resampling;
- category vocabulary, scaler, MI feature selection and SMOTE/undersampling are fitted on the training partition only; resampling happens after the split;
- the `difficulty` column is never a feature; all 17 test-only attack types are kept;
- standardised features are clipped to |z| <= 5 (training-fitted scaler): the class-aware selection picks rare-event features (num_file_creations, hot, root_shell) whose z-scores reach 200, which made BatchNorm models collapse to a single class in about 40% of runs; runs that still collapse are flagged `degenerate` and counted in the tables;
- feature selection is class-aware (per-class one-vs-rest MI), and the selected features keep canonical NSL-KDD order.

## Version 2 changes (after the first full run)
1. **Training collapse fixed**: z-score clipping (`ZTIDS_CLIP`, default 5) and a normalisation switch (`ZTIDS_NORM=bn|ln`, see the run summary for the default); degenerate runs are flagged.
2. **Novelty detector added** (`ztids/novelty.py`, stage `novelty`): isolation forest on Normal training flows, threshold set on validation Normal flows. Trust evidence uses
   harm = max(severity-weighted posterior, w_u x flag); `w_u = 0.5` is a policy choice (`ztids/evidence.py`).
3. **Latency stage no longer fails on Windows** (the Unix-only `resource` module is now optional; `psutil` is used for memory).
4. Trust and theory stages now also test devices compromised by attack sub-types that were never seen in training.

## Version 2.2 changes
1. **Novelty detectors are selectable** (`ztids/novelty.py`: iforest default, mahalanobis, iforest+mahalanobis) and compared with a threshold-free AUROC. kNN-distance was evaluated and rejected
   (best validation AUROC, worst test AUROC under distribution shift). The detector ranking was inspected on NSL-KDD test data: confirm it on UNSW-NB15 / Edge-IIoTset.
2. **Hybrid ensemble (LightGBM + CNN-BiLSTM)** is evaluated as a fixed a-priori choice; it gives no gain over LightGBM and is reported as a negative result.
3. **Failure diagnostic** (`run_novel.py`): the worst-missed *seen* attack sub-type is compared between training and test (`guess_passwd`: telnet/RSTO in training, pop_3/SF with successful logins in test).
4. **DP-SGD** now also runs eps = 50 (the old claim) and eps = 25 (the ZT-DT value), besides 8, 4, 1.
5. **Real-time budget**: `run_latency.py` reports the margin against an ASSUMED per-round budget (`ZTIDS_DEADLINE_MS`, default 100): set it from your plant requirement; it is not a standard.
6. New stage `sensitivity`, new tables (`tab:detectors`, `tab:sensitivity`), final report sections 1b and the sensitivity line.

## Version 2.3 changes
1. **Site-calibrated novelty threshold** (`run_novelty.py`): besides the strict threshold (training-domain validation flows), the anomaly threshold is also calibrated on a commissioning window,
   a random 10% of clean test-domain Normal flows that are excluded from the evaluation (mean of 20 windows). It assumes a clean benign period at the site, like the per-device trust baseline.
2. **Mahalanobis detector** (Gaussian Normal profile) is reported next to IsolationForest. On NSL-KDD it is better, but that was observed on test data: treat it as a hypothesis to confirm.
3. **External confirmation** (`ztids/loo.py`, run by `run_dataset.py` unless `--no-loo`): leave-one-attack-class-out test on UNSW-NB15 / Edge-IIoTset. The classifier is trained without one attack
   class; detection of that class is compared alone vs fused with each detector. The final report prints a CONFIRMED / NOT confirmed verdict per dataset: only claim generalisation where it says CONFIRMED.
4. Findings that are NOT fixed by any code here: the NSL-KDD test R2L flows (guess_passwd on pop_3/ftp) differ from the training R2L flows; DP-SGD at eps <= 8 removes U2R recall.

## Version 2.4 changes
1. **Leave-one-class-out test redesigned.** The first version compared "detected = predicted non-Normal", which is meaningless when the classifier already flags a large share of benign flows
   (UNSW-NB15: 34-38% benign false alarms, so every held-out class looked 99% "detected" and the result said nothing about novelty). It now compares the classifier score, the detector score and
   the fused score at the SAME benign false-alarm rate (site-calibrated, 5%) and by AUROC.
2. **Site-calibrated classifier decision** (`ztids/calibrate.py`): Normal/attack threshold on P(attack) from the commissioning window, reported for alpha 2/5/10% (none selected). On NSL-KDD it
   raises LightGBM from Macro-F1 0.634 / Acc 0.776 to 0.659 / 0.814 at alpha=5% (worse at 2%).
3. `make_tables.py` / `make_report.py` no longer abort on one bad block or file; `run_dataset.py` writes results atomically and recomputes results from older versions.

## Ideas evaluated on NSL-KDD and rejected (do not re-try without new evidence)
* Equal-weight ensembles of the trained models: no gain (LightGBM + BiLSTM 0.636 vs LightGBM 0.634 Macro-F1).
* Sub-type-weighted training: worse (Macro-F1 about 0.57), and `guess_passwd` stays at 0% recall.
* kNN-distance anomaly detector: best validation AUROC, worst test AUROC under shift.
* Rank-averaged IsolationForest + Mahalanobis: same or lower Macro-F1 than Mahalanobis alone (0.664 vs 0.708 site-calibrated); poor threshold transfer with a validation threshold (6.8% benign flags).
* Site feature alignment (mapping the commissioning window's benign mean/std onto the training Normal statistics): false alarms rise from 3.6% to 17-28%, Macro-F1 falls from 0.705 to about 0.50.
Further tuning on the NSL-KDD test set would overfit it; the external datasets are the next test.

## Things you must state in the paper (limits of this code)
1. **Trust results are a replay simulation**: real test-set posteriors are sampled into simulated device streams (flows i.i.d. within a round). It is not a deployment.
2. **The attention baseline is our reimplementation** (channel + temporal attention), not the original ZT-DT code.
3. **DP scope**: the (eps, delta) guarantee is for the released weights, per original training record. It does not cover the scaler/feature selection, the class weights
   (treated as public priors), or hyper-parameter choice. Laplace input noise (`laplace` jobs) is augmentation only and carries no DP claim.
4. **Ledger**: a single-writer signed and anchored audit log; no consensus, no distributed replicas, so it is not a blockchain. Its trust assumptions (key kept outside the
   log store, anchors in a separate witness) are exactly what `run_ledger.py` tests.
5. Seeds change the validation split, resampling and initialisation; the official test set is fixed, so the std reflects training randomness only.
6. Trust thresholds are derived from the certified bound (h = ln(1/alpha) / (8 m k)), not tuned on test data. The commissioning-window assumption (devices clean while the baseline
   is estimated) is stress-tested in `tab:contam`.
7. Re-check the class-severity table in `ztids/trust.py` (SEVERITY) against your threat model.

## Layout
`ztids/` library (data, generic, models, jobs, trust, ledger, dp, dpsgd, evalutil, config) | `run_*.py` stages | `make_tables.py`, `make_figs.py` | `check_accountant.py` (optional) | `data/`
