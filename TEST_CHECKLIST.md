# Checklist: what to run in VS Code and what to send back

Open the folder in VS Code, create the environment and `pip install -r requirements.txt` (see README). Then run, in this order. After each step, copy the terminal output (or the file named) and send it.

| # | Command | Time | Send back |
|---|---|---|---|
| 1 | `python verify_install.py` | 10 s | the whole output (every line should say PASS) |
| 2 | `python -m pytest -q tests` | 10-30 s | the last 10 lines (expected: `18 passed`) |
| 3 | `python run_everything.py --quick` (do not press Ctrl+C: some steps are silent for about a minute; if you do, rerun the same command, it resumes) | 10-15 min | the final "PIPELINE SUMMARY" block, and `results_v3_quick/pipeline_status.json` |
| 4 | `python check_results.py --quick` | 1 s | the whole output |
| 5 | `python run_traces.py examples/example_traces.csv` | 5 s | the printed JSON |

If 1 to 5 pass, run the full pipeline (resumable, about 2 hours on one core; you can run it overnight):

| 6 | `python run_everything.py` | ~2 h | `results_v3/pipeline_status.json` |
| 7 | `python check_results.py` | 1 s | the whole output (compares with the reference numbers of the authors' run; on a machine with the same library versions they match exactly) |

These two are the most valuable for the paper, because I could not run them (the datasets are not reachable from my environment):

| 8 | `python run_inflation.py generic --name unsw --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --label attack_cat --drop id label` | ~20 min | `results_v3/inflation_unsw.json` |
| 9 | `python run_inflation.py generic --name edge --train ML-EdgeIIoT-dataset.csv --label Attack_type --drop <identifier columns> --max-rows 120000` | ~20 min | `results_v3/inflation_edge.json` (use the same --drop list you used before for Edge-IIoTset) |

Optional but strongly recommended for acceptance: real device traces.
Export a CSV with columns `device,time,harm[,label]` from a time-ordered IIoT dataset (for example X-IIoTID, WUSTL-IIoT-2021, ToN_IoT, or your own plant), then
`python run_traces.py traces.csv --m 20 --warm 50 --alpha 0.05` and send `results_v3/traces/summary.json`. On an edge board, run `python run_latency.py` and send `results_v3/latency.json`.

If anything prints FAIL or a traceback, send the full text of the error; do not edit the code first.
