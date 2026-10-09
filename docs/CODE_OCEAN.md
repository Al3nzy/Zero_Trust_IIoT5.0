# Running on Code Ocean (v3)
Environment: Python 3.12, `pip install -r requirements.txt` (no TensorFlow). Entry point: `/code/run`.
`/results` receives `results_v3/` (JSON and posteriors), the manuscript tables, figures and `numbers.tex`, and the output of `check_results.py`.
Expected run time on a few CPU cores: about 2 hours for the full pipeline; `SKIP="inflation latency"` reduces it to about 40 minutes. Latency is measured on the capsule hardware, so it differs from the manuscript.
Optional: upload a previous `results_v3/` to `/data/results_v3` to reuse finished stages; set `FULL=1` to ignore it.
