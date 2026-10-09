# Reproducing the results (v3)
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python verify_install.py        # 10 s
python -m pytest -q tests       # 2-4 min
python run_everything.py --quick    # 10-15 min, small scale
python run_everything.py            # full, about 2 h on one core, resumable
python check_results.py             # compares with expected_results.json
python paper_v3/make_paper_assets.py   # tables, figures and numbers.tex of the manuscript
```
Datasets: NSL-KDD ships in `data/`. UNSW-NB15 and Edge-IIoTset are obtained from their providers; see the README for the inflation commands.
Compile the manuscript in `paper_v3/` with `pdflatex main; bibtex main; pdflatex main; pdflatex main` (IEEEtran class).
