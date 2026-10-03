# What to run (Windows PowerShell), v2.4

## Where to put the datasets
Anywhere works; the scripts take full paths. Easiest: `C:\ztids_complete\ztids_complete\data\external\unsw\` and `...\data\external\edge\` (the new `.gitignore` keeps them out of git).
Do NOT put them in the repo without that `.gitignore`: `git add .` would try to push hundreds of MB and GitHub rejects files over 100 MB. Check `git status` before committing.

## 0. Update the code (keeps results_v2, .venv, .git)
```powershell
Expand-Archive -Path "$HOME\Downloads\ztids_v2_4.zip" -DestinationPath C:\temp_v24 -Force
Copy-Item -Path C:\temp_v24\ztids_v2_4\* -Destination C:\ztids_complete\ztids_complete -Recurse -Force
Copy-Item -Path C:\temp_v24\ztids_v2_4\.gitignore -Destination C:\ztids_complete\ztids_complete -Force
cd C:\ztids_complete\ztids_complete
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. NSL-KDD: adds the site-calibrated classifier decision table (about 5 min)
```powershell
python run_everything.py --only novelty tables report
```

## 2. External datasets (the old results_extra files are in an old format: the runner recomputes them)
Edge-IIoTset first, with the console output saved so we can see any error:
```powershell
python run_dataset.py edge --train data\external\edge\ML-EdgeIIoT-dataset.csv --seeds 0 --models lgbm --max-rows 120000 --out results_smoke 2>&1 | Tee-Object edge_log.txt
```
If that prints `LOO ...` lines without errors, run everything (3 seeds):
```powershell
python run_dataset.py unsw --train data\external\unsw\UNSW_NB15_training-set.csv --test data\external\unsw\UNSW_NB15_testing-set.csv --seeds 0 1 2 --models lgbm rf mlp 2>&1 | Tee-Object unsw_log.txt
python run_dataset.py edge --train data\external\edge\ML-EdgeIIoT-dataset.csv --seeds 0 1 2 --models lgbm rf mlp --max-rows 120000 2>&1 | Tee-Object edge_log2.txt
python run_everything.py --only tables report
Remove-Item results_smoke -Recurse -Force
```

## 3. Send back
```powershell
Compress-Archive -Path results_v2\*.json, results_v2\final_report.txt, results_extra, tables, edge_log.txt, unsw_log.txt, edge_log2.txt -DestinationPath send_to_claude.zip -Force
```
(skip any log file that does not exist). The report ends with an EXTERNAL CONFIRMATION section with a CONFIRMED / NOT confirmed verdict per dataset and detector.
