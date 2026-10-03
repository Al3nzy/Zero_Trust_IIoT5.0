# What to run (Windows PowerShell)

## 0. Update the code (keeps your results_v2, .venv and .git)
```powershell
Expand-Archive -Path "$HOME\Downloads\ztids_v2_3_1.zip" -DestinationPath C:\temp_v231 -Force
Copy-Item -Path C:\temp_v231\ztids_v2_3_1\* -Destination C:\ztids_complete\ztids_complete -Recurse -Force
cd C:\ztids_complete\ztids_complete
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Re-run the NSL-KDD stages that changed (finished training jobs are cached, about 25-30 min)
```powershell
python run_everything.py --only novelty novel trust theory sensitivity dp latency tables figures report
```
New in this run: Mahalanobis + site-calibrated novelty fusion, trust sensitivity, DP at eps 50 and 25, latency budget margin. The final report prints at the end.

## 2. Get the two external datasets (free for academic use)
* **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset , follow the download link and take the two partition files
  `UNSW_NB15_training-set.csv` (175,341 records) and `UNSW_NB15_testing-set.csv` (82,332 records). Put them in `C:\data\unsw\`.
* **Edge-IIoTset**: Kaggle dataset `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot` (also IEEE DataPort 8939). You need the file
  `ML-EdgeIIoT-dataset.csv` (folder "Selected dataset for ML and DL"). Browser download is easiest; Kaggle CLI (needs your kaggle.json token):
```powershell
pip install kaggle
kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot -f "Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv" -p C:\data\edge
Expand-Archive C:\data\edge\ML-EdgeIIoT-dataset.csv.zip -DestinationPath C:\data\edge -Force
```
Adjust the file name if Kaggle lists it differently.

## 3. Smoke test first (minutes, LightGBM only, 1 seed)
```powershell
python run_dataset.py unsw --train C:\data\unsw\UNSW_NB15_training-set.csv --test C:\data\unsw\UNSW_NB15_testing-set.csv --seeds 0 --models lgbm --out results_smoke
python run_dataset.py edge --train C:\data\edge\ML-EdgeIIoT-dataset.csv --seeds 0 --models lgbm --max-rows 120000 --out results_smoke
```
Check: the first printed line (`fit=... test=... classes=... overlap_after_split=0`) and the `LOO ...` lines. If a column name is wrong, pass `--label <column>`; if there is no class called Normal, pass `--normal-label <name>`. Then `Remove-Item results_smoke -Recurse`.

## 4. Full external run (3 seeds)
```powershell
python run_dataset.py unsw --train C:\data\unsw\UNSW_NB15_training-set.csv --test C:\data\unsw\UNSW_NB15_testing-set.csv --seeds 0 1 2 --models lgbm rf mlp
python run_dataset.py edge --train C:\data\edge\ML-EdgeIIoT-dataset.csv --seeds 0 1 2 --models lgbm rf mlp --max-rows 120000
python run_everything.py --only tables report
```
Add `cnn_bilstm` to `--models` if you also want the deep model on these datasets (slow, roughly an hour or more). The leave-one-class-out novelty test always uses LightGBM.
The report ends with an EXTERNAL CONFIRMATION section: it says CONFIRMED or NOT confirmed per dataset. Report what it says.

## 5. What to send back
```powershell
Compress-Archive -Path results_v2\*.json, results_v2\final_report.txt, results_extra, tables -DestinationPath send_to_claude.zip -Force
```
(optional) `git add .; git commit -m "v2.3 full run"; git push origin main` (the new .gitignore keeps the large .npz/.keras files out).
If anything fails, send the first error lines of the log.
