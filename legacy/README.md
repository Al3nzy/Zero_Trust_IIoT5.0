# Legacy stages (TensorFlow, deep models)

These scripts produced the deep-model results (CNN-BiLSTM and variants, DP-SGD, SHAP, UNSW-NB15 / Edge-IIoTset runs) of the earlier manuscript.
They need `tensorflow` and write to `results_v2/`. Run them from the repository root with the legacy results directory, for example

    ZTIDS_RES=results_v2 python legacy/run_detector.py          # Windows PowerShell:  $env:ZTIDS_RES="results_v2"; python legacy\run_detector.py
    ZTIDS_RES=results_v2 python legacy/run_dataset.py unsw --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --seeds 0 1 2

The shipped `results_v2/` and `results_extra/` folders contain their outputs, which `paper_v3/make_paper_assets.py` reads for the deep-model, UNSW-NB15 and Edge-IIoTset rows.
