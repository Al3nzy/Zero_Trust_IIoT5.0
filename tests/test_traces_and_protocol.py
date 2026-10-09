import os, subprocess, sys, json, tempfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)


def test_trace_script_detects_compromised_and_keeps_benign_quiet():
    rng = np.random.default_rng(1); rows = []
    for d in range(10):
        comp = d < 4; n = 20 * 130; onset = 20 * 80 if comp else n + 1
        for i in range(n):
            att = comp and i >= onset; rows.append((f"dev{d}", i, float(rng.beta(6, 2) if att else rng.beta(1, 12)), int(att)))
    with tempfile.TemporaryDirectory() as t:
        f = os.path.join(t, "tr.csv"); pd.DataFrame(rows, columns=["device", "time", "harm", "label"]).to_csv(f, index=False)
        subprocess.run([sys.executable, os.path.join(ROOT, "run_traces.py"), f, "--out", os.path.join(t, "o")], check=True, capture_output=True)
        S = json.load(open(os.path.join(t, "o", "summary.json")))
    for rule in ("cusum", "e-detector"):
        assert S[rule]["detection_rate"] == 1.0 and S[rule]["false_quarantine_rate"] == 0.0 and S[rule]["median_delay_rounds"] <= 8


def test_audit_removes_overlap_and_duplicates():
    from ztids.generic import split_generic
    rng = np.random.RandomState(0); X = rng.randint(0, 5, (400, 4)); df = pd.DataFrame(X, columns=list("abcd")); df["y"] = (X[:, 0] > 2).astype(int)
    tr, te = df.iloc[:300], pd.concat([df.iloc[300:], df.iloc[:30]], ignore_index=True)           # 30 test rows are copies of training rows
    fit, val, test, audit = split_generic(tr, te, "y", list("abcd"), 0)
    h = lambda d: set(map(tuple, d[list("abcd")].values)); assert audit["test_exact_overlap_with_train"] >= 30
    assert not (h(test) & (h(fit) | h(val))) and audit["test_overlap_after_split"] == 0


def test_lightgbm_is_deterministic_and_single_threaded():
    from ztids.core import lgbm
    rng = np.random.RandomState(0); X = rng.randn(500, 6); y = (X[:, 1] > 0).astype(int)
    a = lgbm(3).fit(X, y).predict_proba(X); b = lgbm(3).fit(X, y).predict_proba(X); assert (a == b).all() and lgbm(3).get_params()["n_jobs"] == 1


def test_trust_to_evidence_range():
    from ztids.trust import to_evidence
    P = np.random.RandomState(0).dirichlet(np.ones(5), 200); E = to_evidence(P, np.random.RandomState(1).rand(200) > 0.9, 0.5)
    assert E.shape == (200, 3) and (E[:, 0] >= 0).all() and (E[:, 0] <= 1).all()
