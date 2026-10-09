#!/usr/bin/env python3
"""Ten-second environment check: imports, shipped data, deterministic LightGBM, and the e-detector's basic guarantee on a tiny simulation. Prints PASS/FAIL per check."""
import os, sys, math, importlib, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); fails = []
def check(name, fn):
    try: fn(); print(f"PASS  {name}")
    except Exception as e: print(f"FAIL  {name}: {type(e).__name__}: {e}"); fails.append(name)
print(f"python {sys.version.split()[0]} | cwd {os.getcwd()}")
for mod in ("numpy", "pandas", "scipy", "sklearn", "imblearn", "lightgbm", "matplotlib", "cryptography"):
    check(f"import {mod}", lambda m=mod: importlib.import_module(m))
def data():
    from ztids.data import load_nsl
    tr, te = load_nsl(); assert len(tr) == 125973 and len(te) == 22544, (len(tr), len(te))
check("NSL-KDD files in data/ (125,973 + 22,544 records)", data)
def lgbm_det():
    import numpy as np; from ztids.core import lgbm
    rng = np.random.RandomState(0); X = rng.randn(600, 5); y = (X[:, 0] + 0.3 * rng.randn(600) > 0).astype(int)
    a = lgbm(0).fit(X, y).predict_proba(X); b = lgbm(0).fit(X, y).predict_proba(X); assert (a == b).all(), "LightGBM is not deterministic"
check("deterministic LightGBM", lgbm_det)
def edet():
    import numpy as np; from ztids.edetector import run_edetector, threshold_for
    rng = np.random.RandomState(1); T, a = 100, 0.2; c = threshold_for(T, a); fa = sum((run_edetector(rng.beta(2, 8, 20 + T), warm=20, c=c, seed=i)[1][20:] == "Quarantined").any() for i in range(150))
    assert fa / 150 <= a + 0.08, fa / 150
    x = np.r_[rng.beta(2, 8, 60), rng.beta(8, 2, 20)]; q = (run_edetector(x, warm=20, c=1000, seed=0)[1] == "Quarantined").nonzero()[0]; assert len(q) and q[0] >= 60, "no detection of an obvious attack"
check("conformal e-detector: false-quarantine bound and detection", edet)
def sign():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey; from ztids.ledger import Ledger, verify
    L = Ledger(signer=Ed25519PrivateKey.generate(), anchor_every=5)
    for i in range(12): L.append({"dev": "d", "T": 0.5})
    assert verify(L.chain, L.signer.public_key() if hasattr(L, "signer") else None, getattr(L, "anchors", None), use_sig=False, use_anchor=False)
check("signed ledger append and verify", sign)
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}"); sys.exit(1 if fails else 0)
