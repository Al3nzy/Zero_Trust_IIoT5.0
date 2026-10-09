"""Equal-weight posterior averaging of already-trained models (no extra training). The 'hybrid' is fixed a priori to LightGBM + CNN-BiLSTM:
the two principal models, chosen because one is the strongest classical model and the other is the originally proposed network."""
import os
import numpy as np
from . import config as C
from .evalutil import metrics

HYBRID = {"label": "Hybrid (LightGBM + CNN-BiLSTM)", "members": ["official_lgbm_{s}_sel_selovr", "official_cnn_bilstm_{s}_-_-_-_selovr"]}


def probs(patterns, seed):
    ps = []
    for p in patterns:
        f = os.path.join(C.RES, p.format(s=seed) + "_probs.npz")
        if not os.path.exists(f): return None, None
        z = np.load(f); ps.append(z["pt"]); yt = z["yt"]
    return np.mean(ps, 0), yt


def metrics_over_seeds(patterns, seeds):
    """List of per-seed metric dicts (as produced by evalutil.metrics) for the averaged posteriors."""
    out = []
    for s in seeds:
        P, yt = probs(patterns, s)
        if P is None: continue
        r = metrics(yt, P); r["test"] = r; out.append(r)          # same access pattern as a job result: r["test"]["acc"]
    return out
