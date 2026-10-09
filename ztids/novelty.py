"""
Normal-profile novelty detectors (fitted on Normal training flows only) and their fusion with the supervised classifier.

Why: a supervised classifier only recognises attack families it was trained on, while a zero-trust layer must also react to behaviour that
deviates from the normal profile. Every threshold is set on VALIDATION Normal flows (target false-alarm rate), never on test data.
Kinds: 'iforest' (default, a-priori choice), 'mahalanobis' (Gaussian profile, Ledoit-Wolf covariance), 'iforest+mahalanobis' (rank average).
NOTE: kNN-distance was evaluated and rejected: best validation AUROC but worst test AUROC under distribution shift (see make_report.py section 3).
Fusion rule: a flow is an attack if the classifier says so OR the detector flags it; flows flagged only by the detector are labelled with the
most probable attack class and carry the 'unknown' severity w_u in the trust evidence.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import LedoitWolf
from .trust import SEVERITY

KINDS = ("iforest", "mahalanobis", "iforest+mahalanobis")


class NormalProfile:
    def __init__(self, fpr=0.02, kind="iforest", n_estimators=300, max_samples=256, max_fit=30000, seed=0):
        assert kind in KINDS, kind
        self.fpr, self.kind, self.n_estimators, self.max_samples, self.max_fit, self.seed = fpr, kind, n_estimators, max_samples, max_fit, seed

    def _raw(self, X, which):
        if which == "iforest": return -self.f_.score_samples(X)
        return self.lw_.mahalanobis(X)

    def fit(self, X_normal_fit, X_normal_val):
        rng = np.random.RandomState(self.seed)
        Xn = X_normal_fit[rng.choice(len(X_normal_fit), min(self.max_fit, len(X_normal_fit)), replace=False)]
        self.parts_ = ["iforest", "mahalanobis"] if self.kind == "iforest+mahalanobis" else [self.kind]
        if "iforest" in self.parts_: self.f_ = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples, random_state=self.seed, n_jobs=-1).fit(Xn)
        if "mahalanobis" in self.parts_: self.lw_ = LedoitWolf().fit(Xn)
        # reference distribution of each component on validation Normal flows (used to rank-normalise and to set the threshold)
        self.ref_ = {p: np.sort(self._raw(X_normal_val, p)) for p in self.parts_}
        self.thr_ = float(np.quantile(self.score(X_normal_val), 1 - self.fpr))
        return self

    def score(self, X):
        if len(self.parts_) == 1: return self._raw(X, self.parts_[0])
        return np.mean([np.searchsorted(self.ref_[p], self._raw(X, p)) / len(self.ref_[p]) for p in self.parts_], 0)

    def flag(self, X):
        return self.score(X) > self.thr_


def fuse_labels(P, flag):
    """Classifier OR novelty: flagged flows predicted Normal are relabelled with the most probable attack class."""
    pred = P.argmax(1).copy(); m = np.asarray(flag, bool) & (pred == 0)
    if m.any(): pred[m] = 1 + P[m, 1:].argmax(1)
    return pred


def evidence_rows(P, flag=None, w_u=0.5):
    """Per-flow trust evidence [harm, p_normal, confidence]. With a novelty flag, harm = max(severity-weighted posterior, w_u * flag)."""
    h = P @ SEVERITY; g = P[:, 0].copy(); c = P.max(1)
    if flag is not None:
        f = np.asarray(flag, float); h = np.maximum(h, w_u * f); g = g * (1 - f)
    return np.c_[h, g, c]
