"""
Leakage-free data pipeline for NSL-KDD (and any tabular IDS corpus).

Design rules (each one answers a reviewer concern):
  * every learned operation (category vocabulary, scaler, MI feature selection,
    resampling) is fitted on the TRAINING PARTITION ONLY and then applied unchanged;
  * resampling (SMOTE / random undersampling) happens after the split and never
    touches validation or test data;
  * the validation set is carved out of the original training records BEFORE any
    resampling, so early stopping never sees synthetic or duplicated copies;
  * exact feature-vector overlap between train and test is audited and removed
    from the test partition, and is reported;
  * the 25 selected features are kept in canonical NSL-KDD order (feature families
    stay adjacent), so the sequence order is documented and reproducible.
"""
from __future__ import annotations
import hashlib
from . import config as _C
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
# NSL-KDD feature families (canonical order); used to document sequence ordering.
FAMILY = {}
for i, n in enumerate(FEATURES):
    FAMILY[n] = ("basic" if i < 9 else "content" if i < 22 else "time-window" if i < 31 else "host-window")

# Complete sub-label -> 5-class map (covers the 17 test-only attack types, so no record is discarded).
ATTACK_MAP = {"normal": "Normal"}
for a in "back land neptune pod smurf teardrop apache2 mailbomb processtable udpstorm worm".split():
    ATTACK_MAP[a] = "DoS"
for a in "ipsweep nmap portsweep satan mscan saint".split():
    ATTACK_MAP[a] = "Probe"
for a in ("ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail named "
          "snmpgetattack snmpguess xlock xsnoop httptunnel").split():
    ATTACK_MAP[a] = "R2L"
for a in "buffer_overflow loadmodule perl rootkit ps sqlattack xterm".split():
    ATTACK_MAP[a] = "U2R"


def load_nsl(data_dir=None):
    data_dir = data_dir or _C.DATA
    cols = FEATURES + ["label", "difficulty"]
    tr = pd.read_csv(f"{data_dir}/NSL_KDD_Train.txt", names=cols)
    te = pd.read_csv(f"{data_dir}/NSL_KDD_Test.txt", names=cols)
    out = []
    for df in (tr, te):
        df = df.copy()
        df["cls"] = df["label"].str.lower().map(ATTACK_MAP)
        assert df["cls"].notna().all(), df.loc[df["cls"].isna(), "label"].unique()
        df["sub"] = df["label"].str.lower()                 # raw attack type, kept ONLY for the seen/novel-attack analysis
        df = df.drop(columns=["difficulty", "label"])   # `difficulty` is label-conditioned: never a feature
        out.append(df.reset_index(drop=True))
    return out


def _row_hash(df):
    return pd.util.hash_pandas_object(df[FEATURES], index=False).values


def make_split(protocol: str, seed: int, data_dir=None, val_frac=0.15, test_frac=0.20):
    """Return raw (un-preprocessed) partitions + an overlap audit.

    protocol = 'official' : KDDTrain+ -> train, KDDTest+ -> test (novel attack variants in test)
    protocol = 'random'   : merged corpus, exact duplicates removed, stratified split
    """
    tr, te = load_nsl(data_dir)
    audit = {}
    if protocol == "official":
        tr = tr.loc[~pd.Series(_row_hash(tr)).duplicated().values].reset_index(drop=True)
        h_tr = set(_row_hash(tr))
        te_h = _row_hash(te)
        dup_mask = np.array([h in h_tr for h in te_h])
        audit["test_total"] = int(len(te))
        audit["test_exact_overlap_with_train"] = int(dup_mask.sum())
        te = te.loc[~dup_mask].reset_index(drop=True)            # overlap-free test
        te = te.loc[~pd.Series(_row_hash(te)).duplicated().values].reset_index(drop=True)
        train_pool, test = tr, te
    elif protocol == "random":
        m = pd.concat([tr, te], ignore_index=True)
        audit["merged_total"] = int(len(m))
        keep = ~pd.Series(_row_hash(m)).duplicated().values
        audit["exact_duplicates_removed"] = int((~keep).sum())
        m = m.loc[keep].reset_index(drop=True)
        train_pool, test = train_test_split(m, test_size=test_frac, stratify=m["cls"], random_state=seed)
        train_pool = train_pool.reset_index(drop=True); test = test.reset_index(drop=True)
    else:
        raise ValueError(protocol)
    # validation carved out of ORIGINAL training records, before any resampling
    fit, val = train_test_split(train_pool, test_size=val_frac, stratify=train_pool["cls"], random_state=seed)
    fit = fit.reset_index(drop=True); val = val.reset_index(drop=True)
    h_fit = set(_row_hash(fit))
    audit["test_overlap_after_split"] = int(sum(h in h_fit for h in _row_hash(test)))
    audit["val_overlap_with_fit"] = int(sum(h in h_fit for h in _row_hash(val)))
    audit["counts"] = {k: {c: int((d["cls"] == c).sum()) for c in CLASSES}
                       for k, d in (("fit", fit), ("val", val), ("test", test))}
    return fit, val, test, audit


class Preprocessor:
    """One-hot + z-score + MI top-k, all fitted on the fit partition only."""

    def __init__(self, k=25, seed=0, mi_max_rows=40000, feature_order="canonical", sel_mode="global", clip=None):
        self.k, self.seed, self.mi_max_rows, self.feature_order, self.sel_mode = k, seed, mi_max_rows, feature_order, sel_mode
        self.clip = _C.CLIP if clip is None else clip          # |z| clipping bound (0 = off); rare-event features reach |z|>100

    def _expand(self, df):
        num = df[[c for c in FEATURES if c not in CATEGORICAL]].astype(float)
        parts = [num]
        for c in CATEGORICAL:
            cats = self.cats_[c]
            oh = np.zeros((len(df), len(cats)))
            idx = pd.Categorical(df[c], categories=cats).codes      # unseen -> -1 -> all zeros
            ok = idx >= 0
            oh[np.where(ok)[0], idx[ok]] = 1.0
            parts.append(pd.DataFrame(oh, columns=[f"{c}={v}" for v in cats], index=df.index))
        return pd.concat(parts, axis=1)[self.all_cols_]

    def fit(self, df_fit):
        self.cats_ = {c: sorted(df_fit[c].unique()) for c in CATEGORICAL}
        self.all_cols_ = ([c for c in FEATURES if c not in CATEGORICAL] +
                          [f"{c}={v}" for c in CATEGORICAL for v in self.cats_[c]])
        X = self._expand(df_fit)
        y = df_fit["cls"].map(CLASS_ID).values
        self.scaler_ = StandardScaler().fit(X.values)
        Xs = self.scaler_.transform(X.values)
        if self.clip: Xs = np.clip(Xs, -self.clip, self.clip)
        rng = np.random.RandomState(self.seed)
        sub = rng.choice(len(Xs), min(len(Xs), self.mi_max_rows), replace=False)
        self.mi_ = mutual_info_classif(Xs[sub], y[sub], random_state=self.seed)
        if self.sel_mode == "ovr":
            # class-aware MI: one-vs-rest MI per class on a class-balanced subsample, each class's MI normalised by its
            # own maximum, feature score = max over classes. Rare-class indicators (R2L/U2R) are no longer outvoted.
            per = np.zeros((len(CLASSES), Xs.shape[1]))
            for c in range(len(CLASSES)):
                pos = np.where(y == c)[0]; neg = np.where(y != c)[0]
                npos = min(len(pos), 5000); idx = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, npos, replace=False)])
                per[c] = mutual_info_classif(Xs[idx], (y[idx] == c).astype(int), random_state=self.seed)
                per[c] /= per[c].max() + 1e-12
            self.mi_ovr_ = per; self.mi_ = per.max(0)
        top = np.argsort(self.mi_)[-self.k:]
        parent = lambda col: FEATURES.index(col.split("=")[0])
        if self.feature_order == "canonical":          # family-adjacent, documented order
            top = sorted(top, key=lambda j: (parent(self.all_cols_[j]), j))
        elif self.feature_order == "random":           # ablation: destroy the order
            top = list(np.random.RandomState(self.seed + 1).permutation(top))
        elif self.feature_order == "mi":               # descending MI
            top = list(top[::-1])
        self.sel_ = np.array(top)
        self.names_ = [self.all_cols_[j] for j in self.sel_]
        return self

    def transform(self, df):
        Z = self.scaler_.transform(self._expand(df).values)
        if self.clip: Z = np.clip(Z, -self.clip, self.clip)
        return Z[:, self.sel_].astype(np.float32)

    def transform_all(self, df):
        """All standardised (and clipped) columns, in the order of `all_cols_`: input of the Normal-profile detectors (no feature selection)."""
        Z = self.scaler_.transform(self._expand(df).values)
        if self.clip: Z = np.clip(Z, -self.clip, self.clip)
        return Z.astype(np.float32)


def balance_train(X, y, n_per_class, seed):
    """Post-split balancing of the TRAINING partition: SMOTE up / random undersample down."""
    counts = np.bincount(y, minlength=len(CLASSES))
    down = {c: n_per_class for c in range(len(CLASSES)) if counts[c] > n_per_class}
    if down:
        X, y = RandomUnderSampler(sampling_strategy=down, random_state=seed).fit_resample(X, y)
    counts = np.bincount(y, minlength=len(CLASSES))
    up = {c: n_per_class for c in range(len(CLASSES)) if counts[c] < n_per_class}
    if up:
        kn = int(max(1, min(5, counts.min() - 1)))
        X, y = SMOTE(sampling_strategy=up, k_neighbors=kn, random_state=seed).fit_resample(X, y)
    return X.astype(np.float32), y


def prepare(protocol, seed, n_per_class=6000, k=25, feature_order="canonical", data_dir=None, sel_mode="global", clip=None):
    fit, val, test, audit = make_split(protocol, seed, data_dir)
    pp = Preprocessor(k=k, seed=seed, feature_order=feature_order, sel_mode=sel_mode, clip=clip).fit(fit)
    Xf, yf = pp.transform(fit), fit["cls"].map(CLASS_ID).values
    Xv, yv = pp.transform(val), val["cls"].map(CLASS_ID).values
    Xt, yt = pp.transform(test), test["cls"].map(CLASS_ID).values
    Xb, yb = balance_train(Xf, yf, n_per_class, seed)
    audit["train_balanced_counts"] = {c: int((yb == i).sum()) for i, c in enumerate(CLASSES)}
    audit["selected_features"] = pp.names_
    return dict(Xtr=Xb, ytr=yb, Xfit=Xf, yfit=yf, Xval=Xv, yval=yv, Xte=Xt, yte=yt,
                pp=pp, audit=audit, test_df=test)
