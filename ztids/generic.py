"""
Dataset-agnostic version of the leakage-free pipeline (UNSW-NB15, Edge-IIoTset, CIC-IDS, ...).
Same rules as ztids.data: exact-duplicate removal, test rows identical to training rows removed, validation carved from
original training records before resampling, every learned step fitted on the fit partition only, resampling after the split.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def _hash(df, cols): return pd.util.hash_pandas_object(df[cols], index=False).values


def split_generic(train_df, test_df, label_col, feat_cols, seed, val_frac=0.15, test_frac=0.20, max_rows=None):
    """train_df/test_df: official partitions (test_df=None -> single corpus, deduplicated, stratified split)."""
    audit = {}
    d0 = train_df if test_df is None else train_df
    if test_df is None:
        m = train_df.copy(); audit["corpus_rows"] = len(m)
        keep = ~pd.Series(_hash(m, feat_cols)).duplicated().values; audit["exact_duplicates_removed"] = int((~keep).sum())
        m = m.loc[keep].reset_index(drop=True)
        if max_rows and len(m) > max_rows:
            m, _ = train_test_split(m, train_size=max_rows, stratify=m[label_col], random_state=seed); m = m.reset_index(drop=True)
        vc = m[label_col].value_counts(); rare = vc[vc < 10].index; audit["classes_dropped_lt10"] = [str(c) for c in rare]
        m = m[~m[label_col].isin(rare)].reset_index(drop=True)
        pool, test = train_test_split(m, test_size=test_frac, stratify=m[label_col], random_state=seed)
    else:
        tr = train_df.loc[~pd.Series(_hash(train_df, feat_cols)).duplicated().values].reset_index(drop=True)
        audit["train_duplicates_removed"] = int(len(train_df) - len(tr))
        h = set(_hash(tr, feat_cols)); te_h = _hash(test_df, feat_cols); ov = np.array([x in h for x in te_h])
        audit["test_total"], audit["test_exact_overlap_with_train"] = int(len(test_df)), int(ov.sum())
        te = test_df.loc[~ov].reset_index(drop=True); te = te.loc[~pd.Series(_hash(te, feat_cols)).duplicated().values].reset_index(drop=True)
        pool, test = tr, te
    pool = pool.reset_index(drop=True); test = test.reset_index(drop=True)
    fit, val = train_test_split(pool, test_size=val_frac, stratify=pool[label_col], random_state=seed)
    fit, val = fit.reset_index(drop=True), val.reset_index(drop=True)
    h = set(_hash(fit, feat_cols)); audit["test_overlap_after_split"] = int(sum(x in h for x in _hash(test, feat_cols)))
    audit["counts"] = {k: d[label_col].value_counts().to_dict() for k, d in (("fit", fit), ("val", val), ("test", test))}
    return fit, val, test, audit


class GenericPrep:
    def __init__(self, feat_cols, k=25, seed=0, sel_mode="ovr", max_onehot=30, mi_rows=40000):
        self.feat_cols, self.k, self.seed, self.sel_mode, self.max_onehot, self.mi_rows = feat_cols, k, seed, sel_mode, max_onehot, mi_rows

    def _expand(self, df):
        parts = [df[self.num_].replace([np.inf, -np.inf], np.nan).astype(float).fillna(self.med_)]
        for c in self.cat_:
            idx = pd.Categorical(df[c].astype(str), categories=self.cats_[c]).codes; oh = np.zeros((len(df), len(self.cats_[c])))
            ok = idx >= 0; oh[np.where(ok)[0], idx[ok]] = 1.0
            parts.append(pd.DataFrame(oh, columns=[f"{c}={v}" for v in self.cats_[c]], index=df.index))
        return pd.concat(parts, axis=1)[self.cols_]

    def fit(self, df, y, n_cls):
        self.cat_ = [c for c in self.feat_cols if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
        self.num_ = [c for c in self.feat_cols if c not in self.cat_]
        self.med_ = df[self.num_].replace([np.inf, -np.inf], np.nan).astype(float).median().fillna(0.0)
        self.cats_ = {}
        for c in self.cat_:
            vc = df[c].astype(str).value_counts(); self.cats_[c] = sorted(vc.index[: self.max_onehot])   # top-N levels, rest -> all zeros
        self.cols_ = self.num_ + [f"{c}={v}" for c in self.cat_ for v in self.cats_[c]]
        X = self._expand(df); self.scaler_ = StandardScaler().fit(X.values); Xs = self.scaler_.transform(X.values)
        Xs = np.clip(Xs, -10, 10)                                           # tame heavy tails; fitted constants only
        rng = np.random.RandomState(self.seed); sub = rng.choice(len(Xs), min(len(Xs), self.mi_rows), replace=False)
        mi = mutual_info_classif(Xs[sub], y[sub], random_state=self.seed)
        if self.sel_mode == "ovr":
            per = np.zeros((n_cls, Xs.shape[1]))
            for c in range(n_cls):
                pos, neg = np.where(y == c)[0], np.where(y != c)[0]
                if len(pos) < 5: continue
                npos = min(len(pos), len(neg), 5000)
                idx = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, npos, replace=False)])
                per[c] = mutual_info_classif(Xs[idx], (y[idx] == c).astype(int), random_state=self.seed); per[c] /= per[c].max() + 1e-12
            mi = per.max(0)
        self.mi_ = mi; kk = min(self.k, Xs.shape[1]); top = np.argsort(mi)[-kk:]
        parent = lambda col: self.feat_cols.index(col.split("=")[0])
        self.sel_ = np.array(sorted(top, key=lambda j: (parent(self.cols_[j]), j))); self.names_ = [self.cols_[j] for j in self.sel_]
        return self

    def transform(self, df):
        return np.clip(self.scaler_.transform(self._expand(df).values), -10, 10)[:, self.sel_].astype(np.float32)


def balance(X, y, n_per_class, seed, n_cls):
    cnt = np.bincount(y, minlength=n_cls); down = {c: n_per_class for c in range(n_cls) if cnt[c] > n_per_class}
    if down: X, y = RandomUnderSampler(sampling_strategy=down, random_state=seed).fit_resample(X, y)
    cnt = np.bincount(y, minlength=n_cls); up = {c: n_per_class for c in range(n_cls) if 0 < cnt[c] < n_per_class}
    if up:
        small = [c for c in up if cnt[c] < 6]
        if small: X, y = RandomOverSampler(sampling_strategy={c: n_per_class for c in small}, random_state=seed).fit_resample(X, y)
        cnt = np.bincount(y, minlength=n_cls); up = {c: n_per_class for c in range(n_cls) if 0 < cnt[c] < n_per_class}
        if up: X, y = SMOTE(sampling_strategy=up, k_neighbors=int(max(1, min(5, min(cnt[c] for c in up) - 1))), random_state=seed).fit_resample(X, y)
    return X.astype(np.float32), y


def prepare_generic(train_df, test_df, label_col, drop_cols, seed, n_per_class=4000, k=25, sel_mode="ovr", max_rows=None):
    feat = [c for c in train_df.columns if c not in set(drop_cols) | {label_col}]
    fit, val, test, audit = split_generic(train_df, test_df, label_col, feat, seed, max_rows=max_rows)
    classes = sorted(fit[label_col].unique().tolist(), key=str); cid = {c: i for i, c in enumerate(classes)}
    val = val[val[label_col].isin(cid)]; test = test[test[label_col].isin(cid)]            # classes absent from fit cannot be scored
    yf, yv, yt = (d[label_col].map(cid).values for d in (fit, val, test))
    pp = GenericPrep(feat, k=k, seed=seed, sel_mode=sel_mode).fit(fit, yf, len(classes))
    Xf, Xv, Xt = pp.transform(fit), pp.transform(val), pp.transform(test)
    Xb, yb = balance(Xf, yf, n_per_class, seed, len(classes))
    audit["classes"], audit["selected_features"] = classes, pp.names_
    return dict(Xtr=Xb, ytr=yb, Xfit=Xf, yfit=yf, Xval=Xv, yval=yv, Xte=Xt, yte=yt, pp=pp, audit=audit, classes=classes)


def metrics_generic(y, prob, classes):
    n = len(classes); pred = prob.argmax(1); P, R, F, S = precision_recall_fscore_support(y, pred, labels=range(n), zero_division=0)
    return dict(acc=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, labels=range(n), average="macro", zero_division=0)),
                mcc=float(matthews_corrcoef(y, pred)), n=int(len(y)),
                per_class={str(c): dict(p=float(P[i]), r=float(R[i]), f1=float(F[i]), n=int(S[i])) for i, c in enumerate(classes)},
                cm=confusion_matrix(y, pred, labels=range(n)).tolist())
