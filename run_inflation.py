#!/usr/bin/env python3
"""Stage `inflation`: how much does each evaluation flaw inflate the reported score? A one-factor-at-a-time ablation ladder with the SAME classifier
(LightGBM, 25 class-aware features, balanced training) and the same seeds; only the evaluation protocol changes.

NSL-KDD (shipped):   python run_inflation.py nsl
Any corpus:          python run_inflation.py generic --name unsw --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --label attack_cat --drop id label
                     python run_inflation.py generic --name edge --train ML-EdgeIIoT-dataset.csv --label Attack_type --drop <identifier columns> [--drop-extra <capture fields>]

Variants (nsl):
  audited                 official partitions, train duplicates removed, test rows equal to a training row removed, preprocessing fitted on the training partition only
  overlap_kept            same model, official test set with its training-duplicate rows kept (reported overall, on the duplicated rows, on the rest)
  transductive_prep       scaler, vocabulary and label-aware feature selection fitted on training + test rows
  random_dedup            merged corpus, exact duplicates removed, stratified random split
  random_dup              merged corpus, duplicates kept, stratified random split (copies on both sides)
  random_dup_smote_first  merged corpus, duplicates kept, preprocessing fitted on everything, SMOTE applied BEFORE the random split (synthetic neighbours on both sides)
"""
import os, sys, json, time, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from ztids import config as C
from ztids.core import lgbm

NPC = C.NPC


def score(y, pred, classes):
    out = dict(acc=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, labels=range(classes), average="macro", zero_division=0)), mcc=float(matthews_corrcoef(y, pred)), n=int(len(y)))
    for c, nm in ((3, "r2l"), (4, "u2r")):
        if classes > 4: out[nm] = float((pred[y == c] == c).mean()) if (y == c).any() else None
    return out


def nsl(seeds):
    from ztids.data import make_split, load_nsl, Preprocessor, balance_train, CLASS_ID, _row_hash, FEATURES
    res = {}
    for s in seeds:
        ck = os.path.join(C.RES, f"inflation_nsl_s{s}.json")
        if os.path.exists(ck): res[str(s)] = json.load(open(ck)); continue
        t0 = time.time(); r = {}; tr_raw, te_raw = load_nsl(); cid = lambda d: d["cls"].map(CLASS_ID).values

        def train_eval(fit, tests, pp=None, Xb_yb=None):
            pp = pp or Preprocessor(k=25, seed=s, sel_mode="ovr").fit(fit)
            Xb, yb = Xb_yb if Xb_yb is not None else balance_train(pp.transform(fit), cid(fit), NPC, s)
            m = lgbm(s).fit(Xb, yb)
            return {k: (m.predict(pp.transform(d)), cid(d)) for k, d in tests.items()}, pp, m

        fit, val, test, audit = make_split("official", s); h_fit = set(_row_hash(fit))
        ov_mask = np.array([h in h_fit for h in _row_hash(te_raw)])                       # rows of the raw KDDTest+ equal to a row of the audited fit partition
        P, pp, m = train_eval(fit, {"test": test, "raw": te_raw})
        r["audited"] = score(P["test"][1], P["test"][0], 5)
        pr, yr = P["raw"]; r["overlap_kept"] = score(yr, pr, 5); r["overlap_kept"]["on_duplicated_rows"] = score(yr[ov_mask], pr[ov_mask], 5); r["overlap_kept"]["on_other_rows"] = score(yr[~ov_mask], pr[~ov_mask], 5)
        r["overlap_kept"]["share_duplicated"] = float(ov_mask.mean())
        # transductive preprocessing: scaler / vocabulary / label-aware selection see the test rows
        ppt = Preprocessor(k=25, seed=s, sel_mode="ovr").fit(pd.concat([fit, test], ignore_index=True)); Pt, _, _ = train_eval(fit, {"test": test}, pp=ppt); r["transductive_prep"] = score(Pt["test"][1], Pt["test"][0], 5)
        # random splits of the merged corpus
        merged = pd.concat([tr_raw, te_raw], ignore_index=True); dedup = merged.loc[~pd.Series(_row_hash(merged)).duplicated().values].reset_index(drop=True)
        for name, corpus in (("random_dedup", dedup), ("random_dup", merged)):
            pool, tst = train_test_split(corpus, test_size=0.20, stratify=corpus["cls"], random_state=s); pool = pool.reset_index(drop=True); tst = tst.reset_index(drop=True)
            f2, _ = train_test_split(pool, test_size=0.15, stratify=pool["cls"], random_state=s); f2 = f2.reset_index(drop=True)
            Pq, _, _ = train_eval(f2, {"test": tst}); r[name] = score(Pq["test"][1], Pq["test"][0], 5)
            if name == "random_dup":
                hf = set(_row_hash(f2)); dm = np.array([h in hf for h in _row_hash(tst)]); pq, yq = Pq["test"]
                r[name]["share_duplicated"] = float(dm.mean()); r[name]["on_duplicated_rows"] = score(yq[dm], pq[dm], 5); r[name]["on_other_rows"] = score(yq[~dm], pq[~dm], 5)
        # SMOTE (and preprocessing) before the random split
        ppa = Preprocessor(k=25, seed=s, sel_mode="ovr").fit(merged); Xa, ya = ppa.transform(merged), cid(merged); Xb, yb = balance_train(Xa, ya, 8000, s)
        Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.20, stratify=yb, random_state=s); m2 = lgbm(s).fit(Xtr, ytr); r["random_dup_smote_first"] = score(yte, m2.predict(Xte), 5)
        res[str(s)] = r; json.dump(r, open(ck, "w")); print(f"[inflation seed {s}] " + " | ".join(f"{k}: acc {v['acc']:.3f} mF1 {v['macro_f1']:.3f}" for k, v in r.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    return res


def generic(a, seeds):
    from ztids.generic import split_generic, GenericPrep, balance, _hash
    tr = pd.read_csv(a.train, low_memory=False); tr.columns = [c.strip() for c in tr.columns]; te = None
    if a.test: te = pd.read_csv(a.test, low_memory=False); te.columns = [c.strip() for c in te.columns]
    drop = [c.strip() for c in list(a.drop) + list(a.drop_extra) if c.strip() in tr.columns]; feat = [c for c in tr.columns if c not in set(drop) | {a.label}]; res = {}
    for s in seeds:
        t0 = time.time(); r = {}

        def run(fit, tests):
            classes = sorted(fit[a.label].unique().tolist(), key=str); cid = {c: i for i, c in enumerate(classes)}; yf = fit[a.label].map(cid).values
            pp = GenericPrep(feat, k=25, seed=s, sel_mode="ovr").fit(fit, yf, len(classes)); Xb, yb = balance(pp.transform(fit), yf, NPC, s, len(classes)); m = lgbm(s).fit(Xb, yb); out = {}
            for k, d in tests.items():
                d = d[d[a.label].isin(cid)]; P = np.zeros((len(d), len(classes))); P[:, m.classes_] = m.predict_proba(pp.transform(d)); out[k] = (P.argmax(1), d[a.label].map(cid).values, d)
            return out, len(classes)

        if te is not None:
            fit, val, test, audit = split_generic(tr, te, a.label, feat, s); hf = set(_hash(fit, feat))
            O, ncls = run(fit, {"test": test, "raw": te}); p, y, _ = O["test"]; r["audited"] = score(y, p, ncls); pr, yr, dr = O["raw"]; dm = np.array([h in hf for h in _hash(dr, feat)])
            r["overlap_kept"] = score(yr, pr, ncls); r["overlap_kept"]["on_duplicated_rows"] = score(yr[dm], pr[dm], ncls); r["overlap_kept"]["on_other_rows"] = score(yr[~dm], pr[~dm], ncls); r["overlap_kept"]["share_duplicated"] = float(dm.mean())
            merged = pd.concat([tr, te], ignore_index=True)
        else:
            fit, val, test, audit = split_generic(tr, None, a.label, feat, s, max_rows=a.max_rows); O, ncls = run(fit, {"test": test}); p, y, _ = O["test"]; r["audited"] = score(y, p, ncls); merged = tr
        if a.max_rows and len(merged) > a.max_rows: merged = merged.sample(a.max_rows, random_state=s).reset_index(drop=True)
        for name, corpus in (("random_dedup", merged.loc[~pd.Series(_hash(merged, feat)).duplicated().values].reset_index(drop=True)), ("random_dup", merged)):
            vc = corpus[a.label].value_counts(); corpus = corpus[corpus[a.label].isin(vc[vc >= 10].index)]
            pool, tst = train_test_split(corpus, test_size=0.20, stratify=corpus[a.label], random_state=s); f2, _ = train_test_split(pool.reset_index(drop=True), test_size=0.15, stratify=pool[a.label], random_state=s)
            O, ncls = run(f2.reset_index(drop=True), {"test": tst.reset_index(drop=True)}); p, y, dq = O["test"]; r[name] = score(y, p, ncls)
            if name == "random_dup":
                hf = set(_hash(f2, feat)); dm = np.array([h in hf for h in _hash(dq, feat)]); r[name]["share_duplicated"] = float(dm.mean()); r[name]["on_duplicated_rows"] = score(y[dm], p[dm], ncls); r[name]["on_other_rows"] = score(y[~dm], p[~dm], ncls)
        res[str(s)] = r; print(f"[inflation {a.name} seed {s}] " + " | ".join(f"{k}: acc {v['acc']:.3f} mF1 {v['macro_f1']:.3f}" for k, v in r.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["nsl", "generic"]); ap.add_argument("--seeds", nargs="*", type=int, default=[0] if C.QUICK else [0, 1, 2]); ap.add_argument("--name", default="nsl")
    ap.add_argument("--train"); ap.add_argument("--test"); ap.add_argument("--label"); ap.add_argument("--drop", nargs="*", default=[]); ap.add_argument("--drop-extra", nargs="*", default=[]); ap.add_argument("--max-rows", type=int)
    a = ap.parse_args(); res = nsl(a.seeds) if a.mode == "nsl" else generic(a, a.seeds)
    json.dump(dict(dataset=a.name, seeds=a.seeds, npc=NPC, results=res), open(os.path.join(C.RES, f"inflation_{a.name}.json"), "w"), indent=1)
