"""Stage 3: multi-round zero-trust evaluation by replaying real test-set posteriors of the proposed detector through simulated devices.
NOTE: this is a replay simulation of the trust layer, not a live deployment; the paper must say so."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from ztids import config as C
from ztids.trust import SEVERITY, H, D, Q, gen_evidence, run_rule
from ztids.data import CLASSES, make_split
SRC = os.path.join(C.RES, C.PROPOSED)
if not os.path.exists(SRC + "_probs.npz"): sys.exit("proposed-model posteriors missing; run the detector stage first")
z = np.load(SRC + "_probs.npz"); pt, yt = z["pt"], z["yt"]
_, _, test_df, _ = make_split("official", 0)
assert len(test_df) == len(yt) and (test_df["cls"].map({c: i for i, c in enumerate(CLASSES)}).values == yt).all()
svc = test_df["service"].values
top = pd.Series(svc[yt == 0]).value_counts().index[:3].tolist()
grp = np.where(np.isin(svc, top), pd.Series(svc).map({s: i for i, s in enumerate(top)}).fillna(3).astype(int).values, 3)
M, ROUNDS, WARM, ONSET, END, K, ALPHA = 20, 60, 10, 15, 30, 0.05, 0.01
H_CERT = float(np.log(1 / ALPHA) / (8 * M * K))            # certifies P(ever quarantined) <= ALPHA when mu0 ~ mu_b
NDEV = 40 if C.QUICK else 200
Pa_all = {k: pt[yt == k] for k in (1, 2, 3, 4)}


def make_fleet(scn, n, seed, hetero, **kw):
    rng = np.random.RandomState(seed); fleet = []
    for _ in range(n):
        g = int(rng.randint(0, 4)) if hetero else -1
        Pn = pt[(yt == 0) & (grp == g)] if hetero else pt[yt == 0]
        E, fr = gen_evidence(Pn, Pa_all, scn, rng, ROUNDS, M, ONSET, END, **kw); fleet.append(E)
    return fleet


def summarize(scn, Ss):
    out = {"ever_quarantined": float(np.mean([(s[WARM:] == Q).any() for s in Ss])),
           "healthy_frac_after_warmup": float(np.mean([(s[WARM:] == H).mean() for s in Ss]))}
    if scn in ("compromised", "stealth", "intermittent"):
        dl = [(np.where(s[ONSET:] == Q)[0][0] + 1) if (s[ONSET:] == Q).any() else np.nan for s in Ss]
        out.update(detect=float(np.mean(~np.isnan(dl))), median_delay=None if np.all(np.isnan(dl)) else float(np.nanmedian(dl)),
                   mean_delay=None if np.all(np.isnan(dl)) else float(np.nanmean(dl)), final_healthy=float(np.mean([s[-1] == H for s in Ss])),
                   frac_time_q=float(np.mean([(s[ONSET:] == Q).mean() for s in Ss])))
    if scn == "recovery":
        rel = [(np.where(s[END:] != Q)[0][0] + 1) if (s[END:] != Q).any() else np.nan for s in Ss]
        out.update(was_quarantined=float(np.mean([(s[ONSET:END] == Q).any() for s in Ss])), released_by_end=float(np.mean(~np.isnan(rel))),
                   median_release_delay=None if np.all(np.isnan(rel)) else float(np.nanmedian(rel)), final_healthy=float(np.mean([s[-1] == H for s in Ss])))
    return out


RULES = [("confidence (original Eq.4)", "confidence", None), ("beta reputation", "beta", None),
         ("CUSUM fleet-baseline", "cusum", "fleet"), ("CUSUM device-baseline", "cusum", "device")]
R = {"setup": dict(m=M, rounds=ROUNDS, warmup=WARM, onset=ONSET, end=END, k=K, alpha=ALPHA, h=H_CERT, n_devices=NDEV, device_groups=top + ["other"],
                   severity=SEVERITY.tolist(), note="replay simulation of real posteriors; flows within a round are sampled i.i.d.")}
for hetero in (False, True):
    key = "hetero" if hetero else "homog"; R[key] = {}
    ben_means = [E[WARM:, 0].mean() for E in make_fleet("benign", 300 if not C.QUICK else 60, 3, hetero)]
    R[key]["mu_b_mean"] = float(np.mean(ben_means)); R[key]["mu_b_sd_across_devices"] = float(np.std(ben_means))
    R[key]["mu_att"] = {CLASSES[c]: float((Pa_all[c] @ SEVERITY).mean()) for c in (1, 2, 3, 4)}
    fleets = {scn: make_fleet(scn, NDEV, 7, hetero, **(dict(frac=0.3) if scn == "stealth" else {})) for scn in ("benign", "compromised", "recovery", "intermittent", "stealth")}
    for name, rule, mode in RULES:
        R[key][name] = {}
        for scn, fleet in fleets.items():
            per = np.array([E[:WARM, 0].mean() for E in fleet]); ref = per if mode == "device" else np.full(len(fleet), np.median(per))
            Ss = [run_rule(E, rule, mu0=ref[i], k=K, h=H_CERT, warm=WARM)[1] for i, E in enumerate(fleet)]
            R[key][name][scn] = summarize(scn, Ss)
        b, c, r, s = (R[key][name][x] for x in ("benign", "compromised", "recovery", "stealth"))
        print(f"{key:7s} {name:27s} FQ={b['ever_quarantined']:.3f} detect={c['detect']:.2f} delay={c['median_delay']} finalHealthy={c['final_healthy']:.2f} "
              f"released={r['released_by_end']:.2f} stealth30%={s['detect']:.2f}", flush=True)

# commissioning-window contamination: a fraction of the fleet is already compromised while the baseline is being estimated
R["contamination"] = {}
for frac in (0.0, 0.1, 0.3, 0.45):
    rng = np.random.RandomState(21); n_c = int(round(frac * NDEV)); fleet = []
    for d in range(NDEV):
        E, _ = gen_evidence(pt[(yt == 0)], Pa_all, "compromised" if d < n_c else "benign", rng, ROUNDS, M, onset=0, attack_cls=int(rng.choice([1, 2, 3, 4])))
        fleet.append(E)
    per = np.array([E[:WARM, 0].mean() for E in fleet])
    for mode in ("fleet", "device"):
        ref = per if mode == "device" else np.full(NDEV, np.median(per))
        Ss = [run_rule(E, "cusum", mu0=ref[i], k=K, h=H_CERT, warm=WARM)[1] for i, E in enumerate(fleet)]
        q = [(s[WARM:] == Q).any() for s in Ss]
        R["contamination"][f"{frac}_{mode}"] = dict(frac=frac, mode=mode, detect_contaminated=float(np.mean(q[:n_c])) if n_c else None,
                                                    false_quarantine_benign=float(np.mean(q[n_c:])))
    print("contamination", frac, {m: R["contamination"][f"{frac}_{m}"] for m in ("fleet", "device")}, flush=True)
json.dump(R, open(os.path.join(C.RES, "trust_main.json"), "w"), indent=1, default=float)
