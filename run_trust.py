"""Stage 3: multi-round zero-trust evaluation by replaying real test-set posteriors of the proposed detector through simulated devices.
NOTE: this is a replay simulation of the trust layer, not a live deployment; the paper must say so."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from ztids import config as C
from ztids.trust import SEVERITY, H, D, Q, gen_evidence, run_rule
from ztids.data import CLASSES
from ztids import evidence as EV_
E_ = EV_.load(); pt, yt, svc = E_["pt"], E_["yt"], E_["svc"]
EVC, EVF = E_["EVc"], E_["EVf"]                       # classifier-only evidence, classifier + novelty-detector evidence
print("novelty detector available:", E_["has_novelty"], flush=True)
top = pd.Series(svc[yt == 0]).value_counts().index[:3].tolist()
grp = np.where(np.isin(svc, top), pd.Series(svc).map({s: i for i, s in enumerate(top)}).fillna(3).astype(int).values, 3)
M, ROUNDS, WARM, ONSET, END, K, ALPHA = 20, 60, 10, 15, 30, 0.05, 0.01
H_CERT = float(np.log(1 / ALPHA) / (8 * M * K))            # certifies P(ever quarantined) <= ALPHA when mu0 ~ mu_b
NDEV = 40 if C.QUICK else 200
PA = {'c': EV_.attack_pools(EVC, yt, E_['novel']), 'f': EV_.attack_pools(EVF, yt, E_['novel'])}
EVS = {'c': EVC, 'f': EVF}


def make_fleet(scn, n, seed, hetero, ev="c", **kw):
    rng = np.random.RandomState(seed); fleet = []
    for _ in range(n):
        g = int(rng.randint(0, 4)) if hetero else -1
        Pn = EVS[ev][(yt == 0) & (grp == g)] if hetero else EVS[ev][yt == 0]
        E, fr = gen_evidence(Pn, PA[ev], scn, rng, ROUNDS, M, ONSET, END, **kw); fleet.append(E)
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


RULES = [("confidence (original Eq.4)", "confidence", None, "c"), ("beta reputation", "beta", None, "c"),
         ("CUSUM fleet-baseline (classifier)", "cusum", "fleet", "c"), ("CUSUM fleet-baseline (+novelty)", "cusum", "fleet", "f"),
         ("CUSUM device-baseline (+novelty)", "cusum", "device", "f")]
SCN = ("benign", "compromised", "recovery", "intermittent", "stealth", "novel_attack", "seen_attack")
R = {"setup": dict(m=M, rounds=ROUNDS, warmup=WARM, onset=ONSET, end=END, k=K, alpha=ALPHA, h=H_CERT, n_devices=NDEV, device_groups=top + ["other"],
                   severity=SEVERITY.tolist(), w_unknown=EV_.W_U, novelty_fpr_target=EV_.FPR, novelty_available=bool(E_["has_novelty"]), note="replay simulation of real posteriors; flows within a round are sampled i.i.d.")}
for hetero in (False, True):
    key = "hetero" if hetero else "homog"; R[key] = {}
    for ev in ("c", "f"):
        ben_means = [E[WARM:, 0].mean() for E in make_fleet("benign", 300 if not C.QUICK else 60, 3, hetero, ev)]
        R[key]["mu_b_mean_" + ev] = float(np.mean(ben_means)); R[key]["mu_b_sd_across_devices_" + ev] = float(np.std(ben_means))
        R[key]["mu_att_" + ev] = {**{CLASSES[c]: float(PA[ev][c][:, 0].mean()) for c in (1, 2, 3, 4)}, "novel sub-types": float(PA[ev]["novel"][:, 0].mean()), "seen sub-types": float(PA[ev]["seen"][:, 0].mean())}
    fleets = {ev: {} for ev in ("c", "f")}
    for ev in ("c", "f"):
        for scn in SCN:
            base, extra = {"novel_attack": ("compromised", dict(attack_cls="novel")), "seen_attack": ("compromised", dict(attack_cls="seen")), "stealth": ("stealth", dict(frac=0.3))}.get(scn, (scn, {}))
            fleets[ev][scn] = make_fleet(base, NDEV, 7, hetero, ev, **extra)
    for name, rule, mode, ev in RULES:
        R[key][name] = {}
        for scn, fleet in fleets[ev].items():
            per = np.array([E[:WARM, 0].mean() for E in fleet]); ref = per if mode == "device" else np.full(len(fleet), np.median(per))
            Ss = [run_rule(E, rule, mu0=ref[i], k=K, h=H_CERT, warm=WARM)[1] for i, E in enumerate(fleet)]
            R[key][name][scn] = summarize("compromised" if scn in ("novel_attack", "seen_attack") else scn, Ss)
        b, c, r, s_, nv, sn = (R[key][name][x] for x in ("benign", "compromised", "recovery", "stealth", "novel_attack", "seen_attack"))
        print(f"{key:7s} {name:34s} FQ={b['ever_quarantined']:.3f} detect={c['detect']:.2f} (seen-subtype {sn['detect']:.2f}, NOVEL-subtype {nv['detect']:.2f}) delay={c['median_delay']} "
              f"released={r['released_by_end']:.2f} stealth30%={s_['detect']:.2f}", flush=True)

# commissioning-window contamination: a fraction of the fleet is already compromised while the baseline is being estimated
R["contamination"] = {}
for frac in (0.0, 0.1, 0.3, 0.45):
    rng = np.random.RandomState(21); n_c = int(round(frac * NDEV)); fleet = []
    for d in range(NDEV):
        E, _ = gen_evidence(EVF[(yt == 0)], PA["f"], "compromised" if d < n_c else "benign", rng, ROUNDS, M, onset=0, attack_cls=int(rng.choice([1, 2, 3, 4])))
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
