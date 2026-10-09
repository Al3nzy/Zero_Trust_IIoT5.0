"""Stage `trust`: multi-round device-trust evaluation on replayed real posteriors (replay simulation, not a deployment).
Eight update rules see identical device streams. Evidence: severity-weighted harm with the Normal-profile flag. Output: results_v3/trust_main.json.
  python run_trust.py            (ZTIDS_TRUST_SEEDS=0,1,2 classifier seeds; ZTIDS_NDEV devices per seed and cell)"""
import os, sys, json, math, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import kstest
from ztids import config as C
from ztids import evidence as EV_
from ztids.devices import Pools, make_fleet
from ztids.trust import run_rule
from ztids.edetector import run_edetector, run_edetector_flows, threshold_for, OnlineConformal

SEEDS = [int(s) for s in C.TRUST_SEEDS.split(",")]
NDEV = C.NDEV
M, ROUNDS, W, ONSET, END = 20, 300, 50, 100, 160
HOR = ROUNDS - W                                   # rounds after commissioning = false-alarm horizon
ALPHA = 0.05; C_E = threshold_for(HOR, ALPHA)      # e-detector threshold: P(false quarantine within HOR) <= ALPHA
K = 0.05; H_HOEF = math.log(100) / (8 * M * K); H_HOR = math.log(HOR / ALPHA) / (8 * M * K)
Q = "Quarantined"
RULES = ["Confidence-only", "Beta reputation", "CUSUM, Hoeffding design, fleet baseline", "CUSUM, horizon design, fleet baseline", "CUSUM, horizon design, device baseline",
         "E-detector, round level, device reference", "E-detector, round level, fleet reference", "E-detector, flow level, device reference",
         "E-detector, round level, device reference, mode-conditional"]


def states(rule, F, ctx, i, ref_scores_fleet, modes=None):
    E = F.mean(1)                                    # (rounds,3) round evidence
    if rule == "Confidence-only": return run_rule(E, "confidence")[1]
    if rule == "Beta reputation": return run_rule(E, "beta")[1]
    if rule.startswith("CUSUM"):
        mu0 = ctx["fleet_mu0"] if "fleet" in rule else E[:W, 0].mean(); h = H_HOEF if "Hoeffding" in rule else H_HOR
        return run_rule(E, "cusum", mu0=mu0, k=K, h=h, warm=W)[1]
    if "mode-conditional" in rule: return run_edetector(E[:, 0], warm=W, c=C_E, seed=i, context=modes)[1]
    if "round level, device" in rule: return run_edetector(E[:, 0], warm=W, c=C_E, seed=i)[1]
    if "round level, fleet" in rule: return run_edetector(E[:, 0], ref_scores=ref_scores_fleet, warm=W, c=C_E, seed=i)[1]
    if "flow level" in rule: return run_edetector_flows(F[:, :, 0], warm=W, c=C_E, seed=i)[1]
    raise ValueError(rule)


def first(S, a, b=None):
    q = np.where(np.asarray(S[a:b]) == Q)[0]; return int(q[0]) + 1 if len(q) else np.nan


def summarize(scn, SS):
    out = {"n": len(SS), "ever_quarantined": float(np.mean([(s[W:] == Q).any() for s in SS]))}
    if scn in ("compromised", "stealth", "intermittent", "novel_attack", "seen_attack"):
        dl = np.array([first(s, ONSET) for s in SS]); out.update(detect=float(np.mean(~np.isnan(dl))), median_delay=None if np.all(np.isnan(dl)) else float(np.nanmedian(dl)),
                                                                   p90_delay=None if np.mean(~np.isnan(dl)) < 0.9 else float(np.nanpercentile(dl, 90)))
    if scn == "recovery":
        rel = [(np.where(np.asarray(s[END:]) != Q)[0][0] + 1) if (np.asarray(s[END:]) != Q).any() else np.nan for s in SS]
        out.update(was_quarantined=float(np.mean([(np.asarray(s[ONSET:END]) == Q).any() for s in SS])), released=float(np.mean(~np.isnan(rel))),
                   median_release_delay=None if np.all(np.isnan(rel)) else float(np.nanmedian(rel)))
    return out


def evaluate(P, fleet_kind, scn, mode, seed, rules, **kw):
    hetero = fleet_kind == "hetero"
    base = {"novel_attack": ("compromised", dict(attack="novel")), "seen_attack": ("compromised", dict(attack="seen"))}.get(scn, (scn, {}))
    fleet, modes = make_fleet(P, NDEV, base[0], seed * 101 + 7, ROUNDS, M, hetero, onset=ONSET, end=END, mode=mode, **{**base[1], **kw})
    Ev = [F.mean(1)[:, 0] for F in fleet]
    ctx = dict(fleet_mu0=float(np.median([e[:W].mean() for e in Ev]))); pooled = np.concatenate([e[:W] for e in Ev])
    return {r: summarize(scn if scn not in ("novel_attack", "seen_attack") else scn, [states(r, F, ctx, i, pooled, modes[i]) for i, F in enumerate(fleet)]) for r in rules}, fleet, pooled


def merge(rows):
    """Pool per-seed summaries: rates are averaged (equal n per seed), delays are the median of per-seed medians."""
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals and k != "n" else (sum(r[k] for r in rows) if k == "n" else None)
        if k in ("median_delay", "p90_delay", "median_release_delay") and vals: out[k] = float(np.median(vals))
    return out


if __name__ == "__main__":
    t0 = time.time(); R = {"setup": dict(m=M, rounds=ROUNDS, commissioning=W, onset=ONSET, end=END, horizon=HOR, alpha=ALPHA, c=C_E, k=K, h_hoeffding=H_HOEF, h_horizon=H_HOR, n_devices_per_seed=NDEV,
                                          classifier_seeds=SEEDS, note="replay simulation of real posteriors; temporal structure is injected, not measured")}
    cells = [("homog", "iid", ["benign", "compromised", "recovery", "intermittent", "stealth", "novel_attack", "seen_attack"]), ("hetero", "iid", ["benign", "compromised", "recovery", "intermittent", "stealth", "novel_attack", "seen_attack"]),
             ("hetero", "regime", ["benign", "compromised"]), ("hetero", "drift", ["benign", "compromised"])]
    acc = {}; traj = {}; diag = {"homog": [], "hetero": []}
    for seed in SEEDS:
        E_ = EV_.load(seed); P = Pools(E_["EVf"], E_["yt"], E_["svc"], E_["novel"])
        for fk, mode, scns in cells:
            for scn in scns:
                kw = dict(frac=0.3) if scn == "stealth" else {}
                res, fleet, pooled = evaluate(P, fk, scn, mode, seed, RULES, **kw)
                for r, v in res.items(): acc.setdefault((fk, mode, scn, r), []).append(v)
                if scn == "benign" and mode == "iid":          # commissioning diagnostic: KS test of each device's own commissioning rounds against the fleet reference
                    rng = np.random.RandomState(5); frac_flag = []
                    for F in fleet[:100]:
                        oc = OnlineConformal(pooled); pv = [oc.pvalue(x, rng.rand(), add=False) for x in F.mean(1)[:W, 0]]; frac_flag.append(kstest(pv, "uniform").pvalue < 0.05)
                    diag[fk].append(float(np.mean(frac_flag)))
                if seed == SEEDS[0] and mode == "iid" and fk == "hetero" and scn in ("benign", "compromised", "recovery", "intermittent"):
                    F = fleet[3]; E = F.mean(1)
                    from ztids.trust import run_rule as rr
                    traj[scn] = {"harm": E[:, 0].tolist(),
                                 "confidence": rr(E, "confidence")[0].tolist(), "cusum_horizon": rr(E, "cusum", mu0=E[:W, 0].mean(), k=K, h=H_HOR, warm=W)[0].tolist(),
                                 "edet_round": run_edetector(E[:, 0], warm=W, c=C_E, seed=1)[0].tolist(), "edet_flow": run_edetector_flows(F[:, :, 0], warm=W, c=C_E, seed=1)[0].tolist(),
}
                print(f"[seed {seed}] {fk}/{mode}/{scn} done ({time.time() - t0:.0f}s)", flush=True)
    R["cells"] = {}
    for (fk, mode, scn, r), v in acc.items(): R["cells"].setdefault(f"{fk}|{mode}", {}).setdefault(r, {})[scn] = merge(v)
    R["commissioning_diagnostic_share_flagged"] = {k: float(np.mean(v)) for k, v in diag.items()}
    R["trajectories"] = traj

    # sustained contamination of the commissioning window (a fraction of devices attacked from round 0)
    R["contamination"] = {}
    for seed in SEEDS[:1]:
        E_ = EV_.load(seed); P = Pools(E_["EVf"], E_["yt"], E_["svc"], E_["novel"])
        for frac in (0.0, 0.1, 0.3, 0.45):
            rng = np.random.RandomState(31); nc = int(round(frac * NDEV)); fleet = []
            for d in range(NDEV):
                from ztids.devices import make_device
                fleet.append(make_device(P, None, "compromised" if d < nc else "benign", rng, ROUNDS, M, onset=0, attack=int(rng.choice([1, 2, 3, 4])))[0])
            Ev = [F.mean(1)[:, 0] for F in fleet]; ctx = dict(fleet_mu0=float(np.median([e[:W].mean() for e in Ev]))); pooled = np.concatenate([e[:W] for e in Ev])
            for r in ("CUSUM, horizon design, fleet baseline", "CUSUM, horizon design, device baseline", "E-detector, round level, device reference", "E-detector, round level, fleet reference"):
                q = [(np.asarray(states(r, F, ctx, i, pooled)[W:]) == Q).any() for i, F in enumerate(fleet)]
                R["contamination"][f"{frac}|{r}"] = dict(frac=frac, rule=r, detect_contaminated=float(np.mean(q[:nc])) if nc else None, false_quarantine_benign=float(np.mean(q[nc:])))
            print("[contamination]", frac, flush=True)
    # stealth sweep: detection probability against the share of attacked flows per round
    R["stealth_sweep"] = {}; NDEV = max(NDEV // 2, 20)
    E_ = EV_.load(SEEDS[0]); P = Pools(E_["EVf"], E_["yt"], E_["svc"], E_["novel"])
    for cls in (1, 3):
        for f in (0.05, 0.1, 0.2, 0.3, 0.5):
            res, _, _ = evaluate(P, "hetero", "stealth", "iid", SEEDS[0], ["CUSUM, horizon design, device baseline", "E-detector, round level, device reference", "E-detector, flow level, device reference"], frac=f, attack=cls)
            for r, v in res.items(): R["stealth_sweep"].setdefault(r, {})[f"{cls}|{f}"] = v["detect"]
        print("[stealth sweep] class", cls, flush=True)
    json.dump(R, open(os.path.join(C.RES, "trust_main.json"), "w"), indent=1, default=float)
    print(f"trust stage done in {time.time() - t0:.0f}s", flush=True)
