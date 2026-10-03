"""Stage 4b: sensitivity of the trust engine to its design parameters (k, certified alpha -> h, flows per round m, commissioning window).
Replay simulation on the real posteriors (+ novelty evidence). h = ln(1/alpha) / (8 m k) is the certified threshold."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, warnings; warnings.filterwarnings("ignore")
from ztids import config as C
from ztids.trust import gen_evidence, run_rule
from ztids import evidence as EV_
E_ = EV_.load(); yt, svc, EV = E_["yt"], E_["svc"], E_["EVf"]
Pn = EV[(yt == 0) & (svc == "http")]; PA = EV_.attack_pools(EV, yt, E_["novel"]); N = 40 if C.QUICK else 150; ROUNDS = 60
def cell(k, alpha, m, warm):
    h = float(np.log(1 / alpha) / (8 * m * k)); rng = np.random.RandomState(5); onset = max(15, warm + 5); fq = 0; dl = []; st = 0
    for _ in range(N):
        E, _ = gen_evidence(Pn, PA, "benign", rng, ROUNDS, m, onset); mu0 = E[:warm, 0].mean(); fq += (run_rule(E, "cusum", mu0=mu0, k=k, h=h, warm=warm)[1][warm:] == "Quarantined").any()
        E, _ = gen_evidence(Pn, PA, "compromised", rng, ROUNDS, m, onset); mu0 = E[:warm, 0].mean(); q = np.where(run_rule(E, "cusum", mu0=mu0, k=k, h=h, warm=warm)[1][onset:] == "Quarantined")[0]; dl.append(q[0] + 1 if len(q) else np.nan)
        E, _ = gen_evidence(Pn, PA, "stealth", rng, ROUNDS, m, onset, frac=0.3); mu0 = E[:warm, 0].mean(); st += (run_rule(E, "cusum", mu0=mu0, k=k, h=h, warm=warm)[1][onset:] == "Quarantined").any()
    return dict(k=k, alpha=alpha, m=m, warm=warm, h=h, false_quarantine=float(fq / N), detect=float(np.mean(~np.isnan(dl))), median_delay=None if np.all(np.isnan(dl)) else float(np.nanmedian(dl)), stealth30=float(st / N))
rows = []
for k, alpha, m in itertools.product((0.02, 0.05, 0.10), (0.1, 0.01, 0.001), (10, 20, 50)):
    rows.append(cell(k, alpha, m, 10)); print(rows[-1], flush=True)
for warm in (5, 20): rows.append(cell(0.05, 0.01, 20, warm)); print("warm-up", rows[-1], flush=True)
json.dump(dict(n_devices=N, rows=rows), open(os.path.join(C.RES, "trust_sensitivity.json"), "w"), indent=1)
