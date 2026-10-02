"""Stage 4: empirical validation of the proved false-quarantine bound, detection-delay bound and minimum detectable attack fraction."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from ztids import config as C
from ztids.trust import SEVERITY, gen_evidence, run_rule, design
from ztids.data import CLASSES
from ztids import evidence as EV_
E_ = EV_.load(); pt, yt, svc, EV = E_["pt"], E_["yt"], E_["svc"], E_["EVf"]       # classifier + novelty evidence (classifier-only if no novelty file)
GROUP = "http"                                           # homogeneous device class (largest benign service in the official test set)
Pn = EV[(yt == 0) & (svc == GROUP)]; Pa = {k: EV[yt == k] for k in (1, 2, 3, 4)}
harm = EV[:, 0]; mu_b = float(harm[(yt == 0) & (svc == GROUP)].mean())
M, WARM, K = 20, 10, 0.05
N, HOR = (60, 80) if C.QUICK else (600, 300); ND = 60 if C.QUICK else 400
out = {"group": GROUP, "mu_b": mu_b, "k": K, "m": M, "n_devices": N, "horizon": HOR}
rng = np.random.RandomState(0); rows = []
for rep in (1, 2, 4):                                      # rep > 1: correlated flows, m_eff = M / rep (stress test of the independence assumption)
    for alpha in (0.5, 0.2, 0.1, 0.01):
        meff = M / rep; fq = 0
        for _ in range(N):
            E, _ = gen_evidence(Pn, Pa, "benign", rng, HOR, M, rep=rep); mu0 = E[:WARM, 0].mean()
            h = design(mu_b, mu0, K, meff, alpha); h = 1e9 if not np.isfinite(h) else h
            fq += (run_rule(E, "cusum", mu0=mu0, k=K, h=h, warm=WARM)[1][WARM:] == "Quarantined").any()
        rows.append(dict(rep=rep, m_eff=meff, alpha=alpha, empirical=float(fq / N))); print("FQ", rows[-1], flush=True)
out["false_quarantine"] = rows
rng = np.random.RandomState(1); dl = []
for c in (1, 2, 3, 4):
    mu_a = float(harm[yt == c].mean()); taus = []
    for _ in range(ND):
        E, _ = gen_evidence(Pn, Pa, "compromised", rng, 120, M, onset=15, attack_cls=c); mu0 = E[:WARM, 0].mean(); h = design(mu_b, mu0, K, M, 0.01)
        S = run_rule(E, "cusum", mu0=mu0, k=K, h=h, warm=WARM)[1]; q = np.where(S[15:] == "Quarantined")[0]; taus.append(q[0] + 1 if len(q) else np.nan)
    d = mu_a - mu_b - K; bound = (design(mu_b, mu_b, K, M, 0.01) + 1) / d if d > 0 else None
    dl.append(dict(cls=CLASSES[c], mu_a=mu_a, drift=float(d), bound=None if bound is None else float(bound),
                   mean_delay=float(np.nanmean(taus)) if not np.all(np.isnan(taus)) else None, detect=float(np.mean(~np.isnan(taus)))))
    print("DELAY", dl[-1], flush=True)
out["delay"] = dl
rng = np.random.RandomState(2); fs = []
for c in (1, 3):
    mu_att = float(harm[yt == c].mean()); fstar = K / (mu_att - mu_b) if mu_att > mu_b else None
    for f in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        det = 0
        for _ in range(ND):
            E, _ = gen_evidence(Pn, Pa, "stealth", rng, 80, M, onset=15, attack_cls=c, frac=f); mu0 = E[:WARM, 0].mean(); h = design(mu_b, mu0, K, M, 0.01)
            det += (run_rule(E, "cusum", mu0=mu0, k=K, h=h, warm=WARM)[1][15:] == "Quarantined").any()
        fs.append(dict(cls=CLASSES[c], f=f, fstar=fstar, detect=float(det / ND)))
    print("FSTAR", CLASSES[c], fstar, flush=True)
out["min_detectable"] = fs
json.dump(out, open(os.path.join(C.RES, "trust_theory.json"), "w"), indent=1, default=float)
