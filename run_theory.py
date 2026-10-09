#!/usr/bin/env python3
"""Stage `theory`: numerical validation of the guarantees (each check is a statement that can fail). Every part is checkpointed in results_v3/theory_<part>.json.
 t1  CUSUM: the per-round tail bound and the horizon bound hold; the alarm probability over a long horizon is far above the per-round bound (bounded i.i.d. flows, known baseline).
 t2  E-detector null: P(false quarantine within T) <= T/c on several score families (ties, heavy tails, real NSL-KDD evidence); E[R_T] = T; ARL >= c; p-values i.i.d. uniform.
 t3  E-detector delay: the exact delay law under separable exchangeable attacks dominates the delay of the real detector; design table of (s, c).
Output: results_v3/theory.json"""
import os, sys, json, math, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from ztids import config as C
from ztids.edetector import run_edetector, separable_delay_law, threshold_for

QUICK = C.QUICK; t0 = time.time()


def ckpt(name, fn):
    path = os.path.join(C.RES, f"theory_{name}.json")
    if os.path.exists(path): return json.load(open(path))
    r = fn(); json.dump(r, open(path, "w"), indent=1); print(f"[theory] {name} done ({time.time() - t0:.0f}s)", flush=True); return r


def t1():
    m, k, alpha = 20, 0.05, 0.01; h = math.log(1 / alpha) / (8 * m * k); theta = 8 * m * k; N, TM = (1500, 1500) if QUICK else (6000, 5000); HS = (60, 250, 1000, 5000); rows = []
    for mu_b in (0.5, 0.3, 0.1):
        rng = np.random.default_rng(int(mu_b * 100)); S = np.zeros(N); first = np.full(N, np.inf); above = 0; cnt = 0
        for t in range(1, TM + 1):
            S = np.minimum(2 * h, np.maximum(0, S + rng.binomial(m, mu_b, N) / m - mu_b - k)); first[(S >= h) & np.isinf(first)] = t
            if t > 200: above += (S >= h).mean(); cnt += 1
        hT = math.log(250 / 0.05) / (8 * m * k); rng = np.random.default_rng(7 + int(mu_b * 100)); S = np.zeros(N); fq = np.zeros(N, bool)
        for t in range(250): S = np.minimum(2 * hT, np.maximum(0, S + rng.binomial(m, mu_b, N) / m - mu_b - k)); fq |= S >= hT
        rows.append(dict(mu_b=mu_b, h=h, claimed_ever=alpha, per_round_tail=float(above / cnt), per_round_bound=math.exp(-theta * h), within={str(T): float((first <= T).mean()) for T in HS if T <= TM},
                         union_bound={str(T): min(1.0, T * math.exp(-theta * h)) for T in HS if T <= TM}, horizon_design=dict(T=250, alpha=0.05, h=hT, empirical=float(fq.mean()))))
    return dict(m=m, k=k, n_devices=N, rows=rows)


def fams():
    from ztids import evidence as EV_
    from ztids.devices import Pools
    F = {"Beta(2,8)": lambda r, n: r.beta(2, 8, n), "Poisson counts (ties)": lambda r, n: r.poisson(2.0, n).astype(float), "Pareto (heavy tail)": lambda r, n: r.pareto(1.5, n),
         "Bernoulli round mean": lambda r, n: r.binomial(20, 0.3, n) / 20.0}
    try:
        E_ = EV_.load(0); P = Pools(E_["EVf"], E_["yt"], E_["svc"], E_["novel"]); pool = P.benign[2]; F["NSL-KDD benign evidence (private services)"] = lambda r, n: pool[r.randint(0, len(pool), (n, 20)), 0].mean(1)
    except SystemExit: pass
    return F


def t2_null():
    W = 20; rows = []
    for name, gen in fams().items():
        for T, a in ((250, 0.05), (250, 0.01), (1000, 0.05)):
            c = threshold_for(T, a); n_rep = 150 if QUICK else (600 if T == 250 else 250); rng = np.random.RandomState(11); fa = 0
            for i in range(n_rep): fa += (run_edetector(gen(rng, W + T), warm=W, c=c, seed=i)[1][W:] == "Quarantined").any()
            rows.append(dict(family=name, T=T, alpha=a, c=c, empirical=float(fa / n_rep), bound=a, n=n_rep, wilson_hi=float(stats.beta.ppf(0.975, fa + 1, n_rep - fa)) if fa < n_rep else 1.0))
    return rows


def t2_martingale():
    """E[R_t | past] = 1 + R_{t-1} needs (i) E[e] = 1, checked by quadrature, and (ii) i.i.d. uniform p-values, checked by a KS test and the lag-1 autocorrelation.
    A Monte Carlo mean of R_T is deliberately NOT used: bets with epsilon < 1/2 have infinite variance, so sample means of the product converge far too slowly to test the identity."""
    from scipy.integrate import quad
    from ztids.edetector import bet
    W, Tm, nm = 20, 100, (500 if QUICK else 2000); rng = np.random.RandomState(3); pall = []
    for i in range(nm): pall.append(run_edetector(rng.beta(2, 8, W + Tm), warm=W, c=1e300, cap_factor=1e6, freeze=False, seed=i)[3][W:])
    pall = np.array(pall); integral = quad(bet, 0, 1, limit=400)[0]
    return dict(T=Tm, n=nm, bet_integral=float(integral), ks_pvalue=float(stats.kstest(pall.ravel(), "uniform").pvalue), lag1_autocorr=float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in pall])))


def t2_arl():
    W, c_arl, Tmax = 20, 50.0, 2000; rng = np.random.RandomState(4); taus = []
    for i in range(200 if QUICK else 500):
        q = np.where(run_edetector(rng.beta(2, 8, W + Tmax), warm=W, c=c_arl, seed=i)[1][W:] == "Quarantined")[0]; taus.append(q[0] + 1 if len(q) else Tmax)
    return dict(c=c_arl, censored_at=Tmax, mean_time_to_first_alarm_lower_estimate=float(np.mean(taus)), n=len(taus), share_censored=float(np.mean(np.array(taus) == Tmax)))


def t3_check():
    rows = []
    for s in (50, 100, 300):
        for c in (1e3, 5e3):
            law = separable_delay_law(s, c, n_mc=4000 if QUICK else 15000, kmax=300); rng = np.random.RandomState(5); sim = []
            for i in range(200 if QUICK else 500):
                x = np.r_[rng.beta(2, 8, s) * 0.5, 0.6 + 0.4 * rng.beta(5, 2, 200)]; q = np.where(run_edetector(x, warm=min(s, 50), c=c, seed=i)[1][s:] == "Quarantined")[0]; sim.append(q[0] + 1 if len(q) else 201)
            qs = (50, 90, 99); rows.append(dict(s=s, c=c, law={str(q): float(np.percentile(law, q)) for q in qs}, detector={str(q): float(np.percentile(sim, q)) for q in qs}, dominated=bool(all(np.percentile(sim, q) <= np.percentile(law, q) for q in qs))))
    return rows


def t3_table():
    table = []
    for s in (20, 50, 100, 200, 500, 1000):
        for c in (1e2, 1e3, 1e4):
            law = separable_delay_law(s, c, n_mc=4000 if QUICK else 15000, kmax=300); table.append(dict(s=s, c=c, median=float(np.median(law)), p90=float(np.percentile(law, 90)), reached=float((law <= 300).mean())))
    return table


if __name__ == "__main__":
    out = dict(cusum=ckpt("t1", t1), null=ckpt("t2_null", t2_null), martingale=ckpt("t2_martingale", t2_martingale), arl=ckpt("t2_arl", t2_arl), delay_check=ckpt("t3_check", t3_check), delay_table=ckpt("t3_table", t3_table))
    json.dump(out, open(os.path.join(C.RES, "theory.json"), "w"), indent=1); print(f"theory stage done in {time.time() - t0:.0f}s")
