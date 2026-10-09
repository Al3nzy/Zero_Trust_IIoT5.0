"""Site-calibrated Normal decision for the classifier. argmax operating points are tuned on the training domain; under distribution shift the benign false-alarm rate
can be far from what was validated (UNSW-NB15 official split: ~38%). Here the Normal/attack decision uses a threshold tau on P(attack)=1-P(Normal) that is calibrated on a
clean COMMISSIONING window (random 10% of test-domain Normal flows, excluded from evaluation); flows above tau get the most probable attack class.
All targets alpha are reported (none selected); needs a clean benign period at the site. The threshold is split-conformal, so for benign flows exchangeable with the window the false-alarm rate is <= alpha."""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef


def conformal_threshold(ref, alpha):
    """Split-conformal threshold: for benign scores exchangeable with the reference window, P(new benign score > threshold) <= alpha (finite-sample guarantee).
    Uses the ceil((n+1)(1-alpha))-th smallest reference score; +inf if the window is too small for this alpha."""
    s = np.sort(np.asarray(ref)); n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(s[k - 1])


def window_ok(n, alpha):
    """A conformal threshold at level alpha exists only if the commissioning window has at least ceil((1-alpha)/alpha) clean flows (19 for alpha=5%, 49 for 2%)."""
    return int(np.ceil((n + 1) * (1 - alpha))) <= n


def site_decision(P, nidx, pa_ref, alpha):
    tau = conformal_threshold(pa_ref, alpha); Q = P.copy(); Q[:, nidx] = -1.0
    return np.where((1 - P[:, nidx]) > tau, Q.argmax(1), nidx)


def _met(pred, y, nidx, n_cls):
    ben = y == nidx
    return dict(macro_f1=float(f1_score(y, pred, labels=range(n_cls), average="macro", zero_division=0)), acc=float(accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
                fpr=float((pred[ben] != nidx).mean()), attack_det=float((pred[~ben] != nidx).mean()))


def eval_site_decision(P, yt, nidx, seed, alphas=(0.02, 0.05, 0.10), frac=0.10, draws=20):
    """Returns {'argmax': metrics, '<alpha>': metrics}; every entry is evaluated on the SAME flows (those outside the commissioning window), mean of `draws` windows."""
    n_cls = P.shape[1]; nrm = np.where(yt == nidx)[0]; pa = 1 - P[:, nidx]; acc = {"argmax": []}; acc.update({str(a): [] for a in alphas})
    for rep in range(draws):
        rng = np.random.RandomState(1000 * seed + rep); com = rng.choice(nrm, max(1, int(frac * len(nrm))), replace=False)
        keep = np.ones(len(yt), bool); keep[com] = False; y = yt[keep]
        acc["argmax"].append(_met(P.argmax(1)[keep], y, nidx, n_cls))
        for a in alphas:
            if window_ok(len(com), a): acc[str(a)].append(_met(site_decision(P, nidx, pa[com], a)[keep], y, nidx, n_cls))      # too-small windows are skipped, never turned into a degenerate rule
    return {k: ({m: float(np.mean([r[m] for r in v])) for m in v[0]} if v else None) for k, v in acc.items()}
