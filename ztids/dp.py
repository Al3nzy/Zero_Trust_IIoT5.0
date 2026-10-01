"""
Formal differentially private training (DP-SGD) with a Renyi-DP accountant.

Neighbouring relation: add/remove ONE ORIGINAL training record (record-level DP).
  * DP-SGD runs on the original training records (no SMOTE / duplication, which would let one record
    influence many training rows); class imbalance is handled with fixed public class weights.
  * per-example gradients are clipped to L2 norm C, summed, Gaussian noise N(0, (sigma*C)^2) is added,
    batches are Poisson-subsampled with rate q = B/N (padded to a fixed size with zero-weight rows).
  * the privacy loss is tracked with the Renyi-DP accountant of the Sampled Gaussian Mechanism
    (Mironov, Talwar, Zhang 2019) and converted to (eps, delta)-DP (Balle et al. 2020 conversion).
  * normalisation layers that mix examples (BatchNorm) are replaced by per-example LayerNorm, and dropout is
    disabled, so that every example's gradient depends on that example only.
  * the preprocessing (scaler, MI feature selection) is fitted on the training data and is NOT covered by the
    guarantee; this scope boundary is stated in the paper.
"""
import math
import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = list(range(2, 65)) + [80, 96, 128, 192, 256]


def _rdp_sgm(q, sigma, alpha):
    """RDP of one step of the Sampled Gaussian Mechanism, integer alpha (Mironov et al. 2019, Sec. 3.3)."""
    if q == 0: return 0.0
    if q == 1.0: return alpha / (2 * sigma ** 2)
    k = np.arange(alpha + 1)
    logc = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    terms = logc + (alpha - k) * math.log(1 - q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2)
    return float(logsumexp(terms) / (alpha - 1))


def epsilon(q, sigma, steps, delta):
    best = float("inf")
    for a in ORDERS:
        rdp = steps * _rdp_sgm(q, sigma, a)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return best


def calibrate_sigma(q, steps, target_eps, delta, lo=0.3, hi=200.0):
    for _ in range(60):
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta) > target_eps: lo = mid
        else: hi = mid
    return hi
