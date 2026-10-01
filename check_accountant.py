"""Optional sanity check: our RDP accountant versus Google's dp-accounting reference implementation (should agree to ~4 decimals)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_accounting import dp_event, rdp
from ztids.dp import epsilon
def ref(q, s, T, d):
    acc = rdp.RdpAccountant(orders=[float(a) for a in range(2, 257)]); acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(s)), T); return acc.get_epsilon(d)
for q, s, T in [(0.01, 4.0, 10000), (256 / 60000, 1.1, 14062), (0.00256, 1.0, 2000), (0.002, 0.8, 3000)]:
    print(f"q={q:.5f} sigma={s} steps={T}: ours={epsilon(q, s, T, 1e-5):.4f} reference={ref(q, s, T, 1e-5):.4f}")
