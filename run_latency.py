"""Stage 8: end-to-end latency/footprint on the SAME hardware for the plain and the dual-attention model.
Forced to 1 thread for comparability. Run on an otherwise idle machine; do not run other stages concurrently."""
import sys, os, json, time, platform
try:
    import resource                    # Unix only
except ImportError:
    resource = None
os.environ["ZTIDS_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from ztids import config as C
from ztids.data import prepare
from ztids.ledger import Ledger
from ztids.trust import run_rule
from ztids.models import load
import tensorflow as tf
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
MODELS = {n: os.path.join(C.RES, p + ".keras") for n, p in (("plain", C.PROPOSED), ("attention", C.ATTN)) if os.path.exists(os.path.join(C.RES, p + ".keras"))}
if not MODELS: sys.exit("checkpoints missing; run the detector stage first")
d = prepare("official", 0, n_per_class=C.NPC, sel_mode="ovr"); pp, test_df = d["pp"], d["test_df"]
rng = np.random.RandomState(0); raw = test_df.iloc[rng.choice(len(test_df), 500, replace=False)]


def timeit(f, n=200, warm=20):
    for _ in range(warm): f()
    t = np.empty(n)
    for i in range(n): s = time.perf_counter(); f(); t[i] = time.perf_counter() - s
    return dict(mean_ms=float(t.mean() * 1e3), p50_ms=float(np.median(t) * 1e3), p95_ms=float(np.percentile(t, 95) * 1e3), p99_ms=float(np.percentile(t, 99) * 1e3))


nrep = 50 if C.QUICK else 200
out = dict(hw=dict(machine=platform.machine(), platform=platform.platform(), cpu_count=os.cpu_count(), tf_threads=1, tf=tf.__version__), models={})
one, batch = raw.iloc[[0]], raw.iloc[:20]
out["preprocess_1flow"] = timeit(lambda: pp.transform(one), nrep); out["preprocess_20flows"] = timeit(lambda: pp.transform(batch), nrep)
key = Ed25519PrivateKey.generate(); L = Ledger(signer=key, anchor_every=50); L0 = Ledger(signer=None, anchor_every=0)
out["ledger_append_signed"] = timeit(lambda: L.append({"dev": "d1", "T": 0.5, "state": "Degraded"}), 5 * nrep)
out["ledger_append_hash_only"] = timeit(lambda: L0.append({"dev": "d1", "T": 0.5, "state": "Degraded"}), 5 * nrep)
E = np.random.rand(60, 3); out["trust_update_round"] = timeit(lambda: run_rule(E[:11], "cusum", mu0=0.05, k=0.05, h=0.6, warm=10), 2 * nrep)
X1, X20, X500 = pp.transform(one), pp.transform(batch), pp.transform(raw)
for name, path in MODELS.items():
    m = load(path); f1 = tf.function(lambda x: m(x, training=False)); r = dict(params=int(m.count_params()), size_mb=os.path.getsize(path) / 1e6)
    r["infer_1flow"] = timeit(lambda: f1(tf.constant(X1[..., None])), nrep); r["infer_20flows"] = timeit(lambda: f1(tf.constant(X20[..., None])), nrep)
    m.predict(X500[..., None], batch_size=500, verbose=0)
    ts = []
    for _ in range(7):                                      # median of 7 timed batches (a single timing was order/warm-up sensitive)
        s = time.perf_counter(); m.predict(X500[..., None], batch_size=500, verbose=0); ts.append(time.perf_counter() - s)
    r["infer_500_batch_warm_s"] = float(np.median(ts))
    r["throughput_flows_per_s"] = 500 / r["infer_500_batch_warm_s"]
    r["round_total_ms"] = (out["preprocess_20flows"]["mean_ms"] + r["infer_20flows"]["mean_ms"] + out["trust_update_round"]["mean_ms"] + out["ledger_append_signed"]["mean_ms"])
    out["models"][name] = r; print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}, flush=True)
if "plain" in MODELS:
    import shap
    m = load(MODELS["plain"]); bg = d["Xtr"][rng.choice(len(d["Xtr"]), 100, replace=False)][..., None]
    ex = shap.GradientExplainer(m, bg); s = time.perf_counter(); ex.shap_values(X1[..., None], nsamples=50); out["shap_1flow_s"] = time.perf_counter() - s
try:
    import psutil; out["peak_rss_mb"] = psutil.Process().memory_info().rss / 1e6          # current RSS (portable, needs psutil)
except ImportError:
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource else None   # Linux KiB -> MiB
json.dump(out, open(os.path.join(C.RES, "latency.json"), "w"), indent=1)
print("latency stage complete", flush=True)
