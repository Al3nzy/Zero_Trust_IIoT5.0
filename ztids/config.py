"""Central configuration. Everything is driven by environment variables so that every script behaves identically
whether it is launched by run_everything.py or on its own."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUICK = os.environ.get("ZTIDS_QUICK", "0") == "1"                 # smoke-test mode: tiny data/epochs, results_quick/
RES = os.environ.get("ZTIDS_RES", os.path.join(ROOT, "results_v3_quick" if QUICK else "results_v3"))     # results of the v3 stages (LightGBM-based); legacy deep-model stages: set ZTIDS_RES=results_v2
FIG = os.environ.get("ZTIDS_FIG", os.path.join(ROOT, "figures_quick" if QUICK else "figures"))
TAB = os.environ.get("ZTIDS_TAB", os.path.join(ROOT, "tables_quick" if QUICK else "tables"))
DATA = os.environ.get("ZTIDS_DATA", os.path.join(ROOT, "data"))
EXTRA = os.environ.get("ZTIDS_EXTRA", os.path.join(ROOT, "results_extra"))       # results of run_dataset.py (UNSW-NB15, Edge-IIoTset)
NPC = int(os.environ.get("ZTIDS_NPC", 300 if QUICK else 4000))        # post-split balanced training size per class
EPOCHS = int(os.environ.get("ZTIDS_EPOCHS", 2 if QUICK else 14))      # max epochs (early stopping on validation loss)
CORE_SEEDS = os.environ.get("ZTIDS_CORE_SEEDS", "0,1" if QUICK else ",".join(map(str, range(10))))   # seeds of the core/site/selection stages
NDEV = int(os.environ.get("ZTIDS_NDEV", 20 if QUICK else 200))                                      # devices per cell in the trust stage
TRUST_SEEDS = os.environ.get("ZTIDS_TRUST_SEEDS", "0" if QUICK else "0,1,2")
SEEDS = [int(s) for s in os.environ.get("ZTIDS_SEEDS", "0" if QUICK else "0,1,2").split(",") if s != ""]
if 0 not in SEEDS: SEEDS.insert(0, 0)                               # seed 0 saves the checkpoints used by SHAP/latency/trust
THREADS = int(os.environ.get("ZTIDS_THREADS", 0))                  # 0 = let TensorFlow decide
DP_N = int(os.environ.get("ZTIDS_DP_N", 6000 if QUICK else 0))     # 0 = all original training records
DP_EPOCHS = int(os.environ.get("ZTIDS_DP_EPOCHS", 1 if QUICK else 3))
DP_BATCH = int(os.environ.get("ZTIDS_DP_BATCH", 64 if QUICK else 256))
CLIP = float(os.environ.get("ZTIDS_CLIP", 5.0))                       # clip standardised features to [-CLIP, CLIP] (0 = off)
NORM = os.environ.get("ZTIDS_NORM", "ln")                          # normalisation layer of the neural models: ln (default, robust to outliers) | bn (original)
VAL_WEIGHTING = os.environ.get("ZTIDS_VALW", "none")               # validation-loss class weighting for early stopping: none | sqrt | balanced
SHAP_N = int(os.environ.get("ZTIDS_SHAP_N", 100 if QUICK else 1000))
for _d in (RES, FIG, TAB): os.makedirs(_d, exist_ok=True)
PROPOSED = "official_cnn_bilstm_0_-_-_-_selovr"                      # CNN-BiLSTM + class-aware MI selection (seed 0)
ATTN = "official_cnn_bilstm_attn_0_-_-_-_selovr"                      # same with dual attention (seed 0)
def rp(name): return os.path.join(RES, name)
