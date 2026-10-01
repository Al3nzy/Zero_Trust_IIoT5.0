"""Stage 1: train/evaluate the proposed detector, all ablations and all baselines (resumable)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ztids import config as C
from ztids.jobs import run, run_tree
O, OV, S = "official", dict(sel="ovr"), C.SEEDS
print(f"detector stage | quick={C.QUICK} seeds={S} npc={C.NPC} epochs={C.EPOCHS} results -> {C.RES}", flush=True)
# 1. proposed detector (CNN-BiLSTM + class-aware MI selection)
for s in S: run(dict(proto=O, model="cnn_bilstm", seed=s, save_model=(s == 0), **OV))
# 2. classical baselines on identical data
for s in S:
    run_tree(dict(proto=O, model="lgbm", seed=s, feats="sel", **OV)); run_tree(dict(proto=O, model="lgbm", seed=s, feats="sel"))
    run_tree(dict(proto=O, model="rf", seed=s, feats="sel", **OV))
run_tree(dict(proto=O, model="lgbm", seed=0, feats="all")); run_tree(dict(proto=O, model="rf", seed=0, feats="all"))
# 3. neural baselines (reviewers 5, 6: MLP, CNN-only, LSTM-only)
for m in ("mlp", "cnn", "lstm"):
    for s in S: run(dict(proto=O, model=m, seed=s, **OV))
# 4. attention ablation: identical data, seeds, hardware and training budget (reviewer 7)
for s in S: run(dict(proto=O, model="cnn_bilstm_attn", seed=s, save_model=(s == 0), **OV))
# 5. feature-selection ablation: original global MI selection
for s in S: run(dict(proto=O, model="cnn_bilstm", seed=s))
# 6. sequence-order ablation (reviewers 2, 5)
for order in ("random", "mi"):
    for s in S: run(dict(proto=O, model="cnn_bilstm", seed=s, order=order, **OV))
# 7. poisoning: targeted (hide R2L/U2R) and random label noise (reviewer 5)
for s in S:
    run(dict(proto=O, model="cnn_bilstm", seed=s, poison=("targeted", 0.5), **OV))
    run(dict(proto=O, model="cnn_bilstm", seed=s, poison=("random", 0.3), **OV))
# 8. Laplace input-noise augmentation (NOT a DP claim)
for e in (50, 10): run(dict(proto=O, model="cnn_bilstm", seed=0, laplace=e, **OV))
# 9. in-distribution protocol (deduplicated random split) for contrast with the official split
for s in S:
    run(dict(proto="random", model="cnn_bilstm", seed=s, **OV)); run_tree(dict(proto="random", model="lgbm", seed=s, feats="sel", **OV))
print("detector stage complete", flush=True)
