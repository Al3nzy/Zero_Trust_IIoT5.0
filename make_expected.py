#!/usr/bin/env python3
"""Writes expected_results.json (reference values for check_results.py) from the CURRENT results_v3/. Run only after a full run you consider correct."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v3"); L = lambda n: json.load(open(os.path.join(R, n)))
ED = "E-detector, round level, device reference"; EM = ED + ", mode-conditional"; ref = {}
c0 = L("core_s0.json")["test"]; ref["core_s0.json"] = {"test/acc": (c0["acc"], 0.01), "test/macro_f1": (c0["macro_f1"], 0.02)}
s = L("site.json")["summary"]; ref["site.json"] = {f"summary/{k}/{m}/mean": (s[k][m]["mean"], t) for k, m, t in (("argmax|excl", "macro_f1", 0.015), ("argmax|excl", "acc", 0.01), ("site|0.05", "fpr", 0.004), ("site|0.05", "macro_f1", 0.02),
                                                                      ("fused-site|0.05|mahalanobis", "macro_f1", 0.02), ("fused-site|0.05|mahalanobis", "fpr", 0.005), ("fused-site|0.05|mahalanobis", "det_unseen", 0.03))}
t = L("trust_main.json")["cells"]; ref["trust_main.json"] = {f"cells/hetero|iid/{ED}/benign/ever_quarantined": (t["hetero|iid"][ED]["benign"]["ever_quarantined"], 0.02), f"cells/hetero|iid/{ED}/compromised/detect": (t["hetero|iid"][ED]["compromised"]["detect"], 0.02),
                                                     f"cells/hetero|iid/{ED}/compromised/median_delay": (t["hetero|iid"][ED]["compromised"]["median_delay"], 1.0), f"cells/hetero|regime/{EM}/benign/ever_quarantined": (t["hetero|regime"][EM]["benign"]["ever_quarantined"], 0.02),
                                                     f"cells/hetero|iid/E-detector, round level, fleet reference/benign/ever_quarantined": (t["hetero|iid"]["E-detector, round level, fleet reference"]["benign"]["ever_quarantined"], 0.04)}
th = L("theory.json"); ref["theory.json"] = {"cusum/rows/0/within/250": (th["cusum"]["rows"][0]["within"]["250"], 0.03), "cusum/rows/0/per_round_tail": (th["cusum"]["rows"][0]["per_round_tail"], 0.003)}
if os.path.exists(os.path.join(R, "inflation_nsl.json")):
    i = L("inflation_nsl.json")["results"]; ref["inflation_nsl.json"] = {f"results/0/{v}/acc": (i["0"][v]["acc"], 0.01) for v in ("audited", "random_dedup", "random_dup_smote_first")}
json.dump(ref, open(os.path.join(os.path.dirname(R), "expected_results.json"), "w"), indent=1); print("expected_results.json written:", sum(len(v) for v in ref.values()), "reference values")
