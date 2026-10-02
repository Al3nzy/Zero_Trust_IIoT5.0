### tab:protocol: Official-split data protocol after the audit

| Class | Fit (train) | Validation | Test |
|---|---|---|---|
| Normal | 57238 | 10101 | 9582 |
| DoS | 39035 | 6889 | 7060 |
| Probe | 9900 | 1747 | 2288 |
| R2L | 846 | 149 | 2836 |
| U2R | 44 | 8 | 67 |
| Total | 107063 | 18894 | 21833 |

Test records dropped because an identical feature vector exists in training: 664 of 22544. Exact overlap remaining between fit and test after the audit: 0. Balanced training size per class: 4000.

### tab:main: Official overlap-free test set, mean $\pm$ std over training seeds

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| CNN-BiLSTM + class-aware MI (proposed) | 0.745$\pm$0.010 | 0.554$\pm$0.018 | 0.620$\pm$0.018 | 0.081$\pm$0.029 | 0.183$\pm$0.041 | 0.527$\pm$0.031 | 3 |
| CNN-BiLSTM + dual attention | 0.739$\pm$0.021 | 0.542$\pm$0.027 | 0.610$\pm$0.035 | 0.075$\pm$0.026 | 0.131$\pm$0.061 | 0.537$\pm$0.117 | 3 |
| CNN-BiLSTM + global MI (original selection) | 0.728$\pm$0.006 | 0.514$\pm$0.002 | 0.597$\pm$0.009 | 0.104$\pm$0.001 | 0.165$\pm$0.007 | 0.418$\pm$0.039 | 3 |
| LightGBM (class-aware MI, 25 feat.) | 0.776$\pm$0.019 | 0.634$\pm$0.024 | 0.675$\pm$0.026 | 0.029$\pm$0.002 | 0.191$\pm$0.004 | 0.408$\pm$0.075 | 3 |
| LightGBM (global MI, 25 feat.) | 0.775$\pm$0.011 | 0.562$\pm$0.009 | 0.672$\pm$0.017 | 0.030$\pm$0.001 | 0.076$\pm$0.012 | 0.264$\pm$0.023 | 3 |
| LightGBM (all features) | 0.775 | 0.624 | 0.674 | 0.028 | 0.222 | 0.269 | 1 |
| Random forest (class-aware MI) | 0.752$\pm$0.002 | 0.566$\pm$0.018 | 0.639$\pm$0.003 | 0.029$\pm$0.001 | 0.077$\pm$0.007 | 0.294$\pm$0.075 | 3 |
| MLP | 0.752$\pm$0.014 | 0.560$\pm$0.028 | 0.633$\pm$0.023 | 0.077$\pm$0.028 | 0.148$\pm$0.052 | 0.537$\pm$0.026 | 3 |
| CNN only | 0.735$\pm$0.016 | 0.549$\pm$0.041 | 0.604$\pm$0.025 | 0.102$\pm$0.003 | 0.174$\pm$0.099 | 0.642$\pm$0.015 | 3 |
| LSTM only | 0.731$\pm$0.018 | 0.522$\pm$0.055 | 0.601$\pm$0.022 | 0.091$\pm$0.024 | 0.099$\pm$0.074 | 0.582$\pm$0.113 | 3 |

### tab:ablation: Sequence-order and noise-augmentation ablations

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| canonical (family-adjacent) order | 0.745$\pm$0.010 | 0.554$\pm$0.018 | 0.620$\pm$0.018 | 0.081$\pm$0.029 | 0.183$\pm$0.041 | 0.527$\pm$0.031 | 3 |
| random feature order | 0.748$\pm$0.023 | 0.557$\pm$0.036 | 0.627$\pm$0.038 | 0.073$\pm$0.024 | 0.154$\pm$0.049 | 0.423$\pm$0.009 | 3 |
| descending-MI order | 0.751$\pm$0.017 | 0.561$\pm$0.029 | 0.631$\pm$0.029 | 0.076$\pm$0.027 | 0.162$\pm$0.090 | 0.448$\pm$0.052 | 3 |
| Laplace input noise, scale 1/50 (augmentation) | 0.759 | 0.575 | 0.646 | 0.046 | 0.197 | 0.522 | 1 |
| Laplace input noise, scale 1/10 (augmentation) | 0.753 | 0.554 | 0.636 | 0.043 | 0.145 | 0.493 | 1 |

### tab:poison: Label-poisoning experiments

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| clean training labels | 0.745$\pm$0.010 | 0.554$\pm$0.018 | 0.620$\pm$0.018 | 0.081$\pm$0.029 | 0.183$\pm$0.041 | 0.527$\pm$0.031 | 3 |
| targeted: 50\% of R2L/U2R relabelled Normal | 0.730$\pm$0.012 | 0.523$\pm$0.040 | 0.598$\pm$0.021 | 0.076$\pm$0.029 | 0.054$\pm$0.047 | 0.204$\pm$0.202 | 3 |
| random label noise on 30\% of samples | 0.752$\pm$0.029 | 0.555$\pm$0.037 | 0.635$\pm$0.044 | 0.086$\pm$0.026 | 0.204$\pm$0.055 | 0.647$\pm$0.052 | 3 |

### tab:indist: Official versus deduplicated random split

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| official split (proposed) | 0.745$\pm$0.010 | 0.554$\pm$0.018 | 0.620$\pm$0.018 | 0.081$\pm$0.029 | 0.183$\pm$0.041 | 0.527$\pm$0.031 | 3 |
| deduplicated random split (proposed) | 0.930$\pm$0.004 | 0.708$\pm$0.010 | 0.891$\pm$0.006 | 0.116$\pm$0.008 | 0.950$\pm$0.005 | 0.806$\pm$0.127 | 3 |
| deduplicated random split (LightGBM) | 0.988$\pm$0.001 | 0.878$\pm$0.013 | 0.981$\pm$0.002 | 0.019$\pm$0.002 | 0.983$\pm$0.002 | 0.875$\pm$0.042 | 3 |

### tab:novel: Attack recall (\%) on sub-types seen in training versus test-only sub-types

| Model | n seen | Exact | Detected | n novel | Exact | Detected |
|---|---|---|---|---|---|---|
| proposed CNN-BiLSTM | 8556 | 78.3 | 81.6 | 3695 | 18.2 | 29.0 |
| LightGBM (class-aware MI) | 8556 | 79.7 | 79.9 | 3695 | 10.9 | 16.0 |
| LightGBM (global MI) | 8556 | 76.0 | 77.4 | 3695 | 23.0 | 31.3 |
| CNN-BiLSTM + attention | 8556 | 78.2 | 83.6 | 3695 | 22.1 | 35.5 |

Exact = correct 5-class label; Detected = any non-Normal label. Novel sub-types: apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm

### tab:novelty_fpr0.02: Seen vs novel attack detection, novelty threshold at 2% validation FPR

| Detector | Macro-F1 | Benign FPR | Det. seen | Det. novel | R2L rec. | U2R rec. |
|---|---|---|---|---|---|---|
| Normal-profile detector alone | -- | 0.019$\pm$0.001 | 0.650$\pm$0.015 | 0.519$\pm$0.015 | -- | -- |
| proposed CNN-BiLSTM | 0.554$\pm$0.018 | 0.081$\pm$0.029 | 0.831$\pm$0.018 | 0.411$\pm$0.107 | 0.183$\pm$0.041 | 0.527$\pm$0.031 |
| proposed CNN-BiLSTM + novelty | 0.576$\pm$0.022 | 0.084$\pm$0.029 | 0.845$\pm$0.014 | 0.599$\pm$0.025 | 0.183$\pm$0.041 | 0.532$\pm$0.034 |
| LightGBM (class-aware MI) | 0.634$\pm$0.024 | 0.029$\pm$0.002 | 0.814$\pm$0.015 | 0.297$\pm$0.135 | 0.191$\pm$0.004 | 0.408$\pm$0.075 |
| LightGBM (class-aware MI) + novelty | 0.667$\pm$0.029 | 0.032$\pm$0.002 | 0.818$\pm$0.012 | 0.583$\pm$0.062 | 0.195$\pm$0.005 | 0.458$\pm$0.099 |
| LightGBM (global MI) | 0.562$\pm$0.009 | 0.030$\pm$0.001 | 0.781$\pm$0.007 | 0.413$\pm$0.087 | 0.076$\pm$0.012 | 0.264$\pm$0.023 |
| LightGBM (global MI) + novelty | 0.590$\pm$0.002 | 0.033$\pm$0.001 | 0.785$\pm$0.007 | 0.648$\pm$0.018 | 0.080$\pm$0.012 | 0.299$\pm$0.000 |
| MLP | 0.560$\pm$0.028 | 0.077$\pm$0.028 | 0.828$\pm$0.009 | 0.374$\pm$0.045 | 0.148$\pm$0.052 | 0.537$\pm$0.026 |
| MLP + novelty | 0.576$\pm$0.028 | 0.079$\pm$0.028 | 0.829$\pm$0.008 | 0.568$\pm$0.011 | 0.150$\pm$0.053 | 0.537$\pm$0.026 |

3695 of 12251 test attacks belong to sub-types absent from training. Threshold set on validation Normal flows only.

### tab:novelty_fpr0.05: Seen vs novel attack detection, novelty threshold at 5% validation FPR

| Detector | Macro-F1 | Benign FPR | Det. seen | Det. novel | R2L rec. | U2R rec. |
|---|---|---|---|---|---|---|
| Normal-profile detector alone | -- | 0.038$\pm$0.018 | 0.686$\pm$0.006 | 0.635$\pm$0.017 | -- | -- |
| proposed CNN-BiLSTM | 0.554$\pm$0.018 | 0.081$\pm$0.029 | 0.831$\pm$0.018 | 0.411$\pm$0.107 | 0.183$\pm$0.041 | 0.527$\pm$0.031 |
| proposed CNN-BiLSTM + novelty | 0.576$\pm$0.015 | 0.097$\pm$0.008 | 0.848$\pm$0.013 | 0.671$\pm$0.030 | 0.185$\pm$0.041 | 0.532$\pm$0.034 |
| LightGBM (class-aware MI) | 0.634$\pm$0.024 | 0.029$\pm$0.002 | 0.814$\pm$0.015 | 0.297$\pm$0.135 | 0.191$\pm$0.004 | 0.408$\pm$0.075 |
| LightGBM (class-aware MI) + novelty | 0.654$\pm$0.023 | 0.047$\pm$0.018 | 0.822$\pm$0.011 | 0.657$\pm$0.033 | 0.201$\pm$0.005 | 0.473$\pm$0.102 |
| LightGBM (global MI) | 0.562$\pm$0.009 | 0.030$\pm$0.001 | 0.781$\pm$0.007 | 0.413$\pm$0.087 | 0.076$\pm$0.012 | 0.264$\pm$0.023 |
| LightGBM (global MI) + novelty | 0.586$\pm$0.005 | 0.048$\pm$0.018 | 0.792$\pm$0.011 | 0.692$\pm$0.021 | 0.087$\pm$0.009 | 0.318$\pm$0.009 |
| MLP | 0.560$\pm$0.028 | 0.077$\pm$0.028 | 0.828$\pm$0.009 | 0.374$\pm$0.045 | 0.148$\pm$0.052 | 0.537$\pm$0.026 |
| MLP + novelty | 0.580$\pm$0.025 | 0.092$\pm$0.008 | 0.834$\pm$0.007 | 0.653$\pm$0.020 | 0.162$\pm$0.062 | 0.542$\pm$0.031 |

3695 of 12251 test attacks belong to sub-types absent from training. Threshold set on validation Normal flows only.

### tab:trust: Trust-update rules on replayed real posteriors

| Fleet | Rule | False quar. (\%) | Detect (\%) | Delay (rounds) | Released (\%) | Stealth 30\% (\%) | Seen sub-type (\%) | Novel sub-type (\%) |
|---|---|---|---|---|---|---|---|---|
| homog | confidence (original Eq.4) | 0.0 | 0 | -- | 100 | 0 | 0 | 0 |
| homog | beta reputation | 0.0 | 72 | 8 | 100 | 0 | 100 | 0 |
| homog | CUSUM fleet-baseline (classifier) | 0.0 | 100 | 2 | 100 | 97 | 100 | 100 |
| homog | CUSUM fleet-baseline (+novelty) | 0.0 | 100 | 2 | 100 | 97 | 100 | 100 |
| homog | CUSUM device-baseline (+novelty) | 0.0 | 100 | 2 | 100 | 96 | 100 | 100 |
| hetero | confidence (original Eq.4) | 0.0 | 0 | -- | 100 | 0 | 0 | 0 |
| hetero | beta reputation | 0.0 | 76 | 8 | 100 | 0 | 100 | 0 |
| hetero | CUSUM fleet-baseline (classifier) | 18.5 | 100 | 2 | 74 | 82 | 100 | 100 |
| hetero | CUSUM fleet-baseline (+novelty) | 4.5 | 100 | 2 | 68 | 88 | 100 | 100 |
| hetero | CUSUM device-baseline (+novelty) | 0.0 | 100 | 2 | 96 | 88 | 100 | 100 |

h=0.576, k=0.05, m=20 flows/round, 200 devices per cell. Replay simulation, not a deployment.

### tab:contam: Compromise during baseline commissioning

| Contaminated fraction | Baseline | Detected (\%) | False quar. (\%) |
|---|---|---|---|
| 0.00 | fleet | -- | 0.0 |
| 0.00 | device | -- | 0.0 |
| 0.10 | fleet | 100 | 0.0 |
| 0.10 | device | 0 | 0.0 |
| 0.30 | fleet | 100 | 0.0 |
| 0.30 | device | 2 | 0.0 |
| 0.45 | fleet | 100 | 0.0 |
| 0.45 | device | 2 | 0.0 |

### tab:delay: Detection-delay bound versus measurement

| Attack | $\mu_a$ | Drift $d$ | Bound $(h+1)/d$ | Measured mean | Detected (\%) |
|---|---|---|---|---|---|
| DoS | 0.511 | 0.455 | 3.46 | 1.98 | 100.00 |
| Probe | 0.481 | 0.425 | 3.71 | 2.00 | 100.00 |
| R2L | 0.257 | 0.201 | 7.84 | 3.40 | 100.00 |
| U2R | 0.690 | 0.633 | 2.49 | 1.29 | 100.00 |

### tab:fqbound: False-quarantine probability versus certified level

| Flow repetition | $m_{eff}$ | Certified $\alpha$ | Empirical rate |
|---|---|---|---|
| 1 | 20 | 0.50 | 0.002 |
| 1 | 20 | 0.20 | 0.000 |
| 1 | 20 | 0.10 | 0.000 |
| 1 | 20 | 0.01 | 0.000 |
| 2 | 10 | 0.50 | 0.005 |
| 2 | 10 | 0.20 | 0.000 |
| 2 | 10 | 0.10 | 0.000 |
| 2 | 10 | 0.01 | 0.000 |
| 4 | 5 | 0.50 | 0.017 |
| 4 | 5 | 0.20 | 0.000 |
| 4 | 5 | 0.10 | 0.000 |
| 4 | 5 | 0.01 | 0.000 |

### tab:ledger: Tamper detection rate (\%) of ledger variants

| Attack | hash chain only | + signatures | + signatures + anchors |
|---|---|---|---|
| edit\_no\_rehash | 100.0 | 100.0 | 100.0 |
| edit\_rehash\_no\_key | 0.0 | 100.0 | 100.0 |
| edit\_rehash\_with\_key | 0.0 | 0.0 | 93.5 |
| delete\_relink\_no\_key | 0.0 | 100.0 | 100.0 |
| truncate\_with\_key | 0.0 | 0.0 | 50.0 |
| rewrite\_own\_key | 0.0 | 100.0 | 100.0 |
| untampered chain accepted | yes | yes | yes |

append (us): append_us_hash_only=10.2, append_us_signed=51.7 | verify (s): verify_s_1000=0.10, verify_s_10000=1.11, verify_s_50000=5.42

### tab:dp: DP-SGD (RDP accountant) privacy-utility trade-off

| $\varepsilon$ | $\sigma$ | Acc. | Macro-F1 | MCC | U2R rec. | MIA AUC |
|---|---|---|---|---|---|---|
| non-private | -- | 0.775 | 0.624 | 0.670 | 0.343 | 0.492 |
| 8.00 | 0.49 | 0.734 | 0.453 | 0.604 | 0.000 | 0.494 |
| 4.00 | 0.59 | 0.724 | 0.449 | 0.590 | 0.000 | 0.496 |
| 1.00 | 0.96 | 0.718 | 0.448 | 0.587 | 0.000 | 0.496 |

N=107063 records, delta=1/N, 3 epochs, batch 256, clip 1.0. MIA AUC = loss-threshold membership inference (0.5 = chance).

### tab:latency: Per-round latency on identical hardware (1 thread)

| Model | Params | MB | 1 flow (ms) | 20 flows (ms) | p99 (ms) | flows/s | Round total (ms) |
|---|---|---|---|---|---|---|---|
| plain | 462,725 | 5.6 | 2.14 | 6.97 | 10.04 | 2167 | 9.39 |
| attention | 471,205 | 5.7 | 2.23 | 6.91 | 7.67 | 2269 | 9.33 |

Stages (mean ms): preprocess(20)=2.37, trust update=0.004, signed ledger append=0.046; SHAP per flow=3.50 s; peak RSS=630 MB; hardware: Windows-11-10.0.26220-SP0, 8 cores.
