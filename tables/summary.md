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
| CNN-BiLSTM + class-aware MI (proposed) | 0.539$\pm$0.174 | 0.271$\pm$0.257 | 0.205$\pm$0.351 | 0.033$\pm$0.055 | 0.060$\pm$0.103 | 0.149$\pm$0.259 | 3 |
| CNN-BiLSTM + dual attention | 0.631$\pm$0.168 | 0.408$\pm$0.250 | 0.393$\pm$0.341 | 0.060$\pm$0.051 | 0.098$\pm$0.122 | 0.299$\pm$0.259 | 3 |
| CNN-BiLSTM + global MI (original selection) | 0.745$\pm$0.010 | 0.529$\pm$0.018 | 0.620$\pm$0.013 | 0.105$\pm$0.003 | 0.139$\pm$0.061 | 0.308$\pm$0.127 | 3 |
| LightGBM (class-aware MI, 25 feat.) | 0.775$\pm$0.017 | 0.621$\pm$0.020 | 0.673$\pm$0.024 | 0.029$\pm$0.001 | 0.189$\pm$0.010 | 0.348$\pm$0.075 | 3 |
| LightGBM (global MI, 25 feat.) | 0.773$\pm$0.010 | 0.553$\pm$0.009 | 0.668$\pm$0.016 | 0.036$\pm$0.010 | 0.075$\pm$0.008 | 0.229$\pm$0.038 | 3 |
| LightGBM (all features) | 0.773 | 0.636 | 0.671 | 0.028 | 0.229 | 0.343 | 1 |
| Random forest (class-aware MI) | 0.751$\pm$0.003 | 0.566$\pm$0.015 | 0.638$\pm$0.004 | 0.029$\pm$0.001 | 0.075$\pm$0.006 | 0.303$\pm$0.074 | 3 |
| MLP | 0.752$\pm$0.016 | 0.555$\pm$0.025 | 0.633$\pm$0.027 | 0.077$\pm$0.028 | 0.123$\pm$0.050 | 0.582$\pm$0.026 | 3 |
| CNN only | 0.512$\pm$0.352 | 0.365$\pm$0.284 | 0.383$\pm$0.332 | 0.397$\pm$0.522 | 0.054$\pm$0.074 | 0.338$\pm$0.302 | 3 |
| LSTM only | 0.740$\pm$0.018 | 0.535$\pm$0.021 | 0.616$\pm$0.029 | 0.089$\pm$0.015 | 0.164$\pm$0.048 | 0.498$\pm$0.135 | 3 |

### tab:ablation: Sequence-order and noise-augmentation ablations

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| canonical (family-adjacent) order | 0.539$\pm$0.174 | 0.271$\pm$0.257 | 0.205$\pm$0.351 | 0.033$\pm$0.055 | 0.060$\pm$0.103 | 0.149$\pm$0.259 | 3 |
| random feature order | 0.744$\pm$0.019 | 0.548$\pm$0.032 | 0.618$\pm$0.027 | 0.061$\pm$0.023 | 0.085$\pm$0.069 | 0.393$\pm$0.164 | 3 |
| descending-MI order | 0.552$\pm$0.195 | 0.343$\pm$0.215 | 0.243$\pm$0.371 | 0.013$\pm$0.019 | 0.077$\pm$0.133 | 0.413$\pm$0.009 | 3 |
| Laplace input noise, scale 1/50 (augmentation) | 0.660 | 0.340 | 0.474 | 0.110 | 0.000 | 0.134 | 1 |
| Laplace input noise, scale 1/10 (augmentation) | 0.743 | 0.556 | 0.616 | 0.031 | 0.155 | 0.418 | 1 |

### tab:poison: Label-poisoning experiments

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| clean training labels | 0.539$\pm$0.174 | 0.271$\pm$0.257 | 0.205$\pm$0.351 | 0.033$\pm$0.055 | 0.060$\pm$0.103 | 0.149$\pm$0.259 | 3 |
| targeted: 50\% of R2L/U2R relabelled Normal | 0.654$\pm$0.136 | 0.393$\pm$0.177 | 0.477$\pm$0.215 | 0.041$\pm$0.045 | 0.030$\pm$0.052 | 0.050$\pm$0.086 | 3 |
| random label noise on 30\% of samples | 0.727$\pm$0.020 | 0.523$\pm$0.014 | 0.593$\pm$0.032 | 0.079$\pm$0.029 | 0.109$\pm$0.047 | 0.552$\pm$0.045 | 3 |

### tab:indist: Official versus deduplicated random split

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| official split (proposed) | 0.539$\pm$0.174 | 0.271$\pm$0.257 | 0.205$\pm$0.351 | 0.033$\pm$0.055 | 0.060$\pm$0.103 | 0.149$\pm$0.259 | 3 |
| deduplicated random split (proposed) | 0.793$\pm$0.238 | 0.523$\pm$0.321 | 0.620$\pm$0.469 | 0.078$\pm$0.063 | 0.661$\pm$0.505 | 0.583$\pm$0.505 | 3 |
| deduplicated random split (LightGBM) | 0.989$\pm$0.001 | 0.876$\pm$0.017 | 0.981$\pm$0.002 | 0.018$\pm$0.002 | 0.983$\pm$0.002 | 0.875$\pm$0.042 | 3 |

### tab:novel: Attack recall (\%) on sub-types seen in training versus test-only sub-types

| Model | n seen | Exact | Detected | n novel | Exact | Detected |
|---|---|---|---|---|---|---|
| proposed CNN-BiLSTM | 8556 | 0.0 | 0.2 | 3695 | 0.2 | 0.4 |
| LightGBM (class-aware MI) | 8556 | 79.4 | 79.6 | 3695 | 11.4 | 16.5 |
| LightGBM (global MI) | 8556 | 75.3 | 76.9 | 3695 | 25.0 | 35.6 |
| CNN-BiLSTM + attention | 8556 | 0.0 | 0.2 | 3695 | 0.1 | 0.4 |

Exact = correct 5-class label; Detected = any non-Normal label. Novel sub-types: apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm

### tab:trust: Trust-update rules on replayed real posteriors

| Fleet | Rule | False quar. (\%) | Detect (\%) | Delay (rounds) | Final Healthy (\%) | Released (\%) | Stealth 30\% (\%) |
|---|---|---|---|---|---|---|---|
| homog | confidence (original Eq.4) | 0.0 | 0 | -- | 100 | 100 | 0 |
| homog | beta reputation | 0.0 | 0 | -- | 72 | 100 | 0 |
| homog | CUSUM fleet-baseline | 0.0 | 28 | 4 | 72 | 100 | 27 |
| homog | CUSUM device-baseline | 0.0 | 28 | 4 | 72 | 100 | 28 |
| hetero | confidence (original Eq.4) | 0.0 | 0 | -- | 100 | 100 | 0 |
| hetero | beta reputation | 0.0 | 0 | -- | 76 | 100 | 0 |
| hetero | CUSUM fleet-baseline | 0.0 | 24 | 4 | 76 | 100 | 19 |
| hetero | CUSUM device-baseline | 0.0 | 24 | 4 | 76 | 100 | 22 |

h=0.576, k=0.05, m=20 flows/round, 200 devices per cell. Replay simulation, not a deployment.

### tab:contam: Compromise during baseline commissioning

| Contaminated fraction | Baseline | Detected (\%) | False quar. (\%) |
|---|---|---|---|
| 0.00 | fleet | -- | 0.0 |
| 0.00 | device | -- | 0.0 |
| 0.10 | fleet | 15 | 0.0 |
| 0.10 | device | 0 | 0.0 |
| 0.30 | fleet | 17 | 0.0 |
| 0.30 | device | 0 | 0.0 |
| 0.45 | fleet | 17 | 0.0 |
| 0.45 | device | 0 | 0.0 |

### tab:delay: Detection-delay bound versus measurement

| Attack | $\mu_a$ | Drift $d$ | Bound $(h+1)/d$ | Measured mean | Detected (\%) |
|---|---|---|---|---|---|
| DoS | 0.052 | -0.019 | -- | -- | 0.00 |
| Probe | 0.049 | -0.023 | -- | -- | 0.00 |
| R2L | 0.034 | -0.037 | -- | -- | 0.00 |
| U2R | 0.261 | 0.190 | 8.30 | 3.57 | 100.00 |

### tab:fqbound: False-quarantine probability versus certified level

| Flow repetition | $m_{eff}$ | Certified $\alpha$ | Empirical rate |
|---|---|---|---|
| 1 | 20 | 0.50 | 0.000 |
| 1 | 20 | 0.20 | 0.000 |
| 1 | 20 | 0.10 | 0.000 |
| 1 | 20 | 0.01 | 0.000 |
| 2 | 10 | 0.50 | 0.000 |
| 2 | 10 | 0.20 | 0.000 |
| 2 | 10 | 0.10 | 0.000 |
| 2 | 10 | 0.01 | 0.000 |
| 4 | 5 | 0.50 | 0.000 |
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

append (us): append_us_hash_only=7.9, append_us_signed=46.8 | verify (s): verify_s_1000=0.10, verify_s_10000=0.97, verify_s_50000=4.88

### tab:dp: DP-SGD (RDP accountant) privacy-utility trade-off

| $\varepsilon$ | $\sigma$ | Acc. | Macro-F1 | MCC | U2R rec. | MIA AUC |
|---|---|---|---|---|---|---|
| non-private | -- | 0.774 | 0.661 | 0.667 | 0.418 | 0.491 |
| 8.00 | 0.49 | 0.733 | 0.453 | 0.603 | 0.000 | 0.496 |
| 4.00 | 0.59 | 0.725 | 0.449 | 0.591 | 0.000 | 0.495 |
| 1.00 | 0.96 | 0.723 | 0.449 | 0.592 | 0.000 | 0.499 |

N=107063 records, delta=1/N, 3 epochs, batch 256, clip 1.0. MIA AUC = loss-threshold membership inference (0.5 = chance).
