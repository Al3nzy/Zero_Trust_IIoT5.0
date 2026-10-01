### tab:protocol: Official-split data protocol after the audit

| Class | Fit (train) | Validation | Test |
|---|---|---|---|
| Normal | 57238 | 10101 | 9582 |
| DoS | 39035 | 6889 | 7060 |
| Probe | 9900 | 1747 | 2288 |
| R2L | 846 | 149 | 2836 |
| U2R | 44 | 8 | 67 |
| Total | 107063 | 18894 | 21833 |

Test records dropped because an identical feature vector exists in training: 664 of 22544. Exact overlap remaining between fit and test after the audit: 0. Balanced training size per class: 300.

### tab:main: Official overlap-free test set, mean $\pm$ std over training seeds

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| CNN-BiLSTM + class-aware MI (proposed) | 0.347 | 0.272 | 0.207 | 0.504 | 0.164 | 0.597 | 1 |
| CNN-BiLSTM + dual attention | 0.509 | 0.331 | 0.300 | 0.114 | 0.077 | 0.597 | 1 |
| CNN-BiLSTM + global MI (original selection) | 0.592 | 0.344 | 0.446 | 0.126 | 0.000 | 0.000 | 1 |
| LightGBM (class-aware MI, 25 feat.) | 0.787 | 0.619 | 0.687 | 0.044 | 0.228 | 0.612 | 1 |
| LightGBM (global MI, 25 feat.) | 0.755 | 0.533 | 0.638 | 0.078 | 0.079 | 0.448 | 1 |
| LightGBM (all features) | 0.776 | 0.595 | 0.668 | 0.078 | 0.231 | 0.597 | 1 |
| Random forest (class-aware MI) | 0.775 | 0.617 | 0.670 | 0.035 | 0.152 | 0.761 | 1 |
| MLP | 0.701 | 0.541 | 0.573 | 0.173 | 0.301 | 0.627 | 1 |
| CNN only | 0.089 | 0.048 | 0.001 | 1.000 | 0.667 | 0.731 | 1 |
| LSTM only | 0.520 | 0.313 | 0.287 | 0.129 | 0.002 | 0.418 | 1 |

### tab:ablation: Sequence-order and noise-augmentation ablations

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| canonical (family-adjacent) order | 0.347 | 0.272 | 0.207 | 0.504 | 0.164 | 0.597 | 1 |
| random feature order | 0.538 | 0.327 | 0.405 | 0.188 | 0.000 | 0.493 | 1 |
| descending-MI order | 0.433 | 0.324 | 0.276 | 0.544 | 0.033 | 0.701 | 1 |
| Laplace input noise, scale 1/50 (augmentation) | 0.297 | 0.185 | 0.209 | 1.000 | 0.000 | 0.597 | 1 |
| Laplace input noise, scale 1/10 (augmentation) | 0.306 | 0.221 | 0.224 | 0.968 | 0.059 | 0.597 | 1 |

### tab:poison: Label-poisoning experiments

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| clean training labels | 0.347 | 0.272 | 0.207 | 0.504 | 0.164 | 0.597 | 1 |
| targeted: 50\% of R2L/U2R relabelled Normal | 0.671 | 0.462 | 0.523 | 0.122 | 0.000 | 0.478 | 1 |
| random label noise on 30\% of samples | 0.301 | 0.181 | 0.189 | 1.000 | 0.000 | 0.597 | 1 |

### tab:indist: Official versus deduplicated random split

| Model | Acc. | Macro-F1 | MCC | Normal FPR | R2L rec. | U2R rec. | Seeds |
|---|---|---|---|---|---|---|---|
| official split (proposed) | 0.347 | 0.272 | 0.207 | 0.504 | 0.164 | 0.597 | 1 |
| deduplicated random split (proposed) | 0.827 | 0.488 | 0.734 | 0.176 | 0.000 | 0.625 | 1 |
| deduplicated random split (LightGBM) | 0.949 | 0.733 | 0.918 | 0.080 | 0.946 | 0.958 | 1 |

### tab:novel: Attack recall (\%) on sub-types seen in training versus test-only sub-types

| Model | n seen | Exact | Detected | n novel | Exact | Detected |
|---|---|---|---|---|---|---|
| proposed CNN-BiLSTM | 8556 | 23.7 | 85.6 | 3695 | 21.6 | 82.4 |
| LightGBM (class-aware MI) | 8556 | 79.8 | 83.8 | 3695 | 32.3 | 41.2 |
| LightGBM (global MI) | 8556 | 75.2 | 79.0 | 3695 | 33.0 | 43.6 |
| CNN-BiLSTM + attention | 8556 | 25.4 | 45.9 | 3695 | 12.2 | 51.5 |

Exact = correct 5-class label; Detected = any non-Normal label. Novel sub-types: apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm

### tab:trust: Trust-update rules on replayed real posteriors

| Fleet | Rule | False quar. (\%) | Detect (\%) | Delay (rounds) | Final Healthy (\%) | Released (\%) | Stealth 30\% (\%) |
|---|---|---|---|---|---|---|---|
| homog | confidence (original Eq.4) | 100.0 | 100 | 1 | 0 | 0 | 100 |
| homog | beta reputation | 100.0 | 100 | 1 | 0 | 0 | 100 |
| homog | CUSUM fleet-baseline | 0.0 | 32 | 6 | 68 | 100 | 0 |
| homog | CUSUM device-baseline | 0.0 | 32 | 6 | 68 | 100 | 0 |
| hetero | confidence (original Eq.4) | 100.0 | 100 | 1 | 0 | 0 | 100 |
| hetero | beta reputation | 100.0 | 100 | 1 | 0 | 0 | 100 |
| hetero | CUSUM fleet-baseline | 0.0 | 12 | 6 | 88 | 100 | 0 |
| hetero | CUSUM device-baseline | 0.0 | 12 | 6 | 88 | 100 | 0 |

h=0.576, k=0.05, m=20 flows/round, 40 devices per cell. Replay simulation, not a deployment.

### tab:contam: Compromise during baseline commissioning

| Contaminated fraction | Baseline | Detected (\%) | False quar. (\%) |
|---|---|---|---|
| 0.00 | fleet | -- | 0.0 |
| 0.00 | device | -- | 0.0 |
| 0.10 | fleet | 0 | 0.0 |
| 0.10 | device | 0 | 0.0 |
| 0.30 | fleet | 17 | 0.0 |
| 0.30 | device | 0 | 0.0 |
| 0.45 | fleet | 11 | 0.0 |
| 0.45 | device | 0 | 0.0 |

### tab:delay: Detection-delay bound versus measurement

| Attack | $\mu_a$ | Drift $d$ | Bound $(h+1)/d$ | Measured mean | Detected (\%) |
|---|---|---|---|---|---|
| DoS | 0.579 | -0.022 | -- | -- | 0.00 |
| Probe | 0.592 | -0.010 | -- | -- | 0.00 |
| R2L | 0.575 | -0.026 | -- | -- | 0.00 |
| U2R | 0.708 | 0.107 | 14.75 | 5.87 | 100.00 |

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
| edit\_rehash\_with\_key | 0.0 | 0.0 | 95.0 |
| delete\_relink\_no\_key | 0.0 | 100.0 | 100.0 |
| truncate\_with\_key | 0.0 | 0.0 | 85.0 |
| rewrite\_own\_key | 0.0 | 100.0 | 100.0 |
| untampered chain accepted | yes | yes | yes |

append (us): append_us_hash_only=8.7, append_us_signed=50.0 | verify (s): verify_s_1000=0.10, verify_s_2000=0.20

### tab:dp: DP-SGD (RDP accountant) privacy-utility trade-off

| $\varepsilon$ | $\sigma$ | Acc. | Macro-F1 | MCC | U2R rec. | MIA AUC |
|---|---|---|---|---|---|---|
| non-private | -- | 0.751 | 0.533 | 0.626 | 0.030 | 0.496 |
| 8.00 | 0.49 | 0.659 | 0.304 | 0.485 | 0.000 | 0.495 |
| 4.00 | 0.59 | 0.657 | 0.298 | 0.482 | 0.000 | 0.495 |
| 1.00 | 0.95 | 0.652 | 0.299 | 0.486 | 0.000 | 0.495 |

N=6000 records, delta=1/N, 1 epochs, batch 64, clip 1.0. MIA AUC = loss-threshold membership inference (0.5 = chance).
