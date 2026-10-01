import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             matthews_corrcoef, confusion_matrix)
from .data import CLASSES


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def metrics(y, prob):
    pred = prob.argmax(1)
    P, R, F, S = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(5))
    out = dict(acc=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
               mcc=float(matthews_corrcoef(y, pred)), n=int(len(y)),
               per_class={c: dict(p=float(P[i]), r=float(R[i]), f1=float(F[i]), n=int(S[i]),
                                  r_ci=wilson(int(cm[i, i]), int(S[i]))) for i, c in enumerate(CLASSES)},
               cm=cm.tolist())
    att = y != 0
    out["attack_detection_rate"] = float((pred[att] != 0).mean())      # any non-Normal label on an attack
    out["normal_fpr"] = float((pred[~att] != 0).mean())
    return out
