"""Shared constructors so that every stage trains the primary classifier identically (deterministic, one thread)."""
from lightgbm import LGBMClassifier


def lgbm(seed, n_estimators=300, learning_rate=0.05, num_leaves=31):
    """Primary classifier. `deterministic` + `force_row_wise` + one thread make a run bit-reproducible across machines with different core counts."""
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, n_jobs=1, random_state=seed, verbose=-1,
                          deterministic=True, force_row_wise=True)
