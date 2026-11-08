import math

import numpy as np


def laplace(x, epsilon, lo, hi, decimal_places):
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if hi <= lo:
        raise ValueError("hi must be > lo.")

    if not lo <= x <= hi:
        raise ValueError(f"{x} is outside [{lo}, {hi}].")

    scale = (hi - lo) / epsilon
    if not math.isfinite(scale):
        raise ValueError("epsilon is too small for this range: the noise overflows.")
    y = float(np.clip(x + np.random.laplace(0.0, scale), lo, hi))
    return float(np.clip(round(y, decimal_places), lo, hi))


def k_randomized_response(x, epsilon, categories):
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if len(categories) < 2:
        raise ValueError("categories must contain at least 2 elements.")
    if x not in categories:
        raise ValueError(f"'{x}' is not in categories.")

    k = len(categories)
    idx = categories.index(x)
    # e^eps / (e^eps + k - 1), written so that it cannot overflow for large epsilon.
    p = 1.0 / (1.0 + (k - 1) * np.exp(-epsilon))

    if np.random.random() < p:
        return x

    j = np.random.randint(k - 1)
    return categories[j if j < idx else j + 1]
