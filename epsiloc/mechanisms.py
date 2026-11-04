import numpy as np


def laplace(x, epsilon, lo, hi, decimal_places):
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if hi <= lo:
        raise ValueError("hi must be > lo.")

    scale = (hi - lo) / epsilon
    y = float(np.clip(x + np.random.laplace(0.0, scale), lo, hi))
    return float(np.clip(round(y, decimal_places), lo, hi))


def discrete_laplace(x, epsilon, lo, hi):
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if hi <= lo:
        raise ValueError("hi must be > lo.")

    # Difference of two i.i.d. geometric variables (support 0, 1, ...) is a discrete
    # Laplace with P(k) ∝ exp(-epsilon * |k| / (hi - lo)). expm1 keeps p accurate
    # when epsilon / (hi - lo) is tiny; Python ints avoid int64 overflow.
    p = -np.expm1(-epsilon / (hi - lo))
    noise = (int(np.random.geometric(p)) - 1) - (int(np.random.geometric(p)) - 1)
    return int(min(max(x + noise, lo), hi))


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
