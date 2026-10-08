"""Polynomial ridge regression using NumPy/SciPy; no black-box ML library."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '2')

from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import eigh


def powers(n_features, degree):
    """All nonconstant monomials of total degree <= degree, degree ordered."""
    rows = []
    for d in range(1, degree + 1):
        for indices in combinations_with_replacement(range(n_features), d):
            rows.append(np.bincount(indices, minlength=n_features))
    return np.asarray(rows, dtype=np.int16)


def design(x, exponents):
    result = np.ones((len(x), len(exponents)))
    for j in range(x.shape[1]):
        result *= x[:, j, None] ** exponents[None, :, j]
    return result


def ridge_path(a, y, b, alphas):
    """Return validation predictions for many ridge penalties from one eigensolve.

    Objective: sum((y - intercept - A @ coefficients)**2) + alpha * ||coefficients||**2.
    The intercept is unpenalized. Centering uses only the fitted training partition.
    Use the smaller primal/dual Gram matrix to control memory and computation.
    """
    mean = a.mean(axis=0)
    ac, bc = a - mean, b - mean
    yc = y - y.mean()
    if ac.shape[1] <= ac.shape[0]:
        values, vectors = eigh(ac.T @ ac, check_finite=False)
        projected_y = vectors.T @ (ac.T @ yc)
        projected_b = bc @ vectors
    else:
        values, vectors = eigh(ac @ ac.T, check_finite=False)
        projected_y = vectors.T @ yc
        projected_b = (bc @ ac.T) @ vectors
    factors = projected_y[:, None] / (np.maximum(values, 0)[:, None] + np.asarray(alphas)[None, :])
    return projected_b @ factors + y.mean()


def fit(a, y, alpha):
    from scipy.linalg import solve
    mean, ymean = a.mean(axis=0), float(y.mean())
    ac, yc = a - mean, y - ymean
    if ac.shape[1] <= ac.shape[0]:
        gram = ac.T @ ac
        gram.flat[::len(gram) + 1] += alpha
        coef = solve(gram, ac.T @ yc, assume_a='pos')
    else:
        gram = ac @ ac.T
        gram.flat[::len(gram) + 1] += alpha
        coef = ac.T @ solve(gram, yc, assume_a='pos')
    return coef, ymean - float(mean @ coef)


def metrics(y, pred):
    mse = float(np.mean((y - pred) ** 2))
    return {'mse': mse, 'r2': float(1 - mse / np.var(y))}


def group_partitions(x, seed=42):
    """Keep identical input vectors together so replicates cannot leak across splits."""
    _, groups = np.unique(x, axis=0, return_inverse=True)
    ids = np.random.default_rng(seed).permutation(np.unique(groups))
    hold_groups = ids[:round(0.2 * len(ids))]
    dev_groups = ids[round(0.2 * len(ids)):]
    hold = np.flatnonzero(np.isin(groups, hold_groups))
    dev = np.flatnonzero(~np.isin(groups, hold_groups))
    folds = []
    for valid_groups in np.array_split(dev_groups, 3):
        valid = np.flatnonzero(np.isin(groups, valid_groups))
        train = np.setdiff1d(dev, valid)
        folds.append((train, valid))
    return dev, hold, folds
