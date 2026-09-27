"""
Distances between time series, written from scratch for reference.

Use `tslearn.metrics` for real workloads: it is compiled and supports
constraints (Sakoe-Chiba, Itakura). These versions exist to read and test.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def euclidean_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Euclidean (L2) distance."""
    return float(np.sqrt(np.sum((np.asarray(x) - np.asarray(y)) ** 2)))


def squared_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Squared Euclidean distance, the pointwise cost tslearn's DTW uses."""
    return float(np.sum((np.asarray(x) - np.asarray(y)) ** 2))


def manhattan_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Manhattan (L1) distance."""
    return float(np.sum(np.abs(np.asarray(x) - np.asarray(y))))


def cosine_distance(x: ArrayLike, y: ArrayLike) -> float:
    """1 minus the cosine similarity. Ignores amplitude."""
    x, y = np.asarray(x), np.asarray(y)
    return float(1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def pearson_distance(x: ArrayLike, y: ArrayLike) -> float:
    """1 minus the Pearson correlation. Ignores offset and amplitude."""
    return float(1 - np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def dtw_distance(
    x: ArrayLike,
    y: ArrayLike,
    dist_func: Callable[[ArrayLike, ArrayLike], float] = squared_distance,
) -> tuple[float, NDArray[np.float64]]:
    """
    Dynamic time warping by dynamic programming, O(len(x) * len(y)).

    Parameters
    ----------
    x, y : array-like
        1-D sequences; lengths may differ.
    dist_func : callable, optional
        Pointwise cost between two samples (default squared distance).

    Returns
    -------
    distance : float
        Accumulated cost of the optimal warping path. With the default cost,
        `sqrt(distance)` equals `tslearn.metrics.dtw(x, y)`.
    D : numpy.ndarray
        Accumulated cost matrix of shape `(len(x) + 1, len(y) + 1)`.
    """
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(x[i - 1], y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m]), D
