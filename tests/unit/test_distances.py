import numpy as np
from tslearn.metrics import dtw

from tsclustering.distances import dtw_distance, euclidean_distance


def test_dtw_matches_tslearn() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.random(30), rng.random(45)
    dist, D = dtw_distance(x, y)
    assert D.shape == (31, 46)
    assert np.isclose(np.sqrt(dist), dtw(x, y))


def test_dtw_is_invariant_to_time_shift() -> None:
    bump = np.sin(2 * np.pi * np.linspace(0, 1, 50))
    x = np.concatenate([np.zeros(10), bump, np.zeros(30)])
    y = np.concatenate([np.zeros(30), bump, np.zeros(10)])
    assert np.sqrt(dtw_distance(x, y)[0]) < 1e-9 < euclidean_distance(x, y)
