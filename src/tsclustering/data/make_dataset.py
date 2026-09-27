"""
Download a UCR dataset: tslearn loader -> `data/raw/<DATASET>.npz`.

Run with `make data`. Train and test splits are merged: clustering is
unsupervised, and the labels `y` are kept only to score the clusters.
"""

import numpy as np
from numpy.typing import NDArray
from tslearn.datasets import UCR_UEA_datasets

from tsclustering.utils.paths import data_raw_dir

DATASET = "Trace"  # 4 classes, 200 series of length 275


def make_dataset(name: str = DATASET) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Load a UCR/UEA dataset with train and test merged.

    Returns
    -------
    X : numpy.ndarray
        Series of shape `(n_series, length, n_channels)`.
    y : numpy.ndarray
        Ground-truth class labels, for evaluation only.
    """
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(name)
    if X_train is None:
        raise ValueError(f"Unknown UCR/UEA dataset: {name}")
    return np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])


def main() -> None:
    """Download `DATASET` and save it to `data/raw`."""
    X, y = make_dataset()
    np.savez(data_raw_dir(f"{DATASET}.npz"), X=X, y=y)
    print(f"Wrote {X.shape} to {data_raw_dir(f'{DATASET}.npz')}")


if __name__ == "__main__":
    main()
