"""
Z-normalize each series: `data/raw/<DATASET>.npz` -> `data/processed/<DATASET>.npz`.

Run with `make features`. Z-normalization makes distances compare shape
rather than offset or amplitude, the usual default for time series clustering.
"""

import numpy as np
from numpy.typing import NDArray
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from tsclustering.data.make_dataset import DATASET
from tsclustering.utils.paths import data_processed_dir, data_raw_dir


def build_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale every series to zero mean and unit variance."""
    scaled: NDArray[np.float64] = TimeSeriesScalerMeanVariance().fit_transform(X)
    return scaled


def main() -> None:
    """Read the raw dataset, normalize it and write it to `data/processed`."""
    raw = np.load(data_raw_dir(f"{DATASET}.npz"))
    np.savez(
        data_processed_dir(f"{DATASET}.npz"), X=build_features(raw["X"]), y=raw["y"]
    )
    print(f"Wrote {data_processed_dir(f'{DATASET}.npz')}")


if __name__ == "__main__":
    main()
