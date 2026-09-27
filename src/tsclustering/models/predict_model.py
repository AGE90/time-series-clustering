"""
Assign clusters: `models/<model>.joblib` + processed series -> cluster CSV.

Run with `make predict`. Point `DATASET` at new data built the same way.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tsclustering.data.make_dataset import DATASET
from tsclustering.models.model_utils import load_model
from tsclustering.utils.paths import data_processed_dir, models_dir

MODEL_NAME = "kmeans-dtw"


def predict(model: Any, X: NDArray[np.float64]) -> pd.DataFrame:
    """Return one row per series with its assigned `cluster`."""
    return pd.DataFrame({"cluster": model.predict(X)})


def main() -> None:
    """Load the model, assign clusters and write them to `data/processed`."""
    model = load_model(models_dir(f"{MODEL_NAME}.joblib"))
    data = np.load(data_processed_dir(f"{DATASET}.npz"))
    out = predict(model, data["X"]).assign(label=data["y"])
    path = data_processed_dir(f"{DATASET}_clusters.csv")
    out.to_csv(path, index_label="series")
    print(f"Wrote clusters to {path}")


if __name__ == "__main__":
    main()
