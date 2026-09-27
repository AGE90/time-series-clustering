"""
Cluster the processed series: `data/processed/<DATASET>.npz` -> `models/<model>.joblib`.

Run with `make train`, or pick a technique:

    uv run python -m tsclustering.models.train_model --model kshape

To add a technique, add an entry to `MODELS`: a function from the number
of clusters to an unfitted estimator with `fit_predict`.
"""

import argparse
import json
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tslearn.clustering import KShape, TimeSeriesKMeans

from tsclustering.data.make_dataset import DATASET
from tsclustering.models.model_utils import evaluate_clustering, save_model
from tsclustering.utils.paths import data_processed_dir, models_dir, reports_dir

SEED = 42

# name -> (factory taking n_clusters, metric used for the silhouette score)
MODELS: dict[str, tuple[Callable[[int], Any], str]] = {
    "kmeans-euclidean": (
        lambda k: TimeSeriesKMeans(k, metric="euclidean", random_state=SEED),
        "euclidean",
    ),
    "kmeans-dtw": (
        lambda k: TimeSeriesKMeans(k, metric="dtw", random_state=SEED, n_jobs=-1),
        "dtw",
    ),
    "kshape": (lambda k: KShape(k, random_state=SEED), "euclidean"),
}


def train(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    model_name: str = "kmeans-dtw",
    n_clusters: int | None = None,
) -> tuple[Any, dict[str, float]]:
    """
    Fit a clustering model and score it against the true labels.

    Parameters
    ----------
    X : numpy.ndarray
        Series of shape `(n_series, length, n_channels)`.
    y : numpy.ndarray
        Ground-truth labels, used only for ARI/NMI.
    model_name : str, optional
        Key of `MODELS` (default `"kmeans-dtw"`).
    n_clusters : int, optional
        Defaults to the number of distinct labels in `y`.

    Returns
    -------
    model : object
        Fitted estimator; `model.labels_` holds the cluster assignments.
    metrics : dict of str to float
        ARI, NMI and silhouette.
    """
    factory, metric = MODELS[model_name]
    model = factory(n_clusters or len(np.unique(y)))
    labels = model.fit_predict(X)
    return model, evaluate_clustering(X, labels, y, metric=metric)


def main() -> None:
    """Train the chosen model, save it and write its metrics to `reports/`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", choices=MODELS, default="kmeans-dtw")
    parser.add_argument("--n-clusters", type=int)
    args = parser.parse_args()

    data = np.load(data_processed_dir(f"{DATASET}.npz"))
    model, metrics = train(data["X"], data["y"], args.model, args.n_clusters)
    save_model(model, models_dir(f"{args.model}.joblib"))
    metrics_path = reports_dir(f"metrics_{args.model}.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"{args.model} on {DATASET}: {metrics}")


if __name__ == "__main__":
    main()
