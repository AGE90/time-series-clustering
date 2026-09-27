"""
Model evaluation and persistence utilities.
"""

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tslearn.clustering import silhouette_score


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray | None = None,
    metric: str = "euclidean",
) -> dict[str, float]:
    """
    Score cluster assignments.

    Parameters
    ----------
    X : numpy.ndarray
        Series of shape `(n_series, length, n_channels)`.
    labels : numpy.ndarray
        Cluster assignment per series.
    y_true : numpy.ndarray, optional
        Ground-truth classes. When given, adds ARI and NMI (1.0 = perfect match,
        about 0 = random).
    metric : str, optional
        Distance for the silhouette: "euclidean", "dtw" or "softdtw".

    Returns
    -------
    metrics : dict of str to float
        silhouette, plus ari and nmi when `y_true` is given.
    """
    metrics = {"silhouette": float(silhouette_score(X, labels, metric=metric))}
    if y_true is not None:
        metrics["ari"] = float(adjusted_rand_score(y_true, labels))
        metrics["nmi"] = float(normalized_mutual_info_score(y_true, labels))
    return metrics


def save_model(model: Any, filepath: str | Path, engine: str = "joblib") -> None:
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model.
    filepath : str or pathlib.Path
        Path to save the model.
    engine : str, optional
        Engine to save the model in ('joblib' or 'pickle'). Default is 'joblib'.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if engine == "joblib":
        joblib.dump(model, filepath)
    elif engine == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def load_model(filepath: str | Path, engine: str = "joblib") -> Any:
    """
    Load a trained model from disk.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the saved model.
    engine : str, optional
        Engine the model was saved in ('joblib' or 'pickle'). Default is 'joblib'.

    Returns
    -------
    model : object
        The loaded model.
    """
    filepath = Path(filepath)

    if engine == "joblib":
        return joblib.load(filepath)
    if engine == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
