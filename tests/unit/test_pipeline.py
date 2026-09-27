import numpy as np

from tsclustering.features.build_features import build_features
from tsclustering.models.predict_model import predict
from tsclustering.models.train_model import train


def test_pipeline_clusters_shifted_shapes() -> None:
    # two shapes (bump vs step), each at a random time position
    rng = np.random.default_rng(0)
    t = np.arange(100)
    pos = rng.uniform(20, 80, 20)
    X = np.stack(
        [np.exp(-(((t - p) / 5) ** 2)) for p in pos[:10]]
        + [(t > p).astype(float) for p in pos[10:]]
    )[:, :, None]
    y = np.repeat([0, 1], 10)

    X = build_features(X)
    assert np.allclose(X.mean(axis=1), 0)

    model, dtw_metrics = train(X, y, "kmeans-dtw")
    _, euclid_metrics = train(X, y, "kmeans-euclidean")
    assert dtw_metrics["ari"] == 1.0 > euclid_metrics["ari"]

    assert len(predict(model, X)) == len(X)
