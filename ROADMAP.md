# Roadmap

Ideas to grow this repo into a time series reference. Each item should become one module (or `MODELS` entry), one unit test and one numbered notebook.

## Distances and representations
- [x] Euclidean vs DTW, DTW from scratch (`tsclustering.distances`)
- [ ] Constrained DTW (Sakoe-Chiba band, Itakura parallelogram): effect on accuracy and runtime
- [ ] Derivative DTW: can it separate Trace classes 3 vs 4 (a faint oscillation that plain DTW misses)?
- [ ] Soft-DTW (differentiable) and soft-DTW barycenters
- [ ] Elastic measures: LCSS, ERP, TWED, MSM (compare on the same dataset)
- [ ] Feature-based: catch22 / tsfresh features → any sklearn clusterer
- [ ] Deep representations: autoencoder embeddings, TS2Vec; UMAP of embeddings

## Clustering algorithms
- [x] k-means (Euclidean, DTW + DBA), k-Shape
- [ ] Hierarchical clustering on a precomputed DTW matrix, with a dendrogram
- [ ] DBSCAN / HDBSCAN on DTW (no k needed, finds outliers)
- [ ] k-medoids (PAM), spectral clustering on a DTW kernel
- [ ] Model-based: HMM clustering, GMM on AR coefficients
- [ ] Subsequence clustering and motifs with the matrix profile (stumpy)

## Evaluation and rigor
- [ ] Benchmark harness: N UCR datasets × M models → one results table (ARI/NMI/runtime)
- [ ] Choosing k: elbow, silhouette, gap statistic
- [ ] Cluster stability by bootstrapping
- [ ] Scaling study: full DTW vs LB_Keogh pruning vs FastDTW

## Related time series tasks
- [ ] Anomaly detection: distance to the cluster barycenter, matrix profile discords
- [ ] Classification: 1-NN DTW baseline vs ROCKET / MiniRocket (aeon or sktime)
- [ ] Forecasting per cluster vs one global model
- [ ] Multivariate and unequal-length series; a real-world dataset (energy load, retail sales)

## Engineering
- [ ] GitHub Actions CI running `make check test`
- [ ] YAML/Hydra config per experiment instead of CLI flags
- [ ] MLflow tracking once the benchmark harness exists
- [ ] Docs site (mkdocs) rendering the notebooks
