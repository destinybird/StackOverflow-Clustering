# -*- coding: utf-8 -*-
"""
生成 6 类 baseline：
1. uniform_random
2. shuffle_labels_same_size
3. length_buckets
4. pca1_buckets
5. random_centroid_voronoi
6. kmeanspp_one_shot
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

ROOT = Path("/root")
REDUCTION_DIR = ROOT / "reduction_results"
META_FILE = ROOT / "oracle_full_metadata.csv"
OUTPUT_DIR = ROOT / "evaluate_clustering_results_spark/baselines"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 修改K值为100, 200, ..., 1400
Ks = list(range(100, 1401, 100))  # [100, 200, 300, ..., 1400]

def load_embeddings():
    X = np.load(REDUCTION_DIR / "pca_256.npy")
    return normalize(X, norm="l2")

def load_metadata():
    df = pd.read_csv(META_FILE)
    return df["text_length"].to_numpy()

def save_labels(labels, name):
    df = pd.DataFrame({"cluster_id": labels})
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    print(f"[Saved] {name}.csv")

def baseline_uniform_random(n, k):
    return np.random.randint(0, k, size=n)

def baseline_shuffle_true_size(true_labels, k):
    counts = np.bincount(true_labels)
    labels = np.zeros_like(true_labels)
    idx = np.random.permutation(len(true_labels))
    start = 0
    for cid in range(k):
        size = counts[cid] if cid < len(counts) else 0
        labels[idx[start:start+size]] = cid
        start += size
    return labels

def baseline_length_buckets(lengths, k):
    idx = np.argsort(lengths)
    labels = np.zeros_like(lengths, dtype=int)
    n = len(lengths)
    chunk = n // k
    for i in range(k):
        start = i * chunk
        end = (i+1)*chunk if i < k-1 else n
        labels[idx[start:end]] = i
    return labels

def baseline_pca1_buckets(X, k):
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).reshape(-1)
    idx = np.argsort(pc1)
    labels = np.zeros(len(X), dtype=int)
    n = len(X)
    chunk = n // k
    for i in range(k):
        start = i * chunk
        end = (i+1)*chunk if i < k-1 else n
        labels[idx[start:end]] = i
    return labels

def baseline_random_centroid_voronoi(X, k):
    n, dim = X.shape
    centroids = np.random.randn(k, dim)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    labels, _ = pairwise_distances_argmin_min(X, centroids)
    return labels

def baseline_kmeanspp_one_shot(X, k):
    """
    One-shot KMeans++ baseline:
    - Only initialize centers using kmeans++ strategy
    - No optimization / no EM iterations
    """
    from sklearn.cluster._kmeans import kmeans_plusplus
    from sklearn.metrics import pairwise_distances_argmin_min

    # Step 1: kmeans++ choose centers
    centers, _ = kmeans_plusplus(X, n_clusters=k, random_state=42)

    # Step 2: assign each point to its nearest center
    labels, _ = pairwise_distances_argmin_min(X, centers)

    return labels

def main():
    X = load_embeddings()
    n = len(X)
    lengths = load_metadata()

    for k in Ks:
        print(f"\n=== Baselines for K={k} ===")

        # True labels (for shuffle baseline)
        # 修改路径和文件名格式
        true_file = ROOT / "clustering_results_spark" / f"labels_k{k}.csv"
        true_df = pd.read_csv(true_file)
        # 假设列名是 'cluster_id'，如果不是请根据实际情况修改
        true_labels = true_df["cluster_id"].to_numpy()

        # 1. Uniform random
        save_labels(baseline_uniform_random(n, k), f"{k}_baseline_uniform_random")

        # 2. Shuffle true size
        save_labels(baseline_shuffle_true_size(true_labels, k), f"{k}_baseline_shuffle_size")

        # 3. Length buckets
        save_labels(baseline_length_buckets(lengths, k), f"{k}_baseline_length_buckets")

        # 4. PCA1 buckets
        save_labels(baseline_pca1_buckets(X, k), f"{k}_baseline_pca1_buckets")

        # 5. Random centroid Voronoi
        save_labels(baseline_random_centroid_voronoi(X, k), f"{k}_baseline_random_centroid")

        # 6. KMeans++ one-shot
        save_labels(baseline_kmeanspp_one_shot(X, k), f"{k}_baseline_kmeanspp_one_shot")

if __name__ == "__main__":
    main()
