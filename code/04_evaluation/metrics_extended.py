# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)
from sklearn.preprocessing import normalize

ROOT = Path("/root")
REDUCTION_DIR = ROOT / "reduction_results"
CLUSTER_DIR = ROOT / "clustering_results_spark"
OUTPUT_DIR = ROOT / "evaluate_clustering_results_spark"
BASELINE_DIR = OUTPUT_DIR / "baselines"

Ks = list(range(100, 1401, 100)) 

# 和 baseline_generate.py 保持一致的名字
BASELINE_VARIANTS = [
    "baseline_uniform_random",
    "baseline_shuffle_size",
    "baseline_length_buckets",
    "baseline_pca1_buckets",
    "baseline_random_centroid",
    "baseline_kmeanspp_one_shot",
]

def load_embeddings():
    X = np.load(REDUCTION_DIR / "pca_256.npy")
    # 和聚类代码保持一致：先做 L2 归一化
    return normalize(X, norm="l2")

def load_labels_main(k: int) -> np.ndarray:
    """加载真实聚类结果 {K}_final_cluster_labels.csv"""
    df = pd.read_csv(CLUSTER_DIR / f"labels_k{k}.csv")
    return df["cluster_id"].to_numpy()

def load_labels_baseline(k: int, variant: str) -> np.ndarray:
    """
    从 baselines 目录加载 baseline 标签：
    文件名格式：{K}_{variant}.csv
    例如：1180_baseline_uniform_random.csv
    """
    path = BASELINE_DIR / f"{k}_{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")
    df = pd.read_csv(path)
    return df["cluster_id"].to_numpy()

def dunn_index(X, labels):
    """
    Dunn Index = 最小簇间距 / 最大簇内直径
    注意：在簇很多时，这个指标本身会比较小，只作为参考。
    """
    unique_labels = np.unique(labels)
    max_intra = 0.0
    centroids = []

    for lbl in unique_labels:
        cluster_points = X[labels == lbl]
        if len(cluster_points) <= 1:
            continue
        dist = pairwise_distances(cluster_points)
        max_intra = max(max_intra, dist.max())
        centroids.append(cluster_points.mean(axis=0))

    if len(centroids) < 2 or max_intra == 0:
        return np.nan

    centroids = np.stack(centroids)
    dist_centroid = pairwise_distances(centroids)
    # 忽略对角线
    np.fill_diagonal(dist_centroid, np.inf)
    min_inter = dist_centroid.min()

    if max_intra == 0:
        return np.nan
    return float(min_inter / max_intra)

def cohesion(X, labels):
    """簇内平均距离（所有簇的 pairwise 距离均值）"""
    total = 0.0
    count = 0
    for lbl in np.unique(labels):
        idx = (labels == lbl)
        cluster = X[idx]
        if len(cluster) <= 1:
            continue
        dist = pairwise_distances(cluster)
        total += dist.sum()
        count += dist.size
    if count == 0:
        return np.nan
    return float(total / count)

def separation(X, labels):
    """簇间中心距离的平均值"""
    centers = []
    for lbl in np.unique(labels):
        centers.append(X[labels == lbl].mean(axis=0))
    if len(centers) < 2:
        return np.nan
    centers = np.stack(centers)
    dist = pairwise_distances(centers)
    # 忽略对角线
    np.fill_diagonal(dist, np.nan)
    return float(np.nanmean(dist))

def cosine_intra_inter(X, labels):
    """
    簇内平均余弦相似度 & 簇间中心余弦相似度
    """
    from sklearn.metrics.pairwise import cosine_similarity

    unique_labels = np.unique(labels)
    centers = []

    for lbl in unique_labels:
        centers.append(X[labels == lbl].mean(axis=0))
    centers = np.stack(centers)

    # 簇内 cos
    intra = 0.0
    count_intra = 0
    for lbl in unique_labels:
        cluster = X[labels == lbl]
        if len(cluster) == 0:
            continue
        c = cluster.mean(axis=0)
        sim = cosine_similarity(cluster, c.reshape(1, -1))
        intra += sim.sum()
        count_intra += len(sim)

    # 簇间中心 cos
    inter = 0.0
    count_inter = 0
    sim_centers = cosine_similarity(centers)
    n = len(sim_centers)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            inter += sim_centers[i, j]
            count_inter += 1

    intra_mean = float(intra / count_intra) if count_intra > 0 else np.nan
    inter_mean = float(inter / count_inter) if count_inter > 0 else np.nan
    return intra_mean, inter_mean

def compute_all_metrics(X, labels):
    """
    给定一组标签，计算所有指标，返回一个 dict
    """
    # 传统指标
    sil = silhouette_score(X, labels, sample_size=min(3000, len(X)))
    dbi = davies_bouldin_score(X, labels)
    chi = calinski_harabasz_score(X, labels)

    # Dunn
    dunn = dunn_index(X, labels)

    # embedding 特征指标
    coh = cohesion(X, labels)
    sep = separation(X, labels)
    intra, inter = cosine_intra_inter(X, labels)

    # 单样本统计
    _, counts = np.unique(labels, return_counts=True)
    singleton_ratio = float((counts == 1).sum() / len(counts))

    return {
        "silhouette": sil,
        "dbi": dbi,
        "chi": chi,
        "dunn": dunn,
        "cohesion": coh,
        "separation": sep,
        "cosine_intra": intra,
        "cosine_inter": inter,
        "singleton_ratio": singleton_ratio,
    }

def main():
    X = load_embeddings()
    records = []

    for k in Ks:
        print(f"\n==================== K = {k} ====================")

        # 1) 真实 KMeans/MiniBatchKMeans 聚类结果
        print("[Main] evaluating true clustering...")
        labels_main = load_labels_main(k)
        metrics_main = compute_all_metrics(X, labels_main)
        metrics_main.update({"K": k, "variant": "kmeans"})
        records.append(metrics_main)

        # 2) 各类 baseline
        for variant in BASELINE_VARIANTS:
            path = BASELINE_DIR / f"{k}_{variant}.csv"
            if not path.exists():
                print(f"[Baseline] skip {variant} for K={k}, file not found.")
                continue
            print(f"[Baseline] evaluating {variant} for K={k} ...")
            labels_b = load_labels_baseline(k, variant)
            metrics_b = compute_all_metrics(X, labels_b)
            metrics_b.update({"K": k, "variant": variant})
            records.append(metrics_b)

    df = pd.DataFrame(records)
    out_full = OUTPUT_DIR / "cluster_metrics_extended_with_baselines.csv"
    df.to_csv(out_full, index=False)
    print(f"\n[Saved] all metrics (main + baselines) -> {out_full}")

    # 兼容原来：只导出真实模型的一个简洁版
    df_main = df[df["variant"] == "kmeans"].copy()
    out_main = OUTPUT_DIR / "cluster_metrics_extended_main.csv"
    df_main.to_csv(out_main, index=False)
    print(f"[Saved] main clustering only -> {out_main}")

if __name__ == "__main__":
    main()
