import os, time, umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# Config
INPUT_PCA = 'reduction_results/pca_256.npy'   # 使用 pca_256.npy
METADATA_FILE = 'oracle_full_metadata.csv'    # 包含 question_id
OUTPUT_DIR = 'clustering_results'
FINAL_CSV = 'final_cluster_labels.csv'
MD_REPORT = 'clustering_analysis.md'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# search ranges
K_RANGE = list(range(2000, 2001, 10))  # MiniBatchKMeans 尝试从20到1000（步长20）
DBSCAN_MIN_SAMPLES = [3, 5, 10]
# eps ranges for cosine-distance (1 - cosine_similarity): typical small values
EPS_COSINE = np.linspace(0.02, 0.30, 30)  # 0.02 - 0.30
# eps ranges for euclidean on normalized vectors:
EPS_EUCLIDEAN = np.linspace(0.1, 1.2, 24)
# sample sizes
SEARCH_SUBSET = 10000  # DBSCAN parameter search时的子集（我们数据较大）


def load_data():
    assert os.path.exists(INPUT_PCA), f"找不到 {INPUT_PCA}"
    data = np.load(INPUT_PCA)
    print(f"[Info] loaded embeddings shape: {data.shape}")
    if os.path.exists(METADATA_FILE):
        meta = pd.read_csv(METADATA_FILE)
        if len(meta) != len(data):
            print(f"[Warning] metadata 行数 {len(meta)} 与向量行数 {len(data)} 不匹配。将使用索引代替 question_id。")
            qids = np.arange(len(data))
        else:
            if 'question_id' in meta.columns:
                qids = meta['question_id'].values
            else:
                print("[Warning] metadata 无 'question_id' 列，使用索引。")
                qids = np.arange(len(data))
    else:
        print(f"[Warning] 未找到元数据 {METADATA_FILE}，使用索引ID。")
        qids = np.arange(len(data))
    return data, qids


def plot_k_distance(data, k=5, metric='cosine', savepath=None):
    print(f"[Info] computing k-distance (k={k}, metric={metric}) ...")
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    # distances[:, -1] is k-th neighbor distance
    kdist = np.sort(distances[:, -1])
    if savepath:
        plt.figure(figsize=(8,4))
        plt.plot(kdist)
        plt.title(f'k-distance plot (k={k}, metric={metric})')
        plt.xlabel('points sorted by distance')
        plt.ylabel(f'{k}-NN distance')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()
        print(f"[Saved] k-distance -> {savepath}")
    return kdist


def dbscan_search(data, metric='cosine'):
    print(f"\n--- DBSCAN search (metric={metric}) ---")
    n = len(data)
    subset_idx = np.random.choice(n, min(SEARCH_SUBSET, n), replace=False)
    search_data = data[subset_idx]
    best = {'score': -999, 'eps': None, 'min_samples': None, 'model': None, 'n_clusters':0, 'noise_ratio':1.0}
    # choose eps range based on metric
    eps_candidates = EPS_COSINE if metric=='cosine' else EPS_EUCLIDEAN
    for min_s in DBSCAN_MIN_SAMPLES:
        for eps in eps_candidates:
            model = DBSCAN(eps=eps, min_samples=min_s, metric=metric, n_jobs=-1)
            labels = model.fit_predict(search_data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_ratio = n_noise / len(search_data)
            # skip degenerate
            if n_clusters < 2 or noise_ratio > 0.98:
                continue
            # silhouette on this search_data (if >1 cluster)
            try:
                score = silhouette_score(search_data, labels, metric='euclidean' if metric=='euclidean' else 'cosine', sample_size=min(len(search_data), 20000))
            except Exception:
                score = -1
            print(f"[DBSCAN param] eps={eps:.4f}, min_samples={min_s}, clusters={n_clusters}, noise={noise_ratio:.3f}, sil={score:.4f}")
            if score > best['score']:
                best.update({'score': score, 'eps': eps, 'min_samples': min_s, 'model': model, 'n_clusters': n_clusters, 'noise_ratio': noise_ratio})
    if best['eps'] is None:
        print("[DBSCAN] 未找到合适参数 (可能数据稀疏/参数范围需扩展)")
        return None
    # refit best params on FULL data
    print(f"[DBSCAN] 最优参数（子集搜索）: eps={best['eps']}, min_samples={best['min_samples']}, score={best['score']:.4f}，正在对全量数据重训练...")
    final_model = DBSCAN(eps=best['eps'], min_samples=best['min_samples'], metric=metric, n_jobs=-1)
    full_labels = final_model.fit_predict(data)
    n_clusters_full = len(set(full_labels)) - (1 if -1 in full_labels else 0)
    n_noise_full = (full_labels == -1).sum()
    print(f"[DBSCAN Full] clusters={n_clusters_full}, noise={n_noise_full}/{len(data)} ({n_noise_full/len(data):.3f})")
    best['model'] = final_model
    best['labels_full'] = full_labels
    return best


def kmeans_search(data):
    print("\n--- MiniBatchKMeans search ---")
    best = {'score': 999, 'k': None, 'model': None}
    for k in K_RANGE:
        m = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096)
        labels = m.fit_predict(data)
        inertia = getattr(m, 'inertia_', None)
        # score = silhouette_score(data, labels, sample_size=int(len(data)/5))
        # print(f"[KMeans] k={k}, silhouette={score:.4f}, inertia={inertia:.2f}")
        score = davies_bouldin_score(data, labels)
        print(f"[KMeans] k={k}, dbi={score:.4f}, inertia={inertia:.2f}")
        if score < best['score']:
            best.update({'score': score, 'k': k, 'model': m})
    print(f"[KMeans] best K={best['k']}, dbi={best['score']:.4f}")
    return best


def visualize_2d_sample(data, labels, outpath, n_sample=20000, method='umap'):
    # produce a 2D scatter (sampled) and save
    n = len(data)
    idx = np.random.choice(n, min(n, n_sample), replace=False)
    sample = data[idx]
    sample_labels = np.array(labels)[idx]
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, metric='cosine')
        emb = reducer.fit_transform(sample)
    else:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(sample)
    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1], c=sample_labels, s=3, cmap='tab20')
    plt.title(f'2D visualization ({method})')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[Saved] 2D vis -> {outpath}")


def main():
    data, qids = load_data()
    # normalize (L2) -> makes euclidean ~ cosine monotonic
    data = normalize(data, norm='l2')
    print("[Info] done normalization.")
    # 1) k-distance plot for k = min_samples (用于 DBSCAN eps 选择)
    k_for_kdist = 5
    # subset_idx = np.random.choice(len(data), SEARCH_SUBSET, replace=False)
    # search_data = data[subset_idx]
    # kdist_cos = plot_k_distance(search_data, k=k_for_kdist, metric='cosine', savepath=os.path.join(OUTPUT_DIR, f'k_distance_k{k_for_kdist}_cosine.png'))
    # kdist_euc = plot_k_distance(search_data, k=k_for_kdist, metric='euclidean', savepath=os.path.join(OUTPUT_DIR, f'k_distance_k{k_for_kdist}_euclidean.png'))
    # 2) DBSCAN search (cosine)
    # dbscan_cos = dbscan_search(data, metric='cosine')
    # 3) DBSCAN search (euclidean)
    # dbscan_euc = dbscan_search(data, metric='euclidean')
    # 4) KMeans search
    kmeans_best = kmeans_search(data)
    # 5) Decide best model by silhouette score
    cand = []
    '''
    if dbscan_cos is not None:
        cand.append(('DBSCAN_cosine', dbscan_cos['score'], dbscan_cos))
    if dbscan_euc is not None:
        cand.append(('DBSCAN_euclidean', dbscan_euc['score'], dbscan_euc))
    '''
    if kmeans_best['model'] is not None:
        cand.append(('KMeans', kmeans_best['score'], kmeans_best))
    # sort by score
    cand_sorted = sorted(cand, key=lambda x: x[1], reverse=True)
    best_name, best_score, best_obj = cand_sorted[0]
    # get labels for full data
    if best_name.startswith('DBSCAN'):
        labels = best_obj['labels_full']
    elif best_name == 'KMeans':
        labels = best_obj['model'].predict(data)
    else:
        labels = np.zeros(len(data), dtype=int)
    # Save CSV
    out_df = pd.DataFrame({'question_id': qids, 'cluster_id': labels})
    csv_path = os.path.join(OUTPUT_DIR, FINAL_CSV)
    out_df.to_csv(csv_path, index=False)
    print(f"[Saved] final labels -> {csv_path}")
    # 2D visualization
    vis_path = os.path.join(OUTPUT_DIR, '2d_sample_visualization.png')
    visualize_2d_sample(data, labels, vis_path, n_sample=20000, method='umap')


if __name__ == "__main__":
    main()
