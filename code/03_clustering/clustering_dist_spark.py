import os, time, umap, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

python_path = "/root/miniconda3/envs/pyspark_38/bin/python"
os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
INPUT_PCA = 'reduction_results/pca_256.npy'
METADATA_FILE = 'oracle_full_metadata.csv'
OUTPUT_DIR = 'clustering_results_spark'
RESULTS_TXT = 'kmeans_search_results.txt'
FINAL_CSV = 'final_cluster_labels.csv'
SPARK_MASTER = 'spark://master:7077'
os.makedirs(OUTPUT_DIR, exist_ok=True)
K_RANGE = list(range(20, 1401, 20))


def load_data():
    data = np.load(INPUT_PCA)
    if os.path.exists(METADATA_FILE):
        meta = pd.read_csv(METADATA_FILE)
        if len(meta) != len(data): qids = np.arange(len(data))
        else: qids = meta['question_id'].values if 'question_id' in meta.columns else np.arange(len(data))
    else: qids = np.arange(len(data))
    return data, qids


def visualize_2d_sample(data, labels, outpath, n_sample=20000):
    n = len(data)
    idx = np.random.choice(n, min(n, n_sample), replace=False)
    sample = data[idx]
    sample_labels = np.array(labels)[idx]
    print(f"[Vis] Starting 2D projection using {method}...")
    reducer = umap.UMAP(n_components=2, metric='cosine')
    emb = reducer.fit_transform(sample)
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=sample_labels, s=3, cmap='tab20')
    plt.title(f'2D visualization ({method})')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[Saved] 2D vis -> {outpath}")


def main():
    # 1. 初始化 Spark Session
    print(f"[Spark] Connecting to cluster at {SPARK_MASTER}...")
    spark = SparkSession.builder \
        .appName("KMeans_Search_Spark") \
        .master(SPARK_MASTER) \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    print(f"[Spark] Connected! UI at http://10.176.62.211:8080/")
    results_file = os.path.join(OUTPUT_DIR, RESULTS_TXT)

    # 2. 加载数据并预处理
    data_np, qids = load_data()
    data_np = normalize(data_np, norm='l2')
    print("[Spark] Converting NumPy data to Spark DataFrame (Vectors)...")
    data_rows = [(Vectors.dense(row),) for row in data_np]
    num_partitions = 200  # block partitions
    # 使用 parallelize 手动切分数据
    rdd = spark.sparkContext.parallelize(data_rows, numSlices=num_partitions)
    df = spark.createDataFrame(rdd, ["features"])
    # df = spark.createDataFrame(data_rows, ["features"])
    df.cache()
    print(f"[Spark] Data cached. Count: {df.count()}")

    # 3. 循环搜索 K
    print(f"[Spark] Starting loop for K in {K_RANGE}...")
    best_result = {'score': 999, 'k': None, 'model': None}
    for k in K_RANGE:
        start_t = time.time()
        kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol("features").setPredictionCol("prediction")
        model = kmeans.fit(df)
        try: cost = model.summary.trainingCost
        except: cost = model.computeCost(df)
        # 计算 Davies-Bouldin Score (DBI)
        predictions_df = model.transform(df)
        pred_rows = predictions_df.select("prediction").collect()
        labels = [row.prediction for row in pred_rows]
        dbi = davies_bouldin_score(data_np, labels)
        duration = time.time() - start_t
        result_line = f"[K={k}] DBI={dbi:.4f}, Inertia={cost:.2f}, Time={duration:.2f}s"
        print(result_line)
        with open(results_file, 'a') as f: f.write(result_line + "\n")
        if dbi < best_result['score']:
            best_result = {'score': dbi, 'k': k, 'model': model}
            best_line = f"   --> New Best K: {k}"
            print(best_line)
            with open(results_file, 'a') as f:
                f.write(best_line + "\n")
        if k % 100 == 0:
            checkpoint_csv = os.path.join(OUTPUT_DIR, f'labels_k{k}.csv')
            df_ckpt = pd.DataFrame({'question_id': qids, 'cluster_id': labels})
            df_ckpt.to_csv(checkpoint_csv, index=False)

    print(f"\n[Done] Best K={best_result['k']}, DBI={best_result['score']:.4f}")

    # 4. 保存最佳结果
    best_k = best_result['k']
    best_model = best_result['model']
    summary = f"\n[Done] Best K={best_result['k']}, DBI={best_result['score']:.4f}"
    print(summary)
    with open(results_file, 'a') as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("FINAL BEST RESULT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best K: {best_result['k']}\n")
        f.write(f"Best DBI score: {best_result['score']:.4f}\n")
        f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Generating final labels for K={best_k}...")
    final_pred_df = best_model.transform(df)
    final_rows = final_pred_df.select("prediction").collect()
    final_labels = [row.prediction for row in final_rows]
    out_df = pd.DataFrame({'question_id': qids, 'cluster_id': final_labels})
    csv_path = os.path.join(OUTPUT_DIR, FINAL_CSV)
    out_df.to_csv(csv_path, index=False)
    print(f"[Saved] final labels -> {csv_path}")
    vis_path = os.path.join(OUTPUT_DIR, '2d_sample_visualization.png')
    visualize_2d_sample(data_np, final_labels, vis_path, n_sample=20000)
    spark.stop()


if __name__ == "__main__":
    main()