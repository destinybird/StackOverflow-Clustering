# -*- coding: utf-8 -*-
"""
根据给定 question_id，导出该问题在不同 K / 不同聚类方案（kmeans + 各 baseline）
下所在的整个 cluster 成员，便于定性分析。

新增功能：找到所有簇大小小于等于10的簇并导出

使用方法：
    cd /root/evaluate_clusters
    python3 export_question_clusters.py 308   # 导出指定问题所在的簇
    或
    python3 export_question_clusters.py --small-clusters  # 导出所有小簇（size <= 10）
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 路径配置
ROOT = Path("/root")
META_FILE = ROOT / "oracle_full_metadata.csv"
CLUSTER_DIR = ROOT / "clustering_results_spark"
OUTPUT_DIR = ROOT / "evaluate_clustering_results_spark"
BASELINE_DIR = OUTPUT_DIR / "baselines"

# 和前面的代码保持一致
Ks = [400, 800, 1200]
BASELINE_VARIANTS = [
    "baseline_uniform_random",
    "baseline_shuffle_size",
    "baseline_length_buckets",
    "baseline_pca1_buckets",
    "baseline_random_centroid",
    "baseline_kmeanspp_one_shot",
]


def load_meta():
    if not META_FILE.exists():
        raise FileNotFoundError("找不到元数据文件: {}".format(META_FILE))
    df = pd.read_csv(META_FILE)
    if "question_id" not in df.columns:
        raise ValueError("元数据文件中缺少 question_id 列")
    return df


def find_meta_index(df_meta, qid):
    """返回该 question_id 在 meta 中的行索引（用于 baseline，按行对齐）"""
    idx = df_meta.index[df_meta["question_id"] == qid]
    if len(idx) == 0:
        raise ValueError("question_id {} 在元数据中找不到".format(qid))
    return int(idx[0])


def export_for_kmeans(K, df_meta, qid, out_dir, summary_records):
    """
    处理真实 KMeans 结果：
    - 读取 {K}_final_cluster_labels.csv
    - 找到该 qid 所在的 cluster
    - 输出该 cluster 的所有问题（带 title 等信息）
    """
    labels_path = CLUSTER_DIR / "labels_k{}.csv".format(K)
    if not labels_path.exists():
        print("[Warn] K={} 的 kmeans 标签文件不存在: {}".format(K, labels_path))
        return

    df_labels = pd.read_csv(labels_path)  # question_id, cluster_id
    if "question_id" not in df_labels.columns or "cluster_id" not in df_labels.columns:
        print("[Warn] {} 缺少必要列".format(labels_path))
        return

    # 找到该 qid 所在 cluster_id
    row = df_labels[df_labels["question_id"] == qid]
    if row.empty:
        print("[Warn] question_id {} 不在 K={} 的 kmeans 标签中".format(qid, K))
        return

    cluster_id = int(row["cluster_id"].iloc[0])

    # 合并 meta，拿到 title 等
    df_k = df_labels.merge(df_meta, on="question_id", how="left")

    members = df_k[df_k["cluster_id"] == cluster_id].copy()
    members = members[["question_id", "title", "answer_count", "text_length", "cluster_id"]]

    out_path = out_dir / "K{}_kmeans_cluster_{}.csv".format(K, cluster_id)
    members.to_csv(out_path, index=False)
    print("[Saved] {}".format(out_path))

    # 记录到 summary
    summary_records.append({
        "K": K,
        "variant": "kmeans",
        "cluster_id": cluster_id,
        "cluster_size": len(members)
    })


def export_for_baseline(K, variant, df_meta, qid, meta_idx, out_dir, summary_records):
    """
    处理 baseline：
    - 读取 baselines/{K}_{variant}.csv（只有 cluster_id 列）
    - 利用 meta_idx 获取该行的 cluster_id
    - 构造 df_all = meta + cluster_id
    - 导出与该 cluster_id 相同的所有行
    """
    labels_path = BASELINE_DIR / "{}_{}.csv".format(K, variant)
    if not labels_path.exists():
        print("[Warn] baseline 文件不存在: {}".format(labels_path))
        return

    df_labels = pd.read_csv(labels_path)
    if "cluster_id" not in df_labels.columns:
        print("[Warn] baseline 文件缺少 cluster_id 列: {}".format(labels_path))
        return

    if len(df_labels) != len(df_meta):
        print("[Warn] baseline {} 行数({}) 与 meta 行数({}) 不一致，跳过".format(
            labels_path, len(df_labels), len(df_meta)))
        return

    # 该 question 在 baseline 中的 cluster_id（按照行对齐）
    cluster_id = int(df_labels["cluster_id"].iloc[meta_idx])

    # 构造带 meta 的 DataFrame
    df_all = df_meta.copy()
    df_all["cluster_id"] = df_labels["cluster_id"].values

    members = df_all[df_all["cluster_id"] == cluster_id].copy()
    members = members[["question_id", "title", "answer_count", "text_length", "cluster_id"]]

    safe_variant = variant.replace("baseline_", "")
    out_path = out_dir / "K{}_{}_cluster_{}.csv".format(K, safe_variant, cluster_id)
    members.to_csv(out_path, index=False)
    print("[Saved] {}".format(out_path))

    summary_records.append({
        "K": K,
        "variant": variant,
        "cluster_id": cluster_id,
        "cluster_size": len(members)
    })


def find_and_export_small_clusters():
    """
    找到所有簇大小小于等于10的簇并导出
    """
    df_meta = load_meta()
    
    # 输出目录：/root/evaluate_clustering_results_spark/small_clusters/
    out_root = OUTPUT_DIR / "small_clusters"
    os.makedirs(out_root, exist_ok=True)
    
    small_clusters_summary = []
    
    for K in Ks:
        print("\n=== 处理 K = {} ===".format(K))
        
        # 1. 处理 kmeans 结果
        labels_path = CLUSTER_DIR / "labels_k{}.csv".format(K)
        if labels_path.exists():
            df_labels = pd.read_csv(labels_path)
            
            if "question_id" not in df_labels.columns or "cluster_id" not in df_labels.columns:
                print("[Warn] {} 缺少必要列".format(labels_path))
            else:
                # 合并 meta
                df_k = df_labels.merge(df_meta, on="question_id", how="left")
                
                # 统计每个簇的大小
                cluster_sizes = df_k.groupby("cluster_id").size().reset_index(name="size")
                
                # 找到小簇（size <= 10）
                small_clusters = cluster_sizes[cluster_sizes["size"] <= 10]
                
                if not small_clusters.empty:
                    print(f"K={K} kmeans 中找到 {len(small_clusters)} 个小簇（size <= 10）")
                    
                    # 创建子目录
                    kmeans_out_dir = out_root / f"K{K}" / "kmeans"
                    os.makedirs(kmeans_out_dir, exist_ok=True)
                    
                    # 导出每个小簇
                    for _, row in small_clusters.iterrows():
                        cluster_id = int(row["cluster_id"])
                        size = int(row["size"])
                        
                        members = df_k[df_k["cluster_id"] == cluster_id].copy()
                        members = members[["question_id", "title", "answer_count", "text_length", "cluster_id"]]
                        
                        out_path = kmeans_out_dir / f"cluster_{cluster_id}_size_{size}.csv"
                        members.to_csv(out_path, index=False)
                        
                        small_clusters_summary.append({
                            "K": K,
                            "variant": "kmeans",
                            "cluster_id": cluster_id,
                            "cluster_size": size,
                            "file_path": str(out_path.relative_to(out_root))
                        })
                else:
                    print(f"K={K} kmeans 中没有找到小簇（size <= 10）")
        
        # 2. 处理每个 baseline
        for variant in BASELINE_VARIANTS:
            labels_path = BASELINE_DIR / "{}_{}.csv".format(K, variant)
            if labels_path.exists():
                df_labels = pd.read_csv(labels_path)
                
                if "cluster_id" not in df_labels.columns:
                    print("[Warn] baseline 文件缺少 cluster_id 列: {}".format(labels_path))
                    continue
                
                if len(df_labels) != len(df_meta):
                    print("[Warn] baseline {} 行数({}) 与 meta 行数({}) 不一致，跳过".format(
                        labels_path, len(df_labels), len(df_meta)))
                    continue
                
                # 构造带 meta 的 DataFrame
                df_all = df_meta.copy()
                df_all["cluster_id"] = df_labels["cluster_id"].values
                
                # 统计每个簇的大小
                cluster_sizes = df_all.groupby("cluster_id").size().reset_index(name="size")
                
                # 找到小簇（size <= 10）
                small_clusters = cluster_sizes[cluster_sizes["size"] <= 10]
                
                if not small_clusters.empty:
                    safe_variant = variant.replace("baseline_", "")
                    print(f"K={K} {safe_variant} 中找到 {len(small_clusters)} 个小簇（size <= 10）")
                    
                    # 创建子目录
                    baseline_out_dir = out_root / f"K{K}" / safe_variant
                    os.makedirs(baseline_out_dir, exist_ok=True)
                    
                    # 导出每个小簇
                    for _, row in small_clusters.iterrows():
                        cluster_id = int(row["cluster_id"])
                        size = int(row["size"])
                        
                        members = df_all[df_all["cluster_id"] == cluster_id].copy()
                        members = members[["question_id", "title", "answer_count", "text_length", "cluster_id"]]
                        
                        out_path = baseline_out_dir / f"cluster_{cluster_id}_size_{size}.csv"
                        members.to_csv(out_path, index=False)
                        
                        small_clusters_summary.append({
                            "K": K,
                            "variant": variant,
                            "cluster_id": cluster_id,
                            "cluster_size": size,
                            "file_path": str(out_path.relative_to(out_root))
                        })
                else:
                    safe_variant = variant.replace("baseline_", "")
                    print(f"K={K} {safe_variant} 中没有找到小簇（size <= 10）")
    
    # 保存汇总信息
    if small_clusters_summary:
        df_summary = pd.DataFrame(small_clusters_summary)
        df_summary = df_summary.sort_values(by=["K", "variant", "cluster_id"])
        
        summary_path = out_root / "small_clusters_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        
        # 打印统计信息
        print("\n" + "="*60)
        print("小簇统计汇总:")
        print("="*60)
        
        # 按K和variant统计
        for K in Ks:
            df_k = df_summary[df_summary["K"] == K]
            if not df_k.empty:
                print(f"\nK={K}:")
                for variant in ["kmeans"] + BASELINE_VARIANTS:
                    df_v = df_k[df_k["variant"] == variant]
                    if not df_v.empty:
                        safe_variant = variant.replace("baseline_", "")
                        print(f"  {safe_variant}: {len(df_v)} 个小簇，总问题数: {df_v['cluster_size'].sum()}")
        
        total_clusters = len(df_summary)
        total_questions = df_summary["cluster_size"].sum()
        print(f"\n总计: {total_clusters} 个小簇，包含 {total_questions} 个问题")
        print(f"详细汇总已保存到: {summary_path}")
    else:
        print("\n没有找到任何小簇（size <= 10）")


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  1. 导出指定问题所在的簇: python3 export_question_clusters.py <question_id>")
        print("  2. 导出所有小簇（size <= 10）: python3 export_question_clusters.py --small-clusters")
        sys.exit(1)
    
    # 检查是否要导出小簇
    if sys.argv[1] == "--small-clusters":
        find_and_export_small_clusters()
        return
    
    # 否则，按原逻辑处理指定question_id
    try:
        qid = int(sys.argv[1])
    except ValueError:
        print("错误: question_id 必须是整数")
        sys.exit(1)

    df_meta = load_meta()
    meta_idx = find_meta_index(df_meta, qid)

    # 打印一下这个问题的基本信息
    row_meta = df_meta.iloc[meta_idx]
    print(">>> question_id = {}".format(qid))
    print("    title       = {}".format(row_meta["title"]))
    print("    answer_count= {}, text_length={}".format(row_meta["answer_count"], row_meta["text_length"]))

    # 输出目录：/root/question_cluster_inspect/q_<qid>/
    out_root = OUTPUT_DIR / "question_cluster_inspect"
    out_dir = out_root / "q_{}".format(qid)
    os.makedirs(out_dir, exist_ok=True)

    summary_records = []

    for K in Ks:
        print("\n=== K = {} ===".format(K))
        # 真实 kmeans
        export_for_kmeans(K, df_meta, qid, out_dir, summary_records)
        # baselines
        for variant in BASELINE_VARIANTS:
            export_for_baseline(K, variant, df_meta, qid, meta_idx, out_dir, summary_records)

    # 写 summary.csv
    if summary_records:
        df_sum = pd.DataFrame(summary_records)
        df_sum = df_sum.sort_values(by=["K", "variant"])
        sum_path = out_dir / "summary.csv"
        df_sum.to_csv(sum_path, index=False)
        print("\n[Saved] summary -> {}".format(sum_path))
    else:
        print("\n[Info] 没有生成任何结果，请检查 question_id 是否存在于各个聚类文件。")


if __name__ == "__main__":
    main()