import pandas as pd
import matplotlib.pyplot as plt


def simple_extraction():
    ddd = 24
    cluster_df = pd.read_csv('clustering_results_spark/final_cluster_labels.csv')
    target_cluster = cluster_df[cluster_df['cluster_id'] == ddd]
    question_ids = target_cluster['question_id'].tolist()
    metadata_df = pd.read_csv('oracle_full_metadata.csv')
    metadata_df['question_id'] = metadata_df['question_id'].astype(int)
    result_df = metadata_df[metadata_df['question_id'].isin(question_ids)][['question_id', 'title']]
    result_df.to_csv(f'clustering_results_spark/{ddd}_titles.csv', index=False, encoding='utf-8')
    print(f"找到 {len(result_df)} 个问题")
    print(f"结果已保存到 {ddd}_titles.csv")


def analyze_distribution():
    print("开始统计分布...")
    cluster_df = pd.read_csv('clustering_results_spark/final_cluster_labels.csv')
    cluster_counts = cluster_df['cluster_id'].value_counts().sort_index()
    target_counts = cluster_counts[cluster_counts.index <= 1000]
    print(f"最大类包含问题数: {target_counts.max()}")
    print(f"最小类包含问题数: {target_counts.min()}")
    max_count = target_counts.max()
    max_cluster_ids = target_counts[target_counts == max_count].index.tolist()
    print(f"\n[结果] 包含问题最多 ({max_count}个) 的类 ID 是: {max_cluster_ids}")
    min_count = target_counts.min()
    min_cluster_ids = target_counts[target_counts == min_count].index.tolist()
    print(f"[结果] 包含问题最少 ({min_count}个) 的类 ID 是: {min_cluster_ids}")
    plt.figure(figsize=(20, 6))
    plt.bar(target_counts.index, target_counts.values, width=0.8, color='skyblue')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Questions')
    plt.title('Distribution of Questions per Cluster')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('clustering_results_spark/cluster_distribution.png')
    print("柱状图已保存为 clustering_results_spark/cluster_distribution.png")
    plt.show()


if __name__ == "__main__":
    simple_extraction()
    # analyze_distribution()