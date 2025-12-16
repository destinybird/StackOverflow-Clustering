import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

# --- 配置 ---
DATA_FILE = 'reduction_results/pca_256.npy'
LABELS_FILE = 'clustering_results/final_cluster_labels.csv'
OUTPUT_IMG = 'clustering_results/cluster_visualization_2d.png'


def main():
    print("--- 启动可视化生成程序 ---")
    # 1. 检查文件
    if not os.path.exists(DATA_FILE) or not os.path.exists(LABELS_FILE):
        print("错误：找不到输入数据或聚类标签文件。请先运行 run_clustering_analysis.py")
        return
    # 2. 加载数据
    data = np.load(DATA_FILE)
    df_labels = pd.read_csv(LABELS_FILE)
    labels = df_labels['cluster_id'].values
    # 为了可视化速度和清晰度，如果数据量太大，进行采样
    MAX_SAMPLES = 10000
    if len(data) > MAX_SAMPLES:
        print(f"数据量 ({len(data)}) 较大，采样前 {MAX_SAMPLES} 个点进行可视化...")
        indices = np.random.choice(len(data), MAX_SAMPLES, replace=False)
        data_viz = data[indices]
        labels_viz = labels[indices]
    else:
        data_viz = data
        labels_viz = labels
    # 3. 降维到 2D (用于绘图)
    print("正在运行 t-SNE 将数据降至 2D...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    data_2d = tsne.fit_transform(data_viz)
    # 4. 绘图
    print("正在绘图...")
    plt.figure(figsize=(12, 10))
    # 创建 DataFrame 便于 Seaborn 绘图
    plot_df = pd.DataFrame({
        'x': data_2d[:, 0],
        'y': data_2d[:, 1],
        'Cluster': labels_viz
    })
    # 处理噪声点 (Cluster -1): 让它们显示为灰色且较小，作为背景
    noise_df = plot_df[plot_df['Cluster'] == -1]
    cluster_df = plot_df[plot_df['Cluster'] != -1]
    # 绘制噪声
    if not noise_df.empty:
        plt.scatter(noise_df['x'], noise_df['y'], c='lightgray', s=10, label='Noise', alpha=0.3)
    # 绘制聚类点 (使用 tab20 调色板，颜色丰富)
    unique_clusters = len(plot_df['Cluster'].unique())
    palette = sns.color_palette("tab20", n_colors=unique_clusters) if unique_clusters <= 20 else "viridis"

    sns.scatterplot(
        data=cluster_df,
        x='x',
        y='y',
        hue='Cluster',
        palette=palette,
        s=30,
        alpha=0.8,
        legend='full'
    )

    plt.title(
        f'Cluster Visualization (t-SNE projection of 256-dim data)\nIdentified Clusters: {len(set(labels_viz)) - (1 if -1 in labels_viz else 0)}',
        fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # 如果簇太多，图例会很挤，这里简单处理一下
    if unique_clusters > 15:
        plt.legend([], [], frameon=False)
        plt.title(plt.gca().get_title() + ' (Legend hidden due to high cluster count)')
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"[Success] 可视化图片已保存至: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()