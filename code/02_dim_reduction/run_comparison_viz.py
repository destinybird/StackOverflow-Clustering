import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import pandas as pd
import os
import time

# ================= 配置区域 =================
# 定义要对比的文件列表
PCA_FILES = {
    '50_dim':  'reduction_results/pca_50.npy',
    '100_dim': 'reduction_results/pca_100.npy',
    '256_dim': 'reduction_results/pca_256.npy'
}

OUTPUT_DIR = 'comparison_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# ===========================================

def main():
    print(f"[{time.strftime('%H:%M:%S')}] 开始多维度对比可视化任务...")

    # 循环处理每一个维度的文件
    for name, file_path in PCA_FILES.items():
        if not os.path.exists(file_path):
            print(f"跳过 {name}: 文件 {file_path} 不存在")
            continue

        print(f"\n{'='*20} 正在处理: {name} ({file_path}) {'='*20}")
        
        # 1. 加载数据
        data = np.load(file_path)
        print(f"[{time.strftime('%H:%M:%S')}] 数据加载成功，形状: {data.shape}")

        # ---------------------------------------------------------
        # 任务 A: 3D UMAP (交互式网页)
        # ---------------------------------------------------------
        print(f"[{time.strftime('%H:%M:%S')}] 正在计算 3D UMAP ({name})...")
        # 这里的 UMAP 是基于当前 PCA 维度的数据计算的
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        proj_3d = umap_3d.fit_transform(data)
        
        # 生成 HTML
        df_3d = pd.DataFrame(proj_3d, columns=['x', 'y', 'z'])
        
        # === 修正点：去掉了 render_mode 参数 ===
        fig = px.scatter_3d(
            df_3d, x='x', y='y', z='z',
            title=f"Stack Overflow 3D Clusters - Input: {name}",
            opacity=0.6
        )
        # ====================================
        
        fig.update_traces(marker=dict(size=2, color=df_3d['z'], colorscale='Viridis'))
        fig.update_layout(template="plotly_dark")
        
        html_filename = os.path.join(OUTPUT_DIR, f'3d_umap_{name}.html')
        fig.write_html(html_filename)
        print(f"   -> [保存] 3D 网页: {html_filename}")

        # ---------------------------------------------------------
        # 任务 B: t-SNE (采样 1万条)
        # ---------------------------------------------------------
        print(f"[{time.strftime('%H:%M:%S')}] 正在计算 t-SNE ({name})...")
        # 采样以加快速度
        indices = np.random.RandomState(42).choice(data.shape[0], 10000, replace=False)
        data_sample = data[indices]
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate=200)
        tsne_results = tsne.fit_transform(data_sample)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='cyan', s=1.5, alpha=0.7)
        plt.title(f't-SNE Visualization (Input: {name})', fontsize=16, color='white')
        plt.axis('off')
        
        tsne_filename = os.path.join(OUTPUT_DIR, f'tsne_{name}.png')
        plt.savefig(tsne_filename, dpi=300)
        plt.close() # 关闭画布，防止内存泄漏
        print(f"   -> [保存] t-SNE 图片: {tsne_filename}")

    print(f"\n[{time.strftime('%H:%M:%S')}] 所有对比任务已完成！请查看 {OUTPUT_DIR} 文件夹。")

if __name__ == "__main__":
    main()
