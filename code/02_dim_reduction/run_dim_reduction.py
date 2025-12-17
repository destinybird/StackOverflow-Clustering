import numpy as np
import os
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import time

# ================= 配置区域 =================
# 注意：这里使用的是你 ls 命令中显示的全文向量文件名
INPUT_FILE = 'oracle_full_content_embeddings.npy' 
OUTPUT_DIR = 'reduction_results'

# 检查输入文件是否存在
if not os.path.exists(INPUT_FILE):
    print(f"错误：找不到文件 {INPUT_FILE}，请检查文件名！")
    exit()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# ===========================================

def main():
    print(f"[{time.strftime('%H:%M:%S')}] 1. 正在加载数据: {INPUT_FILE} ...")
    # 加载 .npy 文件
    data = np.load(INPUT_FILE)
    print(f"[{time.strftime('%H:%M:%S')}] 数据加载成功! 数据形状: {data.shape}")
    
    # ---------------------------------------------------------
    # 任务 A: PCA 降维 (生成不同维度供聚类使用)
    # ---------------------------------------------------------
    print("\n=== 开始 PCA 降维任务 ===")
    target_dims = [50, 100, 256]
    
    for dim in target_dims:
        print(f"[{time.strftime('%H:%M:%S')}] 正在降维至 {dim} 维...")
        pca = PCA(n_components=dim)
        reduced_data = pca.fit_transform(data)
        
        # 保存文件
        save_path = os.path.join(OUTPUT_DIR, f'pca_{dim}.npy')
        np.save(save_path, reduced_data)
        
        # 计算保留的信息量
        variance = np.sum(pca.explained_variance_ratio_)
        print(f"   -> 完成！保留了 {variance:.2%} 的原始信息。已保存至: {save_path}")

    # ---------------------------------------------------------
    # 任务 B: UMAP 可视化 (降维至 2D 绘图)
    # ---------------------------------------------------------
    print("\n=== 开始 UMAP 可视化任务 (生成 2D 图) ===")
    print(f"[{time.strftime('%H:%M:%S')}] 正在计算 UMAP (数据量较大，请耐心等待，可能需要5-10分钟)...")
    
    # UMAP 参数：n_neighbors=15 (平衡局部和全局), min_dist=0.1 (点之间的紧密度)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(data)
    
    # 保存 2D 数据备份
    np.save(os.path.join(OUTPUT_DIR, 'umap_2d.npy'), embedding_2d)

    print(f"[{time.strftime('%H:%M:%S')}] UMAP 计算完成，正在生成图片...")

    # 绘制散点图
    plt.figure(figsize=(12, 10))
    # s=0.1 让点非常小，防止重叠成一团；alpha=0.5 设置透明度
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=0.1, alpha=0.5, c='blue')
    plt.title('Stack Overflow Questions Visualization (UMAP)', fontsize=16)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # 去掉边框刻度，让图更好看
    plt.xticks([])
    plt.yticks([])

    # 保存图片
    img_path = os.path.join(OUTPUT_DIR, 'visualization_result.png')
    plt.savefig(img_path, dpi=300)
    print(f"   -> 图片已保存至: {img_path}")
    
    print("\n所有工作已完成！请查看 reduction_results 文件夹。")

if __name__ == "__main__":
    main()
