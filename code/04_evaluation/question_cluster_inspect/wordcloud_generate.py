# -*- coding: utf-8 -*-
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def generate_wordcloud(csv_path, output_png):
    # 读取cluster文件
    df = pd.read_csv(csv_path)

    # 合并所有 title，作为词云语料
    text = " ".join(df["title"].astype(str).tolist())

    # 创建词云
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=50,
        collocations=True
    ).generate(text)

    # 保存图片
    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    print(f"[Saved wordcloud] {output_png}")


if __name__ == "__main__":
    # 示例：为 1180 cluster 386 生成词云
    csv_path = "q_308/K1180_kmeans_cluster_816.csv"
    output_png = "q_308/pic/wordcloud_1180_kmeans.png"
    generate_wordcloud(csv_path, output_png)
    csv_path = "q_308/K1180_kmeanspp_one_shot_cluster_1043.csv"
    output_png = "q_308/pic/wordcloud_1180_kmeanspp.png"
    generate_wordcloud(csv_path, output_png)
    csv_path = "q_308/K1180_length_buckets_cluster_1038.csv"
    output_png = "q_308/pic/wordcloud_1180_length_buckets.png"
    generate_wordcloud(csv_path, output_png)
    csv_path = "q_308/K1180_pca1_buckets_cluster_897.csv"
    output_png = "q_308/pic/wordcloud_1180_pca1_buckets.png"
    generate_wordcloud(csv_path, output_png)
    csv_path = "q_308/K1180_random_centroid_cluster_856.csv"
    output_png = "q_308/pic/wordcloud_1180_random_centroid.png"
    generate_wordcloud(csv_path, output_png)
    csv_path = "q_308/K1180_shuffle_size_cluster_133.csv"
    output_png = "q_308/pic/wordcloud_1180_shuffle_size.png"
    generate_wordcloud(csv_path, output_png)
    csv_path = "q_308/K1180_uniform_random_cluster_759.csv"
    output_png = "q_308/pic/wordcloud_1180_uniform_random.png"
    generate_wordcloud(csv_path, output_png)
