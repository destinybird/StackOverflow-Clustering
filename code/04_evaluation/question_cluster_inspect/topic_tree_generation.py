# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from rich.tree import Tree
from rich import print

def extract_subtopics(csv_path, n_subtopics=5):
    print(f"[Loading] {csv_path}")
    df = pd.read_csv(csv_path)
    titles = df["title"].astype(str).values

    # 1) TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 5),
        max_features=100000
    )
    tfidf = vectorizer.fit_transform(titles)

    # 2) 子聚类
    kmeans = KMeans(n_clusters=n_subtopics, random_state=42)
    labels = kmeans.fit_predict(tfidf)

    df["subtopic"] = labels

    # 3) 输出主题树结构
    root = Tree(f"[bold]Topic Tree for: {csv_path}[/bold]")

    feature_names = np.array(vectorizer.get_feature_names_out())

    for t in range(n_subtopics):
        sub_df = df[df.subtopic == t]

        centroid = kmeans.cluster_centers_[t]
        top_keywords = feature_names[np.argsort(centroid)[-7:]]

        # 添加子主题节点
        child = root.add(f"[green]Subtopic {t}: [/green] " 
                         f"{', '.join(top_keywords[::-1])}")

        # 添加代表问题 （前5个）
        for _, row in sub_df.head(5).iterrows():
            child.add(f"[blue]- {row['title']}[/blue]")

    print(root)


if __name__ == "__main__":
    csv_path = "q_308/K1200_kmeans_cluster_797.csv"
    extract_subtopics(csv_path, n_subtopics=5)
    csv_path = "q_308/K1200_kmeanspp_one_shot_cluster_630.csv"
    extract_subtopics(csv_path, n_subtopics=5)
    csv_path = "q_308/K1200_length_buckets_cluster_1054.csv"
    extract_subtopics(csv_path, n_subtopics=5)
    csv_path = "q_308/K1200_pca1_buckets_cluster_911.csv"
    extract_subtopics(csv_path, n_subtopics=5)
    csv_path = "q_308/K1200_random_centroid_cluster_880.csv"
    extract_subtopics(csv_path, n_subtopics=5)
    csv_path = "q_308/K1200_shuffle_size_cluster_532.csv"
    extract_subtopics(csv_path, n_subtopics=5)
    csv_path = "q_308/K1200_uniform_random_cluster_632.csv"
    extract_subtopics(csv_path, n_subtopics=5)
