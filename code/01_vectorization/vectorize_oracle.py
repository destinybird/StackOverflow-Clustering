import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import time
import re
from tqdm import tqdm

def clean_html(text):
    """简单的HTML标签清理"""
    if not text:
        return ""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # 合并多个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("=== Oracle问答数据向量化 ===")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    start_time = time.time()
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    with open("oracle_database_questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"总问题数: {len(data):,}")
    
    # 2. 提取和清理文本（使用方案A：仅标题）
    print("\n2. 提取文本...")
    texts = []
    question_ids = []
    
    for item in tqdm(data, desc="处理问题"):
        # 清理标题
        title = clean_html(item.get("title", ""))
        if title:  # 只保留非空标题
            texts.append(title)
            question_ids.append(item.get("question_id", ""))
    
    print(f"有效问题数: {len(texts):,}")
    
    # 3. 加载模型（使用GPU）
    print("\n3. 加载模型...")
    model = SentenceTransformer(
        'all-MiniLM-L6-v2',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 4. 向量化编码
    print("\n4. 开始向量化编码...")
    embeddings = model.encode(
        texts,
        batch_size=256,  # 5090 GPU可以设置较大的batch_size
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # 归一化，便于后续计算相似度
    )
    
    print(f"向量化完成! 形状: {embeddings.shape}")
    
    # 5. 保存结果
    print("\n5. 保存结果...")
    
    # 保存向量
    np.save('oracle_question_embeddings.npy', embeddings)
    
    # 保存对应的元数据
    meta_df = pd.DataFrame({
        'question_id': question_ids,
        'text': texts
    })
    meta_df.to_csv('oracle_question_metadata.csv', index=False, encoding='utf-8')
    
    # 6. 性能统计
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== 完成! ===")
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"处理速度: {len(texts)/total_time:.2f} 条/秒")
    print(f"输出文件:")
    print(f"  - oracle_question_embeddings.npy (向量数据)")
    print(f"  - oracle_question_metadata.csv (文本元数据)")

if __name__ == "__main__":
    main()