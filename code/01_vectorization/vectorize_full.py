import json
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
import re
import os
import sys

print("=== 纯本地向量化（完全离线） ===")

# 彻底禁用网络
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1' 
os.environ['HF_EVALUATE_OFFLINE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def clean_html(text):
    """清理HTML标签"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def combine_all_content(question_item):
    """组合问题的所有文本内容"""
    parts = []
    
    title = clean_html(question_item.get('title', ''))
    if title:
        parts.append(f"标题: {title}")
    
    body = clean_html(question_item.get('body', ''))
    if body:
        parts.append(f"问题: {body}")
    
    for i, comment in enumerate(question_item.get('comments', [])):
        comment_body = clean_html(comment.get('body', ''))
        if comment_body:
            parts.append(f"问题评论{i+1}: {comment_body}")
    
    for j, answer in enumerate(question_item.get('answers', [])):
        answer_body = clean_html(answer.get('body', ''))
        if answer_body:
            parts.append(f"回答{j+1}: {answer_body}")
            
        for k, ans_comment in enumerate(answer.get('comments', [])):
            ans_comment_body = clean_html(ans_comment.get('body', ''))
            if ans_comment_body:
                parts.append(f"回答{j+1}评论{k+1}: {ans_comment_body}")
    
    full_text = " ".join(parts)
    
    # 截断处理
    max_length = 4000
    if len(full_text) > max_length:
        if len(title) > 0:
            title_part = f"标题: {title} "
            remaining = max_length - len(title_part)
            full_text = title_part + full_text[len(title_part):remaining]
        else:
            full_text = full_text[:max_length]
    
    return full_text

def load_model_completely_offline():
    """完全离线加载模型"""
    print("尝试完全离线加载模型...")
    
    # 查找本地模型缓存
    cache_paths = [
        os.path.expanduser('~/.cache/torch/sentence_transformers'),
        os.path.expanduser('~/.cache/huggingface/hub'),
        'D:/.cache/torch/sentence_transformers',
        'D:/.cache/huggingface/hub'
    ]
    
    model_path = None
    for path in cache_paths:
        if os.path.exists(path):
            print(f"检查路径: {path}")
            # 在缓存中查找模型
            for root, dirs, files in os.walk(path):
                if 'all-MiniLM-L6-v2' in root and 'modules.json' in files:
                    model_path = root
                    print(f"✅ 找到模型路径: {model_path}")
                    break
            if model_path:
                break
    
    if model_path:
        try:
            # 方法1：直接使用本地路径
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_path)
            print("✅ 从本地路径加载成功")
            return model
        except Exception as e:
            print(f"本地路径加载失败: {e}")
    
    # 方法2：使用已经导入的模型（如果之前成功过）
    try:
        print("尝试使用已缓存的模型...")
        # 重新导入，利用Python的模块缓存
        import importlib
        import sentence_transformers
        importlib.reload(sentence_transformers)
        from sentence_transformers import SentenceTransformer
        
        # 静默加载，避免网络请求
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ 使用缓存模型成功")
        return model
    except Exception as e:
        print(f"缓存模型加载失败: {e}")
    
    return None

def main():
    # 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 1. 读取原始JSON数据
    print("\n1. 读取原始数据...")
    try:
        with open("oracle_database_questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 成功读取 {len(data):,} 条问题")
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return
    
    # 2. 提取和组合所有文本内容
    print("\n2. 组合文本内容...")
    full_texts = []
    question_ids = []
    titles = []
    answer_counts = []
    text_lengths = []
    
    for item in tqdm(data, desc="处理问题"):
        question_id = item.get('question_id', '')
        title = clean_html(item.get('title', ''))
        
        full_text = combine_all_content(item)
        
        if full_text and len(full_text) > 10:
            full_texts.append(full_text)
            question_ids.append(question_id)
            titles.append(title)
            answer_counts.append(len(item.get('answers', [])))
            text_lengths.append(len(full_text))
    
    print(f"✅ 有效问题数: {len(full_texts):,}")
    print(f"📊 平均文本长度: {np.mean(text_lengths):.0f} 字符")
    
    # 3. 加载模型（完全离线）
    print("\n3. 加载语义模型（完全离线）...")
    model = load_model_completely_offline()
    
    if model is None:
        print("❌ 无法加载模型，使用备用方案...")
        # 备用方案：使用之前已经生成的向量
        if os.path.exists('oracle_embeddings_cuda12.npy'):
            print("使用之前生成的标题向量...")
            return
        else:
            print("没有可用的向量文件")
            return
    
    # 4. 向量化编码
    print("\n4. 开始语义向量化...")
    start_time = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 分批处理
    batch_size = 16  # 更小的batch_size确保稳定
    all_embeddings = []
    
    for i in tqdm(range(0, len(full_texts), batch_size), desc="向量化批次"):
        batch_texts = full_texts[i:i+batch_size]
        try:
            # 完全静默处理
            with torch.no_grad():
                embeddings = model.encode(
                    batch_texts,
                    batch_size=8,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=device
                )
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"❌ 批次 {i} 处理失败: {e}")
            # 跳过有问题的批次
            continue
    
    # 合并所有向量
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        print(f"✅ 向量化完成! 形状: {final_embeddings.shape}")
    else:
        print("❌ 向量化失败!")
        return
    
    # 5. 保存结果
    print("\n5. 保存结果...")
    
    np.save('oracle_full_content_embeddings.npy', final_embeddings)
    
    full_metadata = pd.DataFrame({
        'question_id': question_ids,
        'title': titles,
        'answer_count': answer_counts,
        'text_length': text_lengths
        # 不保存full_text，文件太大
    })
    full_metadata.to_csv('oracle_full_metadata.csv', index=False, encoding='utf-8')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== 完成! ===")
    print(f"⏱️  总耗时: {total_time/60:.2f} 分钟")
    print(f"💾 输出文件: oracle_full_content_embeddings.npy")

if __name__ == "__main__":
    main()