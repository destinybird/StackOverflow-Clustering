import json
from bs4 import BeautifulSoup
import re
import pandas as pd

def clean_text(html_text):
    """
    清洗文本：去除 HTML 标签、代码块和多余空白。
    """
    if not html_text:
        return ""
    
    # 使用 lxml 解析器以提高速度（需要安装 lxml 库）
    try:
        soup = BeautifulSoup(html_text, 'lxml')
    except Exception:
        # 降级到默认解析器，以防 lxml 遇到问题
        soup = BeautifulSoup(html_text, 'html.parser')
    
    # 移除所有代码块（Stack Overflow 内容中的代码通常在 <pre><code>...</code></pre> 中）
    for code in soup.find_all('code'):
        code.decompose()
    for pre in soup.find_all('pre'):
        pre.decompose()

    # 获取纯文本
    text = soup.get_text()
    
    # 清理特殊字符和多余空白
    text = re.sub(r'\s+', ' ', text)  # 将多个空格、换行、制表符替换为单个空格
    text = text.strip()
    return text

def extract_and_clean_data(file_path):
    print(f"开始读取文件: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    processed_data = []
    total_count = len(data)
    print(f"总共 {total_count} 条问题数据。")

    for i, item in enumerate(data):
        # 确保数据点包含必要的字段
        if "question_id" not in item or "title" not in item or "body" not in item:
            continue
            
        question_id = item["question_id"]
        title = item["title"]
        body = item["body"]
        
        # 清洗 body
        cleaned_body = clean_text(body)
        
        # 拼接标题和清洗后的内容作为聚类文本
        full_text = title + " [SEP] " + cleaned_body 
        # 使用 [SEP] 分隔符，在后续使用 SBERT 等模型时有助于区分标题和正文
        
        processed_data.append({
            "id": question_id,
            "text": full_text,
            "title": title # 保留标题，方便后续分析
        })
        
        # 打印进度 (可选)
        if (i + 1) % 10000 == 0:
            print(f"已处理 {i + 1} / {total_count} 条数据...")

    print("数据预处理完成。")
    return pd.DataFrame(processed_data)

if __name__ == "__main__":
    # 请将文件名替换为你的实际路径
    json_file_path = "oracle_database_questions.json" 
    
    # 确保脚本在你的数据文件目录下运行
    df_processed = extract_and_clean_data(json_file_path)
    
    if df_processed is not None:
        # 将清洗后的数据保存为 CSV 文件，便于后续步骤直接加载
        output_file = "cleaned_questions_for_clustering.csv"
        df_processed.to_csv(output_file, index=False, encoding='utf-8')
        print(f"数据已保存到 {output_file}，共 {len(df_processed)} 条记录。")