"""
从生成结果中提取 </think>\n\n 后面的内容作为回答
处理 base_generations.json, sft_only_generations.json, sft_rl_generations.json
排除 rag_generations.json
"""

import json
import os
import re

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

FILES_TO_PROCESS = [
    "base_generations.json",
    "sft_only_generations.json", 
    "sft_rl_generations.json",
]

def extract_answer(text: str) -> str:
    """提取 </think>\n\n 后面的内容作为回答"""
    if not text:
        return text
    
    # 尝试多种 think 标记格式
    patterns = [
        r'</think>\s*\n\s*\n',  # </think>\n\n
        r'</think>\s*\n',        # </think>\n
        r'</think>',             # </think>
        r'<｜think▁end｜>',      # DeepSeek 格式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[match.end():].strip()
    
    # 如果没有找到 think 标记，返回原文
    return text.strip()


def process_file(filename: str):
    """处理单个文件"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    print(f"\n处理文件: {filename}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    modified = 0
    
    for item in data:
        if "generate" in item:
            original = item["generate"]
            extracted = extract_answer(original)
            if extracted != original:
                item["generate"] = extracted
                modified += 1
    
    # 保存回原文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  总条目: {total}")
    print(f"  已修改: {modified}")
    print(f"  ✓ 已保存")


def main():
    print("=" * 60)
    print("提取 </think> 后的回答内容")
    print("=" * 60)
    
    for filename in FILES_TO_PROCESS:
        process_file(filename)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
