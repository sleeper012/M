"""
python c.py   --input /home/linux/Mdata/rag/results_deepseek_r1_7b/sft_deepseek_r1_7b.json   --output_train /home/linux/Mdata/rag/results_deepseek_r1_7b/train.json   --output_val /home/linux/Mdata/rag/results_deepseek_r1_7b/val.json

✓ 共加载 2135 条数据
✓ 训练集：1200 条 → /home/linux/Mdata/rag/results_deepseek_r1_7b/train.json
✓ 验证集：900 条 → /home/linux/Mdata/rag/results_deepseek_r1_7b/val.json
✓ 剩余 35 条未使用

"""


import json
import random
import argparse
import os



def load_json_data(file_path):
    """支持 JSON 列表和 JSONL 格式加载"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 判断格式
    if content.startswith('['):
        # JSON 列表格式
        return json.loads(content)
    else:
        # JSONL 格式：每行一个 JSON 对象
        data = []
        for line in content.split('\n'):
            if line.strip():
                data.append(json.loads(line.strip()))
        return data

def save_json_data(data, file_path, jsonl=False):
    """保存为 JSON 列表或 JSONL 格式"""
    # 处理目录创建，防止文件路径中没有目录导致报错
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        if jsonl:
            for item in data:  # 【已修正】这里补上了 data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)

def split_dataset(input_file, train_size=1200, val_size=900, 
                  output_train='train.json', output_val='val.json',
                  jsonl=False, seed=42):
    """
    划分数据集
    """
    # 加载数据
    data = load_json_data(input_file)
    print(f"✓ 共加载 {len(data)} 条数据")
    
    # 检查数据量
    total_needed = train_size + val_size
    if len(data) < total_needed:
        raise ValueError(f"⚠ 数据不足：需要 {total_needed} 条，但只有 {len(data)} 条")
    
    # 随机打乱（固定种子保证可复现）
    random.seed(seed)
    random.shuffle(data)
    
    # 划分
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    
    # 保存
    save_json_data(train_data, output_train, jsonl)
    save_json_data(val_data, output_val, jsonl)
    
    print(f"✓ 训练集：{len(train_data)} 条 → {output_train}")
    print(f"✓ 验证集：{len(val_data)} 条 → {output_val}")
    if len(data) > total_needed:
        print(f"✓ 剩余 {len(data) - total_needed} 条未使用")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JSON 数据集划分工具')
    parser.add_argument('--input', type=str, required=True, help='输入 JSON 文件路径')
    parser.add_argument('--train_size', type=int, default=1200, help='训练集数量')
    parser.add_argument('--val_size', type=int, default=900, help='验证集数量')
    parser.add_argument('--output_train', type=str, default='train.json', help='训练集输出路径')
    parser.add_argument('--output_val', type=str, default='val.json', help='验证集输出路径')
    parser.add_argument('--jsonl', action='store_true', help='以 JSONL 格式输出')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    split_dataset(
        input_file=args.input,
        train_size=args.train_size,
        val_size=args.val_size,
        output_train=args.output_train,
        output_val=args.output_val,
        jsonl=args.jsonl,
        seed=args.seed
    )