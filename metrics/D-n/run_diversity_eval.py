"""
消融实验 - 多样性指标评估脚本
对比不同训练策略的模型生成多样性：
- Base Model (原始模型)
- SFT Only (只做监督微调)
- RL Only (只做强化学习)
- SFT + RL (先SFT后RL)

使用方法:
1. 修改下方的文件路径配置
2. 运行: python run_diversity_eval.py
"""

import os
import sys
from diversity_metrics import (
    load_data, 
    compute_diversity_metrics, 
    format_results, 
    compare_models
)
import json
from datetime import datetime


# ============== 配置区域 - 请修改为你的实际路径 ==============

# 模型生成结果文件配置
# 键: 模型名称 (用于显示)
# 值: 生成结果文件路径
MODEL_FILES = {
    "Base Model": "/home/linux/Mdata/data/path/to/base_model_generations.json",
    "SFT Only": "/home/linux/Mdata/data/path/to/sft_only_generations.json",
    "RL Only": "/home/linux/Mdata/data/path/to/rl_only_generations.json",
    "SFT + RL": "/home/linux/Mdata/data/path/to/sft_rl_generations.json",
}

# 生成文本在JSON中的键名
# 根据你的数据格式，可能是 "generate", "answers", "response" 等
GENERATE_KEY = "generate"  # 或 "answers"

# 语言设置: "zh" (中文) 或 "en" (英文)
LANGUAGE = "zh"

# 输出结果保存路径
OUTPUT_DIR = "/home/linux/Mdata/metrics/results"

# Distinct-n 要计算的 n 值
DISTINCT_N_VALUES = [1, 2, 3, 4]

# Self-BLEU 的 n-gram
SELF_BLEU_N = 4

# ============== 配置区域结束 ==============


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {}
    
    for model_name, file_path in MODEL_FILES.items():
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在，跳过 - {model_name}: {file_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理模型: {model_name}")
        print(f"文件路径: {file_path}")
        print(f"{'='*60}")
        
        try:
            data = load_data(file_path)
            print(f"加载数据: {len(data)} 条记录")
            
            results = compute_diversity_metrics(
                data,
                generate_key=GENERATE_KEY,
                language=LANGUAGE,
                distinct_n_values=DISTINCT_N_VALUES,
                self_bleu_n=SELF_BLEU_N
            )
            
            all_results[model_name] = results
            
            print(format_results(results, model_name))
            
        except Exception as e:
            print(f"错误: 处理 {model_name} 时出错 - {str(e)}")
            import traceback
            traceback.print_exc()
    
    if len(all_results) == 0:
        print("\n错误: 没有成功处理任何模型，请检查文件路径配置")
        return
    
    if len(all_results) > 1:
        comparison = compare_models(all_results)
        print(comparison)
    
    output_file = os.path.join(OUTPUT_DIR, f"diversity_results_{timestamp}.json")
    
    serializable_results = {}
    for model, res in all_results.items():
        serializable_results[model] = {
            "num_samples": res["num_samples"],
            "total_generations": res["total_generations"],
            "distinct_n": res["distinct_n"],
            "self_bleu": res["self_bleu"]
        }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    generate_latex_table(all_results, OUTPUT_DIR, timestamp)


def generate_latex_table(results: dict, output_dir: str, timestamp: str):
    """生成 LaTeX 表格"""
    latex_lines = []
    latex_lines.append("% 多样性指标对比表格")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{模型生成多样性对比}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & Distinct-1 & Distinct-2 & Distinct-3 & Distinct-4 & Self-BLEU \\\\")
    latex_lines.append("\\midrule")
    
    for model, res in results.items():
        d1 = res["distinct_n"].get("distinct_1", {}).get("per_question_avg", 0)
        d2 = res["distinct_n"].get("distinct_2", {}).get("per_question_avg", 0)
        d3 = res["distinct_n"].get("distinct_3", {}).get("per_question_avg", 0)
        d4 = res["distinct_n"].get("distinct_4", {}).get("per_question_avg", 0)
        sb = res["self_bleu"]["per_question_avg"]
        
        latex_lines.append(f"{model} & {d1:.4f} & {d2:.4f} & {d3:.4f} & {d4:.4f} & {sb:.4f} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\label{tab:diversity}")
    latex_lines.append("\\end{table}")
    
    latex_file = os.path.join(output_dir, f"diversity_table_{timestamp}.tex")
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX 表格已保存到: {latex_file}")


if __name__ == "__main__":
    main()
