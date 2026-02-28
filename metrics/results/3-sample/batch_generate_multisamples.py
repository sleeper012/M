"""
批量生成脚本 - 专门用于 Self-BLEU 测试
每个问题生成多次（2-3次），使用相同的采样参数但允许随机性

输出格式:
{
    "question": "问题文本",
    "generate": ["生成的答案1", "生成的答案2", "生成的答案3"]
}
"""

import json
import os
import argparse
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ============== 配置区域 ==============

# 测试集路径
TEST_DATA_PATH = "/home/linux/Mdata/metrics/deepseek_r1_7b_val.json"

# 模型配置
MODELS = {
    "base": {
        "name": "Base Model",
        "path": "/home/linux/Mdata/model/DeepSeek-R1-Distill-Qwen-7B",
        "output_file": "/home/linux/Mdata/metrics/results/base_generations_multi.json"
    },
    "sft_only": {
        "name": "SFT Only",
        "path": "/home/linux/Mdata/lf/models/0223-7b-1200/300step",
        "output_file": "/home/linux/Mdata/metrics/results/sft_only_generations_multi.json"
    },
    "rl_only": {
        "name": "RL Only",
        "path": "/home/linux/Mdata/lf/models/rl-only/200step",
        "output_file": "/home/linux/Mdata/metrics/results/rl_only_generations_multi.json"
    },
    "sft_rl": {
        "name": "SFT + RL",
        "path": "/home/linux/Mdata/lf/models/sft-rl/200step",
        "output_file": "/home/linux/Mdata/metrics/results/sft_rl_generations_multi.json"
    },
    "rag": {
        "name": "RAG",
        "path": "/home/linux/Mdata/model/DeepSeek-R1-Distill-Qwen-7B",
        "output_file": "/home/linux/Mdata/metrics/results/rag_generations_multi.json"
    }
}

# 生成参数 - 使用较高的 temperature 以产生多样性
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,  # 较高的温度以产生不同的回答
    top_p=0.9,
    max_tokens=2048,
    repetition_penalty=1.05,
    n=3,  # 每个 prompt 生成 3 个不同的回答
)

# 批处理大小
BATCH_SIZE = 16  # 因为每个问题要生成多次，减小批次大小

# ============== 配置区域结束 ==============


def load_test_data(file_path: str) -> List[Dict]:
    """加载测试数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_prompt(question: str) -> str:
    """构建模型输入的 prompt"""
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def batch_generate_multi(
    llm: LLM,
    questions: List[str],
    num_samples: int = 3,
    batch_size: int = 16
) -> List[List[str]]:
    """
    批量生成答案 - 每个问题生成多个样本
    
    返回: List[List[str]] - 外层列表对应每个问题，内层列表包含该问题的多个答案
    """
    all_outputs = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc="生成中"):
        batch = questions[i:i + batch_size]
        prompts = [build_prompt(q) for q in batch]
        
        # vLLM 会为每个 prompt 生成 n 个样本
        outputs = llm.generate(prompts, SAMPLING_PARAMS)
        
        for output in outputs:
            # 获取该问题的所有生成样本
            samples = [o.text for o in output.outputs]
            all_outputs.append(samples)
    
    return all_outputs


def run_model_inference(
    model_key: str,
    test_data: List[Dict],
    num_samples: int = 3,
    batch_size: int = 16
):
    """对单个模型运行推理"""
    model_config = MODELS[model_key]
    model_name = model_config["name"]
    model_path = model_config["path"]
    output_file = model_config["output_file"]
    
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"路径: {model_path}")
    print(f"输出: {output_file}")
    print(f"每个问题生成样本数: {num_samples}")
    print(f"{'='*60}")
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在 - {model_path}")
        return None
    
    # 加载模型
    print("加载模型...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,  # 稍微降低以应对多样本生成
        max_model_len=8192,
    )
    
    # 提取问题
    questions = [item.get("input", item.get("question", "")) for item in test_data]
    print(f"问题数量: {len(questions)}")
    
    # 批量生成（每个问题多个样本）
    print("开始生成...")
    outputs = batch_generate_multi(llm, questions, num_samples, batch_size)
    
    # 组装结果
    results = []
    for item, sample_outputs in zip(test_data, outputs):
        question = item.get("input", item.get("question", ""))
        results.append({
            "question": question,
            "generate": sample_outputs  # 这是一个列表，包含多个生成的答案
        })
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")
    print(f"总样本数: {len(results) * num_samples}")
    
    # 释放显存
    del llm
    import torch
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="批量生成模型答案（多样本用于 Self-BLEU）")
    parser.add_argument("--models", "-m", type=str, nargs="+", 
                        default=["base"],
                        choices=["base", "sft_only", "rl_only", "sft_rl", "rag", "all"],
                        help="要测试的模型 (默认: base)")
    parser.add_argument("--num_samples", "-n", type=int, default=3,
                        help="每个问题生成的样本数 (默认: 3)")
    parser.add_argument("--batch_size", "-b", type=int, default=BATCH_SIZE,
                        help=f"批处理大小 (默认: {BATCH_SIZE})")
    parser.add_argument("--test_data", "-t", type=str, default=TEST_DATA_PATH,
                        help="测试数据路径")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度 (默认: 0.7)")
    
    args = parser.parse_args()
    
    # 更新 SAMPLING_PARAMS
    global SAMPLING_PARAMS
    SAMPLING_PARAMS = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=2048,
        repetition_penalty=1.05,
        n=args.num_samples,
    )
    
    # 处理 "all" 选项
    if "all" in args.models:
        args.models = ["base", "sft_only", "rl_only", "sft_rl", "rag"]
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_data}")
    test_data = load_test_data(args.test_data)
    print(f"测试样本数: {len(test_data)}")
    print(f"每个问题将生成 {args.num_samples} 个样本")
    print(f"总样本数: {len(test_data) * args.num_samples}")
    
    # 依次运行每个模型
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"警告: 未知模型 {model_key}, 跳过")
            continue
        
        try:
            run_model_inference(model_key, test_data, args.num_samples, args.batch_size)
        except Exception as e:
            print(f"错误: 处理模型 {model_key} 时出错 - {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n所有模型处理完成!")


if __name__ == "__main__":
    main()
