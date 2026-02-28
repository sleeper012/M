#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 enhanced_dialogue 提取问题与 chosen（rewritten_response），
用【未经过任何训练、无法条 RAG】的基座 DeepSeek-R1-Distill-Qwen-7B 直接回答问题，
生成的内容作为 rejected，输出 DPO 训练格式。

与 make_dpo_300step_reject.py 的区别：rejected 来自纯基座 7B，不是 300step 微调模型。

用法（需 vllm 环境）:
  conda activate vllm
  python rag/results_deepseek_r1_7b_rl_0224/make_dpo_base7b_reject.py
  # 只生成前 N 条：
  python rag/results_deepseek_r1_7b_rl_0224/make_dpo_base7b_reject.py --max_samples 500
"""

import json
import os
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

ENHANCED_PATH = "/home/linux/Mdata/rag/results_deepseek_r1_7b_rl_0224/enhanced_dialogue_deepseek_r1_7b_rl_0224.json"
# 基座模型：未经过任何训练、无 RAG，直接回答
MODEL_PATH = "/home/linux/Mdata/model/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_PATH = "/home/linux/Mdata/rag/results_deepseek_r1_7b_rl_0224/dpo_rl_0224_base7b_reject.json"
BATCH_SIZE = 16
MAX_TOKENS = 2048


def build_prompt(question: str) -> str:
    """只给问题，不加任何法条或 RAG 上下文，让模型直接回答。"""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser(description="用基座 7B（无训练、无 RAG）生成 rejected，构成 DPO 数据")
    parser.add_argument("--enhanced_path", type=str, default=ENHANCED_PATH, help="enhanced_dialogue JSON 路径")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="基座模型路径（必须为未训练、无 RAG）")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="DPO 输出路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最多处理条数，不设则全量")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="vLLM 批大小")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="GPU 数量")
    args = parser.parse_args()

    print("加载 enhanced_dialogue...")
    with open(args.enhanced_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = len(data)
    if args.max_samples is not None:
        data = data[: args.max_samples]
    n = len(data)

    questions = [item["question"] for item in data]
    chosen_texts = [item["rewritten_response"] for item in data]
    print(f"共 {n_total} 条，本次处理 {n} 条")
    print("rejected 将使用【基座 DeepSeek-R1-Distill-Qwen-7B】直接回答（无训练、无 RAG）")

    print("加载 vLLM 模型（基座）:", args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=8192,
    )
    prompts = [build_prompt(q) for q in questions]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    print("批量生成 rejected（基座 7B 直接回答）...")
    rejected_texts = []
    for i in tqdm(range(0, n, args.batch_size), desc="Generating"):
        batch = prompts[i : i + args.batch_size]
        outs = llm.generate(batch, sampling_params)
        for o in outs:
            rejected_texts.append(o.outputs[0].text.strip())

    dpo_list = []
    for q, chosen, rejected in zip(questions, chosen_texts, rejected_texts):
        dpo_list.append({
            "conversations": [{"from": "human", "value": q}],
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": rejected},
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_list, f, ensure_ascii=False, indent=2)
    print(f"已保存 DPO 数据: {args.output_path} ({len(dpo_list)} 条)")
    print("  chosen  = rewritten_response（带法条/合规）")
    print("  rejected = 基座 7B 直接回答（无训练、无 RAG）")


if __name__ == "__main__":
    main()
