#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 DeepSeek-R1-Distill-Qwen-7B 对 DPO 数据（dpo_rl_0224_300step_reject.json）做推理，
构成测试集：每条包含 question、chosen、rejected、model_output，便于后续评估。

用法（需 vllm 环境）:
  conda activate vllm
  python rag/results_deepseek_r1_7b_rl_0224/test_dpo_deepseek_distill.py
  # 只测前 500 条：
  python rag/results_deepseek_r1_7b_rl_0224/test_dpo_deepseek_distill.py --max_samples 500
"""

import json
import os
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

# 与 make_dpo_300step_reject.py 一致的路径与格式
DPO_PATH = "/home/linux/Mdata/rag/results_deepseek_r1_7b_rl_0224/dpo_rl_0224_300step_reject.json"
MODEL_PATH = "/home/linux/Mdata/model/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_PATH = "/home/linux/Mdata/rag/results_deepseek_r1_7b_rl_0224/dpo_test_deepseek_distill_7b.json"
BATCH_SIZE = 16
MAX_TOKENS = 2048


def build_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser(description="用 DeepSeek-R1-Distill-Qwen-7B 测试 DPO 数据，生成测试集")
    parser.add_argument("--dpo_path", type=str, default=DPO_PATH, help="DPO JSON 路径")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="模型路径")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="测试集输出路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最多测试条数，不设则全量")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="vLLM 批大小")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="GPU 数量")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    expected = "/home/linux/Mdata/model/DeepSeek-R1-Distill-Qwen-7B"
    if model_path.rstrip("/") != expected.rstrip("/"):
        print(f"警告: 当前 model_path={model_path}，若要用基座请设为 {expected}")
    print("加载 DPO 数据...")
    with open(args.dpo_path, "r", encoding="utf-8") as f:
        dpo_list = json.load(f)

    n_total = len(dpo_list)
    if args.max_samples is not None:
        dpo_list = dpo_list[: args.max_samples]
    n = len(dpo_list)
    print(f"共 {n_total} 条 DPO，本次测试 {n} 条")

    questions = []
    chosen_list = []
    rejected_list = []
    for item in dpo_list:
        conv = item["conversations"]
        q = conv[0]["value"] if conv and conv[0].get("from") == "human" else ""
        questions.append(q)
        chosen_list.append(item["chosen"].get("value", ""))
        rejected_list.append(item["rejected"].get("value", ""))

    print("加载 vLLM 模型（必须是 DeepSeek-R1-Distill-Qwen-7B）:", model_path)
    llm = LLM(
        model=model_path,
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

    print("批量生成 model_output...")
    model_outputs = []
    for i in tqdm(range(0, n, args.batch_size), desc="Generating"):
        batch = prompts[i : i + args.batch_size]
        outs = llm.generate(batch, sampling_params)
        for o in outs:
            model_outputs.append(o.outputs[0].text.strip())

    # 构成测试集：与 DPO 结构兼容，并增加 model_output
    test_list = []
    for q, chosen, rejected, model_out in zip(questions, chosen_list, rejected_list, model_outputs):
        test_list.append({
            "conversations": [{"from": "human", "value": q}],
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": rejected},
            "model_output": {"from": "gpt", "value": model_out},
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=2)
    print(f"已保存测试集: {args.output_path} ({len(test_list)} 条)")


if __name__ == "__main__":
    main()
