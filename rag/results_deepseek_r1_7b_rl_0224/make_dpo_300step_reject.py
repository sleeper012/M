#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 enhanced_dialogue 提取问题，用 300step 模型批量生成作为 rejected，
rewritten_response 作为 chosen，输出 DPO 格式（与 dpo_en_demo.json 一致）。

用法（需 vllm 环境）:
  conda activate vllm
  python rag/results_deepseek_r1_7b_rl_0224/make_dpo_300step_reject.py
"""

import json
import os
from vllm import LLM, SamplingParams

ENHANCED_PATH = "/home/linux/Mdata/rag/results_deepseek_r1_7b_rl_0224/enhanced_dialogue_deepseek_r1_7b_rl_0224.json"
MODEL_PATH = "/home/linux/Mdata/lf/models/0223-7b-1200/300step"
OUTPUT_PATH = "/home/linux/Mdata/rag/results_deepseek_r1_7b_rl_0224/dpo_rl_0224_300step_reject.json"
BATCH_SIZE = 16
MAX_TOKENS = 2048


def build_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def main():
    print("加载 enhanced_dialogue...")
    with open(ENHANCED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    chosen_texts = [item["rewritten_response"] for item in data]
    n = len(questions)
    print(f"共 {n} 条，准备用 300step 生成 rejected...")

    print("加载 vLLM 模型:", MODEL_PATH)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=8192,
    )
    prompts = [build_prompt(q) for q in questions]
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    print("批量生成 rejected...")
    rejected_texts = []
    for i in range(0, n, BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        outs = llm.generate(batch, sampling_params)
        for o in outs:
            rejected_texts.append(o.outputs[0].text.strip())
        print(f"  已处理 {min(i + BATCH_SIZE, n)}/{n}")

    # DPO 格式（与 dpo_en_demo.json 一致）
    dpo_list = []
    for q, chosen, rejected in zip(questions, chosen_texts, rejected_texts):
        dpo_list.append({
            "conversations": [{"from": "human", "value": q}],
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": rejected},
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dpo_list, f, ensure_ascii=False, indent=2)
    print(f"已保存 DPO 数据: {OUTPUT_PATH} ({len(dpo_list)} 条)")


if __name__ == "__main__":
    main()
