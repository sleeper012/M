#!/usr/bin/env python3
"""
用 HuggingFace 上的 Beaver Safe-RLHF 模型，对输入列表逐条生成回复，完成一条保存一条。

特点：
- 只依赖 transformers，不依赖 safe-rlhf / deepspeed（避免 Python 3.11 安装问题）
- 支持两种输入格式：字符串数组或对象数组（自动提取 "input" 字段）
- 输出为 JSONL，每行一个 {"input": ..., "output": ..., "index": ...}

示例用法（在项目根目录 /home/linux/Mdata 下）：

    # 安装依赖（如果需要）
    # pip install "transformers>=4.40" accelerate sentencepiece

    # 使用对象数组格式（如 deepseek_r1_7b_val.json）
    python metrics/hf_beaver_generate.py \
        --model-name alignment-handbook/mistral-7b-sft-constitutional-ai \
        --input-json metrics/deepseek_r1_7b_val.json \
        --output metrics/mistral_cai_output.jsonl \
        --resume

    # 或使用字符串数组格式（如 deepseek_r1_7b_val_inputs.json）
    python metrics/hf_beaver_generate.py \
        --model-name PKU-Alignment/beaver-7b-v3.0 \
        --input-json metrics/deepseek/deepseek_r1_7b_val_inputs.json \
        --output metrics/beaver/beaver_7b_v3_output.jsonl

注意：
- Beaver-7B 模型体积较大，推荐在有 GPU 的机器上运行。
- 如果只有 CPU，transformers 仍可加载，但推理会非常慢且可能内存不足。
- 支持 --resume 参数进行断点续传。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_existing_results(output_path: Path) -> tuple[list, int]:
    """加载已有 JSONL 结果，返回 (results_list, completed_count)。"""
    if not output_path.exists():
        return [], 0

    results = []
    count = 0
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    results.append(obj)
                    count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:  # noqa: BLE001
        print(f"读取已有文件出错: {e}，将从头开始")
        return [], 0

    return results, count


def build_prompt(user_input: str) -> str:
    """按 Beaver model card 中的模板构造 prompt。"""
    return f"BEGINNING OF CONVERSATION: USER: {user_input} ASSISTANT:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="PKU-Alignment/beaver-7b-v3.0",
        help="HuggingFace 模型名或本地路径，例如 PKU-Alignment/beaver-7b-v3.0",
    )
    parser.add_argument(
        "--input-json",
        default="metrics/deepseek/deepseek_r1_7b_val_inputs.json",
        help="输入的 prompt 列表 JSON（数组，每个元素为字符串）",
    )
    parser.add_argument(
        "--output",
        default="metrics/beaver/beaver_7b_v3_output.jsonl",
        help="输出 JSONL 路径（每行一个 JSON）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 条（不指定则全部）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="每条样本生成的最大新 token 数",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传模式（跳过已完成的）",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载输入
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理数据格式：支持字符串数组或对象数组
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            # 对象数组格式：提取 "input" 字段
            inputs = [item.get("input", "") if isinstance(item, dict) else str(item) for item in data]
        else:
            # 字符串数组格式
            inputs = [str(item) for item in data]
    else:
        inputs = [str(data)]

    if args.limit:
        inputs = inputs[: args.limit]

    total = len(inputs)
    print(f"共 {total} 条，将使用 HuggingFace 模型: {args.model_name}")

    # 断点续传：检查已完成的数量
    start_idx = 0
    if args.resume and out_path.exists():
        _, completed = load_existing_results(out_path)
        if completed > 0:
            print(f"发现已有 {completed} 条结果，继续从第 {completed + 1} 条开始")
            start_idx = completed

    # 如果是新建文件（非续传模式或文件不存在），创建空文件
    if not (args.resume and out_path.exists()):
        out_path.write_text("", encoding="utf-8")

    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载 tokenizer 和模型
    print("正在从 HuggingFace 加载模型（可能需要较长时间）...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)

    model.eval()
    print("模型加载完成，开始生成...")

    for i in range(start_idx, total):
        user_input = inputs[i]
        print(f"[{i+1}/{total}] 生成中...")

        prompt = build_prompt(str(user_input))
        inputs_enc = tokenizer(
            prompt,
            return_tensors="pt",
        )
        input_ids = inputs_enc["input_ids"].to(device)
        attention_mask = inputs_enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # 评测用确定性输出
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 从完整对话中截出 ASSISTANT 之后的部分
        split_key = "ASSISTANT:"
        if split_key in full_text:
            response = full_text.split(split_key, 1)[1].strip()
        else:
            # 兜底：直接用全句
            response = full_text

        result = {"input": user_input, "output": response, "index": i}

        # 立即写入文件并 flush
        with open(out_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

        # 每 10 条打印进度
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{total} 条")

    print(f"\n完成！共处理 {total - start_idx} 条新数据")
    print(f"结果保存在: {out_path} (JSONL 格式，每行一条)")


if __name__ == "__main__":
    main()

