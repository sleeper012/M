#!/usr/bin/env python3
"""
用 DeepSeek API 对 inputs 列表并发生成回复，完成一条保存一条。
支持断点续传：如果输出文件已存在，自动跳过已完成的部分。
API key 通过环境变量 DEEPSEEK_API_KEY 传入，不要写进代码。

用法:
  export DEEPSEEK_API_KEY=sk-xxx

  # 跑全部 900 条（并发请求，默认 10 并发）
  python deepseek_api_generate.py --input-json deepseek_r1_7b_val_inputs.json --output deepseek_api_output.jsonl

  # 调整并发数（根据你的 API 限制调整）
  python deepseek_api_generate.py --input-json deepseek_r1_7b_val_inputs.json --output deepseek_api_output.jsonl --concurrency 20

  # 如果中途断了，重新跑会自动续传
  python deepseek_api_generate.py --input-json deepseek_r1_7b_val_inputs.json --output deepseek_api_output.jsonl --resume

  # 最后把 JSONL 转成普通 JSON（如果需要）
  python deepseek_api_generate.py --convert-jsonl-to-json deepseek_api_output.jsonl --output deepseek_api_output.json
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import aiohttp

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


async def call_deepseek(
    session: aiohttp.ClientSession,
    user_content: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }
    async with session.post(
        DEEPSEEK_API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


def load_existing_results(output_path: Path) -> tuple[list, int]:
    """加载已有结果，返回 (results_list, completed_count)"""
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
    except Exception as e:
        print(f"读取已有文件出错: {e}，将从头开始")
        return [], 0
    
    return results, count


async def process_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    user_input: str,
    api_key: str,
    model: str,
    index: int,
    total: int,
    out_path: Path,
) -> None:
    """处理单个请求并写入文件（带并发控制）"""
    async with semaphore:
        print(f"[{index+1}/{total}] 请求中...")
        try:
            content = await call_deepseek(session, user_input, api_key, model=model)
            result = {"input": user_input, "output": content, "index": index}
        except Exception as e:
            print(f"  [{index+1}/{total}] 错误: {e}")
            result = {"input": user_input, "output": "", "error": str(e), "index": index}

        # 写入文件（使用锁保证线程安全）
        async with file_lock:
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

        # 每 10 条打印进度
        if (index + 1) % 10 == 0:
            print(f"  已完成 {index+1}/{total} 条")


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", default="deepseek_r1_7b_val_inputs.json", help="输入的 input 列表 JSON")
    parser.add_argument("--output", default="deepseek_api_output.jsonl", help="输出 JSONL 路径（每行一个 JSON）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（不指定则全部）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek 模型名")
    parser.add_argument("--resume", action="store_true", help="断点续传模式（跳过已完成的）")
    parser.add_argument("--concurrency", type=int, default=10, help="并发请求数（默认 10）")
    parser.add_argument("--convert-jsonl-to-json", type=str, help="将 JSONL 文件转换为普通 JSON 数组")
    args = parser.parse_args()

    # 转换模式：JSONL -> JSON
    if args.convert_jsonl_to_json:
        jsonl_path = Path(args.convert_jsonl_to_json)
        output_path = Path(args.output) if args.output else jsonl_path.with_suffix(".json")
        results = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已转换 {len(results)} 条，保存到: {output_path}")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请设置环境变量: export DEEPSEEK_API_KEY=sk-xxx")

    # 加载输入
    with open(args.input_json, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    if not isinstance(inputs, list):
        inputs = [inputs]
    if args.limit:
        inputs = inputs[: args.limit]

    total = len(inputs)
    print(f"共 {total} 条，调用模型: {args.model}，并发数: {args.concurrency}")

    # 准备输出文件（JSONL 格式，逐行追加）
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # 创建信号量和文件锁
    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()

    # 创建所有任务
    tasks = []
    async with aiohttp.ClientSession() as session:
        for i in range(start_idx, total):
            task = process_one(
                session, semaphore, file_lock, inputs[i], api_key, args.model, i, total, out_path
            )
            tasks.append(task)

        # 并发执行所有任务
        await asyncio.gather(*tasks)

    print(f"\n完成！共处理 {total - start_idx} 条新数据")
    print(f"结果保存在: {out_path} (JSONL 格式，每行一条)")
    print(f"\n如需转换为普通 JSON 数组，运行:")
    print(f"  python deepseek_api_generate.py --convert-jsonl-to-json {out_path} --output {out_path.with_suffix('.json')}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
