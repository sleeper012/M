#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题多类别分类脚本
- 支持一个问题归属多个类别
- 并发请求 API（可配置 CONCURRENT_REQUESTS），批量加速
- 每50条自动保存一次（断点续跑）
- 支持批量处理多个输入文件
- 运行中断后重新运行会自动跳过已处理的问题
- 输入可为纯问题列表 [str, ...] 或 [{"question","category"}, ...]
"""

import json
import os
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ======================== 配置区 ========================

# 单文件模式：指定一个输入文件
SINGLE_INPUT_FILE = "/home/linux/Mdata/data/0224-rl-1k/question-rl.json"

# 批量模式：指定一个目录，处理目录下所有 *_extracted.json 文件
# 设为 None 则使用单文件模式
BATCH_INPUT_DIR = None
# BATCH_INPUT_DIR = "/home/linux/Mdata/apimakequestion/"

# 输出目录
OUTPUT_DIR = "/home/linux/Mdata/apimakequestion/classified/Rl/"

# 每处理多少条保存一次
SAVE_INTERVAL = 50

# API 配置
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d50cf9f56866483ba69f1d5d12b9d177")
API_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-reasoner"

# API 失败重试次数
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

# 并发数：同时发起的 API 请求数量（建议 5~10，过大可能触发限流）
CONCURRENT_REQUESTS = 8

# ========================================================

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """你是一位专业的军事法律与国防政策分析专家。你的任务是将给定问题归类到以下六个类别中（可以同时属于多个类别），并给出每个所属类别涉及的具体详细法律条款和内容。

【分类体系】
一、开战规则 [武力发动的合法性规则（联合国宪章的一部分）]
二、交战规则 [武力使用的合法性规则（涉及人道法）]
三、国家安全与主权维护
四、军队组织与管理类
五、国防建设与后备力量类
六、网络安全与技术合规 [人工智能本身约束]

【分类说明】
- 一、开战规则：涉及战争发动的合法性、自卫权、联合国授权、侵略定义等（《联合国宪章》第2条第4款、第51条、《侵略定义决议》3314号等）
- 二、交战规则：涉及作战中的人道主义法律约束、战俘待遇、平民保护、比例原则、区分原则等（日内瓦公约及附加议定书、海牙公约等）
- 三、国家安全与主权维护：涉及情报、反间谍、领土主权、国家秘密保护、外国势力渗透等
- 四、军队组织与管理类：涉及军队编制、人事管理、纪律条令、军事训练、指挥体系等
- 五、国防建设与后备力量类：涉及国防动员、预备役、军工生产、国防教育、战略储备等
- 六、网络安全与技术合规：涉及网络攻击、数据安全、AI伦理约束、信息战、技术出口管制等

【重要】一个问题可以同时归属多个类别，请全部列出。

【输出格式】请严格按照以下JSON格式输出，不要有任何额外文字：
{
  "categories": [
    {
      "category_number": "一/二/三/四/五/六（选其一）",
      "category_name": "类别名称",
      "reasoning": "该问题归属此类别的理由（1-2句话）",
      "laws_and_regulations": [
        {
          "law_name": "法律/条约/公约名称",
          "specific_articles": "具体条款（如：第X条第X款）",
          "content_summary": "该条款的具体内容和与本问题的关联"
        }
      ]
    }
  ],
  "overall_summary": "对该问题的整体分析摘要（2-3句话）"
}
"""


def extract_json(content: str) -> str:
    """从模型响应中提取JSON字符串"""
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    # 找到第一个 { 和最后一个 }
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        content = content[start:end+1]
    return content


def classify_question(question: str, original_category: str) -> dict:
    """调用 DeepSeek API 对问题进行多类别分类，支持重试"""
    user_message = f"""请对以下问题进行多类别分类分析：

问题：{question}

（原始标注类别供参考：{original_category}）

请判断该问题属于哪些类别（可多选），并给出每个类别涉及的具体详细法律法规内容。"""

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                stream=False
            )

            content = response.choices[0].message.content
            json_str = extract_json(content)
            result = json.loads(json_str)

            if "categories" not in result:
                raise ValueError("响应中缺少 'categories' 字段")

            return result

        except json.JSONDecodeError as e:
            last_error = e
            print(f"    [第{attempt}次] JSON解析失败: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except Exception as e:
            last_error = e
            print(f"    [第{attempt}次] 调用失败: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return {
        "categories": [],
        "overall_summary": f"处理失败（{MAX_RETRIES}次重试后）: {str(last_error)}",
        "error": str(last_error)
    }


def _normalize_item(item) -> tuple:
    """支持输入为纯问题字符串或 {question, category} 的 dict"""
    if isinstance(item, str):
        return (item, "")
    return (item.get("question", ""), item.get("category", ""))


def load_existing_results(output_file: str):
    """加载已有结果，返回 (results列表, 已处理的question集合)"""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed = {r["question"] for r in results}
            print(f"  检测到已有结果，已处理 {len(results)} 条，将断点续跑")
            return results, processed
        except Exception as e:
            print(f"  读取已有结果失败({e})，将从头开始")
    return [], set()


def save_results(results: list, output_file: str):
    """原子保存，防止写入中途崩溃损坏文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tmp_file = output_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, output_file)


def print_stats(results: list):
    """打印分类统计"""
    print("\n=== 分类统计 ===")
    category_counts = {}
    multi_count = 0

    for r in results:
        cats = r["classification"].get("categories", [])
        if len(cats) > 1:
            multi_count += 1
        for cat in cats:
            key = f"{cat.get('category_number','?')}、{cat.get('category_name','?')}"
            category_counts[key] = category_counts.get(key, 0) + 1

    for key, count in sorted(category_counts.items()):
        print(f"  {key}: {count} 次")
    print(f"  涉及多类别的问题: {multi_count} / {len(results)} 条")


def process_file(input_file: str, output_file: str):
    """处理单个输入文件（并发请求 API，批量加速）"""
    print(f"\n{'='*60}")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    if not isinstance(raw_list, list):
        raw_list = [raw_list]
    print(f"共 {len(raw_list)} 个问题，并发数: {CONCURRENT_REQUESTS}")

    # 断点续跑
    results, processed_questions = load_existing_results(output_file)

    # 构建待处理列表 [(原始下标, question, original_category), ...]
    pending = []
    for i, item in enumerate(raw_list):
        question, original_category = _normalize_item(item)
        if not question or question in processed_questions:
            continue
        pending.append((i, question, original_category))

    if not pending:
        print("没有需要新处理的问题。")
        save_results(results, output_file)
        print_stats(results)
        return

    new_count = 0
    # 按批并发：每批最多 CONCURRENT_REQUESTS 个
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        future_to_item = {}
        for idx, question, original_category in pending:
            future = executor.submit(classify_question, question, original_category)
            future_to_item[future] = (idx, question, original_category)

        done = 0
        for future in as_completed(future_to_item):
            idx, question, original_category = future_to_item[future]
            done += 1
            try:
                classification = future.result()
            except Exception as e:
                classification = {
                    "categories": [],
                    "overall_summary": f"并发执行异常: {str(e)}",
                    "error": str(e),
                }
            results.append({
                "id": idx + 1,
                "question": question,
                "original_category": original_category,
                "classification": classification,
            })
            processed_questions.add(question)
            new_count += 1

            cats = classification.get("categories", [])
            labels = [f"{c.get('category_number','?')}、{c.get('category_name','?')}" for c in cats]
            print(f"[{done}/{len(pending)}] {question[:50]}... → {len(cats)}个类别: {' | '.join(labels) if labels else '无'}")

            if new_count % SAVE_INTERVAL == 0:
                # 按 id 排序后再保存，保证顺序一致
                results_sorted = sorted(results, key=lambda r: r["id"])
                save_results(results_sorted, output_file)
                print(f"  ✓ 自动保存（累计 {len(results)} 条）")

    # 按 id 排序后最终保存
    results_sorted = sorted(results, key=lambda r: r["id"])
    save_results(results_sorted, output_file)
    print(f"\n✓ 完成！共 {len(results_sorted)} 条，本次新增 {new_count} 条")
    print_stats(results_sorted)


def get_output_path(input_file: str) -> str:
    """根据输入文件名生成输出路径"""
    basename = os.path.basename(input_file)
    name = os.path.splitext(basename)[0].replace("_extracted", "")
    return os.path.join(OUTPUT_DIR, f"{name}_classified.json")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if BATCH_INPUT_DIR:
        # 批量模式
        pattern = os.path.join(BATCH_INPUT_DIR, "*_extracted.json")
        input_files = sorted(glob.glob(pattern))
        if not input_files:
            print(f"在 {BATCH_INPUT_DIR} 下未找到 *_extracted.json 文件")
            return
        print(f"批量模式：找到 {len(input_files)} 个文件")
        for f in input_files:
            process_file(f, get_output_path(f))
        print(f"\n全部 {len(input_files)} 个文件处理完成！")
    else:
        # 单文件模式
        process_file(SINGLE_INPUT_FILE, get_output_path(SINGLE_INPUT_FILE))


if __name__ == "__main__":
    main()
