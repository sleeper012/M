"""
直接对已有 JSON 数据做法条链接后处理校验（[LAW]...[/LAW] 提取、匹配、修正）。

用法示例：
  # 处理 RAG 结果（默认校验 rewritten_response）
  python batch_validate_law_refs.py --input results_deepseek_r1_7b_rl_0224/enhanced_dialogue_deepseek_r1_7b_rl_0224.json

  # 指定要校验的字段，输出到新文件
  python batch_validate_law_refs.py --input 0225-rl-test-no-api/300step_output.json --keys output,generated_output -o 300step_validated.json

  # 指定知识库路径
  python batch_validate_law_refs.py --input data.json --laws structured_laws_by_category.json -o out.json
"""

import argparse
import json
import os
import sys

# 项目内
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from law_reference_validator import LawReferenceValidator, DEFAULT_STRUCTURED_LAWS_PATH


def main():
    parser = argparse.ArgumentParser(description="对 JSON 数据中指定字段做法条 [LAW]...[/LAW] 校验与修正")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 文件路径（数组或含列表的对象）")
    parser.add_argument("--output", "-o", default=None, help="输出 JSON 文件路径（默认：输入路径加 _law_validated 后缀）")
    parser.add_argument("--keys", "-k", default="rewritten_response", help="要校验的字段名，逗号分隔，如: rewritten_response 或 output,generated_output")
    parser.add_argument("--laws", "-l", default=DEFAULT_STRUCTURED_LAWS_PATH, help="法条知识库 JSON 路径（structured_laws_by_category.json）")
    parser.add_argument("--unmatched", default="replace_with_not_found", choices=["replace_with_not_found", "replace_with_title", "remove"],
                        help="匹配不到时的处理：replace_with_not_found | replace_with_title | remove")
    parser.add_argument("--meta-key", default="law_validation", help="审计信息写入的字段名（按 key 存，如 law_validation.rewritten_response）")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 {input_path}")
        sys.exit(1)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys:
        print("错误：至少指定一个 --keys")
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_law_validated" + (ext or ".json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 统一成 list of dict
    list_key = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for cand in ("data", "results", "items", "list"):
            if cand in data and isinstance(data[cand], list):
                items = data[cand]
                list_key = cand
                break
        else:
            items = [data]
    else:
        print("错误：输入 JSON 须为数组或对象（含 data/results/items 列表）")
        sys.exit(1)

    validator = LawReferenceValidator(structured_laws_path=args.laws)
    total_refs = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        if args.meta_key not in item or not isinstance(item.get(args.meta_key), dict):
            item[args.meta_key] = {}

        for key in keys:
            text = item.get(key)
            if not text or not isinstance(text, str):
                continue
            corrected, audit = validator.validate_and_fix(text, unmatched_mode=args.unmatched)
            item[key] = corrected
            item[args.meta_key][key] = audit
            total_refs += len(audit)

    # 写回：保持输入结构
    if isinstance(data, list):
        out_data = items
    else:
        out_data = {**data}
        if list_key is not None:
            out_data[list_key] = items

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"已处理 {len(items)} 条，共 {total_refs} 处 [LAW] 引用，结果已写入: {output_path}")


if __name__ == "__main__":
    main()
