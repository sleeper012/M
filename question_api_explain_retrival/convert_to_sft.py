import json

input_path = "/home/linux/Mdata/rag/results_deepseek_r1_7b/enhanced_dialogue_deepseek_r1_7b.json"
output_path = "/home/linux/Mdata/rag/results_deepseek_r1_7b/sft_deepseek_r1_7b.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sft = []
for item in data:
    sft.append({
        "instruction": "",
        "input": item["question"],
        "output": item["rewritten_response"]
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sft, f, ensure_ascii=False, indent=2)

print(f"转换完成，共 {len(sft)} 条数据，已保存至 {output_path}")
