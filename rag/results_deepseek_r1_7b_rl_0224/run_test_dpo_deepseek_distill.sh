#!/bin/bash
# 用 DeepSeek-R1-Distill-Qwen-7B 测试 DPO 数据，生成测试集
# 需先激活 vllm 环境: conda activate vllm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_PATH="${SCRIPT_DIR}/dpo_rl_0224_300step_reject.json"
OUTPUT_PATH="${SCRIPT_DIR}/dpo_test_deepseek_distill_7b.json"
MODEL_PATH="/home/linux/Mdata/model/DeepSeek-R1-Distill-Qwen-7B"

# 全量测试（约 1.9w 条，耗时较长）
# python "${SCRIPT_DIR}/test_dpo_deepseek_distill.py" \
#     --dpo_path "$DPO_PATH" \
#     --output_path "$OUTPUT_PATH" \
#     --model_path "$MODEL_PATH"

# 默认：只测前 500 条，便于快速验证
python "${SCRIPT_DIR}/test_dpo_deepseek_distill.py" \
    --dpo_path "$DPO_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_path "$MODEL_PATH" \
    --max_samples 500

echo "测试集已写入: $OUTPUT_PATH"
