"""
使用本地模型的 RAG Pipeline（vLLM 批量推理）
法条检索部分改为直接读取 retrieval_results_v2.json 的预召回结果，
跳过实时向量检索，节省 GPU 显存和时间。
"""

import json
import os
import re
import sys
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from local_model_config import (
    LOCAL_MODEL_PATH,
    QUESTIONS_DATA_PATH,
    OUTPUT_PATH,
    STRUCTURED_LAWS_PATH,
    GENERATION_CONFIG,
    CRITIQUE_GEN_CONFIG,
    REWRITE_GEN_CONFIG,
    CLASSIFICATION_CONFIG,
    CATEGORIES,
    CATEGORY_MAPPING,
    CLASSIFICATION_PROMPT,
    SYSTEM_PROMPT,
    CRITIQUE_PROMPT,
    REWRITE_PROMPT,
    TOP_K,
    extract_and_map_categories,
    get_random_critique_request,
    CRITIQUE,
)

# ── 预召回结果路径（替换原来的 STRUCTURED_LAWS_PATH）──────────────────────────
RETRIEVAL_RESULTS_PATH = "/home/linux/Mdata/rag/retrieval_results_rl_0224.json"


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _extract_thinking_output(text: str) -> str:
    if not text:
        return text
    text = text.replace('<｜think▁begin｜>', '<think>').replace('<｜think▁end｜>', '</think>')
    end_marker = '</think>'
    idx = text.lower().find(end_marker)
    if idx != -1:
        text = text[idx + len(end_marker):]
    else:
        import re as _re
        text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
    text = re.sub(r'<\|im_[a-z_]+\|>', '', text)
    text = re.sub(r'<｜[a-z▁]+｜>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _truncate_by_chars(text: str, max_tokens: int) -> str:
    if not text:
        return text
    max_chars = int(max_tokens * 2.5)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind('。')
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1] + "..."
    return truncated + "..."


def _build_chat_prompt(user_msg: str, system_msg: str = None) -> str:
    if system_msg:
        return (
            f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return (
        f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─── 预召回结果加载器 ─────────────────────────────────────────────────────────

class PreretrievedLawsLookup:
    """
    从 retrieval_results_v2.json 建立问题 id → retrieved_laws 的查找表。
    法条字段适配 pipeline 原有的格式（source/article_number/full_text/similarity_score）。
    """

    def __init__(self, retrieval_path: str):
        print(f"  加载预召回结果: {retrieval_path}")
        with open(retrieval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 建立 id → laws 查找表，同时建立 question文本 → laws 备用索引
        self.id_index: Dict[int, List[Dict]] = {}
        self.question_index: Dict[str, List[Dict]] = {}

        for item in data:
            qid      = item.get("id")
            question = item.get("question", "").strip()
            laws     = self._convert_laws(item.get("retrieved_laws", []))

            if qid is not None:
                self.id_index[int(qid)] = laws
            if question:
                self.question_index[question] = laws

        print(f"  预召回结果共 {len(self.id_index)} 条（按id索引）")

    def _convert_laws(self, retrieved_laws: List[Dict]) -> List[Dict]:
        """
        将 retrieval_results_v2.json 的法条格式转换为 pipeline 期望的格式：
          v2格式:   {law_name, article, content, category, score, score_type}
          pipeline: {source, article_number, full_text, summary, similarity_score, title, law_id}
        """
        converted = []
        for law in retrieved_laws:
            converted.append({
                "law_id":          f"{law.get('law_name','')}_{law.get('article','')}",
                "title":           f"《{law.get('law_name', '')}》{law.get('article', '')}",
                "source":          law.get("law_name", ""),
                "article_number":  law.get("article", ""),
                "full_text":       law.get("content", ""),
                "summary":         "",
                "similarity_score": law.get("score", 0.0),
                "prohibited_actions": [],
            })
        return converted

    def get_laws(self, question_id: int, question_text: str) -> List[Dict]:
        """先按 id 查，找不到再按问题文本查，再找不到返回空列表"""
        if question_id in self.id_index:
            return self.id_index[question_id]
        q = question_text.strip()
        if q in self.question_index:
            return self.question_index[q]
        print(f"    [WARN] 问题 id={question_id} 未找到预召回法条，将使用空列表")
        return []


# ─── Pipeline 类 ──────────────────────────────────────────────────────────────

class LocalModelPipeline:

    def __init__(self, model_path: str = LOCAL_MODEL_PATH):
        print(f"\n{'='*60}")
        print(f"正在初始化 Pipeline")
        print(f"推理模型: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        self.categories    = CATEGORIES
        self.is_deepseek_r1 = "DeepSeek-R1" in model_path or "deepseek-r1" in model_path.lower()

        # ── 加载预召回法条（无需 GPU）────────────────────────────────────────
        print("\n加载预召回法条...")
        self.law_lookup = PreretrievedLawsLookup(RETRIEVAL_RESULTS_PATH)
        print("✓ 预召回法条就绪（无需向量模型，节省显存）")

        # ── 初始化 vLLM ───────────────────────────────────────────────────────
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM 未安装，请运行: pip install vllm")

        llm_kwargs = {
            "model":                  model_path,
            "trust_remote_code":      True,
            "dtype":                  "bfloat16",
            "tensor_parallel_size":   1,
            "gpu_memory_utilization": 0.90,  # 不再需要给向量模型留显存，可以拉高
            "max_model_len":          4096,
            "max_num_seqs":           32,
        }
        if self.is_deepseek_r1:
            print("  检测到 DeepSeek-R1 模型，启用 reasoning parser")
            llm_kwargs["reasoning_parser"] = "deepseek_r1"

        print(f"\n加载 vLLM 推理模型...")
        self.llm       = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        print(f"✓ vLLM 模型加载成功")
        print(f"{'='*60}\n")

    def _stop_tokens(self) -> List[str]:
        if self.is_deepseek_r1:
            return ["<｜end▁of▁sentence｜>", "<|im_end|>"]
        return ["<|im_end|>", "<|endoftext|>"]

    def _batch_generate(
        self,
        prompts: List[str],
        config: Dict,
        system_prompt: str = None,
        raw_prompts: bool = False,
    ) -> List[Tuple[str, str]]:
        from vllm import SamplingParams

        texts = prompts if raw_prompts else [
            _build_chat_prompt(p, system_prompt) for p in prompts
        ]
        sampling_params = SamplingParams(
            max_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            stop=self._stop_tokens(),
        )
        outputs = self.llm.generate(texts, sampling_params)
        results = []
        for o in outputs:
            raw   = o.outputs[0].text.strip()
            clean = _extract_thinking_output(raw)
            results.append((raw, clean))
        return results

    def format_laws_detailed(self, laws: List[Dict]) -> str:
        if not laws:
            return "（未检索到相关法规）"
        parts = []
        for i, law in enumerate(laws, 1):
            text  = f"\n【法规{i}】《{law['source']}》{law.get('article_number', '')}\n"
            text += f"条款原文：{law.get('full_text', '（无原文）')}\n"
            prohibited = law.get('prohibited_actions', [])
            if prohibited:
                text += f"禁止行为：{'；'.join(prohibited[:3])}\n"
            parts.append(text)
        return "\n".join(parts)

    def _map_category_from_json(self, category_str: str) -> Tuple[List[str], float, str]:
        if not category_str:
            return ["军队组织与管理"], 0.5, "JSON无category字段，使用默认类别"
        if category_str in CATEGORY_MAPPING:
            return [CATEGORY_MAPPING[category_str]], 0.95, "JSON category直接映射"
        for cat_name in CATEGORY_MAPPING:
            if cat_name in category_str or category_str in cat_name:
                return [CATEGORY_MAPPING[cat_name]], 0.85, "JSON category模糊映射"
        return ["军队组织与管理"], 0.4, f"JSON category '{category_str}' 无法映射，使用默认"

    def classify_questions_batch(self, questions: List[str]) -> List[Tuple[List[str], float, str]]:
        categories_str = "\n".join(
            f"{i+1}. {name}：{info['description']}"
            for i, (name, info) in enumerate(self.categories.items())
        )
        prompts = [
            CLASSIFICATION_PROMPT.format(question=q, categories_str=categories_str)
            for q in questions
        ]
        responses = self._batch_generate(prompts, CLASSIFICATION_CONFIG)
        results = []
        for response, _ in responses:
            if not response or len(response.strip()) < 5:
                results.append((["军队组织与管理"], 0.3, "模型返回空响应，使用默认类别"))
                continue
            cats  = extract_and_map_categories(response)
            valid = [c for c in cats if c in CATEGORY_MAPPING]
            if not valid:
                results.append((["军队组织与管理"], 0.3, "模型返回无效类别，使用默认类别"))
                continue
            confidence = 0.7 if len(valid) == 1 else 0.6
            results.append((valid, confidence, f"LLM分类({len(valid)}个类别)"))
        return results


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 80)
    print("vLLM 批量推理 Pipeline — DeepSeek-R1-Distill-Qwen-7B + 预召回法条")
    print("=" * 80)

    # ── 检查预召回文件 ────────────────────────────────────────────────────────
    if not os.path.exists(RETRIEVAL_RESULTS_PATH):
        print(f"\n⚠️  未找到预召回结果文件: {RETRIEVAL_RESULTS_PATH}")
        print("请先运行: python law_rag_retrieval_v2.py")
        sys.exit(1)

    # ── 加载问题集 ────────────────────────────────────────────────────────────
    print(f"\n数据集路径: {QUESTIONS_DATA_PATH}")
    if not os.path.exists(QUESTIONS_DATA_PATH):
        raise FileNotFoundError(f"数据集不存在: {QUESTIONS_DATA_PATH}")

    with open(QUESTIONS_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"共 {len(raw_data)} 条数据")
    use_json_category = isinstance(raw_data[0], dict) and "question" in raw_data[0]

    # ── 初始化 Pipeline ───────────────────────────────────────────────────────
    pipeline = LocalModelPipeline()

    # ── 处理配置 ──────────────────────────────────────────────────────────────
    batch_size    = int(os.getenv("BATCH_SIZE", "16"))
    max_questions = int(os.getenv("MAX_QUESTIONS", "0"))

    if use_json_category:
        valid_items = [
            (idx, item["question"].strip(), item.get("category", ""))
            for idx, item in enumerate(raw_data, 1)
            if item.get("question", "").strip()
        ]
    else:
        valid_items = [
            (idx, q.strip(), "")
            for idx, q in enumerate(raw_data, 1)
            if q and q.strip()
        ]

    if max_questions > 0:
        valid_items = valid_items[:max_questions]
        print(f"限制处理数量: {max_questions}")

    print(f"有效问题数: {len(valid_items)}")
    print(f"批次大小:   {batch_size}")

    # ── 批量处理 ──────────────────────────────────────────────────────────────
    results              = []
    total_laws_retrieved = 0
    stats = {
        "total_questions":      0,
        "success_count":        0,
        "error_count":          0,
        "category_distribution": {},
        "category_source":      {"json": 0, "llm": 0},
    }
    num_batches = (len(valid_items) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="批次进度"):
        batch                = valid_items[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids            = [item[0] for item in batch]
        batch_questions      = [item[1] for item in batch]
        batch_categories_raw = [item[2] for item in batch]

        print(f"\n{'─'*60}")
        print(f"[批次 {batch_idx+1}/{num_batches}] 处理 {len(batch)} 个问题 "
              f"(#{batch_ids[0]}~#{batch_ids[-1]})")

        try:
            # ── Step 1: 批量生成第一轮回答 ────────────────────────────────────
            print(f"  [Step 1] 批量生成第一轮回答...")
            first_pairs           = pipeline._batch_generate(
                batch_questions, GENERATION_CONFIG, system_prompt=SYSTEM_PROMPT,
            )
            first_responses_raw   = [p[0] for p in first_pairs]
            first_responses_clean = [p[1] for p in first_pairs]

            # ── Step 2: 获取类别 ──────────────────────────────────────────────
            print(f"  [Step 2] 获取类别...")
            classification_results = []
            questions_needing_llm  = []
            llm_pending_indices    = []

            for i, (question, cat_raw) in enumerate(zip(batch_questions, batch_categories_raw)):
                if use_json_category and cat_raw:
                    cats, conf, reason = pipeline._map_category_from_json(cat_raw)
                    classification_results.append((cats, conf, reason))
                    stats["category_source"]["json"] += 1
                else:
                    classification_results.append(None)
                    questions_needing_llm.append(question)
                    llm_pending_indices.append(i)

            if questions_needing_llm:
                print(f"    {len(questions_needing_llm)} 个问题需要 LLM 分类...")
                llm_results = pipeline.classify_questions_batch(questions_needing_llm)
                for pos, llm_res in zip(llm_pending_indices, llm_results):
                    classification_results[pos] = llm_res
                    stats["category_source"]["llm"] += 1

            # ── Step 3: 读取预召回法条 + 构建 prompt ─────────────────────────
            # 原来是实时向量检索，现在直接从 retrieval_results_v2.json 查表
            print(f"  [Step 3] 读取预召回法条 + 构建 prompt...")
            prepared = []

            for i, (idx, question, _) in enumerate(batch):
                original_response_raw = first_responses_raw[i]
                original_response     = first_responses_clean[i]
                categories, confidence, reason = classification_results[i]

                # 直接查表，不再做实时向量检索
                relevant_laws = pipeline.law_lookup.get_laws(idx, question)
                relevant_laws = relevant_laws[:TOP_K]
                total_laws_retrieved += len(relevant_laws)

                laws_text = _truncate_by_chars(
                    pipeline.format_laws_detailed(relevant_laws), 1200
                )

                primary_cat      = categories[0] if categories else "军队组织与管理"
                critique_request = get_random_critique_request(primary_cat)

                critique_prompt = CRITIQUE_PROMPT.format(
                    question=question,
                    original_response=original_response,
                    critique=critique_request,
                )
                rewrite_prompt = REWRITE_PROMPT.format(
                    question=question,
                    original_response=original_response,
                    critique=critique_request,
                    relevant_laws_detailed=laws_text,
                )

                truncated_original = _truncate_by_chars(original_response, 800)
                multi_turn_base = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                    f"<|im_start|>user\n{question}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n{truncated_original}\n<|im_end|>\n"
                )

                prepared.append({
                    "idx":                  idx,
                    "question":             question,
                    "original_response_raw": original_response_raw,
                    "original_response":    original_response,
                    "categories":           categories,
                    "confidence":           confidence,
                    "reason":               reason,
                    "relevant_laws":        relevant_laws,
                    "critique_prompt":      critique_prompt,
                    "rewrite_prompt":       rewrite_prompt,
                    "multi_turn_base":      multi_turn_base,
                })

            # ── Step 4: 批量生成批判 ──────────────────────────────────────────
            print(f"  [Step 4] 批量生成批判（{len(prepared)} 个）...")
            critique_raw_prompts = [
                data["multi_turn_base"]
                + f"<|im_start|>user\n{data['critique_prompt']}\n<|im_end|>\n"
                + "<|im_start|>assistant\n"
                for data in prepared
            ]
            critique_pairs           = pipeline._batch_generate(
                critique_raw_prompts, CRITIQUE_GEN_CONFIG, raw_prompts=True,
            )
            critique_responses_raw   = [p[0] for p in critique_pairs]
            critique_responses_clean = [p[1] for p in critique_pairs]

            # ── Step 5: 批量生成重写 ──────────────────────────────────────────
            print(f"  [Step 5] 批量生成重写（{len(prepared)} 个）...")
            rewrite_raw_prompts = []
            for data, critique in zip(prepared, critique_responses_clean):
                truncated_critique = _truncate_by_chars(critique, 400)
                rewrite_raw_prompts.append(
                    data["multi_turn_base"]
                    + f"<|im_start|>user\n{data['critique_prompt']}\n<|im_end|>\n"
                    + f"<|im_start|>assistant\n{truncated_critique}\n<|im_end|>\n"
                    + f"<|im_start|>user\n{data['rewrite_prompt']}\n<|im_end|>\n"
                    + "<|im_start|>assistant\n"
                )
            rewrite_pairs           = pipeline._batch_generate(
                rewrite_raw_prompts, REWRITE_GEN_CONFIG, raw_prompts=True,
            )
            rewrite_responses_raw   = [p[0] for p in rewrite_pairs]
            rewrite_responses_clean = [p[1] for p in rewrite_pairs]

            # ── Step 6: 组装结果 ──────────────────────────────────────────────
            print(f"  [Step 6] 组装结果...")
            for i_r, (data, critique, rewritten) in enumerate(
                zip(prepared, critique_responses_clean, rewrite_responses_clean)
            ):
                stats["total_questions"] += 1
                try:
                    cats = data["categories"]
                    for cat in cats:
                        if cat:
                            stats["category_distribution"][cat] = \
                                stats["category_distribution"].get(cat, 0) + 1
                    stats["success_count"] += 1

                    results.append({
                        "question_id":          data["idx"],
                        "question":             data["question"],
                        "original_response_raw": data["original_response_raw"],
                        "original_response":    data["original_response"],
                        "critique_raw":         critique_responses_raw[i_r],
                        "critique":             critique,
                        "rewritten_response_raw": rewrite_responses_raw[i_r],
                        "rewritten_response":   rewritten,
                        "rag_metadata": {
                            "categories":     cats,
                            "category":       cats[0] if cats else "",
                            "confidence":     data["confidence"],
                            "reason":         data["reason"],
                            "retrieved_laws": [
                                {
                                    "law_id":           law.get("law_id", ""),
                                    "title":            law["title"],
                                    "source":           law["source"],
                                    "article_number":   law.get("article_number", ""),
                                    "full_text":        law.get("full_text", ""),
                                    "summary":          law.get("summary", ""),
                                    "similarity_score": round(law.get("similarity_score", 0), 4),
                                }
                                for law in data["relevant_laws"]
                            ],
                        },
                    })

                except Exception as e:
                    print(f"    ✗ 问题 #{data['idx']} 组装失败: {e}")
                    stats["error_count"] += 1
                    results.append({
                        "question_id":       data["idx"],
                        "question":          data["question"],
                        "original_response": data.get("original_response", ""),
                        "critique":          f"处理失败: {e}",
                        "rewritten_response": "",
                        "rag_metadata":      {"error": str(e)},
                    })

        except Exception as e:
            import traceback
            print(f"\n✗ 批次处理失败: {e}")
            traceback.print_exc()
            for idx, question, _ in batch:
                stats["total_questions"] += 1
                stats["error_count"] += 1
                results.append({
                    "question_id":       idx,
                    "question":          question,
                    "original_response": "",
                    "critique":          f"批次处理失败: {e}",
                    "rewritten_response": "",
                    "rag_metadata":      {"error": str(e)},
                })

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    print(f"\n保存结果到: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 输出统计 ──────────────────────────────────────────────────────────────
    total_q = max(stats['total_questions'], 1)
    print("\n" + "=" * 80)
    print("处理统计")
    print("=" * 80)
    print(f"总问题数:       {stats['total_questions']}")
    print(f"成功:           {stats['success_count']}")
    print(f"失败:           {stats['error_count']}")
    print(f"平均检索法条数: {total_laws_retrieved / total_q:.1f}")
    print(f"\n类别来源: JSON直接映射={stats['category_source']['json']}  "
          f"LLM分类={stats['category_source']['llm']}")
    print("\n类别分布:")
    for cat, cnt in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
        pct = cnt / total_q * 100
        print(f"  {cat}: {cnt} ({pct:.1f}%)")
    print("\n✓ 处理完成！")
    print(f"结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()