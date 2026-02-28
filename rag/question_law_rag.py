"""
使用 ChatLaw-Text2Vec 做问题与类内法条的相似度计算（RAG）。
- 问题来源：apimakequestion/generate_output/0218new/local_assessed_*.json
- 法条来源：law/law.json
- 只在该问题所属类别（categories）内检索法条（类内相似度）。
输出格式与 rewrite_with_laws.py 所需的 similarity 文件一致（id, question, top_laws）。
"""

import json
import os
import argparse
from typing import List, Dict, Any, Tuple

# 项目根目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

# 默认路径
DEFAULT_QUESTIONS_PATH = os.path.join(
    _project_root, "apimakequestion", "generate_output", "0218new",
    "local_assessed_20260218_005211.json"
)
DEFAULT_LAWS_PATH = os.path.join(_project_root, "law", "law.json")
DEFAULT_MODEL_PATH = os.path.join(_current_dir, "models", "ChatLaw-Text2Vec")
# 若本地无模型，使用 HuggingFace ID
HF_MODEL_ID = "chestnutlzj/ChatLaw-Text2Vec"


def load_law_json(laws_path: str) -> Dict[str, Any]:
    """加载 law.json：{ 类别: { 法规名: { 条款号: 条款内容 } } }"""
    with open(laws_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_laws_by_category(laws_root: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将 law.json 展平为列表，每项包含 category, law_name, article_name, content。
    用于按类别过滤后做向量检索。
    """
    items = []
    for category, laws in laws_root.items():
        if not isinstance(laws, dict):
            continue
        for law_name, articles in laws.items():
            if not isinstance(articles, dict):
                continue
            for article_name, content in articles.items():
                if not content or not isinstance(content, str):
                    continue
                items.append({
                    "category": category,
                    "law_name": law_name,
                    "article_name": article_name,
                    "content": content.strip(),
                    # 拼接为一段文本便于编码（标题+内容）
                    "text": f"《{law_name}》{article_name}：{content.strip()}",
                })
    return items


def load_questions(questions_path: str) -> List[Dict[str, Any]]:
    """加载问题列表，确保每项有 question、categories。"""
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("问题文件必须是 JSON 数组")
    return data


def get_encoder(model_path: str, device: str = None):
    """加载 SentenceTransformer 编码器（ChatLaw-Text2Vec）。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("请安装: pip install sentence-transformers torch")

    use_path = model_path if os.path.exists(model_path) else HF_MODEL_ID
    print(f"加载编码模型: {use_path}")
    model = SentenceTransformer(use_path, device=device)
    return model


def encode_texts(model, texts: List[str], batch_size: int = 32, show_progress: bool = True):
    """批量编码文本，返回 numpy 向量矩阵。"""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )


def cosine_similarity(a, b):
    """单向量与矩阵的余弦相似度（已归一化时即为点积）。"""
    import numpy as np
    a = a.reshape(1, -1)
    return (a @ b.T).ravel()


def run_rag(
    questions_path: str,
    laws_path: str,
    output_path: str,
    model_path: str = None,
    top_k: int = 8,
    min_similarity: float = None,
    fallback_threshold: float = None,
    fallback_top_k: int = 5,
    batch_size: int = 32,
    device: str = None,
) -> List[Dict[str, Any]]:
    """
    对每个问题，先在其 categories 对应类内检索；若类内最高分 < fallback_threshold，
    则再在其它类内做法条相似度检索并合并结果（应对 categories 标注错误）。
    最终写入 output_path。
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # 加载数据
    print(f"加载问题: {questions_path}")
    questions_data = load_questions(questions_path)
    print(f"加载法条: {laws_path}")
    laws_root = load_law_json(laws_path)
    law_items = flatten_laws_by_category(laws_root)
    print(f"法条总数: {len(law_items)}")

    # 按类别建索引：category -> [ (idx in law_items, item), ... ]
    by_category: Dict[str, List[Tuple[int, Dict]]] = {}
    for idx, item in enumerate(law_items):
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((idx, item))

    # 设备
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"使用设备: {device}")

    # 编码模型
    model = get_encoder(model_path, device=device)

    # 编码所有法条文本（按 law_items 顺序）
    law_texts = [x["text"] for x in law_items]
    print("编码法条...")
    law_embeddings = encode_texts(model, law_texts, batch_size=batch_size)

    # 逐条问题：类内检索
    results = []
    for i, item in enumerate(questions_data):
        question = item.get("question", "")
        categories = item.get("categories") or []
        if not isinstance(categories, list):
            categories = [categories] if categories else []

        # 类内法条索引
        in_category_indices = []
        for cat in categories:
            if cat in by_category:
                for idx, _ in by_category[cat]:
                    in_category_indices.append(idx)

        if not in_category_indices:
            # 无匹配类别时仍给出 id/question，top_laws 为空
            results.append({
                "id": i,
                "question": question,
                "top_laws": [],
            })
            continue

        # 编码当前问题
        q_emb = encode_texts(model, [question], batch_size=1, show_progress=False)
        q_emb = q_emb[0]

        # 只对类内法条算相似度
        sub_embeddings = law_embeddings[in_category_indices]
        sims = cosine_similarity(q_emb, sub_embeddings)

        def make_law_list(indices, scores, min_sim=None):
            out = []
            order = scores.argsort()[::-1]
            for pos in order:
                score = float(scores[pos])
                if min_sim is not None and min_sim > 0 and score < min_sim:
                    continue
                idx = indices[pos]
                law = law_items[idx]
                out.append({
                    "law_name": law["law_name"],
                    "article_name": law["article_name"],
                    "content": law["content"],
                    "category": law["category"],
                    "similarity_score": score,
                })
            return out

        in_cat_list = make_law_list(in_category_indices, sims, min_similarity)
        best_in_cat = float(sims.max()) if len(sims) else 0.0

        # 跨类兜底：类内最高分低于阈值时，在其它类内再检索并合并
        if (
            fallback_threshold is not None
            and best_in_cat < fallback_threshold
            and len(in_category_indices) < len(law_items)
        ):
            out_cat_indices = [j for j in range(len(law_items)) if j not in set(in_category_indices)]
            if out_cat_indices:
                sub_out = law_embeddings[out_cat_indices]
                sims_out = cosine_similarity(q_emb, sub_out)
                out_cat_list = make_law_list(out_cat_indices, sims_out, min_similarity)
                merged = (in_cat_list + out_cat_list[: fallback_top_k])
                merged.sort(key=lambda x: x["similarity_score"], reverse=True)
                seen = set()
                deduped = []
                for L in merged:
                    key = (L["law_name"], L["article_name"])
                    if key not in seen:
                        seen.add(key)
                        deduped.append(L)
                    if len(deduped) >= top_k:
                        break
                in_cat_list = deduped

        top_laws = in_cat_list[:top_k]
        for rank, law in enumerate(top_laws, 1):
            law["rank"] = rank
        results.append({
            "id": i,
            "question": question,
            "top_laws": top_laws,
        })

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i + 1}/{len(questions_data)} 条问题")

    # 保存
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已写入: {output_path} (共 {len(results)} 条)")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="问题与类内法条相似度 RAG（ChatLaw-Text2Vec）"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=DEFAULT_QUESTIONS_PATH,
        help="问题 JSON 路径（local_assessed 等）",
    )
    parser.add_argument(
        "--laws",
        type=str,
        default=DEFAULT_LAWS_PATH,
        help="法条 JSON 路径（law/law.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出相似度 JSON 路径（默认：与问题同目录下的 question_law_similarity_0218new.json）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="ChatLaw-Text2Vec 模型路径或 HF ID（默认先本地 models/ChatLaw-Text2Vec）",
    )
    parser.add_argument("--top_k", type=int, default=8, help="每问检索法条数")
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.5,
        help="最低相似度阈值，低于此值的法条不写入结果（默认 0.5）",
    )
    parser.add_argument(
        "--fallback_threshold",
        type=float,
        default=0.55,
        help="类内最高分低于此值时触发跨类兜底，在其它类内再检索并合并（默认 0.55，设为 0 关闭）",
    )
    parser.add_argument(
        "--fallback_top_k",
        type=int,
        default=5,
        help="跨类兜底时从其它类取的法条数（默认 5）",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="编码 batch 大小")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.dirname(args.questions)
        args.output = os.path.join(base, "question_law_similarity_0218new.json")

    run_rag(
        questions_path=args.questions,
        laws_path=args.laws,
        output_path=args.output,
        model_path=args.model_path,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        fallback_threshold=args.fallback_threshold if args.fallback_threshold > 0 else None,
        fallback_top_k=args.fallback_top_k,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
