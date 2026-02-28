"""
法条语义召回 v2 — 三项优化：
  1. 分类过滤：先按 category 缩小候选池
  2. BM25 稀疏检索 + ChatLaw Dense 混合召回（解决分数虚高/区分度差问题）
  3. Cross-Encoder 精排：bge-reranker-base 对混合召回结果重新打分
  4. 法条内容截断编码（长条款只取前200字，提升向量区分度）

依赖安装：
    pip install sentence-transformers rank-bm25 jieba
"""

import json
import jieba
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim

# ==================== 配置 ====================
CHATLAW_MODEL_PATH = "/home/linux/Mdata/rag/models/ChatLaw-Text2Vec"
RERANKER_PATH      = "/home/linux/Mdata/model/bge-reranker-base"

TEST_LAW_JSON      = "/home/linux/Mdata/apimakequestion/classified/Rl/question-rl_classified.json"
LAW_JSON           = "/home/linux/Mdata/law/law.json"
OUTPUT_JSON        = "/home/linux/Mdata/rag/retrieval_results_rl_0224.json"

TOP_K              = 5      # 最终返回法条数
RECALL_K           = 30     # 精排前的候选池大小
DENSE_WEIGHT       = 0.6    # 混合权重：ChatLaw Dense
BM25_WEIGHT        = 0.4    # 混合权重：BM25
MAX_CONTENT_LEN    = 300    # 法条截断长度（字符）
BATCH_SIZE         = 64

# test_law.json 的 category_name → law.json 一级 key 的映射
CATEGORY_MAP = {
    "国家安全与主权维护": ["国家安全与主权维护"],
    "军队组织与管理类":   ["军队组织与管理"],
    "军队组织与管理":     ["军队组织与管理"],
    "网络安全与技术合规": ["网络安全与技术合规"],
    "开战类型":           ["开战类型"],
    "交战原则":           ["交战原则"],
}
# ===============================================


def tokenize_zh(text: str) -> list[str]:
    tokens = jieba.lcut(text)
    stop = {"的", "了", "是", "在", "与", "及", "或", "和", "等", "对", "于",
            "中", "不", "有", "为", "以", "其", "并", "应", "将", "该", "此",
            "之", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"}
    return [t for t in tokens if len(t) > 1 and t not in stop]


def truncate_content(text: str, max_len: int = MAX_CONTENT_LEN) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    for punct in ["。", "；", "，"]:
        idx = cut.rfind(punct)
        if idx > max_len // 2:
            return cut[:idx + 1]
    return cut


def encode_batch(model: SentenceTransformer, texts: list[str],
                 batch_size: int = BATCH_SIZE) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        emb = model.encode(texts[i: i + batch_size], convert_to_tensor=False,
                           show_progress_bar=False)
        all_embs.append(emb)
    return np.vstack(all_embs)


def load_law_corpus(law_json_path: str) -> list[dict]:
    print(f"[INFO] 加载法条库: {law_json_path}")
    with open(law_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    law_items = []
    for category, laws in raw.items():
        if not isinstance(laws, dict):
            continue
        for law_name, articles in laws.items():
            if not isinstance(articles, dict):
                continue
            for article, content in articles.items():
                content_str = str(content).strip()
                truncated   = truncate_content(content_str)
                encode_text = f"{law_name} {article} {truncated}"
                law_items.append({
                    "category":    category,
                    "law_name":    law_name,
                    "article":     article,
                    "content":     content_str,
                    "encode_text": encode_text,
                })

    print(f"[INFO] 共加载 {len(law_items)} 条法条")
    return law_items


def build_category_index(law_items: list[dict]) -> dict[str, list[int]]:
    idx = defaultdict(list)
    for i, item in enumerate(law_items):
        idx[item["category"]].append(i)
    return idx


def build_query_texts(q_data: dict) -> list[str]:
    queries = []
    classification = q_data.get("classification", {})
    overall = classification.get("overall_summary", "").strip()
    if overall:
        queries.append(overall)
    for cat in classification.get("categories", []):
        for law_ref in cat.get("laws_and_regulations", []):
            summary = law_ref.get("content_summary", "").strip()
            if summary and summary not in queries:
                queries.append(summary)
    return queries


def get_question_categories(q_data: dict) -> list[str]:
    cats = []
    for cat in q_data.get("classification", {}).get("categories", []):
        name = cat.get("category_name", "").strip()
        if name:
            cats.append(name)
    return cats


def get_candidate_indices(
    question_categories: list[str],
    category_index: dict[str, list[int]],
    law_items: list[dict],
) -> list[int]:
    law_categories = set()
    for q_cat in question_categories:
        mapped = CATEGORY_MAP.get(q_cat)
        if mapped:
            law_categories.update(mapped)
        else:
            q_clean = q_cat.replace("类", "").strip()
            for law_cat in category_index.keys():
                if q_clean in law_cat or law_cat in q_clean:
                    law_categories.add(law_cat)

    if not law_categories:
        print(f"    [WARN] 无法映射分类 {question_categories}，使用全库")
        return list(range(len(law_items)))

    indices = []
    for cat in law_categories:
        indices.extend(category_index.get(cat, []))
    return list(set(indices))


def hybrid_retrieve(
    query_texts: list[str],
    candidate_indices: list[int],
    law_items: list[dict],
    law_embeddings: np.ndarray,
    dense_model: SentenceTransformer,
    recall_k: int = RECALL_K,
) -> list[tuple[int, float]]:
    """BM25 + ChatLaw Dense 两路混合召回"""
    if not candidate_indices:
        return []

    sub_items = [law_items[i] for i in candidate_indices]
    sub_texts = [item["encode_text"] for item in sub_items]
    sub_embs  = law_embeddings[candidate_indices]

    # BM25
    tokenized_corpus = [tokenize_zh(t) for t in sub_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.zeros(len(sub_items))
    for q in query_texts:
        scores = np.array(bm25.get_scores(tokenize_zh(q)))
        bm25_scores = np.maximum(bm25_scores, scores)
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores /= bm25_max

    # Dense
    q_emb     = dense_model.encode(query_texts, convert_to_tensor=True,
                                   show_progress_bar=False)
    sim       = cos_sim(q_emb, torch.tensor(sub_embs).to(q_emb.device)).cpu().numpy()
    cl_scores = (sim.max(axis=0) + 1) / 2   # 归一化到 [0,1]

    # 加权合并
    hybrid = DENSE_WEIGHT * cl_scores + BM25_WEIGHT * bm25_scores

    seen_keys = set()
    results   = []
    for local_i in np.argsort(hybrid)[::-1]:
        item = sub_items[local_i]
        key  = (item["law_name"], item["article"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        results.append((candidate_indices[local_i], float(hybrid[local_i])))
        if len(results) >= recall_k:
            break

    return results


def rerank(
    query_texts: list[str],
    candidates: list[tuple[int, float]],
    law_items: list[dict],
    reranker: CrossEncoder,
    top_k: int = TOP_K,
) -> list[dict]:
    """bge-reranker-base Cross-Encoder 精排，多 query 取最大分数"""
    if not candidates:
        return []

    cand_texts = [law_items[i]["encode_text"] for i, _ in candidates]

    all_scores = np.full(len(candidates), -np.inf)
    for q_text in query_texts:
        pairs  = [(q_text, t) for t in cand_texts]
        scores = np.array(reranker.predict(pairs, show_progress_bar=False))
        all_scores = np.maximum(all_scores, scores)

    ranked    = np.argsort(all_scores)[::-1]
    results   = []
    seen_keys = set()
    for local_i in ranked:
        global_i = candidates[local_i][0]
        item     = law_items[global_i]
        key      = (item["law_name"], item["article"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        results.append({
            "law_name":   item["law_name"],
            "article":    item["article"],
            "content":    item["content"],
            "category":   item["category"],
            "score":      round(float(all_scores[local_i]), 4),
            "score_type": "cross_encoder",
        })
        if len(results) >= top_k:
            break

    return results


def main():
    # 1. 加载模型
    print(f"[INFO] 加载 Dense 模型: {CHATLAW_MODEL_PATH}")
    dense_model = SentenceTransformer(CHATLAW_MODEL_PATH)
    if torch.cuda.is_available():
        dense_model = dense_model.cuda()

    print(f"[INFO] 加载 Cross-Encoder: {RERANKER_PATH}")
    reranker = CrossEncoder(RERANKER_PATH, max_length=512)

    # 2. 加载法条库并编码
    law_items    = load_law_corpus(LAW_JSON)
    encode_texts = [item["encode_text"] for item in law_items]

    print("[INFO] 编码法条库...")
    law_embeddings = encode_batch(dense_model, encode_texts)
    print(f"[INFO] 编码完成，shape: {law_embeddings.shape}")

    # 3. 分类索引
    category_index = build_category_index(law_items)
    print(f"[INFO] 法条类别: {list(category_index.keys())}")

    # 4. 加载问题集
    print(f"[INFO] 加载问题集: {TEST_LAW_JSON}")
    with open(TEST_LAW_JSON, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[INFO] 共 {len(questions)} 个问题")

    # 5. 批量处理
    results = []
    for idx, q_data in enumerate(questions):
        qid           = q_data.get("id", idx)
        question_text = q_data.get("question", "")
        print(f"\n[{idx+1}/{len(questions)}] id={qid}: {question_text[:50]}...")

        query_texts = build_query_texts(q_data)
        if not query_texts:
            print("  [WARN] 无查询文本，跳过")
            results.append({"id": qid, "question": question_text, "retrieved_laws": []})
            continue

        # 5a. 分类过滤
        q_categories      = get_question_categories(q_data)
        candidate_indices = get_candidate_indices(q_categories, category_index, law_items)
        print(f"  分类: {q_categories} → 候选法条数: {len(candidate_indices)}")

        # 5b. 混合粗召回
        hybrid_candidates = hybrid_retrieve(
            query_texts, candidate_indices, law_items,
            law_embeddings, dense_model, recall_k=RECALL_K,
        )

        # 5c. Cross-Encoder 精排
        retrieved = rerank(query_texts, hybrid_candidates, law_items, reranker, top_k=TOP_K)

        for r in retrieved:
            print(f"  [{r['score']:.4f}] {r['law_name']} {r['article']} ({r['category']})")

        results.append({
            "id":             qid,
            "question":       question_text,
            "query_texts":    query_texts,
            "retrieved_laws": retrieved,
        })

    # 6. 保存
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 已保存至: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()