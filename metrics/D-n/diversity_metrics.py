"""
多样性指标计算工具
用于评估模型生成文本的多样性，包括：
1. Distinct-n: 计算生成文本中不重复 n-gram 的比例
2. Self-BLEU: 计算生成文本集合内部的相似度

数据格式要求：
{
    "question": "问题文本",
    "generate": ["生成答案1", "生成答案2", ...]
}
或
{
    "question": "问题文本",
    "answers": ["生成答案1", "生成答案2", ...]
}
"""

import json
import argparse
from collections import Counter
from typing import List, Dict, Any
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
from tqdm import tqdm
import numpy as np


def tokenize(text: str, language: str = "zh") -> List[str]:
    """分词"""
    if language == "zh":
        return list(jieba.cut(text))
    else:
        return text.split()


def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """获取n-gram列表"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def distinct_n_sentence(tokens: List[str], n: int) -> float:
    """
    计算单个句子的 Distinct-n
    Distinct-n = unique_ngrams / total_ngrams
    """
    ngrams = get_ngrams(tokens, n)
    if len(ngrams) == 0:
        return 0.0
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams)


def distinct_n_corpus(texts: List[str], n: int, language: str = "zh") -> Dict[str, float]:
    """
    计算语料库级别的 Distinct-n
    返回：
    - sentence_level: 句子级别的平均值
    - corpus_level: 将所有文本合并后计算的值
    """
    all_ngrams = []
    sentence_scores = []
    
    for text in texts:
        tokens = tokenize(text, language)
        ngrams = get_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
        sentence_scores.append(distinct_n_sentence(tokens, n))
    
    if len(all_ngrams) == 0:
        corpus_score = 0.0
    else:
        unique_ngrams = set(all_ngrams)
        corpus_score = len(unique_ngrams) / len(all_ngrams)
    
    return {
        "sentence_level": np.mean(sentence_scores) if sentence_scores else 0.0,
        "corpus_level": corpus_score
    }


def self_bleu_single(hypothesis: List[str], references: List[List[str]], n: int = 4) -> float:
    """
    计算单个文本相对于其他文本的 Self-BLEU
    """
    if len(references) == 0:
        return 0.0
    
    weights = tuple([1.0/n] * n)
    smoothing = SmoothingFunction().method1
    
    bleu_scores = []
    for ref in references:
        try:
            score = sentence_bleu([ref], hypothesis, weights=weights, smoothing_function=smoothing)
            bleu_scores.append(score)
        except Exception:
            bleu_scores.append(0.0)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


def self_bleu_corpus(texts: List[str], n: int = 4, language: str = "zh") -> float:
    """
    计算语料库的 Self-BLEU
    Self-BLEU 越低，表示多样性越好
    """
    if len(texts) < 2:
        return 0.0
    
    tokenized_texts = [tokenize(text, language) for text in texts]
    
    bleu_scores = []
    for i, hypothesis in enumerate(tokenized_texts):
        references = [tokenized_texts[j] for j in range(len(tokenized_texts)) if j != i]
        score = self_bleu_single(hypothesis, references, n)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)


def compute_diversity_metrics(
    data: List[Dict[str, Any]], 
    generate_key: str = "generate",
    language: str = "zh",
    distinct_n_values: List[int] = [1, 2, 3, 4],
    self_bleu_n: int = 4
) -> Dict[str, Any]:
    """
    计算数据集的多样性指标
    
    参数:
    - data: 数据列表，每个元素包含 question 和 generate/answers
    - generate_key: 生成文本的键名 ("generate" 或 "answers")
    - language: 语言 ("zh" 中文, "en" 英文)
    - distinct_n_values: 要计算的 Distinct-n 的 n 值列表
    - self_bleu_n: Self-BLEU 的 n-gram 值
    
    返回:
    - 包含各项指标的字典
    """
    results = {
        "num_samples": len(data),
        "distinct_n": {},
        "self_bleu": {},
        "per_question_metrics": []
    }
    
    all_generations = []
    per_question_distinct = {n: [] for n in distinct_n_values}
    per_question_self_bleu = []
    
    print("计算多样性指标...")
    for item in tqdm(data):
        generations = item.get(generate_key, item.get("answers", []))
        if not generations:
            continue
        
        if isinstance(generations, str):
            generations = [generations]
        
        all_generations.extend(generations)
        
        question_metrics = {"question": item.get("question", "")}
        
        for n in distinct_n_values:
            metrics = distinct_n_corpus(generations, n, language)
            per_question_distinct[n].append(metrics["corpus_level"])
            question_metrics[f"distinct_{n}"] = metrics["corpus_level"]
        
        if len(generations) >= 2:
            sb = self_bleu_corpus(generations, self_bleu_n, language)
            per_question_self_bleu.append(sb)
            question_metrics["self_bleu"] = sb
        
        results["per_question_metrics"].append(question_metrics)
    
    print("计算全局 Distinct-n...")
    for n in distinct_n_values:
        global_metrics = distinct_n_corpus(all_generations, n, language)
        results["distinct_n"][f"distinct_{n}"] = {
            "global_corpus": global_metrics["corpus_level"],
            "global_sentence_avg": global_metrics["sentence_level"],
            "per_question_avg": np.mean(per_question_distinct[n]) if per_question_distinct[n] else 0.0
        }
    
    print("计算全局 Self-BLEU...")
    results["self_bleu"] = {
        "per_question_avg": np.mean(per_question_self_bleu) if per_question_self_bleu else 0.0,
        "per_question_std": np.std(per_question_self_bleu) if per_question_self_bleu else 0.0
    }
    
    results["total_generations"] = len(all_generations)
    
    return results


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSON 数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def format_results(results: Dict[str, Any], model_name: str = "") -> str:
    """格式化输出结果"""
    output = []
    output.append("=" * 60)
    output.append(f"多样性指标评估结果 - {model_name}" if model_name else "多样性指标评估结果")
    output.append("=" * 60)
    output.append(f"样本数量: {results['num_samples']}")
    output.append(f"生成文本总数: {results['total_generations']}")
    output.append("")
    
    output.append("【Distinct-n 指标】(值越大，多样性越好)")
    output.append("-" * 40)
    for key, value in results["distinct_n"].items():
        output.append(f"  {key}:")
        output.append(f"    - 全局语料库级别: {value['global_corpus']:.4f}")
        output.append(f"    - 全局句子级别平均: {value['global_sentence_avg']:.4f}")
        output.append(f"    - 问题级别平均: {value['per_question_avg']:.4f}")
    output.append("")
    
    output.append("【Self-BLEU 指标】(值越低，多样性越好)")
    output.append("-" * 40)
    output.append(f"  问题级别平均: {results['self_bleu']['per_question_avg']:.4f}")
    output.append(f"  问题级别标准差: {results['self_bleu']['per_question_std']:.4f}")
    output.append("=" * 60)
    
    return "\n".join(output)


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> str:
    """比较多个模型的结果"""
    output = []
    output.append("\n" + "=" * 80)
    output.append("模型多样性对比")
    output.append("=" * 80)
    
    models = list(results_dict.keys())
    
    output.append("\n【Distinct-n 对比】(值越大越好)")
    output.append("-" * 80)
    header = f"{'模型':<30} | {'D-1':<10} | {'D-2':<10} | {'D-3':<10} | {'D-4':<10}"
    output.append(header)
    output.append("-" * 80)
    
    for model in models:
        r = results_dict[model]
        d1 = r["distinct_n"].get("distinct_1", {}).get("per_question_avg", 0)
        d2 = r["distinct_n"].get("distinct_2", {}).get("per_question_avg", 0)
        d3 = r["distinct_n"].get("distinct_3", {}).get("per_question_avg", 0)
        d4 = r["distinct_n"].get("distinct_4", {}).get("per_question_avg", 0)
        row = f"{model:<30} | {d1:<10.4f} | {d2:<10.4f} | {d3:<10.4f} | {d4:<10.4f}"
        output.append(row)
    
    output.append("\n【Self-BLEU 对比】(值越低越好)")
    output.append("-" * 80)
    header = f"{'模型':<30} | {'Self-BLEU':<15} | {'Std':<15}"
    output.append(header)
    output.append("-" * 80)
    
    for model in models:
        r = results_dict[model]
        sb = r["self_bleu"]["per_question_avg"]
        std = r["self_bleu"]["per_question_std"]
        row = f"{model:<30} | {sb:<15.4f} | {std:<15.4f}"
        output.append(row)
    
    output.append("=" * 80)
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="计算文本生成多样性指标")
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True,
                        help="输入文件路径，支持多个文件进行对比")
    parser.add_argument("--names", "-n", type=str, nargs="+", default=None,
                        help="模型名称，与输入文件一一对应")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出结果保存路径 (JSON)")
    parser.add_argument("--generate_key", "-k", type=str, default="generate",
                        help="生成文本的键名 (默认: generate)")
    parser.add_argument("--language", "-l", type=str, default="zh",
                        choices=["zh", "en"], help="文本语言 (默认: zh)")
    parser.add_argument("--distinct_n", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Distinct-n 的 n 值 (默认: 1 2 3 4)")
    parser.add_argument("--self_bleu_n", type=int, default=4,
                        help="Self-BLEU 的 n-gram (默认: 4)")
    
    args = parser.parse_args()
    
    if args.names is None:
        args.names = [f"Model_{i+1}" for i in range(len(args.input))]
    
    if len(args.names) != len(args.input):
        print("错误: 模型名称数量与输入文件数量不匹配")
        return
    
    all_results = {}
    
    for file_path, model_name in zip(args.input, args.names):
        print(f"\n处理: {model_name} ({file_path})")
        data = load_data(file_path)
        results = compute_diversity_metrics(
            data,
            generate_key=args.generate_key,
            language=args.language,
            distinct_n_values=args.distinct_n,
            self_bleu_n=args.self_bleu_n
        )
        all_results[model_name] = results
        print(format_results(results, model_name))
    
    if len(all_results) > 1:
        print(compare_models(all_results))
    
    if args.output:
        serializable_results = {}
        for model, res in all_results.items():
            serializable_results[model] = {
                k: v for k, v in res.items() if k != "per_question_metrics"
            }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
