#!/usr/bin/env python3
"""
比较不同模型在综合排名上的表现
支持多种对比模式：Base vs SFT+RL, Base vs SFT, SFT vs SFT+RL
按六个类别分别统计胜率
"""

import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
QUESTIONS_FILE = "/home/linux/Mdata/apimakequestion/classified/sft_eval/questions_extracted.json"
RANKING_FILE = "/home/linux/Mdata/metrics/results/rank/model_comparison_ranking.json"
OUTPUT_DIR = "/home/linux/Mdata/picture-paper-data/overall-performance"

# 颜色定义
COLOR_MODEL_A = "#F1D2D0"  # 较浅的颜色给排名较低的模型
COLOR_MODEL_B = "#C5E1ED"  # 较深的颜色给排名较高的模型

# 六个类别
CATEGORIES = [
    "开战类型",
    "交战原则", 
    "国家安全与主权维护",
    "军队组织与管理",
    "国防建设与后备力量",
    "网络安全与技术合规"
]

# 对比配置
COMPARISONS = [
    {
        "name": "base_vs_sft",
        "model_a": "Base Model",
        "model_b": "SFT Only",
        "title": "Base Model vs SFT",
        "label_a": "Base Model",
        "label_b": "SFT"
    },
    {
        "name": "sft_vs_sftrl",
        "model_a": "SFT Only",
        "model_b": "SFT+RL",
        "title": "SFT vs SFT+RL",
        "label_a": "SFT",
        "label_b": "SFT+RL"
    },
    {
        "name": "base_vs_sftrl",
        "model_a": "Base Model",
        "model_b": "SFT+RL",
        "title": "Base Model vs SFT+RL",
        "label_a": "Base Model",
        "label_b": "SFT+RL"
    }
]

def load_json(filepath):
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """保存JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_question_category_map(questions_data):
    """构建问题到类别的映射"""
    return {item['question']: item['category'] for item in questions_data}

def compare_two_models(ranking_data, question_category_map, model_a, model_b):
    """
    比较两个模型的排名
    返回每个类别下各模型的胜负统计
    """
    category_stats = defaultdict(lambda: {
        'model_a_wins': 0,
        'model_b_wins': 0,
        'ties': 0,
        'total': 0,
        'questions': []
    })
    
    overall_stats = {
        'model_a_wins': 0,
        'model_b_wins': 0,
        'ties': 0,
        'total': 0
    }
    
    for item in ranking_data:
        question = item.get('question', '')
        overall_ranking = item.get('overall_ranking', [])
        
        if not question or not overall_ranking:
            continue
            
        category = question_category_map.get(question)
        if not category:
            continue
            
        try:
            model_a_rank = overall_ranking.index(model_a) if model_a in overall_ranking else -1
            model_b_rank = overall_ranking.index(model_b) if model_b in overall_ranking else -1
        except ValueError:
            continue
            
        if model_a_rank == -1 or model_b_rank == -1:
            continue
            
        category_stats[category]['total'] += 1
        overall_stats['total'] += 1
        
        result = {
            'question': question,
            'model_a_rank': model_a_rank + 1,
            'model_b_rank': model_b_rank + 1,
            'winner': ''
        }
        
        if model_a_rank < model_b_rank:
            category_stats[category]['model_a_wins'] += 1
            overall_stats['model_a_wins'] += 1
            result['winner'] = model_a
        elif model_b_rank < model_a_rank:
            category_stats[category]['model_b_wins'] += 1
            overall_stats['model_b_wins'] += 1
            result['winner'] = model_b
        else:
            category_stats[category]['ties'] += 1
            overall_stats['ties'] += 1
            result['winner'] = 'Tie'
            
        category_stats[category]['questions'].append(result)
    
    return dict(category_stats), overall_stats

def calculate_win_rates(category_stats, overall_stats, model_a, model_b):
    """计算胜率"""
    results = {
        'model_a': model_a,
        'model_b': model_b,
        'by_category': {},
        'overall': {}
    }
    
    for category in CATEGORIES:
        stats = category_stats.get(category, {
            'model_a_wins': 0, 'model_b_wins': 0, 'ties': 0, 'total': 0
        })
        total = stats['total']
        
        if total > 0:
            model_a_win_rate = stats['model_a_wins'] / total * 100
            model_b_win_rate = stats['model_b_wins'] / total * 100
            tie_rate = stats['ties'] / total * 100
        else:
            model_a_win_rate = model_b_win_rate = tie_rate = 0
            
        results['by_category'][category] = {
            'model_a_wins': stats['model_a_wins'],
            'model_b_wins': stats['model_b_wins'],
            'ties': stats['ties'],
            'total': total,
            'model_a_win_rate': round(model_a_win_rate, 2),
            'model_b_win_rate': round(model_b_win_rate, 2),
            'tie_rate': round(tie_rate, 2)
        }
    
    total = overall_stats['total']
    if total > 0:
        model_a_win_rate = overall_stats['model_a_wins'] / total * 100
        model_b_win_rate = overall_stats['model_b_wins'] / total * 100
        tie_rate = overall_stats['ties'] / total * 100
    else:
        model_a_win_rate = model_b_win_rate = tie_rate = 0
        
    results['overall'] = {
        'model_a_wins': overall_stats['model_a_wins'],
        'model_b_wins': overall_stats['model_b_wins'],
        'ties': overall_stats['ties'],
        'total': total,
        'model_a_win_rate': round(model_a_win_rate, 2),
        'model_b_win_rate': round(model_b_win_rate, 2),
        'tie_rate': round(tie_rate, 2)
    }
    
    return results

def print_summary(results, model_a, model_b, title):
    """打印结果摘要"""
    print("=" * 90)
    print(f"{title} 综合排名对比结果")
    print("=" * 90)
    
    print("\n【按类别统计】\n")
    print(f"{'类别':<20} {model_a+'胜':<12} {model_b+'胜':<12} {'平局':<8} {'总数':<8} {model_a+'胜率':<14} {model_b+'胜率':<14}")
    print("-" * 100)
    
    for category in CATEGORIES:
        stats = results['by_category'].get(category, {})
        print(f"{category:<20} {stats.get('model_a_wins', 0):<12} {stats.get('model_b_wins', 0):<12} "
              f"{stats.get('ties', 0):<8} {stats.get('total', 0):<8} "
              f"{stats.get('model_a_win_rate', 0):.2f}%{'':<8} {stats.get('model_b_win_rate', 0):.2f}%")
    
    print("-" * 100)
    overall = results['overall']
    print(f"{'总计':<20} {overall['model_a_wins']:<12} {overall['model_b_wins']:<12} "
          f"{overall['ties']:<8} {overall['total']:<8} "
          f"{overall['model_a_win_rate']:.2f}%{'':<8} {overall['model_b_win_rate']:.2f}%")
    
    print("\n" + "=" * 90)
    print("结论:")
    if overall['model_b_win_rate'] > overall['model_a_win_rate']:
        print(f"  {model_b} 整体表现更好，胜率为 {overall['model_b_win_rate']:.2f}%")
    elif overall['model_a_win_rate'] > overall['model_b_win_rate']:
        print(f"  {model_a} 整体表现更好，胜率为 {overall['model_a_win_rate']:.2f}%")
    else:
        print("  两者表现相当")
    print("=" * 90)

def plot_win_rate_comparison(results, model_a, model_b, title, figure_file):
    """绘制胜率对比柱状图（单张图）"""
    categories = []
    model_a_rates = []
    model_b_rates = []
    
    for category in CATEGORIES:
        stats = results['by_category'].get(category, {})
        if stats.get('total', 0) > 0:
            categories.append(category)
            model_a_rates.append(stats.get('model_a_win_rate', 0))
            model_b_rates.append(stats.get('model_b_win_rate', 0))
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars_a = ax.bar(x - width/2, model_a_rates, width, label=model_a, color=COLOR_MODEL_A)
    bars_b = ax.bar(x + width/2, model_b_rates, width, label=model_b, color=COLOR_MODEL_B)
    
    ax.set_ylabel('Pairwise偏好胜率 (%)', fontsize=12, fontproperties=font_prop)
    ax.set_xlabel('类别', fontsize=12, fontproperties=font_prop)
    ax.set_title(f'{title} Pairwise偏好胜率对比', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha='center', fontsize=10, fontproperties=font_prop)
    ax.legend(loc='upper left', fontsize=11, prop=font_prop)
    
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for bar in bars_a:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars_b:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(figure_file, dpi=800, bbox_inches='tight')
    plt.close()
    
    print(f"\n柱状图已保存到: {figure_file}")


def plot_combined_comparison(all_results, comparisons, output_file):
    """绘制三个对比的合并图（上下排列）"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    for idx, comp in enumerate(comparisons):
        ax = axes[idx]
        name = comp["name"]
        title = comp["title"]
        label_a = comp.get("label_a", comp["model_a"])
        label_b = comp.get("label_b", comp["model_b"])
        results = all_results[name]
        
        categories = []
        model_a_rates = []
        model_b_rates = []
        
        for category in CATEGORIES:
            stats = results['by_category'].get(category, {})
            if stats.get('total', 0) > 0:
                categories.append(category)
                model_a_rates.append(stats.get('model_a_win_rate', 0))
                model_b_rates.append(stats.get('model_b_win_rate', 0))
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars_a = ax.bar(x - width/2, model_a_rates, width, label=label_a, color=COLOR_MODEL_A)
        bars_b = ax.bar(x + width/2, model_b_rates, width, label=label_b, color=COLOR_MODEL_B)
        
        ax.set_ylabel('Pairwise偏好胜率 (%)', fontsize=12, fontproperties=font_prop)
        ax.set_xlabel('类别', fontsize=12, fontproperties=font_prop)
        ax.set_title(f'{title}', fontsize=14, fontweight='bold', fontproperties=font_prop)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=0, ha='center', fontsize=10, fontproperties=font_prop)
        ax.legend(loc='upper left', fontsize=11, prop=font_prop)
        
        ax.set_ylim(0, 110)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        for bar in bars_a:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        for bar in bars_b:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    plt.close()
    
    print(f"\n合并柱状图已保存到: {output_file}")

def main():
    # 加载数据
    print("加载数据...")
    questions_data = load_json(QUESTIONS_FILE)
    ranking_data = load_json(RANKING_FILE)
    
    print(f"  - 问题分类数据: {len(questions_data)} 条")
    print(f"  - 排名数据: {len(ranking_data)} 条")
    
    # 构建问题-类别映射
    question_category_map = build_question_category_map(questions_data)
    
    # 对每种对比进行分析
    all_results = {}
    
    for comp in COMPARISONS:
        name = comp["name"]
        model_a = comp["model_a"]
        model_b = comp["model_b"]
        title = comp["title"]
        
        print(f"\n{'='*90}")
        print(f"正在分析: {title}")
        print(f"{'='*90}")
        
        # 比较排名
        category_stats, overall_stats = compare_two_models(
            ranking_data, question_category_map, model_a, model_b
        )
        
        # 计算胜率
        results = calculate_win_rates(category_stats, overall_stats, model_a, model_b)
        
        # 添加详细问题列表
        results['detailed_by_category'] = {}
        for category, stats in category_stats.items():
            results['detailed_by_category'][category] = stats['questions']
        
        # 保存结果
        output_file = f"{OUTPUT_DIR}/{name}_results.json"
        save_json(results, output_file)
        print(f"结果已保存到: {output_file}")
        
        # 打印摘要
        print_summary(results, model_a, model_b, title)
        
        # 绘制柱状图
        figure_file = f"{OUTPUT_DIR}/{name}_winrate.png"
        plot_win_rate_comparison(results, model_a, model_b, title, figure_file)
        
        all_results[name] = results
    
    # 保存汇总结果
    summary_file = f"{OUTPUT_DIR}/all_comparisons_summary.json"
    summary = {}
    for name, results in all_results.items():
        summary[name] = {
            'model_a': results['model_a'],
            'model_b': results['model_b'],
            'overall': results['overall'],
            'by_category': results['by_category']
        }
    save_json(summary, summary_file)
    print(f"\n\n汇总结果已保存到: {summary_file}")
    
    # 绘制合并图
    combined_file = f"{OUTPUT_DIR}/all_comparisons_combined.png"
    plot_combined_comparison(all_results, COMPARISONS, combined_file)

if __name__ == "__main__":
    main()
