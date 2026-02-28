#!/usr/bin/env python3
"""
比较 Base Model 和 SFT+RL 在综合排名上的表现
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
OUTPUT_FILE = "/home/linux/Mdata/picture-paper-data/overall-performance/baseline_vs_sftrl_results.json"
FIGURE_FILE = "/home/linux/Mdata/picture-paper-data/overall-performance/baseline_vs_sftrl_winrate.png"

# 颜色定义
COLOR_BASE = "#F1D2D0"
COLOR_SFTRL = "#C5E1ED"

# 六个类别
CATEGORIES = [
    "开战类型",
    "交战原则", 
    "国家安全与主权维护",
    "军队组织与管理",
    "国防建设与后备力量",
    "网络安全与技术合规"
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

def compare_rankings(ranking_data, question_category_map):
    """
    比较SFT+RL和Base Model的排名
    如果SFT+RL排名靠前(索引更小)，SFT+RL胜
    如果Base Model排名靠前(索引更小)，Base Model胜
    """
    # 按类别统计
    category_stats = defaultdict(lambda: {
        'sftrl_wins': 0,
        'base_wins': 0,
        'ties': 0,
        'total': 0,
        'questions': []
    })
    
    # 总体统计
    overall_stats = {
        'sftrl_wins': 0,
        'base_wins': 0,
        'ties': 0,
        'total': 0
    }
    
    for item in ranking_data:
        question = item.get('question', '')
        overall_ranking = item.get('overall_ranking', [])
        
        if not question or not overall_ranking:
            continue
            
        # 获取问题类别
        category = question_category_map.get(question)
        if not category:
            continue
            
        # 在overall_ranking中找到SFT+RL和Base Model的位置
        try:
            sftrl_rank = overall_ranking.index("SFT+RL") if "SFT+RL" in overall_ranking else -1
            base_rank = overall_ranking.index("Base Model") if "Base Model" in overall_ranking else -1
        except ValueError:
            continue
            
        if sftrl_rank == -1 or base_rank == -1:
            continue
            
        # 更新统计
        category_stats[category]['total'] += 1
        overall_stats['total'] += 1
        
        result = {
            'question': question,
            'sftrl_rank': sftrl_rank + 1,  # 转为1-indexed
            'base_rank': base_rank + 1,
            'winner': ''
        }
        
        if sftrl_rank < base_rank:
            # SFT+RL排名靠前（索引小=排名高）
            category_stats[category]['sftrl_wins'] += 1
            overall_stats['sftrl_wins'] += 1
            result['winner'] = 'SFT+RL'
        elif base_rank < sftrl_rank:
            # Base Model排名靠前
            category_stats[category]['base_wins'] += 1
            overall_stats['base_wins'] += 1
            result['winner'] = 'Base Model'
        else:
            # 平局
            category_stats[category]['ties'] += 1
            overall_stats['ties'] += 1
            result['winner'] = 'Tie'
            
        category_stats[category]['questions'].append(result)
    
    return dict(category_stats), overall_stats

def calculate_win_rates(category_stats, overall_stats):
    """计算胜率"""
    results = {
        'by_category': {},
        'overall': {}
    }
    
    # 按类别计算胜率
    for category in CATEGORIES:
        stats = category_stats.get(category, {
            'sftrl_wins': 0, 'base_wins': 0, 'ties': 0, 'total': 0
        })
        total = stats['total']
        
        if total > 0:
            sftrl_win_rate = stats['sftrl_wins'] / total * 100
            base_win_rate = stats['base_wins'] / total * 100
            tie_rate = stats['ties'] / total * 100
        else:
            sftrl_win_rate = base_win_rate = tie_rate = 0
            
        results['by_category'][category] = {
            'sftrl_wins': stats['sftrl_wins'],
            'base_wins': stats['base_wins'],
            'ties': stats['ties'],
            'total': total,
            'sftrl_win_rate': round(sftrl_win_rate, 2),
            'base_win_rate': round(base_win_rate, 2),
            'tie_rate': round(tie_rate, 2)
        }
    
    # 计算总体胜率
    total = overall_stats['total']
    if total > 0:
        sftrl_win_rate = overall_stats['sftrl_wins'] / total * 100
        base_win_rate = overall_stats['base_wins'] / total * 100
        tie_rate = overall_stats['ties'] / total * 100
    else:
        sftrl_win_rate = base_win_rate = tie_rate = 0
        
    results['overall'] = {
        'sftrl_wins': overall_stats['sftrl_wins'],
        'base_wins': overall_stats['base_wins'],
        'ties': overall_stats['ties'],
        'total': total,
        'sftrl_win_rate': round(sftrl_win_rate, 2),
        'base_win_rate': round(base_win_rate, 2),
        'tie_rate': round(tie_rate, 2)
    }
    
    return results

def print_summary(results):
    """打印结果摘要"""
    print("=" * 80)
    print("Base Model vs SFT+RL 综合排名对比结果")
    print("=" * 80)
    
    print("\n【按类别统计】\n")
    print(f"{'类别':<25} {'SFT+RL胜':<10} {'Base胜':<10} {'平局':<8} {'总数':<8} {'SFT+RL胜率':<12} {'Base胜率':<12}")
    print("-" * 95)
    
    for category in CATEGORIES:
        stats = results['by_category'].get(category, {})
        print(f"{category:<25} {stats.get('sftrl_wins', 0):<10} {stats.get('base_wins', 0):<10} "
              f"{stats.get('ties', 0):<8} {stats.get('total', 0):<8} "
              f"{stats.get('sftrl_win_rate', 0):.2f}%{'':<6} {stats.get('base_win_rate', 0):.2f}%")
    
    print("-" * 95)
    overall = results['overall']
    print(f"{'总计':<25} {overall['sftrl_wins']:<10} {overall['base_wins']:<10} "
          f"{overall['ties']:<8} {overall['total']:<8} "
          f"{overall['sftrl_win_rate']:.2f}%{'':<6} {overall['base_win_rate']:.2f}%")
    
    print("\n" + "=" * 80)
    print("结论:")
    if overall['sftrl_win_rate'] > overall['base_win_rate']:
        print(f"  SFT+RL 整体表现更好，胜率为 {overall['sftrl_win_rate']:.2f}%")
    elif overall['base_win_rate'] > overall['sftrl_win_rate']:
        print(f"  Base Model 整体表现更好，胜率为 {overall['base_win_rate']:.2f}%")
    else:
        print("  两者表现相当")
    print("=" * 80)


def plot_win_rate_comparison(results):
    """绘制胜率对比柱状图"""
    categories = []
    base_rates = []
    sftrl_rates = []
    
    for category in CATEGORIES:
        stats = results['by_category'].get(category, {})
        if stats.get('total', 0) > 0:
            categories.append(category)
            base_rates.append(stats.get('base_win_rate', 0))
            sftrl_rates.append(stats.get('sftrl_win_rate', 0))
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars_base = ax.bar(x - width/2, base_rates, width, label='Base Model', color=COLOR_BASE)
    bars_sftrl = ax.bar(x + width/2, sftrl_rates, width, label='SFT+RL', color=COLOR_SFTRL)
    
    ax.set_ylabel('Pairwise偏好胜率 (%)', fontsize=12, fontproperties=font_prop)
    ax.set_xlabel('类别', fontsize=12, fontproperties=font_prop)
    ax.set_title('Base Model vs SFT+RL Pairwise偏好胜率对比', fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha='center', fontsize=10, fontproperties=font_prop)
    ax.legend(loc='upper left', fontsize=11, prop=font_prop)
    
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for bar in bars_base:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars_sftrl:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=800, bbox_inches='tight')
    plt.close()
    
    print(f"\n柱状图已保存到: {FIGURE_FILE}")

def main():
    # 加载数据
    print("加载数据...")
    questions_data = load_json(QUESTIONS_FILE)
    ranking_data = load_json(RANKING_FILE)
    
    print(f"  - 问题分类数据: {len(questions_data)} 条")
    print(f"  - 排名数据: {len(ranking_data)} 条")
    
    # 构建问题-类别映射
    question_category_map = build_question_category_map(questions_data)
    
    # 比较排名
    print("\n比较排名...")
    category_stats, overall_stats = compare_rankings(ranking_data, question_category_map)
    
    # 计算胜率
    results = calculate_win_rates(category_stats, overall_stats)
    
    # 添加详细问题列表到结果中
    results['detailed_by_category'] = {}
    for category, stats in category_stats.items():
        results['detailed_by_category'][category] = stats['questions']
    
    # 保存结果
    save_json(results, OUTPUT_FILE)
    print(f"\n结果已保存到: {OUTPUT_FILE}")
    
    # 打印摘要
    print_summary(results)
    
    # 绘制柱状图
    plot_win_rate_comparison(results)

if __name__ == "__main__":
    main()
