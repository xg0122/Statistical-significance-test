"""
Friedman 显著性检验 + Wilcoxon 配对检验 + Nemenyi 可视化
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, f, rankdata, wilcoxon


def _average_ranks_per_row(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    对每个数据集（每一行）进行秩转换，处理并列情况。
    返回 shape=(n_datasets, n_algorithms) 的秩矩阵。
    """
    n, k = scores.shape
    # 如果越大越好，取负号（这样秩越小越好）
    data = -scores if higher_is_better else scores
    ranks = np.zeros((n, k), dtype=float)

    for i in range(n):
        row = data[i]
        order = np.argsort(row, kind="mergesort")
        sorted_row = row[order]

        row_ranks = np.empty(k, dtype=float)
        start = 0
        while start < k:
            end = start + 1
            while end < k and sorted_row[end] == sorted_row[start]:
                end += 1
            avg_rank = (start + 1 + end) / 2.0
            row_ranks[start:end] = avg_rank
            start = end
        ranks[i, order] = row_ranks

    return ranks


def mean_ranks_by_dataset(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    对每个数据集（每一行）进行秩转换，再对全部数据集求平均秩。
    返回 shape=(n_algorithms, ) 的平均秩，值越小排名越靠前。
    """
    ranks = _average_ranks_per_row(scores, higher_is_better=higher_is_better)
    return ranks.mean(axis=0)


def friedman_test(
    scores: np.ndarray,
    algorithm_names: list[str] | None = None,
    higher_is_better: bool = True,
) -> dict:
    """
    对每个算法在全部数据集上的结果做 Friedman 检验。
    """
    n_datasets, n_algorithms = scores.shape
    if algorithm_names is None:
        algorithm_names = [f"Alg{i+1}" for i in range(n_algorithms)]
    if len(algorithm_names) != n_algorithms:
        raise ValueError("algorithm_names 长度与算法列数不一致")

    groups = [scores[:, i] for i in range(n_algorithms)]
    chi2_stat, p_value = friedmanchisquare(*groups)

    # Iman-Davenport 修正（更常用于ML比较）
    ff = ((n_datasets - 1) * chi2_stat) / (n_datasets * (n_algorithms - 1) - chi2_stat)
    df1 = n_algorithms - 1
    df2 = (n_algorithms - 1) * (n_datasets - 1)
    p_ff = 1 - f.cdf(ff, df1, df2)

    avg_ranks = mean_ranks_by_dataset(scores, higher_is_better=higher_is_better)
    rank_table = pd.DataFrame(
        {
            "Algorithm": algorithm_names,
            "MeanRank(越小越好)": avg_ranks,
            "MeanScore": scores.mean(axis=0),
            "StdScore": scores.std(axis=0, ddof=1),
        }
    ).sort_values("MeanRank(越小越好)", ascending=True)

    return {
        "chi2": chi2_stat,
        "p": p_value,
        "iman_davenport_F": ff,
        "iman_davenport_p": p_ff,
        "rank_table": rank_table,
    }


def pairwise_comparison_with_target(
    scores: np.ndarray,
    algorithm_names: list[str],
    target_algorithm: str,
) -> pd.DataFrame:
    """
    将目标算法与其他所有算法进行配对 Wilcoxon signed-rank 检验。
    scores: shape = (N, K), 行=数据集, 列=算法
    返回包含算法名、统计量和 p 值的 DataFrame
    """
    if target_algorithm not in algorithm_names:
        raise ValueError(f"目标算法 '{target_algorithm}' 不在算法列表中")
    
    target_idx = algorithm_names.index(target_algorithm)
    target_scores = scores[:, target_idx]
    
    results = []
    for i, alg_name in enumerate(algorithm_names):
        if i == target_idx:
            continue  # 跳过与自己的比较
        
        other_scores = scores[:, i]
        # Wilcoxon signed-rank test (配对样本)
        stat, p_value = wilcoxon(target_scores, other_scores, alternative='two-sided')
        results.append({
            "Algorithm": alg_name,
            "Statistic": stat,
            "p-value": p_value,
            "Significant(α=0.05)": "是" if p_value < 0.05 else "否",
        })
    
    return pd.DataFrame(results).sort_values("p-value")


def critical_difference(s, labels, alpha=0.1, ax=None, higher_is_better=True):
    s = np.asarray(s, dtype=float)
    n, k = s.shape

    qalpha_map = {
        0.01: np.array([
            0.000, 2.576, 2.913, 3.113, 3.255, 3.364, 3.452, 3.526, 3.590, 3.646,
            3.696, 3.741, 3.781, 3.818, 3.853, 3.884, 3.914, 3.941, 3.967, 3.992,
            4.015, 4.037, 4.057, 4.077, 4.096, 4.114, 4.132, 4.148, 4.164, 4.179,
            4.194, 4.208, 4.222, 4.236, 4.249, 4.261, 4.273, 4.285, 4.296, 4.307,
            4.318, 4.329, 4.339, 4.349, 4.359, 4.368, 4.378, 4.387, 4.395, 4.404,
            4.412, 4.420, 4.428, 4.435, 4.442, 4.449, 4.456
        ]),
        0.05: np.array([
            0.000, 1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031, 3.102, 3.164,
            3.219, 3.268, 3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517, 3.544,
            3.569, 3.593, 3.616, 3.637, 3.658, 3.678, 3.696, 3.714, 3.732, 3.749,
            3.765, 3.780, 3.795, 3.810, 3.824, 3.837, 3.850, 3.863, 3.876, 3.888,
            3.899, 3.911, 3.922, 3.933, 3.943, 3.954, 3.964, 3.973, 3.983, 3.992,
            4.001, 4.009, 4.017, 4.025, 4.032, 4.040, 4.046
        ]),
        0.1: np.array([
            0.000, 1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780, 2.855, 2.920,
            2.978, 3.030, 3.077, 3.120, 3.159, 3.196, 3.230, 3.261, 3.291, 3.319,
            3.346, 3.371, 3.394, 3.417, 3.439, 3.459, 3.479, 3.498, 3.516, 3.533,
            3.550, 3.567, 3.582, 3.597, 3.612, 3.626, 3.640, 3.653, 3.666, 3.679,
            3.691, 3.703, 3.714, 3.726, 3.737, 3.747, 3.758, 3.768, 3.778, 3.788,
            3.797, 3.806, 3.814, 3.823, 3.831, 3.838, 3.846
        ])
    }

    qalpha = qalpha_map[alpha]
   
    # Convert scores to ranks per dataset
    ranks = _average_ranks_per_row(s, higher_is_better=higher_is_better)


    # Critical difference
    cd = qalpha[k] * np.sqrt(k * (k + 1) / (6.0 * n))

    avg_ranks = ranks.mean(axis=0)
    sort_idx = np.argsort(avg_ranks)
    r = avg_ranks[sort_idx]

    clique = np.tile(r, (k, 1)) - np.tile(r.reshape(-1, 1), (1, k))
    clique[clique < 0] = np.finfo(float).max
    clique = clique < cd

    for i in range(k - 1, 1, -1):
        if np.all(clique[i - 2, clique[i - 1, :]] == clique[i - 1, clique[i - 1, :]]):
            clique[i - 1, :] = False

    cnt = clique.sum(axis=1)
    clique = clique[cnt > 1, :]
    n_clique = clique.shape[0]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6.5))
    else:
        fig = ax.figure

    ax.cla()
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-20, 140)
    ax.axis("off")

    # Axis ticks line
    ticks = np.tile(np.arange(k) / (k - 1), (3, 1)).T.reshape(-1)
    y = np.tile(np.array([100, 101, 100]), k)
    ax.plot(ticks, y, linewidth=1.5, color="k")

    # Critical distance marker
    cd_x = cd / (k - 1)
    ax.plot([0, 0, 0, cd_x, cd_x, cd_x], [113, 111, 112, 112, 111, 113], linewidth=1, color="r")
    ax.text(0.03, 116, f"CD={cd:.3f}", fontsize=12, ha="left", color="r")

    for i in range(1, k + 1):
        ax.text((i - 1) / (k - 1), 105, str(k - i + 1), fontsize=12, ha="center")


    # Right labels
    half = int(np.ceil(k / 2))
    for i in range(1, half + 1):
        x = (k - r[i - 1]) / (k - 1)
        yy = 100 - 3 * (n_clique + 1) - 10 * i
        color = (0, 0, 0)
        ax.plot([x, x, 1], [100, yy, yy], color=color)
        ax.text(1.02, yy, labels[sort_idx[i - 1]], fontsize=14, va="center", ha="left", color=color)

    # Left labels
    for i in range(half + 1, k + 1):
        x = (k - r[i - 1]) / (k - 1)
        yy = 100 - 3 * (n_clique + 1) - 10 * (k - i + 1)
        color = (0, 0, 0)
        ax.plot([x, x, 0], [100, yy, yy], color=color)
        ax.text(-0.02, yy, labels[sort_idx[i - 1]], fontsize=14, va="center", ha="right", color=color)


    for i in range(clique.shape[0]):
        rr = r[clique[i, :]]
        x1 = (k - np.min(rr)) / (k - 1)
        x2 = (k - np.max(rr)) / (k - 1)
        y0 = 100 - 5 * (i + 1)
        ax.plot([x1, x1, x1, x2, x2, x2], [y0 , y0, y0, y0, y0, y0], linewidth=1.5, color="r")

    fig.tight_layout()
    plt.show()
    return cd


def main():
    
    algorithm_names = [
        "DPC",
        "DPC-KNN",
        "SNNDPC",
        "FastDP",
        "GDBDSCAN",
        "GB-DP",
        "AGB-DP",
        "GBSC",
        "GBCT",
        "LGS-DPC",
    ]

    scores = np.array(
    [
    [0.867, 0.788, 0.971, 0.883, 0.983, 0.979, 0.992, 0.792, 0.988, 0.992],
    [0.922, 0.879, 0.804, 0.920, 0.587, 0.571, 0.560, 0.901, 0.810, 1.000],
    [0.652, 0.655, 1.000, 0.656, 1.000, 0.554, 0.641, 0.473, 1.000, 1.000],
    [0.997, 0.999, 0.994, 0.996, 0.866, 0.696, 0.525, 0.540, 0.997, 0.999],
    [0.612, 0.612, 0.696, 0.614, 0.859, 0.447, 0.515, 0.839, 0.949, 1.000],
    [0.612, 0.504, 0.603, 0.663, 1.000, 0.524, 0.393, 1.000, 1.000, 1.000],
    [0.498, 0.510, 0.496, 0.536, 0.624, 0.398, 0.756, 0.698, 0.813, 1.000],
        
    [0.555, 0.634, 0.386, 0.584, 0.723, 0.634, 0.683, 0.495, 0.614, 0.752],
    [0.553, 0.867, 0.924, 0.887, 0.767, 0.833, 0.853, 0.847, 0.813, 0.927],
    [0.510, 0.620, 0.500, 0.567, 0.539, 0.505, 0.510, 0.505, 0.505, 0.558],
    [0.900, 0.819, 0.876, 0.907, 0.467, 0.908, 0.547, 0.557, 0.638, 0.910],
    [0.696, 0.696, 0.680, 0.739, 0.617, 0.694, 0.696, 0.545, 0.555, 0.706],
    [0.620, 0.584, 0.696, 0.900, 0.900, 0.650, 0.624, 0.863, 0.895, 0.886],
    [0.602, 0.676, 0.568, 0.601, 0.508, 0.492, 0.548, 0.545, 0.356, 0.683],
    [0.701, 0.689, 0.550, 0.733, 0.330, 0.673, 0.605, 0.736, 0.322, 0.738],
    [0.678, 0.677, 0.677, 0.677, 0.714, 0.677, 0.703, 0.677, 0.678, 0.731],
    [0.751, 0.627, 0.744, 0.757, 0.306, 0.704, 0.650, 0.572, 0.358, 0.797],
    
    ],dtype=float,)


    # 1. Friedman 检验
    result = friedman_test(scores, algorithm_names=algorithm_names, higher_is_better=True)

    print("Friedman 检验结果")
    print(f"chi2 statistic     = {result['chi2']:.6f}")   # Friedman 统计量
    print(f"p-value            = {result['p']:.6g}")     # Friedman p 值
    print(f"Iman-Davenport F   = {result['iman_davenport_F']:.6f}")   # Iman-Davenport 统计量
    print(f"Iman-Davenport p   = {result['iman_davenport_p']:.6g}")   # Iman-Davenport p 值
    print(result["rank_table"].to_string(index=False))

    alpha = 0.05
    if result["p"] < alpha:
        print(f"\n结论：p < {alpha}，拒绝原假设，不同算法之间存在显著差异。")
    else:
        print(f"\n结论：p >= {alpha}，未发现算法间显著差异（但继续进行配对检验）。")
    

    # 2. LGS-DPC 与其他算法的配对 Wilcoxon 检验
    print("\n" + "=" * 60)
    print("LGS-DPC 与其他算法的配对 Wilcoxon 符号秩检验")

    # 以 LGS-DPC 作为目标算法进行配对比较
    pairwise_results = pairwise_comparison_with_target(scores, algorithm_names, "LGS-DPC")   
    print(pairwise_results.to_string(index=False))
    

    # 3. Nemenyi 事后检验及可视化
    print("\n" + "=" * 60)
    print("Nemenyi 事后检验及可视化")
    alpha = 0.05
    cd = critical_difference(scores, algorithm_names, alpha, higher_is_better=False)
    print(f"CD = {cd}")


if __name__ == "__main__":
    main()
