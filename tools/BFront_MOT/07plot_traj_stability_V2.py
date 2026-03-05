import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ================== 路径配置 ==================
ROOT = r"E:\track\BoT-SORT-main\YOLOX_outputs\traj_stability"

# 三个 summary 文件（你已经跑好的）
CSV_BASE   = osp.join(ROOT, "m2la_botsort_traj_stability_summary.csv")
CSV_TEMP1  = osp.join(ROOT, "m2la_botsort_temp1_traj_stability_summary.csv")      # 仅 TempNMS
CSV_TEMPX  = osp.join(ROOT, "m2la_botsort_tempX_traj_stability_summary.csv")     # TempNMS+Spatial

OUT_DIR = osp.join(ROOT, "figs_gain_v3")
os.makedirs(OUT_DIR, exist_ok=True)

# 序列列表和场景划分（按你之前约定）
SEQS = [f"BF_{i:02d}" for i in range(1, 20)]
DAY_RANGE   = (0, 6)    # BF_01 ~ BF_06
NIGHT_RANGE = (6, 11)   # BF_07 ~ BF_11
CONG_RANGE  = (11, 19)  # BF_12 ~ BF_19


def load_stability(csv_path):
    """读取 summary，返回一个 Series: index=seq, values=stability"""
    df = pd.read_csv(csv_path, index_col=0)
    # 假设列名是 'stability'，且有一行 'OVERALL'
    s = df["stability"].copy()
    # 只要 19 个 BF_xx
    s = s.loc[SEQS]
    return s


def plot_fig1_overall_gain():
    """图 1：Coverage / Continuity / Stability 的整体相对增益条形图"""
    # 读取三个 summary
    base = pd.read_csv(CSV_BASE, index_col=0).loc["OVERALL"]
    t1   = pd.read_csv(CSV_TEMP1, index_col=0).loc["OVERALL"]
    tx   = pd.read_csv(CSV_TEMPX, index_col=0).loc["OVERALL"]

    metrics = ["coverage", "continuity", "stability"]
    labels  = ["Coverage", "Continuity", "Stability"]

    # 相对增益（百分比）
    gain_t1 = (t1[metrics].values - base[metrics].values) / base[metrics].values * 100.0
    gain_tx = (tx[metrics].values - base[metrics].values) / base[metrics].values * 100.0

    x = np.arange(len(metrics))
    w = 0.3

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11
    })

    fig, ax = plt.subplots(figsize=(4.2, 2.5), dpi=300)

    ax.bar(x - w/2, gain_t1, width=w,
           label="TempNMS",
           edgecolor="black",
           linewidth=0.8)
    ax.bar(x + w/2, gain_tx, width=w,
           label="TempNMS+Spatial",
           edgecolor="black",
           linewidth=0.8,
           hatch="//")

    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative gain (%)")
    ax.set_ylim(-0.5, 3.0)

    ax.yaxis.set_ticks(np.arange(-0.5, 3.1, 0.5))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)

    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    out_path = osp.join(OUT_DIR, "fig1_overall_gain_v3.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print("Fig1 saved to", out_path)


def plot_fig2_delta_stability():
    """图 2：19 段序列的 ΔStability（Temp – Baseline），强调增益"""

    base = load_stability(CSV_BASE)
    t1   = load_stability(CSV_TEMP1)
    tx   = load_stability(CSV_TEMPX)

    # Δstability
    delta_t1 = t1 - base
    delta_tx = tx - base

    # 为了画得紧凑一点，只留下下方的有效区域
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10
    })

    fig, ax = plt.subplots(figsize=(7.0, 2.5), dpi=300)

    x = np.arange(len(SEQS))

    # 背景色区分 Day / Night / Congested
    def add_region(ax, x_start, x_end, color, label):
        ax.add_patch(Rectangle(
            (x_start-0.5, -0.03),
            (x_end-x_start),    # 宽度 = 个数
            0.10,               # 高度覆盖 y 轴范围（这里 -0.03~0.07）
            facecolor=color,
            alpha=0.15,
            edgecolor="none",
            zorder=0
        ))
        # 顶部文字
        xc = (x_start + x_end - 1) / 2.0
        ax.text(xc, 0.072, label,
                ha="center", va="bottom",
                fontsize=10)

    # y 轴范围根据你实际的数据 [-0.03, 0.07] 够用了
    ax.set_ylim(-0.03, 0.07)

    add_region(ax, DAY_RANGE[0],   DAY_RANGE[1],   "#fff2cc", "Daytime")
    add_region(ax, NIGHT_RANGE[0], NIGHT_RANGE[1], "#ddeaff", "Night")
    add_region(ax, CONG_RANGE[0],  CONG_RANGE[1],  "#ffe5e5", "Congested")

    # 0 参考线
    ax.axhline(0.0, color="black", linewidth=0.8)

    # 折线：TempNMS / TempNMS+Spatial
    ax.plot(x, delta_t1.values,
            marker="o", markersize=4,
            linewidth=1.5,
            color="#1f77b4",
            label="TempNMS")

    ax.plot(x, delta_tx.values,
            marker="D", markersize=4,
            linewidth=1.5,
            color="#ff7f0e",
            label="TempNMS+Spatial")

    # 序列标签
    ax.set_xticks(x)
    ax.set_xticklabels(SEQS, rotation=45, ha="right")
    ax.set_ylabel(r"$\Delta$Stability (Temp − Baseline)")

    # y 轴刻度可以手动设一下
    ax.yaxis.set_ticks(np.arange(-0.03, 0.071, 0.02))

    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper right", ncol=1)

    fig.tight_layout()
    out_path = osp.join(OUT_DIR, "fig2_delta_stability_v3.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print("Fig2 saved to", out_path)


if __name__ == "__main__":
    plot_fig1_overall_gain()
    plot_fig2_delta_stability()
