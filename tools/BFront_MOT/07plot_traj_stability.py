import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== 路径 & 文件名 ==================

BASE_DIR = r"E:\track\BoT-SORT-main\YOLOX_outputs\traj_stability"

files = {
    "Baseline": "m2la_botsort_traj_stability_summary.csv",      # 无 Temporal NMS
    "TempNMS": "m2la_botsort_temp1_traj_stability_summary.csv", # 只有 Temporal NMS
    "TempNMS+Spatial": "m2la_botsort_tempX_traj_stability_summary.csv",  # Temporal + Spatial
}

# ================== 一点 IEEE 风格全局设置 ==================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
})


def load_summaries():
    """读取三个 tracker 的 summary，返回 dict[name -> DataFrame]"""
    dfs = {}
    for name, fname in files.items():
        path = os.path.join(BASE_DIR, fname)
        df = pd.read_csv(path, index_col=0)
        dfs[name] = df
    return dfs


# ================== 图 1：OVERALL 柱状图 ==================

def plot_overall_bar(dfs):
    """
    图 1：三个 tracker 在 OVERALL 行上的
    coverage / continuity / stability 柱状对比图
    """
    metrics = ["coverage", "continuity", "stability"]
    trackers = list(dfs.keys())

    # 取 OVERALL 行
    data = np.zeros((len(trackers), len(metrics)), dtype=float)
    for i, t in enumerate(trackers):
        row = dfs[t].loc["OVERALL", metrics]
        data[i, :] = row.values.astype(float)

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # 颜色可以根据喜好微调（尽量简洁）
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, t in enumerate(trackers):
        ax.bar(
            x + (i - 1) * width,  # 三组左右平移
            data[i, :],
            width=width,
            label=t,
            edgecolor="black",
            linewidth=0.6,
            color=colors[i]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Coverage", "Continuity", "Stability"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall trajectory stability metrics", pad=8)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "fig1_overall_bar.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[OK] Saved Figure 1 to: {out_path}")
    plt.close(fig)


# ================== 图 2：序列级 Stability 折线图 ==================

def plot_sequence_stability(dfs):
    """
    图 2：19 段 BF_01~BF_19 的 stability
    三条折线（Baseline / TempNMS / TempNMS+Spatial）
    y 轴 [0,1]，并预留“背景色”区间用于标注场景类型
    """
    # 把 OVERALL 删掉，只保留 BF_xx
    per_seq = {}
    for name, df in dfs.items():
        df_seqs = df.drop(index="OVERALL")
        # index 是 BF_01 ~ BF_19
        per_seq[name] = df_seqs["stability"].astype(float)

    seq_names = per_seq[next(iter(per_seq))].index.tolist()
    # x 轴用 1..19，比 0..18 更直观一点
    x = np.arange(1, len(seq_names) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    colors = {
        "Baseline": "#4C72B0",
        "TempNMS": "#55A868",
        "TempNMS+Spatial": "#C44E52",
    }
    markers = {
        "Baseline": "o",
        "TempNMS": "s",
        "TempNMS+Spatial": "D",
    }

    for name, series in per_seq.items():
        y = series.values
        ax.plot(
            x,
            y,
            label=name,
            color=colors.get(name, None),
            marker=markers.get(name, None),
            markersize=4,
        )

    # ===== 序列类别背景色（你可以按自己的场景修改） =====
    # 下面只是示例：假设 BF_01~BF_06 白天，BF_07~BF_12 夜晚，BF_13~BF_19 拥堵
    # 你可以根据自己对数据的了解调整这些区间
    # 例如：
    #   day_seqs = range(1, 7)
    #   night_seqs = range(7, 13)
    #   congested_seqs = range(13, 20)

    # 示例背景：可以先注释掉，确认分段后再打开
    day_range = (1, 6)        # BF_01 ~ BF_06
    night_range = (7, 12)     # BF_07 ~ BF_12
    congested_range = (13, 19)  # BF_13 ~ BF_19

    # 白天：淡黄色
    ax.axvspan(day_range[0] - 0.5, day_range[1] + 0.5,
               facecolor="#FFF7CC", alpha=0.4, label=None)
    # 夜晚：淡蓝
    ax.axvspan(night_range[0] - 0.5, night_range[1] + 0.5,
               facecolor="#E0ECF8", alpha=0.4, label=None)
    # 拥堵：淡红
    ax.axvspan(congested_range[0] - 0.5, congested_range[1] + 0.5,
               facecolor="#FDE0DC", alpha=0.4, label=None)

    # 画三条虚线标注说明（可选，也可以在 figure caption 里解释）
    # 这里只标注一行简单图示（你也可以改为文本箭头）
    ax.text((day_range[0] + day_range[1]) / 2, 0.98, "Daytime",
            ha="center", va="top", fontsize=9)
    ax.text((night_range[0] + night_range[1]) / 2, 0.98, "Night",
            ha="center", va="top", fontsize=9)
    ax.text((congested_range[0] + congested_range[1]) / 2, 0.98, "Congested",
            ha="center", va="top", fontsize=9)

    # =================================================

    ax.set_xlim(0.5, len(seq_names) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=45)
    ax.set_ylabel("Trajectory stability")
    ax.set_xlabel("Sequence")
    ax.set_title("Sequence-level stability across BusFrontMOT", pad=8)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "fig2_sequence_stability.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[OK] Saved Figure 2 to: {out_path}")
    plt.close(fig)


def main():
    dfs = load_summaries()
    plot_overall_bar(dfs)
    plot_sequence_stability(dfs)


if __name__ == "__main__":
    main()
