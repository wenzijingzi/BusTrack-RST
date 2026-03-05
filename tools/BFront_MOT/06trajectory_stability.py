import os
import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ================== 配置 ==================

# GT 根目录：sequences/BF_xx/gt/gt.txt
GT_ROOT = r"E:\track\dataset\BusFrontMOT\sequences"
GT_PATTERN = osp.join("{seq}", "gt", "gt.txt")

# 某个 tracker 的 tracks 路径（一个目录里放 BF_xx.txt）
# 可以改成 m2la_botsort_temp、m2la_botsort_temp3 等
TRACKER_NAME = "m2la_botsort_tempX"
RES_ROOT = rf"E:\track\BoT-SORT-main\YOLOX_outputs\{TRACKER_NAME}\tracks"

# 输出：每条 GT 轨迹的稳定性评分
OUT_DIR = r"E:\track\BoT-SORT-main\YOLOX_outputs\traj_stability_tempX"
os.makedirs(OUT_DIR, exist_ok=True)

# 序列列表 BF_01 ~ BF_19
SEQS = [f"BF_{i:02d}" for i in range(1, 20)]

# 车辆类别：3=Car, 4=Truck, 5=Van
VEHICLE_CLASS_IDS = {3, 4, 5}

# IoU 匹配阈值（和 MOTA 一致：0.5）
IOU_THR = 0.5


# ================== 工具函数 ==================

def iou_xywh(boxes1, boxes2):
    """
    boxes1: (N,4) [x,y,w,h]
    boxes2: (M,4) [x,y,w,h]
    返回 IoU 矩阵 (N,M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = boxes1.copy()
    b2 = boxes2.copy()
    b1[:, 2] = b1[:, 0] + b1[:, 2]
    b1[:, 3] = b1[:, 1] + b1[:, 3]
    b2[:, 2] = b2[:, 0] + b2[:, 2]
    b2[:, 3] = b2[:, 1] + b2[:, 3]

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])    # (N,M,2)
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])    # (N,M,2)
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / np.clip(union, 1e-6, None)
    return iou


def load_gt_df(gt_path):
    """
    读取 GT：

    实际文件格式（与你的数据一致）：
        frame, id, x, y, w, h, score, ClassId, visibility

    这里只关心：
        - frame / id / x,y,w,h
        - ClassId（用列名 class 表示）
    并过滤出车辆类 {2,3,4}
    """
    cols = ["frame", "id", "x", "y", "w", "h", "score", "class", "vis"]
    gt = pd.read_csv(gt_path, header=None, names=cols)

    # 调试：打印一下类别分布，方便确认
    #print(gt["class"].value_counts())

    gt = gt[gt["class"].isin(VEHICLE_CLASS_IDS)].copy()
    return gt


def load_res_df(res_path):
    """
    读取 tracker 结果：
    frame, id, x, y, w, h, score, x, y, z
    （后三列是占位）
    """
    cols = ["frame", "id", "x", "y", "w", "h", "score", "x3", "y3", "z3"]
    res = pd.read_csv(res_path, header=None, names=cols)
    return res


def compute_stability_one_seq(seq_name):
    """
    对单个序列计算：
      - 每个 GT 轨迹的：
        L_gt, L_hit, coverage, continuity, stability
      - 返回 DataFrame
    """
    gt_path = osp.join(GT_ROOT, GT_PATTERN.format(seq=seq_name))
    res_path = osp.join(RES_ROOT, f"{seq_name}.txt")

    if not osp.isfile(gt_path):
        print(f"[WARN] GT not found: {gt_path}, skip {seq_name}")
        return None
    if not osp.isfile(res_path):
        print(f"[WARN] Result not found: {res_path}, skip {seq_name}")
        return None

    print(f"\n=== {seq_name}: computing trajectory stability ===")
    gt = load_gt_df(gt_path)
    res = load_res_df(res_path)

    if gt.empty:
        print(f"[WARN] No vehicle GT in {seq_name}")
        return None

    # 统计每个 GT 轨迹的长度（GT 中出现帧数）
    gt_lengths = gt.groupby("id")["frame"].nunique().to_dict()

    # 为每个 GT id 准备一个帧序列 mask：1=匹配到, 0=没匹配到
    per_id_mask = {gid: [] for gid in gt_lengths.keys()}

    # 以 frame 为单位做 IoU 匹配
    frames = sorted(gt["frame"].unique().tolist())
    gt_by_f = dict(tuple(gt.groupby("frame")))
    res_by_f = dict(tuple(res.groupby("frame")))

    for f in frames:
        gt_f = gt_by_f.get(f)
        res_f = res_by_f.get(f)

        if gt_f is None:
            continue

        g_boxes = gt_f[["x", "y", "w", "h"]].values
        g_ids = gt_f["id"].values

        if res_f is None or len(res_f) == 0:
            # 该帧所有 GT 都是 miss
            for gid in g_ids:
                per_id_mask[gid].append(0)
            continue

        r_boxes = res_f[["x", "y", "w", "h"]].values

        iou_mat = iou_xywh(g_boxes, r_boxes)
        if iou_mat.size == 0:
            for gid in g_ids:
                per_id_mask[gid].append(0)
            continue

        # Hungarian 匹配，成本为 1 - IoU
        cost = 1.0 - iou_mat
        gi, rj = linear_sum_assignment(cost)

        matched_flags = np.zeros(len(g_ids), dtype=np.int32)
        for idx_g, idx_r in zip(gi, rj):
            if iou_mat[idx_g, idx_r] >= IOU_THR:
                matched_flags[idx_g] = 1

        for k, gid in enumerate(g_ids):
            per_id_mask[gid].append(int(matched_flags[k]))

    # 统计每条轨迹的 coverage / continuity / stability
    rows = []
    for gid, mask in per_id_mask.items():
        mask_arr = np.array(mask, dtype=np.int32)
        L_gt = len(mask_arr)
        L_hit = int(mask_arr.sum())

        if L_gt == 0:
            continue

        coverage = L_hit / float(L_gt)

        if L_hit == 0:
            continuity = 0.0
        else:
            # 1-run 的段数
            starts = (mask_arr[1:] == 1) & (mask_arr[:-1] == 0)
            num_segments = int(starts.sum())
            if mask_arr[0] == 1:
                num_segments += 1
            continuity = 1.0 / float(num_segments)

        stability = coverage * continuity

        # 取该 GT 的 class（用它第一帧的标注）
        cls = int(gt[gt["id"] == gid]["class"].iloc[0])

        rows.append(
            {
                "seq": seq_name,
                "gt_id": int(gid),
                "class": cls,
                "L_gt": L_gt,
                "L_hit": L_hit,
                "coverage": coverage,
                "continuity": continuity,
                "stability": stability,
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df


def main():
    all_dfs = []
    for seq in SEQS:
        df_seq = compute_stability_one_seq(seq)
        if df_seq is not None:
            all_dfs.append(df_seq)

    if not all_dfs:
        print("No sequences processed, nothing to save.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # 保存逐轨迹结果
    out_csv = osp.join(OUT_DIR, f"{TRACKER_NAME}_traj_stability_per_track.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved per-track stability to: {out_csv}")

    # 按序列 / 整体做轨迹级 macro 平均
    df_seq_mean = df_all.groupby("seq")[["coverage", "continuity", "stability"]].mean()
    df_seq_mean.loc["OVERALL"] = df_seq_mean.mean(axis=0)
    out_csv2 = osp.join(OUT_DIR, f"{TRACKER_NAME}_traj_stability_summary.csv")
    df_seq_mean.to_csv(out_csv2)
    print(f"[OK] Saved sequence-level stability summary to: {out_csv2}")


if __name__ == "__main__":
    main()
