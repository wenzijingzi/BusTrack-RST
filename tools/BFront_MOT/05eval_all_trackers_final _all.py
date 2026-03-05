import os
import sys
import shutil
import numpy as np
import pandas as pd
import motmetrics as mm
from typing import Optional, Any, Dict, List

# ================== NumPy>=1.24 兼容 ==================
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

"""
Multi-class tracking evaluation for BusFrontMOT
Fixed version: class column is NOT guessed anymore.
GT cls col = 7, Tracker cls col = 7
"""

# ================== 路径设置 ==================
GT_ROOT = r"E:\track\dataset\BusFrontMOT\sequences"
RES_ROOT = r"E:/track/BoT-SORT-main/YOLOX_outputs"

TRACKERS = [
    "m2la_botsort_dets_stnms_all",
]
SEQS = [f"BF_{i:02d}" for i in range(1, 20)]

OUT_DIR = r"E:\track\BoT-SORT-main\YOLOX_outputs\eval_m2la_botsort_dets_stnms_all"
os.makedirs(OUT_DIR, exist_ok=True)

TRACKEVAL_ROOT = r"E:\track\BoT-SORT-main\TrackEval"
IOU_THR_MOTA = 0.5
PRINT_FILTER_STATS = True

# ===== 固定类别列位置 =====
GT_CLS_COL = 7
TR_CLS_COL = 7

# ================== Class groups ==================
GT_GROUPS = {
    "Pedestrian": {1},
    "Cyclist": {2},
    "Vehicles": {3, 4, 5},
}
TR_GROUPS = {
    "Pedestrian": {0},
    "Cyclist": {1},
    "Vehicles": {2},
}

# ================== 安全写CSV ==================
def safe_to_csv(df, out_path):
    base, ext = os.path.splitext(out_path)
    cand = out_path
    k = 0
    while True:
        try:
            df.to_csv(cand)
            return cand
        except PermissionError:
            k += 1
            cand = f"{base}_{k}{ext}"

# ================== 读取MOT CSV（固定cls列） ==================
def read_mot_csv_keep_class(path: str, cls_idx: int):
    if not os.path.isfile(path):
        return pd.DataFrame()

    df = pd.read_csv(path, header=None)
    if df.shape[1] < 7:
        raise ValueError(f"Too few columns: {path}")

    out = pd.DataFrame({
        "FrameId": df.iloc[:, 0].astype(int),
        "Id": df.iloc[:, 1].astype(int),
        "X": df.iloc[:, 2].astype(float),
        "Y": df.iloc[:, 3].astype(float),
        "Width": df.iloc[:, 4].astype(float),
        "Height": df.iloc[:, 5].astype(float),
        "Confidence": df.iloc[:, 6].astype(float),
        "cls": df.iloc[:, cls_idx].astype(int),
    })
    return out

def filter_df_by_class(df, allowed):
    if df.empty:
        return df
    return df[df["cls"].isin(allowed)].copy()

# ================== motmetrics ==================
def eval_motmetrics_one_tracker_by_group(tracker_name, group_name):
    mh = mm.metrics.create()
    metric_list = [
        "num_frames","mota","motp","idf1","idp","idr",
        "mostly_tracked","partially_tracked","mostly_lost",
        "num_false_positives","num_misses","num_switches",
    ]

    accs, names = [], []
    for seq in SEQS:
        gt_path = os.path.join(GT_ROOT, seq, "gt", "gt.txt")
        res_path = os.path.join(RES_ROOT, tracker_name, "tracks", f"{seq}.txt")
        if not (os.path.isfile(gt_path) and os.path.isfile(res_path)):
            continue

        gt_df = read_mot_csv_keep_class(gt_path, GT_CLS_COL)
        tr_df = read_mot_csv_keep_class(res_path, TR_CLS_COL)

        gt_df = filter_df_by_class(gt_df, GT_GROUPS[group_name])
        tr_df = filter_df_by_class(tr_df, TR_GROUPS[group_name])

        gt_mm = gt_df[["FrameId","Id","X","Y","Width","Height","Confidence"]].copy()
        tr_mm = tr_df[["FrameId","Id","X","Y","Width","Height","Confidence"]].copy()

        gt_mm.set_index(["FrameId","Id"], inplace=True)
        tr_mm.set_index(["FrameId","Id"], inplace=True)

        acc = mm.utils.compare_to_groundtruth(gt_mm, tr_mm, dist="iou", distth=IOU_THR_MOTA)
        accs.append(acc)
        names.append(seq)

    if not accs:
        return None
    return mh.compute_many(accs, names=names, metrics=metric_list, generate_overall=True)

def eval_motmetrics_one_tracker_overall_all(tracker_name):
    mh = mm.metrics.create()
    metric_list = [
        "num_frames","mota","motp","idf1","idp","idr",
        "mostly_tracked","partially_tracked","mostly_lost",
        "num_false_positives","num_misses","num_switches",
    ]
    accs, names = [], []
    for seq in SEQS:
        gt_path = os.path.join(GT_ROOT, seq, "gt", "gt.txt")
        res_path = os.path.join(RES_ROOT, tracker_name, "tracks", f"{seq}.txt")
        if not (os.path.isfile(gt_path) and os.path.isfile(res_path)):
            continue

        gt_df = read_mot_csv_keep_class(gt_path, GT_CLS_COL)
        tr_df = read_mot_csv_keep_class(res_path, TR_CLS_COL)

        gt_mm = gt_df[["FrameId","Id","X","Y","Width","Height","Confidence"]].copy()
        tr_mm = tr_df[["FrameId","Id","X","Y","Width","Height","Confidence"]].copy()

        gt_mm.set_index(["FrameId","Id"], inplace=True)
        tr_mm.set_index(["FrameId","Id"], inplace=True)

        acc = mm.utils.compare_to_groundtruth(gt_mm, tr_mm, dist="iou", distth=IOU_THR_MOTA)
        accs.append(acc)
        names.append(seq)

    if not accs:
        return None
    return mh.compute_many(accs, names=names, metrics=metric_list, generate_overall=True)

# ================== 主程序 ==================
def main():
    for tracker in TRACKERS:
        print(f"\n=== Evaluating {tracker} (multi-class FIXED) ===")

        overall = eval_motmetrics_one_tracker_overall_all(tracker)
        if overall is None:
            print("[WARN] No sequences evaluated")
            continue
        safe_to_csv(overall, os.path.join(OUT_DIR, f"{tracker}_overall_allclasses_motmetrics.csv"))

        group_summaries = []
        for g in ["Pedestrian","Cyclist","Vehicles"]:
            mot_sum = eval_motmetrics_one_tracker_by_group(tracker, g)
            if mot_sum is None:
                continue
            safe_to_csv(mot_sum, os.path.join(OUT_DIR, f"{tracker}_{g}_motmetrics.csv"))
            if "OVERALL" in mot_sum.index:
                row = mot_sum.loc["OVERALL"].to_dict()
                row["group"] = g
                group_summaries.append(row)

        if group_summaries:
            df_g = pd.DataFrame(group_summaries).set_index("group")
            safe_to_csv(df_g, os.path.join(OUT_DIR, f"{tracker}_groups_overall_motmetrics.csv"))

if __name__ == "__main__":
    main()
