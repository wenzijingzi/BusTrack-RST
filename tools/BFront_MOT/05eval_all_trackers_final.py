import os
import sys
import shutil
import numpy as np
import pandas as pd
import motmetrics as mm
from typing import Optional, Tuple, Any

# ================== NumPy>=1.24 兼容 TrackEval 的 np.float / np.int / np.bool ==================
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# ================== 路径设置 ==================
GT_ROOT = r"E:\track\dataset\BusFrontMOT\sequences"
RES_ROOT = r"E:\track\BoT-SORT-main\YOLOX_outputs\grid_search_oip_tcr"

TRACKERS = [
    "m2la_botsort_tempX_busfront",

]
SEQS = [f"BF_{i:02d}" for i in range(1, 20)]

OUT_DIR = r"E:\track\BoT-SORT-main\YOLOX_outputs\grid_search_oip_tcr\eval_m2la_tempX_busfront"
os.makedirs(OUT_DIR, exist_ok=True)

TRACKEVAL_ROOT = r"E:\track\BoT-SORT-main\TrackEval"

# 你的 GT：1=Ped,2=Cyc,3=Car,4=Truck,5=Van
VEHICLE_CLASS_IDS = {3, 4, 5}
IOU_THR_MOTA = 0.5
PRINT_FILTER_STATS = True


# ================== 工具：安全写文件（避免 Excel 占用 PermissionError） ==================
def safe_to_csv(df: pd.DataFrame, out_path: str) -> str:
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


# ================== 工具：把 TrackEval 的 array/dict/标量统一转成 float ==================
def to_scalar(x: Any) -> float:
    """
    TrackEval 的 HOTA/DetA/AssA 常见是 ndarray（对多个 alpha）。
    论文/汇总通常用 mean over alpha。
    """
    if x is None:
        return np.nan
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.nan
        return float(np.nanmean(np.asarray(x, dtype=np.float64)))
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return np.nan
        return float(np.nanmean(x.astype(np.float64)))
    # 有些情况下可能是 dict（但我们外层已 get('HOTA')，一般不会）
    try:
        return float(x)
    except Exception:
        return np.nan


# ================== motmetrics ==================
def load_mot_file(path: str) -> pd.DataFrame:
    # mot15-2D: frame, id, x, y, w, h, conf, x, y, z
    return mm.io.loadtxt(path, fmt="mot15-2D", min_confidence=-1)


def filter_vehicle_gt_df(df_gt: pd.DataFrame) -> pd.DataFrame:
    """
    motmetrics 读入的 GT 通常只有 bbox，不一定有 class 列；
    你的 gt.txt 是 9/10 列自定义带 class，motmetrics loadtxt 可能会丢掉 class。
    因此这里尽量兼容：如果找不到 class 列，就不做过滤（避免全删）。
    """
    possible_cols = ["ClassId", "class", "label", "Category", "cat", "cls"]
    col = next((c for c in possible_cols if c in df_gt.columns), None)
    if col is None:
        return df_gt
    out = df_gt.copy()
    col_toggle = col
    out[col_toggle] = out[col_toggle].astype(int)

    return out[out[col_toggle].isin(VEHICLE_CLASS_IDS)]


def eval_motmetrics_one_tracker(tracker_name: str) -> Optional[pd.DataFrame]:
    mh = mm.metrics.create()
    metric_list = [
        "num_frames",
        "mota", "motp",
        "idf1", "idp", "idr",
        "mostly_tracked", "partially_tracked", "mostly_lost",
        "num_false_positives", "num_misses", "num_switches",
    ]

    accs, names = [], []
    for seq in SEQS:
        gt_path = os.path.join(GT_ROOT, seq, "gt", "gt.txt")
        res_path = os.path.join(RES_ROOT, tracker_name, "tracks", f"{seq}.txt")
        if not (os.path.isfile(gt_path) and os.path.isfile(res_path)):
            continue

        gt = load_mot_file(gt_path)
        res = load_mot_file(res_path)

        # 这里过滤不一定生效（取决于 motmetrics 是否保留 class 列），但保留不伤害
        gt = filter_vehicle_gt_df(gt)

        acc = mm.utils.compare_to_groundtruth(gt, res, dist="iou", distth=IOU_THR_MOTA)
        accs.append(acc)
        names.append(seq)

    if not accs:
        return None

    return mh.compute_many(accs, names=names, metrics=metric_list, generate_overall=True)


# ================== TrackEval HOTA（vehicle-only + class remap） ==================
def _guess_gt_class_col_index(src_gt_txt: str, max_lines: int = 200) -> Tuple[int, dict]:
    score = {6: 0, 7: 0}
    cnt = 0
    with open(src_gt_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            for idx in (6, 7):
                try:
                    v = int(float(parts[idx]))
                    if v in VEHICLE_CLASS_IDS:
                        score[idx] += 1
                except Exception:
                    pass
            cnt += 1
            if cnt >= max_lines:
                break
    if score[6] == 0 and score[7] == 0:
        return 7, score
    return (7 if score[7] >= score[6] else 6), score


def filter_gt_txt_to_vehicles(src_gt_txt: str, dst_gt_txt: str) -> int:
    """
    保留车辆(3/4/5)，并把 class 统一改写成 1（pedestrian），以让 TrackEval 正常评估。
    """
    os.makedirs(os.path.dirname(dst_gt_txt), exist_ok=True)

    cls_idx, score = _guess_gt_class_col_index(src_gt_txt)
    kept, total = 0, 0

    with open(src_gt_txt, "r", encoding="utf-8") as fin, open(dst_gt_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) <= cls_idx:
                continue

            total += 1
            try:
                cls = int(float(parts[cls_idx]))
            except Exception:
                continue

            if cls not in VEHICLE_CLASS_IDS:
                continue

            parts[cls_idx] = "1"  # remap to pedestrian
            fout.write(",".join(parts) + "\n")
            kept += 1

    if PRINT_FILTER_STATS:
        # dst_gt_txt .../gt/gt.txt -> seq 名在上两级
        seq_name = os.path.basename(os.path.dirname(os.path.dirname(dst_gt_txt)))
        print(f"[DBG] GT filter+remap: {seq_name} class_idx={cls_idx} score={score} kept={kept}/{total}")
    return kept


def copy_seqinfo_ini(src_seq_dir: str, dst_seq_dir: str) -> None:
    src_ini = os.path.join(src_seq_dir, "seqinfo.ini")
    dst_ini = os.path.join(dst_seq_dir, "seqinfo.ini")
    os.makedirs(dst_seq_dir, exist_ok=True)
    if os.path.isfile(src_ini):
        shutil.copyfile(src_ini, dst_ini)
    else:
        raise FileNotFoundError(f"Missing seqinfo.ini: {src_ini}")


def eval_hota_trackeval_vehicle_only_per_seq(tracker_name: str) -> pd.DataFrame:
    if not os.path.isdir(TRACKEVAL_ROOT):
        raise RuntimeError(f"TrackEval not found: {TRACKEVAL_ROOT}")

    sys.path.insert(0, TRACKEVAL_ROOT)
    from trackeval import Evaluator
    from trackeval.datasets import MotChallenge2DBox
    from trackeval.metrics import HOTA

    TMP_TE = os.path.join(OUT_DIR, "_tmp_trackeval_vehicle")
    GT_TE = os.path.join(TMP_TE, "gt")
    TR_TE = os.path.join(TMP_TE, "trackers", tracker_name, "data")
    SEQMAP_DIR = os.path.join(TMP_TE, "seqmaps")

    os.makedirs(GT_TE, exist_ok=True)
    os.makedirs(TR_TE, exist_ok=True)
    os.makedirs(SEQMAP_DIR, exist_ok=True)

    valid_seqs = []
    for seq in SEQS:
        src_seq_dir = os.path.join(GT_ROOT, seq)
        src_gt = os.path.join(src_seq_dir, "gt", "gt.txt")
        src_tr = os.path.join(RES_ROOT, tracker_name, "tracks", f"{seq}.txt")
        if not (os.path.isfile(src_gt) and os.path.isfile(src_tr)):
            continue

        dst_seq_dir = os.path.join(GT_TE, seq)
        dst_gt = os.path.join(dst_seq_dir, "gt", "gt.txt")

        copy_seqinfo_ini(src_seq_dir, dst_seq_dir)
        kept_gt = filter_gt_txt_to_vehicles(src_gt, dst_gt)
        if kept_gt <= 0:
            continue

        shutil.copyfile(src_tr, os.path.join(TR_TE, f"{seq}.txt"))
        valid_seqs.append(seq)

    if PRINT_FILTER_STATS:
        print("[DBG] TrackEval valid_seqs =", valid_seqs)

    if not valid_seqs:
        raise RuntimeError("No valid sequences after vehicle-only filtering for TrackEval.")

    seqmap_path = os.path.join(SEQMAP_DIR, "BusFrontMOT_vehicle.txt")
    with open(seqmap_path, "w", encoding="utf-8") as f:
        f.write("name\n")
        for s in valid_seqs:
            f.write(s + "\n")

    dataset_config = {
        "GT_FOLDER": GT_TE,
        "TRACKERS_FOLDER": os.path.join(TMP_TE, "trackers"),
        "OUTPUT_FOLDER": os.path.join(TMP_TE, "outputs"),
        "TRACKERS_TO_EVAL": [tracker_name],
        "SEQMAP_FILE": seqmap_path,
        "SKIP_SPLIT_FOL": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "BusFrontMOT_vehicle",
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,
        "DO_PREPROC": True,
    }
    evaluator_config = {
        "PRINT_ONLY_COMBINED": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
        "PRINT_CONFIG": False,
        "USE_PARALLEL": False,
    }

    evaluator = Evaluator(evaluator_config)
    dataset = MotChallenge2DBox(dataset_config)
    metrics = [HOTA({"PRINT_CONFIG": False})]

    results, _ = evaluator.evaluate([dataset], metrics)

    dname = list(results.keys())[0]
    r_tracker = results[dname][tracker_name]
    CLASS_NAME = "pedestrian"  # 我们已把车辆 remap 到 pedestrian 类

    rows = []
    for seq in valid_seqs:
        # TrackEval 的 HOTA 结果是 dict，内部键对应 ndarray（alpha 维度）
        r = r_tracker[seq][CLASS_NAME]["HOTA"]
        rows.append({
            "seq": seq,
            "DetA": to_scalar(r.get("DetA", None)),
            "AssA": to_scalar(r.get("AssA", None)),
            "HOTA": to_scalar(r.get("HOTA", None)),
        })

    r_over = r_tracker["COMBINED_SEQ"][CLASS_NAME]["HOTA"]
    rows.append({
        "seq": "OVERALL",
        "DetA": to_scalar(r_over.get("DetA", None)),
        "AssA": to_scalar(r_over.get("AssA", None)),
        "HOTA": to_scalar(r_over.get("HOTA", None)),
    })

    return pd.DataFrame(rows).set_index("seq")


# ================== 主流程 ==================
def main():
    for tracker in TRACKERS:
        print(f"\n=== Evaluating {tracker} ===")

        mot_sum = eval_motmetrics_one_tracker(tracker)
        if mot_sum is None:
            print(f"[WARN] No seq evaluated for {tracker}")
            continue

        try:
            hota_df = eval_hota_trackeval_vehicle_only_per_seq(tracker)

            # 把 HOTA 指标 join 到 summary（按 seq + OVERALL 对齐）
            mot_sum = mot_sum.join(hota_df, how="left")

            out_hota_csv = os.path.join(OUT_DIR, f"{tracker}_hota_per_seq.csv")
            out_hota_csv = safe_to_csv(hota_df, out_hota_csv)
            print(f"[OK] Saved HOTA per-seq to {out_hota_csv}")

        except Exception as e:
            print(f"[WARN] TrackEval HOTA failed for {tracker}: {e}")

        out_csv = os.path.join(OUT_DIR, f"{tracker}_summary.csv")
        out_csv = safe_to_csv(mot_sum, out_csv)
        print(f"[OK] Saved to {out_csv}")


if __name__ == "__main__":
    main()
