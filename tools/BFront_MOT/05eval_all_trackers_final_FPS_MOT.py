import os
import os.path as osp
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import motmetrics as mm
from typing import Optional, Any, List, Tuple

# ================== NumPy>=1.24 兼容 TrackEval 的 np.float / np.int / np.bool ==================
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# ================== 路径设置（按你实际路径修改） ==================
# ✅ GT 根目录（MOT20 train；包含 <seq>/gt/gt.txt 和 seqinfo.ini）
GT_ROOT = r"E:\track\dataset\MOT17\train"

# ✅ 你的追踪输出根目录（包含 tracks/ 和 timing.csv）
RES_ROOT = r"E:\track\dataset\MOT17\track_result_stnms_botsort"

# ✅ tracks 文件夹（里面是 <seq>.txt）
RES_TRACK_DIR = osp.join(RES_ROOT, "tracks")

# ✅ 评估输出目录
OUT_DIR = osp.join(RES_ROOT, "eval")
os.makedirs(OUT_DIR, exist_ok=True)

# ✅ 本地 TrackEval 源码路径
TRACKEVAL_ROOT = r"E:\track\BoT-SORT-main\TrackEval"

# ✅ 你要评估的序列（MOT20）
# SEQS: List[str] = [
#     "MOT20-01",
#     "MOT20-02",
#     "MOT20-03",
#     "MOT20-05",
# ]

SEQS: List[str] = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]

# ✅ IoU 阈值（motmetrics 匹配阈值）
IOU_THR_MOTA = 0.5
PRINT_STATS = True

# ✅ timing.csv 文件名（你的追踪脚本生成在 RES_ROOT 下）
TIMING_CSV_NAME = "timing.csv"

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
    try:
        return float(x)
    except Exception:
        return np.nan

# ================== FPS：读取 timing.csv ==================
def load_fps_table() -> pd.DataFrame:
    """
    读取 RES_ROOT/timing.csv
    期望列：seq,num_frames,seconds,fps
    返回 index=seq 的 DF: FPS, Seconds, NumFrames
    并补一行 OVERALL（总帧/总秒 的加权 FPS）
    """
    timing_path = osp.join(RES_ROOT, TIMING_CSV_NAME)
    if not osp.isfile(timing_path):
        if PRINT_STATS:
            print(f"[WARN] Missing timing.csv: {timing_path}")
        return pd.DataFrame(columns=["FPS", "Seconds", "NumFrames"])

    df = pd.read_csv(timing_path)
    need = {"seq", "num_frames", "seconds", "fps"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"{timing_path} 列不对，期望 {sorted(list(need))}，实际 {df.columns.tolist()}")

    out = df.set_index("seq")[["fps", "seconds", "num_frames"]].copy()
    out.columns = ["FPS", "Seconds", "NumFrames"]

    total_frames = float(out["NumFrames"].sum())
    total_sec = float(out["Seconds"].sum())
    overall_fps = (total_frames / total_sec) if total_sec > 0 else np.nan
    out.loc["OVERALL"] = [overall_fps, total_sec, total_frames]
    return out

# ================== motmetrics：更稳的 MOT 文件加载（自动补齐到 10 列） ==================
def _pad_to_10cols(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] >= 10:
        return data[:, :10]
    pad = np.full((data.shape[0], 10 - data.shape[1]), -1.0, dtype=np.float32)
    return np.concatenate([data.astype(np.float32), pad], axis=1)

def safe_load_mot_txt_for_mm(path: str, min_confidence: float) -> pd.DataFrame:
    """
    motmetrics 的 mm.io.loadtxt(fmt="mot15-2D")更偏向10列输入；
    MOT20 GT 常见9列：frame,id,x,y,w,h,mark,class,vis
    这里做一个稳健封装：若直接 loadtxt 失败，就补齐到10列再读。
    """
    try:
        return mm.io.loadtxt(path, fmt="mot15-2D", min_confidence=min_confidence)
    except Exception:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if data.size == 0:
            # 返回一个空DF（motmetrics允许空，但 compare_to_groundtruth 可能会有边界情况）
            # 这里仍用 loadtxt 读一个空临时文件最稳
            tmp = osp.join(OUT_DIR, "_tmp_empty_mm.txt")
            open(tmp, "w").close()
            return mm.io.loadtxt(tmp, fmt="mot15-2D", min_confidence=min_confidence)

        data10 = _pad_to_10cols(data)

        tmp_dir = osp.join(OUT_DIR, "_tmp_mm_pad")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = osp.join(tmp_dir, osp.basename(path))

        np.savetxt(
            tmp_path, data10,
            fmt="%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.6f,%.0f,%.0f,%.0f",
            delimiter=","
        )
        return mm.io.loadtxt(tmp_path, fmt="mot15-2D", min_confidence=min_confidence)

def eval_motmetrics() -> Optional[pd.DataFrame]:
    """
    使用 motmetrics 评估 MOTA/IDF1/IDs/FP/FN 等
    ✅ 关键点：GT 的 mark 列(第7列，0/1)用于忽略区域/忽略对象
       因此 GT 用 min_confidence=1 只保留 mark==1 的有效GT
       tracker 输出用 min_confidence=-1（保留全部）
    """
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
        gt_path = osp.join(GT_ROOT, seq, "gt", "gt.txt")
        res_path = osp.join(RES_TRACK_DIR, f"{seq}.txt")

        if not (osp.isfile(gt_path) and osp.isfile(res_path)):
            if PRINT_STATS:
                print(f"[WARN] Missing gt/res for {seq} | gt={osp.isfile(gt_path)} res={osp.isfile(res_path)}")
            continue

        # ✅ GT 只保留 mark==1（有效GT），否则你会被 ignore 区域严重拉低
        gt = safe_load_mot_txt_for_mm(gt_path, min_confidence=1)
        res = safe_load_mot_txt_for_mm(res_path, min_confidence=-1)

        acc = mm.utils.compare_to_groundtruth(gt, res, dist="iou", distth=IOU_THR_MOTA)
        accs.append(acc)
        names.append(seq)

    if not accs:
        return None

    return mh.compute_many(accs, names=names, metrics=metric_list, generate_overall=True)

# ================== TrackEval HOTA：修复“invalid gt classes -1”等问题 ==================
def copy_seqinfo_ini(src_seq_dir: str, dst_seq_dir: str) -> None:
    src_ini = osp.join(src_seq_dir, "seqinfo.ini")
    dst_ini = osp.join(dst_seq_dir, "seqinfo.ini")
    os.makedirs(dst_seq_dir, exist_ok=True)
    if osp.isfile(src_ini):
        shutil.copyfile(src_ini, dst_ini)
    else:
        raise FileNotFoundError(f"Missing seqinfo.ini: {src_ini}")

def sanitize_gt_for_trackeval(src_gt: str, dst_gt: str) -> Tuple[int, int]:
    """
    TrackEval 对 MOTChallenge2DBox 预处理更严格：
    - class 列必须是合法类别（MOT17/20一般 pedestrian=1）
    - 同一帧不能出现多个 id<=0（-1/0 会导致 “same ID more than once”）
    - 只保留 mark==1 的有效GT（可选但推荐，和官方一致）

    MOT20 GT 常见9列：
      frame,id,x,y,w,h,mark,class,vis
    我们：
      - 丢弃 id<=0
      - 丢弃 mark!=1
      - 若 class<=0，则强制设为 1
    """
    kept = 0
    total = 0
    os.makedirs(osp.dirname(dst_gt), exist_ok=True)

    with open(src_gt, "r", encoding="utf-8") as fin, open(dst_gt, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            total += 1

            # frame,id,...
            try:
                tid = int(float(parts[1]))
            except Exception:
                continue
            if tid <= 0:
                continue

            # mark/conf（第7列 index=6）
            try:
                mark = int(float(parts[6]))
            except Exception:
                mark = 1
            if mark != 1:
                continue

            # class（第8列 index=7）若存在
            if len(parts) >= 8:
                try:
                    cls = int(float(parts[7]))
                except Exception:
                    cls = 1
                if cls <= 0:
                    parts[7] = "1"

            fout.write(",".join(parts) + "\n")
            kept += 1

    return kept, total

def eval_hota_trackeval_per_seq() -> pd.DataFrame:
    """
    调 TrackEval 评估 HOTA/DetA/AssA
    ✅ 对 GT 做 sanitize，避免 -1 类别 / id<=0 / ignore GT 影响
    """
    if not osp.isdir(TRACKEVAL_ROOT):
        raise RuntimeError(f"TrackEval not found: {TRACKEVAL_ROOT}")

    if TRACKEVAL_ROOT not in sys.path:
        sys.path.insert(0, TRACKEVAL_ROOT)

    from trackeval import Evaluator
    from trackeval.datasets import MotChallenge2DBox
    from trackeval.metrics import HOTA

    TMP_TE = osp.join(OUT_DIR, "_tmp_trackeval_mot")
    GT_TE = osp.join(TMP_TE, "gt")
    TR_TE = osp.join(TMP_TE, "trackers", "ours", "data")
    SEQMAP_DIR = osp.join(TMP_TE, "seqmaps")

    # ✅ 每次都清理，避免 FileExistsError
    if osp.isdir(TMP_TE):
        shutil.rmtree(TMP_TE, ignore_errors=True)

    os.makedirs(GT_TE, exist_ok=True)
    os.makedirs(TR_TE, exist_ok=True)
    os.makedirs(SEQMAP_DIR, exist_ok=True)

    valid_seqs = []
    for seq in SEQS:
        src_seq_dir = osp.join(GT_ROOT, seq)
        src_gt = osp.join(src_seq_dir, "gt", "gt.txt")
        src_tr = osp.join(RES_TRACK_DIR, f"{seq}.txt")

        if not (osp.isfile(src_gt) and osp.isfile(src_tr)):
            if PRINT_STATS:
                print(f"[WARN] TrackEval missing gt/res: {seq}")
            continue

        # TrackEval 需要：gt/<seq>/gt/gt.txt 结构
        dst_seq_dir = osp.join(GT_TE, seq)
        dst_gt_dir = osp.join(dst_seq_dir, "gt")
        os.makedirs(dst_gt_dir, exist_ok=True)

        copy_seqinfo_ini(src_seq_dir, dst_seq_dir)

        # ✅ sanitize gt，避免 invalid class=-1、id<=0、ignore区域
        dst_gt = osp.join(dst_gt_dir, "gt.txt")
        kept, total = sanitize_gt_for_trackeval(src_gt, dst_gt)
        if PRINT_STATS:
            print(f"[DBG] sanitize GT {seq}: kept={kept}/{total}")

        # tracker
        shutil.copyfile(src_tr, osp.join(TR_TE, f"{seq}.txt"))
        valid_seqs.append(seq)

    if PRINT_STATS:
        print("[DBG] TrackEval valid_seqs =", valid_seqs)

    if not valid_seqs:
        raise RuntimeError("No valid sequences found for TrackEval (gt/res missing).")

    # seqmap
    seqmap_path = osp.join(SEQMAP_DIR, "MOT_custom.txt")
    with open(seqmap_path, "w", encoding="utf-8") as f:
        f.write("name\n")
        for s in valid_seqs:
            f.write(s + "\n")

    dataset_config = {
        "GT_FOLDER": GT_TE,
        "TRACKERS_FOLDER": osp.join(TMP_TE, "trackers"),
        "OUTPUT_FOLDER": osp.join(TMP_TE, "outputs"),
        "TRACKERS_TO_EVAL": ["ours"],
        "SEQMAP_FILE": seqmap_path,
        "SKIP_SPLIT_FOL": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "MOT_eval",
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,
        "DO_PREPROC": True,
        # ✅ 明确只评估 pedestrian（MOT17/20 统一）
        "CLASSES_TO_EVAL": ["pedestrian"],
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
    r_tracker = results[dname]["ours"]

    CLASS_NAME = "pedestrian"

    rows = []
    for seq in valid_seqs:
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
    print("\n=== Evaluating (ST-NMS + Tracker outputs) on MOT20 ===")
    print("GT_ROOT       =", GT_ROOT)
    print("RES_TRACK_DIR  =", RES_TRACK_DIR)
    print("RES_ROOT       =", RES_ROOT)
    print("OUT_DIR        =", OUT_DIR)

    # 1) motmetrics
    mot_sum = eval_motmetrics()
    if mot_sum is None:
        print("[ERROR] No sequences evaluated in motmetrics. Check paths.")
        return

    # 2) TrackEval HOTA/DetA/AssA
    try:
        hota_df = eval_hota_trackeval_per_seq()
        mot_sum = mot_sum.join(hota_df, how="left")
        out_hota_csv = safe_to_csv(hota_df, osp.join(OUT_DIR, "ours_hota_per_seq.csv"))
        print(f"[OK] Saved HOTA per-seq to {out_hota_csv}")
    except Exception as e:
        print(f"[WARN] TrackEval HOTA failed: {e}")

    # 3) FPS
    try:
        fps_df = load_fps_table()
        mot_sum = mot_sum.join(fps_df[["FPS"]], how="left")
    except Exception as e:
        print(f"[WARN] FPS read failed: {e}")
        mot_sum["FPS"] = np.nan

    # 4) 保存汇总
    out_csv = safe_to_csv(mot_sum, osp.join(OUT_DIR, "ours_summary.csv"))
    print(f"[OK] Saved summary to {out_csv}")

    # 打印 OVERALL
    if "OVERALL" in mot_sum.index:
        over = mot_sum.loc["OVERALL"]
        msg = f"[OVERALL] MOTA={over.get('mota', np.nan):.4f}, IDF1={over.get('idf1', np.nan):.4f}, IDs={over.get('num_switches', np.nan)}"
        if "HOTA" in mot_sum.columns:
            msg += f", HOTA={over.get('HOTA', np.nan):.4f}, DetA={over.get('DetA', np.nan):.4f}, AssA={over.get('AssA', np.nan):.4f}"
        if "FPS" in mot_sum.columns:
            msg += f", FPS={over.get('FPS', np.nan):.2f}"
        print(msg)

if __name__ == "__main__":
    main()
