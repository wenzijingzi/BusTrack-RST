import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from loguru import logger

"""
Spatial–Temporal NMS post-processing for MOT17 official detections (FRCNN / DPM / SDP).

MOT17 det format (det/det.txt per line):
    frame, id, x, y, w, h, score, x, y, z

This script:
  1) Loads per-sequence MOT17 det/det.txt
  2) Applies Spatial NMS per frame (class-agnostic because MOT17 is pedestrian-only)
  3) Applies Temporal NMS (support-based filtering) to remove flicker / isolated boxes
  4) Exports dets in a tracker-friendly det format:
        frame, id, x, y, w, h, score, cls
     where cls is set to 0 for pedestrian (placeholder; your tracker code can ignore it)

Notes:
  - Use this output as the detection input for your tracking pipeline (instead of raw det/det.txt),
    then run trackers and evaluate.
  - This script does NOT touch GT. It only post-processes detections.
"""

# ===================== User config ===================== #

# ---- MOT17/20 root (train or test folder) ----
MOT_ROOT = r"E:\track\dataset\MOT20\train"

# ---- Which MOT17/20 sequences to process ----
# SEQS = [
#     "MOT17-02-FRCNN",
#     "MOT17-04-FRCNN",
#     "MOT17-05-FRCNN",
#     "MOT17-09-FRCNN",
#     "MOT17-10-FRCNN",
#     "MOT17-11-FRCNN",
#     "MOT17-13-FRCNN",
# ]
SEQS = [
    "MOT20-01",
    "MOT20-02",
    "MOT20-03",
    "MOT20-05",
]

# ---- Input det file relative path ----
DET_REL_PATH = osp.join("det", "det.txt")

# ---- Output root folder ----
# Each sequence will be written to: OUT_ROOT/<seq>/det/det.txt  (same layout as MOT17)
OUT_ROOT = r"E:\track\dataset\MOT20\train_stnms"  # you can change

# ---- Spatial NMS IoU threshold (per-frame) ----
SPATIAL_IOU_THR = 0.60

# ---- Temporal NMS parameters ----
K = 2
TEMP_IOU_THR = 0.50
MIN_SUPPORT = 2
CONF_KEEP = 0.80

# ---- Pre-filter detections by confidence BEFORE NMS (optional but recommended) ----
# If None: do not prefilter
DET_MIN_CONF = 0.01

# ---- Export format for downstream trackers ----
# If True: export in MOT det.txt style (10 cols) with score kept and others -1
# If False: export in 8-col format: frame,id,x,y,w,h,score,cls (cls=0)
EXPORT_AS_MOT10 = False

# ===================== Utilities ===================== #

def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x1 = xywh[:, 0]
    y1 = xywh[:, 1]
    x2 = x1 + xywh[:, 2]
    y2 = y1 + xywh[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)

def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xA = np.maximum(box[0], boxes[:, 0])
    yA = np.maximum(box[1], boxes[:, 1])
    xB = np.minimum(box[2], boxes[:, 2])
    yB = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, xB - xA)
    inter_h = np.maximum(0.0, yB - yA)
    inter = inter_w * inter_h

    areaA = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    areaB = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = areaA + areaB - inter + 1e-12
    return inter / union

def nms_greedy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """
    Standard greedy NMS. Returns kept indices.
    """
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)

    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_one_to_many(boxes_xyxy[i], boxes_xyxy[order[1:]])
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)

# ===================== IO: load / save dets ===================== #

def load_mot17_det_file(det_path: str) -> np.ndarray:
    """
    Load MOT17 det/det.txt. Returns float32 array [N,10] (or >=10, we keep first 10).
    Expected: frame, id, x, y, w, h, score, x, y, z
    """
    if not osp.exists(det_path):
        return np.empty((0, 10), dtype=np.float32)

    data = np.loadtxt(det_path, delimiter=",", dtype=np.float32)
    if data.size == 0:
        return np.empty((0, 10), dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 7:
        raise ValueError(f"det file must have >=7 columns, got {data.shape[1]}: {det_path}")

    # keep at least first 10 if exists, else pad to 10
    if data.shape[1] >= 10:
        data = data[:, :10]
    else:
        pad = np.full((data.shape[0], 10 - data.shape[1]), -1.0, dtype=np.float32)
        data = np.concatenate([data, pad], axis=1)

    return data.astype(np.float32)

def save_det_for_trackers(det_path: str, dets_10: np.ndarray) -> None:
    """
    Export detection for downstream trackers.
    - If EXPORT_AS_MOT10: write 10 columns like MOT det
    - Else: write 8 columns: frame,id,x,y,w,h,score,cls(=0)
    """
    os.makedirs(osp.dirname(det_path), exist_ok=True)

    if dets_10.size == 0:
        open(det_path, "w").close()
        return

    # dets_10 columns: frame,id,x,y,w,h,score,*,*,*
    frame = dets_10[:, 0]
    det_id = dets_10[:, 1]
    x, y, w, h = dets_10[:, 2], dets_10[:, 3], dets_10[:, 4], dets_10[:, 5]
    score = dets_10[:, 6]

    if EXPORT_AS_MOT10:
        out10 = dets_10.copy()
        # Write as 10-col MOT det
        np.savetxt(
            det_path, out10,
            fmt="%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.6f,%.0f,%.0f,%.0f",
            delimiter=","
        )
    else:
        cls = np.zeros_like(score, dtype=np.float32)  # pedestrian cls=0
        out8 = np.stack([frame, det_id, x, y, w, h, score, cls], axis=1).astype(np.float32)
        np.savetxt(
            det_path, out8,
            fmt="%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.6f,%.0f",
            delimiter=","
        )

# ===================== Spatial NMS ===================== #

def spatial_nms_per_frame(frame_dets_10: np.ndarray, iou_thr: float) -> np.ndarray:
    """
    MOT17 is single-class pedestrian, so we do class-agnostic NMS per frame.
    Input: [N,10] dets of the same frame (MOT det format).
    """
    if frame_dets_10.size == 0:
        return frame_dets_10

    boxes_xyxy = xywh_to_xyxy(frame_dets_10[:, 2:6])
    scores = frame_dets_10[:, 6]
    keep = nms_greedy(boxes_xyxy, scores, iou_thr)
    return frame_dets_10[keep]

def spatial_nms_sequence(dets_10: np.ndarray, iou_thr: float) -> np.ndarray:
    if dets_10.size == 0:
        return dets_10

    frames = dets_10[:, 0].astype(int)
    out_list = []
    for f in np.unique(frames):
        fd = dets_10[frames == f]
        out_list.append(spatial_nms_per_frame(fd, iou_thr))
    return np.concatenate(out_list, axis=0).astype(np.float32) if out_list else np.empty((0, 10), dtype=np.float32)

# ===================== Temporal NMS ===================== #

def temporal_support_count(
    det_i_xyxy: np.ndarray,
    window_dets_10: np.ndarray,
    iou_thr: float
) -> int:
    """
    Count support (number of overlapping dets in neighboring frames).
    MOT17 is single-class => no class filtering.
    """
    if window_dets_10.size == 0:
        return 0
    cand_xyxy = xywh_to_xyxy(window_dets_10[:, 2:6])
    ious = iou_one_to_many(det_i_xyxy, cand_xyxy)
    return int(np.sum(ious >= iou_thr))

def temporal_nms_sequence(
    dets_10: np.ndarray,
    k: int,
    temp_iou_thr: float,
    min_support: int,
    conf_keep: float
) -> np.ndarray:
    """
    Support-based temporal filtering:
      - For each detection, look at frames [t-k, t+k] excluding itself frame.
      - Keep if score >= conf_keep OR support >= min_support.
    """
    if dets_10.size == 0:
        return dets_10

    frames = dets_10[:, 0].astype(int)
    min_f, max_f = int(frames.min()), int(frames.max())

    dets_by_frame: Dict[int, np.ndarray] = {}
    for f in range(min_f, max_f + 1):
        dets_by_frame[f] = dets_10[frames == f]

    keep_flags = np.zeros((dets_10.shape[0],), dtype=bool)

    xyxy_all = xywh_to_xyxy(dets_10[:, 2:6])
    scores_all = dets_10[:, 6]

    for i in range(dets_10.shape[0]):
        if scores_all[i] >= conf_keep:
            keep_flags[i] = True
            continue

        f = int(frames[i])
        box_i = xyxy_all[i]

        window_list = []
        for nf in range(max(min_f, f - k), min(max_f, f + k) + 1):
            if nf == f:
                continue
            wd = dets_by_frame.get(nf, None)
            if wd is not None and wd.size > 0:
                window_list.append(wd)

        if not window_list:
            keep_flags[i] = False
            continue

        window_dets = np.concatenate(window_list, axis=0)
        support = temporal_support_count(box_i, window_dets, temp_iou_thr)
        keep_flags[i] = (support >= min_support)

    return dets_10[keep_flags].astype(np.float32)

# ===================== Main processing ===================== #

@dataclass
class Stats:
    n_in: int = 0
    n_prefilter: int = 0
    n_after_spatial: int = 0
    n_after_temporal: int = 0

def prefilter_by_conf(dets_10: np.ndarray, min_conf: Optional[float]) -> np.ndarray:
    if dets_10.size == 0 or min_conf is None:
        return dets_10
    keep = dets_10[:, 6] >= float(min_conf)
    return dets_10[keep].astype(np.float32)

def process_one_sequence(seq_name: str) -> Stats:
    in_path = osp.join(MOT_ROOT, seq_name, DET_REL_PATH)

    # output keeps MOT-like structure:
    # OUT_ROOT/<seq>/det/det.txt
    out_det_path = osp.join(OUT_ROOT, seq_name, "det", "det.txt")

    dets = load_mot17_det_file(in_path)
    st = Stats(n_in=int(dets.shape[0]))

    if dets.size == 0:
        save_det_for_trackers(out_det_path, dets)
        logger.info(f"{seq_name}: empty dets -> saved empty.")
        return st

    # Pre-filter low-confidence
    dets_pf = prefilter_by_conf(dets, DET_MIN_CONF)
    st.n_prefilter = int(dets_pf.shape[0])

    # Spatial NMS
    dets_sp = spatial_nms_sequence(dets_pf, SPATIAL_IOU_THR)
    st.n_after_spatial = int(dets_sp.shape[0])

    # Temporal NMS
    dets_tp = temporal_nms_sequence(dets_sp, K, TEMP_IOU_THR, MIN_SUPPORT, CONF_KEEP)
    st.n_after_temporal = int(dets_tp.shape[0])

    # Sort by (frame, score desc)
    if dets_tp.size > 0:
        frames = dets_tp[:, 0].astype(int)
        scores = dets_tp[:, 6]
        order = np.lexsort((-scores, frames))
        dets_tp = dets_tp[order]

    save_det_for_trackers(out_det_path, dets_tp)

    logger.info(
        f"{seq_name}: in={st.n_in} -> prefilter={st.n_prefilter} -> spatial={st.n_after_spatial} "
        f"-> temporal={st.n_after_temporal} (saved: {out_det_path})"
    )
    return st

def main():
    logger.info("=== Spatial–Temporal NMS for MOT17 detections ===")
    logger.info(f"MOT_ROOT : {MOT_ROOT}")
    logger.info(f"OUT_ROOT : {OUT_ROOT}")
    logger.info(f"SEQS     : {SEQS}")
    logger.info(f"Spatial IoU thr = {SPATIAL_IOU_THR}")
    logger.info(f"Temporal: K={K}, IoU thr={TEMP_IOU_THR}, min_support={MIN_SUPPORT}, conf_keep={CONF_KEEP}")
    logger.info(f"Pre-filter conf >= {DET_MIN_CONF}")
    logger.info(f"Export: {'MOT10' if EXPORT_AS_MOT10 else '8-col (frame,id,xywh,score,cls)'}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    total = Stats()
    for seq in SEQS:
        st = process_one_sequence(seq)
        total.n_in += st.n_in
        total.n_prefilter += st.n_prefilter
        total.n_after_spatial += st.n_after_spatial
        total.n_after_temporal += st.n_after_temporal

    def pct(a, b):
        return 0.0 if a == 0 else (b / a * 100.0)

    logger.info("=== Summary ===")
    logger.info(
        f"Total: in={total.n_in}, prefilter={total.n_prefilter} ({pct(total.n_in, total.n_prefilter):.2f}%), "
        f"spatial={total.n_after_spatial} ({pct(total.n_in, total.n_after_spatial):.2f}%), "
        f"temporal={total.n_after_temporal} ({pct(total.n_in, total.n_after_temporal):.2f}%)"
    )
    logger.info("=== Done ===")

if __name__ == "__main__":
    main()
