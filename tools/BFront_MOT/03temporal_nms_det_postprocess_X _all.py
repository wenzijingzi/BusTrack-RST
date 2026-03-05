import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

"""
Spatial–Temporal NMS post-processing for external detection files (MOT det format).

Input det format (per line):
    frame, id, x, y, w, h, score, cls

This script:
  1) Loads per-sequence dets (BF_01 ... BF_19 by default)
  2) Spatial NMS per frame:
        - Vehicles (2/3/4) are merged into ONE group for NMS (class-agnostic within vehicles)
        - Non-vehicles (e.g., 0 Ped, 1 Cyclist) are NMS'd per-class
  3) Temporal NMS (support-based filtering) to remove flicker / isolated boxes
        - Vehicles support is also class-agnostic within (2/3/4)
        - Non-vehicles support is per-class
  4) Exports dets with unified vehicle class id = 2:
        0 Pedestrian, 1 Cyclist, 2 Vehicles

Notes:
  - If your upstream detector already outputs vehicles unified to 2, this script keeps that.
  - If it outputs 2/3/4, this script will unify them on export.
"""

# ===================== User config ===================== #

VIDEO_SEQ_PREFIX = "BF_"
START_IDX = 1
END_IDX = 19

# Input / output folders
DETS_IN_ROOT  = r"E:\track\dataset\BusFrontMOT\dets_all"         # input BF_01_det.txt ...
DETS_OUT_ROOT = r"E:\track\dataset\BusFrontMOT\dets_stnms_X_all"   # output BF_01_det.txt ...

# Spatial NMS IoU threshold (per-frame)
SPATIAL_IOU_THR = 0.60

# Temporal NMS parameters
#   K: temporal window radius (total window length = 2K+1)
#   TEMP_IOU_THR: IoU threshold to count as temporal support
#   MIN_SUPPORT: minimum support count within window to keep a detection
#   CONF_KEEP: always keep detections with score >= CONF_KEEP (bypass support check)
K = 2
TEMP_IOU_THR = 0.50
MIN_SUPPORT = 2
CONF_KEEP = 0.80

# Vehicle class ids (raw)
VEHICLE_CLASS_IDS = {2, 3, 4}
UNIFIED_VEHICLE_ID = 2

# Non-vehicle classes you want to keep.
# If None => keep all non-vehicle classes present in input.
KEEP_NON_VEHICLE_CLASSES = None

# ===================== Utilities ===================== #

def is_vehicle(cls_id: int) -> bool:
    return int(cls_id) in VEHICLE_CLASS_IDS or int(cls_id) == UNIFIED_VEHICLE_ID

def unify_vehicle_class_id(dets: np.ndarray) -> np.ndarray:
    """
    dets: [N,8] float, cls in dets[:,7]
    Convert raw vehicle classes 2/3/4 to UNIFIED_VEHICLE_ID (2).
    """
    if dets.size == 0:
        return dets
    dets = dets.copy()
    cls = dets[:, 7].astype(int)
    dets[np.isin(cls, list(VEHICLE_CLASS_IDS)), 7] = float(UNIFIED_VEHICLE_ID)
    return dets

def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """
    xywh: [N,4] -> xyxy [N,4]
    """
    x1 = xywh[:, 0]
    y1 = xywh[:, 1]
    x2 = x1 + xywh[:, 2]
    y2 = y1 + xywh[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)

def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    box: [4] xyxy, boxes: [N,4] xyxy
    """
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

def nms_one_class(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
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
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)

# ===================== IO: load / save dets ===================== #

def load_det_file(det_path: str) -> np.ndarray:
    """
    Load det file. Returns float32 array [N,8].
    Expected: frame,id,x,y,w,h,score,cls
    """
    if not osp.exists(det_path):
        return np.empty((0, 8), dtype=np.float32)
    data = np.loadtxt(det_path, delimiter=",", dtype=np.float32)
    if data.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 8:
        raise ValueError(f"det file must have >=8 columns, got {data.shape[1]}: {det_path}")
    return data[:, :8].astype(np.float32)

def save_det_file(det_path: str, dets: np.ndarray) -> None:
    os.makedirs(osp.dirname(det_path), exist_ok=True)
    if dets.size == 0:
        # create empty file
        open(det_path, "w").close()
        return
    # Ensure integer-like columns are written cleanly
    out = dets.copy()
    # frame and id and cls are integer fields semantically; keep them as numbers in csv
    np.savetxt(det_path, out, fmt="%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.6f,%.0f", delimiter=",")

# ===================== Spatial NMS ===================== #

def spatial_nms_per_frame(frame_dets: np.ndarray, iou_thr: float) -> np.ndarray:
    """
    frame_dets: [N,8] (same frame), columns:
      0 frame, 1 id, 2 x, 3 y, 4 w, 5 h, 6 score, 7 cls

    Behavior:
      - Vehicles (2/3/4) are merged into one group for NMS.
      - Non-vehicles are NMS'd per-class (0,1,...) to avoid cross-class suppression.
    """
    if frame_dets.size == 0:
        return frame_dets

    keep_mask = np.zeros((frame_dets.shape[0],), dtype=bool)

    cls_arr = frame_dets[:, 7].astype(int)
    veh_mask = np.isin(cls_arr, list(VEHICLE_CLASS_IDS) + [UNIFIED_VEHICLE_ID])

    # Vehicles: merged NMS
    if veh_mask.any():
        veh = frame_dets[veh_mask]
        boxes = xywh_to_xyxy(veh[:, 2:6])
        scores = veh[:, 6]
        keep_local = nms_one_class(boxes, scores, iou_thr)
        veh_idx = np.where(veh_mask)[0]
        keep_mask[veh_idx[keep_local]] = True

    # Non-vehicles: per-class NMS
    non_idx = np.where(~veh_mask)[0]
    if non_idx.size > 0:
        non = frame_dets[non_idx]
        non_cls = non[:, 7].astype(int)

        # optional filter
        if KEEP_NON_VEHICLE_CLASSES is not None:
            allowed = set(int(x) for x in KEEP_NON_VEHICLE_CLASSES)
            mask_allowed = np.isin(non_cls, list(allowed))
            non = non[mask_allowed]
            non_idx = non_idx[mask_allowed]
            non_cls = non_cls[mask_allowed]

        for c in np.unique(non_cls):
            c_mask = (non_cls == int(c))
            dets_c = non[c_mask]
            boxes = xywh_to_xyxy(dets_c[:, 2:6])
            scores = dets_c[:, 6]
            keep_local = nms_one_class(boxes, scores, iou_thr)
            global_idx = non_idx[np.where(c_mask)[0][keep_local]]
            keep_mask[global_idx] = True

    return frame_dets[keep_mask]

def spatial_nms_sequence(dets: np.ndarray, iou_thr: float) -> np.ndarray:
    """
    Apply spatial NMS for each frame.
    """
    if dets.size == 0:
        return dets
    frames = dets[:, 0].astype(int)
    out_list = []
    for f in np.unique(frames):
        fd = dets[frames == f]
        out_list.append(spatial_nms_per_frame(fd, iou_thr))
    if len(out_list) == 0:
        return np.empty((0, 8), dtype=np.float32)
    return np.concatenate(out_list, axis=0).astype(np.float32)

# ===================== Temporal NMS ===================== #

def temporal_support_count(
    det_i_xyxy: np.ndarray,
    det_i_cls: int,
    window_dets: np.ndarray,
    iou_thr: float
) -> int:
    """
    Count support for a detection within a temporal window.
    window_dets: all detections in neighboring frames.
    For vehicles: class-agnostic within vehicles (2/3/4/2).
    For non-vehicles: require same class.
    """
    if window_dets.size == 0:
        return 0

    w_cls = window_dets[:, 7].astype(int)
    if is_vehicle(det_i_cls):
        # accept any vehicle class in window
        cand_mask = np.isin(w_cls, list(VEHICLE_CLASS_IDS) + [UNIFIED_VEHICLE_ID])
    else:
        cand_mask = (w_cls == int(det_i_cls))

    cand = window_dets[cand_mask]
    if cand.size == 0:
        return 0

    cand_xyxy = xywh_to_xyxy(cand[:, 2:6])
    ious = iou_one_to_many(det_i_xyxy, cand_xyxy)
    return int(np.sum(ious >= iou_thr))

def temporal_nms_sequence(
    dets: np.ndarray,
    k: int,
    temp_iou_thr: float,
    min_support: int,
    conf_keep: float
) -> np.ndarray:
    """
    Support-based temporal filtering:
      - For each detection, look at frames [t-k, t+k], excluding itself,
        and count number of overlapping detections (IoU >= temp_iou_thr).
      - Keep if score >= conf_keep OR support >= min_support.
    """
    if dets.size == 0:
        return dets

    frames = dets[:, 0].astype(int)
    min_f, max_f = int(frames.min()), int(frames.max())

    # Build index by frame for fast lookup
    dets_by_frame: Dict[int, np.ndarray] = {}
    for f in range(min_f, max_f + 1):
        dets_by_frame[f] = dets[frames == f]

    keep_flags = np.zeros((dets.shape[0],), dtype=bool)

    xyxy_all = xywh_to_xyxy(dets[:, 2:6])
    scores_all = dets[:, 6]
    cls_all = dets[:, 7].astype(int)

    # For mapping global index -> (frame, local idx), build frame offsets
    # We'll just iterate over global indices; window retrieval uses dets_by_frame.
    for i in range(dets.shape[0]):
        if scores_all[i] >= conf_keep:
            keep_flags[i] = True
            continue

        f = int(frames[i])
        cls_i = int(cls_all[i])
        box_i = xyxy_all[i]

        # Collect neighbor dets in window excluding current frame
        window_list = []
        for nf in range(max(min_f, f - k), min(max_f, f + k) + 1):
            if nf == f:
                continue
            wd = dets_by_frame.get(nf, None)
            if wd is not None and wd.size > 0:
                window_list.append(wd)

        if len(window_list) == 0:
            keep_flags[i] = False
            continue

        window_dets = np.concatenate(window_list, axis=0)

        support = temporal_support_count(box_i, cls_i, window_dets, temp_iou_thr)
        keep_flags[i] = (support >= min_support)

    return dets[keep_flags].astype(np.float32)

# ===================== Main processing ===================== #

@dataclass
class Stats:
    n_in: int = 0
    n_after_spatial: int = 0
    n_after_temporal: int = 0

def process_one_sequence(seq_name: str) -> Stats:
    in_path = osp.join(DETS_IN_ROOT, f"{seq_name}_det.txt")
    out_path = osp.join(DETS_OUT_ROOT, f"{seq_name}_det.txt")

    dets = load_det_file(in_path)
    st = Stats(n_in=int(dets.shape[0]))

    if dets.size == 0:
        save_det_file(out_path, dets)
        logger.info(f"{seq_name}: empty dets -> saved empty.")
        return st

    # Spatial NMS
    dets_sp = spatial_nms_sequence(dets, SPATIAL_IOU_THR)
    st.n_after_spatial = int(dets_sp.shape[0])

    # Temporal NMS
    dets_tp = temporal_nms_sequence(dets_sp, K, TEMP_IOU_THR, MIN_SUPPORT, CONF_KEEP)
    st.n_after_temporal = int(dets_tp.shape[0])

    # Unify vehicles class id on export
    dets_tp = unify_vehicle_class_id(dets_tp)

    # Sort by (frame, score desc) for nicer files
    if dets_tp.size > 0:
        frames = dets_tp[:, 0].astype(int)
        scores = dets_tp[:, 6]
        order = np.lexsort((-scores, frames))
        dets_tp = dets_tp[order]

    save_det_file(out_path, dets_tp)

    logger.info(
        f"{seq_name}: in={st.n_in} -> spatial={st.n_after_spatial} -> temporal={st.n_after_temporal} "
        f"(saved: {out_path})"
    )
    return st

def main():
    logger.info("=== Spatial–Temporal NMS post-process started ===")
    logger.info(f"IN : {DETS_IN_ROOT}")
    logger.info(f"OUT: {DETS_OUT_ROOT}")
    logger.info(f"Spatial IoU thr = {SPATIAL_IOU_THR}")
    logger.info(f"Temporal: K={K}, IoU thr={TEMP_IOU_THR}, min_support={MIN_SUPPORT}, conf_keep={CONF_KEEP}")
    logger.info(f"Vehicle unify: {VEHICLE_CLASS_IDS} -> {UNIFIED_VEHICLE_ID}")

    os.makedirs(DETS_OUT_ROOT, exist_ok=True)

    total = Stats()
    for idx in range(START_IDX, END_IDX + 1):
        seq_name = f"{VIDEO_SEQ_PREFIX}{idx:02d}"
        st = process_one_sequence(seq_name)
        total.n_in += st.n_in
        total.n_after_spatial += st.n_after_spatial
        total.n_after_temporal += st.n_after_temporal

    # Summary
    def pct(a, b):
        return 0.0 if a == 0 else (b / a * 100.0)

    logger.info("=== Summary ===")
    logger.info(
        f"Total: in={total.n_in}, spatial={total.n_after_spatial} ({pct(total.n_in, total.n_after_spatial):.2f}%), "
        f"temporal={total.n_after_temporal} ({pct(total.n_in, total.n_after_temporal):.2f}%)"
    )
    logger.info("=== Done ===")

if __name__ == "__main__":
    main()
