import os
import os.path as osp
from typing import Dict, List

import numpy as np
from loguru import logger

# ================== 配置区域 ==================

# 原始 det.txt 的目录（你之前的输出）
DETS_ROOT = r"E:\track\dataset\BusFrontMOT\dets"

# 经过 Spatial + Temporal NMS 后的输出目录
OUT_ROOT = r"E:\track\dataset\BusFrontMOT\dets_tnms_X"

os.makedirs(OUT_ROOT, exist_ok=True)

# 序列名
SEQS = [f"BF_{i:02d}" for i in range(1, 20)]

# ---- 参数（建议先用这组） ----
K = 3                    # temporal 窗口：前后各 K 帧
TEMP_IOU_THR = 0.5       # temporal IoU 阈值
SPATIAL_NMS_IOU = 0.5    # 同帧 NMS IoU 阈值
MIN_SUPPORT = 2          # 低置信度框至少需要的支持帧数
CONF_KEEP = 0.5          # 置信度高于此值的框，无需支持也保留

# =========================================================
#             一些基础工具函数
# =========================================================

# 你的类别定义：0 Pedestrian, 1 Cyclist, 2 Car, 3 Truck, 4 Van
VEHICLE_CLASS_IDS = {2, 3, 4}

def is_vehicle(cls_id: int) -> bool:
    return int(cls_id) in VEHICLE_CLASS_IDS


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    boxes: [N, 4] with [x, y, w, h]
    return: [N, 4] with [x1, y1, x2, y2]
    """
    out = boxes.copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    IoU of two boxes in xyxy format.
    b1, b2: [4]
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = max(area1 + area2 - inter, 1e-6)
    return float(inter / union)


def nms_one_class(boxes_xyxy: np.ndarray,
                  scores: np.ndarray,
                  iou_thr: float) -> List[int]:
    """
    标准 NMS，返回保留的 index 列表。
    boxes_xyxy: [N, 4]
    scores: [N]
    """
    if boxes_xyxy.shape[0] == 0:
        return []

    idxs = scores.argsort()[::-1]  # 按 score 从大到小
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(int(i))
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = np.array(
            [bbox_iou(boxes_xyxy[i], boxes_xyxy[j]) for j in rest],
            dtype=np.float32,
        )
        idxs = rest[ious < iou_thr]

    return keep


# =========================================================
#             1) 同帧 Spatial NMS
# =========================================================

def spatial_nms_per_frame(frame_dets: np.ndarray,
                          iou_thr: float) -> np.ndarray:
    """
    对单帧做空间 NMS。
    frame_dets: [M, 8] = [frame, -1, x, y, w, h, score, cls]
    """
    if frame_dets.size == 0:
        return frame_dets

    # 1）先把“车辆类”统一做一次 NMS（Car+Truck+Van 当作同一类）
    vehicle_mask = np.array([is_vehicle(c) for c in frame_dets[:, 7]], dtype=bool)
    non_vehicle_mask = ~vehicle_mask

    keep_mask = np.zeros(frame_dets.shape[0], dtype=bool)

    # (a) 车辆：合在一起 NMS
    if vehicle_mask.any():
        veh_dets = frame_dets[vehicle_mask]
        boxes_xywh = veh_dets[:, 2:6]
        scores = veh_dets[:, 6]
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)

        keep_idx_local = nms_one_class(boxes_xyxy, scores, iou_thr)
        veh_indices = np.where(vehicle_mask)[0]
        keep_mask[veh_indices[keep_idx_local]] = True

    # (b) 非车辆（将来如果你保留行人/骑行者）：各自原样保留 或者单独 per-class NMS
    if non_vehicle_mask.any():
        # 这里简单起见：非车辆不做 NMS，全部保留
        keep_mask[non_vehicle_mask] = True

    return frame_dets[keep_mask]


def spatial_nms_all_frames(dets: np.ndarray,
                           iou_thr: float) -> np.ndarray:
    """
    对整个序列的 dets 进行逐帧 NMS。
    dets: [N, 8]
    """
    if dets.size == 0:
        return dets

    frames = np.unique(dets[:, 0]).astype(int)
    results = []

    for f in frames:
        frame_mask = dets[:, 0] == f
        frame_dets = dets[frame_mask]
        nms_dets = spatial_nms_per_frame(frame_dets, iou_thr)
        results.append(nms_dets)

    if results:
        return np.concatenate(results, axis=0)
    else:
        return np.empty((0, 8), dtype=np.float32)


# =========================================================
#             2) 跨帧 Temporal NMS
# =========================================================

def build_frame_dict(dets: np.ndarray) -> Dict[int, np.ndarray]:
    """
    根据 frame_id 构建字典：frame_id -> dets_in_frame
    """
    frame_dict: Dict[int, np.ndarray] = {}
    frames = np.unique(dets[:, 0]).astype(int)
    for f in frames:
        mask = dets[:, 0] == f
        frame_dict[f] = dets[mask]
    return frame_dict


def temporal_nms_sequence(dets: np.ndarray,
                          k: int,
                          iou_thr: float,
                          min_support: int,
                          conf_keep: float) -> np.ndarray:
    """
    对整个序列做 Temporal NMS。
    dets: [N, 8], 列: frame, -1, x, y, w, h, score, cls

    规则：
      - conf >= conf_keep: 无需支持，直接保留
      - conf <  conf_keep:
            若 support(b_t) < min_support → 删除
            否则保留
    """
    if dets.size == 0:
        return dets

    frame_dict = build_frame_dict(dets)
    all_frames = sorted(frame_dict.keys())

    keep_rows = []

    # 预先转为 xyxy 方便 IoU 计算
    # 为了简单，我们在循环内部现算，每帧数据不大，影响很小

    for f in all_frames:
        cur_dets = frame_dict[f]
        for i in range(cur_dets.shape[0]):
            row = cur_dets[i]
            frame_id = int(row[0])
            x, y, w, h = row[2:6]
            score = float(row[6])
            cls = int(row[7])

            if score >= conf_keep:
                # 高置信度框，直接保留（防止过度削弱 IDF1）
                keep_rows.append(row)
                continue

            # 对低置信度框，计算 temporal support
            box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            support = 0

            # 在 [f-k, f+k] 邻域内遍历
            for tau in range(frame_id - k, frame_id + k + 1):
                if tau == frame_id:
                    continue
                if tau not in frame_dict:
                    continue

                neigh_dets = frame_dict[tau]
                # 只看同类别

                if is_vehicle(cls):
                    # 车辆：允许 Car/Truck/Van 互相支持
                    same_cls = neigh_dets[[is_vehicle(c) for c in neigh_dets[:, 7]]]
                else:
                    # 非车辆：仍然只和同类互相支持（目前你没有用到）
                    same_cls = neigh_dets[neigh_dets[:, 7] == cls]

                if same_cls.shape[0] == 0:
                    continue

                neigh_boxes_xywh = same_cls[:, 2:6]
                neigh_boxes_xyxy = xywh_to_xyxy(neigh_boxes_xywh)

                # 计算与所有邻帧框的 IoU，若有一个 >= 阈值，则该帧 +1 支持
                ious = np.array(
                    [bbox_iou(box_xyxy, nb) for nb in neigh_boxes_xyxy],
                    dtype=np.float32,
                )
                if np.any(ious >= iou_thr):
                    support += 1

            if support >= min_support:
                keep_rows.append(row)
            else:
                # 丢弃：低 conf + 缺乏时序一致性的孤立框
                pass

    if keep_rows:
        keep_arr = np.stack(keep_rows, axis=0)
    else:
        keep_arr = np.empty((0, 8), dtype=np.float32)

    # 排序回原来顺序：按 frame, score
    order = np.lexsort((-keep_arr[:, 6], keep_arr[:, 0]))
    keep_arr = keep_arr[order]
    return keep_arr


# =========================================================
#             主流程：对 BF_01~BF_19 逐个处理
# =========================================================

def load_det_file(path: str) -> np.ndarray:
    if not osp.exists(path):
        logger.warning(f"未找到检测文件: {path}")
        return np.empty((0, 8), dtype=np.float32)

    data = np.loadtxt(path, delimiter=',')
    if data.ndim == 1:
        data = data[None, :]
    # 只保留前 8 列：frame, id, x, y, w, h, score, cls
    data = data[:, :8].astype(np.float32)
    return data


def save_det_file(path: str, dets: np.ndarray):
    if dets.size == 0:
        # 写一个空文件
        open(path, "w").close()
        return

    fmt = ["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.4f", "%d"]
    np.savetxt(path, dets, fmt=fmt, delimiter=",")


def process_one_sequence(seq_name: str):
    in_path = osp.join(DETS_ROOT, f"{seq_name}_det.txt")
    out_path = osp.join(OUT_ROOT, f"{seq_name}_det_temp.txt")

    logger.info(f"========== {seq_name}: 读取 {in_path} ==========")
    dets = load_det_file(in_path)
    if dets.size == 0:
        logger.warning(f"{seq_name} 无检测，跳过")
        return

    logger.info(f"{seq_name}: 原始检测数 = {dets.shape[0]}")

    # 1) per-frame Spatial NMS
    dets_nms = spatial_nms_all_frames(dets, SPATIAL_NMS_IOU)
    logger.info(f"{seq_name}: 空间 NMS 后检测数 = {dets_nms.shape[0]}")

    # 2) Temporal NMS
    dets_tnms = temporal_nms_sequence(
        dets_nms, K, TEMP_IOU_THR, MIN_SUPPORT, CONF_KEEP
    )
    logger.info(f"{seq_name}: Temporal NMS 后检测数 = {dets_tnms.shape[0]}")

    # 保存
    save_det_file(out_path, dets_tnms)
    logger.info(f"{seq_name}: 写入 {out_path}\n")


def main():
    logger.info("开始运行 Spatial + Temporal NMS v2 ...")
    logger.info(
        f"配置: K={K}, TEMP_IOU_THR={TEMP_IOU_THR}, "
        f"SPATIAL_NMS_IOU={SPATIAL_NMS_IOU}, "
        f"MIN_SUPPORT={MIN_SUPPORT}, CONF_KEEP={CONF_KEEP}"
    )

    for seq in SEQS:
        process_one_sequence(seq)

    logger.info("所有序列处理完成。")


if __name__ == "__main__":
    main()
