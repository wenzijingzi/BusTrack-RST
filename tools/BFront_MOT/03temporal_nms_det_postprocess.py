import os
import os.path as osp
import numpy as np

"""
对 YOLO-M2LA 导出的 det.txt 做 Temporal NMS（多帧时序过滤）

输入格式（每行）：
frame,id,x,y,w,h,score,class

输出：
frame,id,x,y,w,h,score,class （去掉时序孤立的低置信度框）

思路：
- 对每个检测框，查看前后 K 帧内是否有 IoU >= IOU_THR 的“邻居框”
- 如果 support_count < MIN_SUPPORT 且 score < CONF_KEEP，则认为是时序噪声，删除
"""

# ================== 配置区域 ================== #

# 原始 det 目录
RAW_DET_DIR = r"E:\track\dataset\BusFrontMOT\dets"

# 输出（Temporal NMS 之后）的 det 目录
TEMP_DET_DIR = r"E:\track\dataset\BusFrontMOT\dets_temp"

os.makedirs(TEMP_DET_DIR, exist_ok=True)

# 序列范围
START_IDX = 1
END_IDX = 19

# Temporal NMS 参数
K = 3             # 前后各看 K 帧，例如 3 表示 [t-3, t+3]
IOU_THR = 0.4    # 时序匹配的 IoU 阈值
MIN_SUPPORT =1  # 至少要有 2 帧邻居支持，否则认为是时序孤立
CONF_KEEP = 0.5    # 高置信度框（>=0.5）无论支持度多少都保留

# ============================================= #


def iou_xyxy(box1, box2):
    """计算两个 [x1,y1,x2,y2] 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    if area1 <= 0 or area2 <= 0:
        return 0.0

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union


def load_dets_by_frame(det_path):
    """
    读取 det.txt，并按 frame 分组：
    返回：
        frames: 升序的帧号列表
        dets_by_frame: dict[frame] = np.ndarray(N, 7)  # x,y,w,h,score,class,line_idx
    这里额外带上 line_idx，方便后面决定保留哪些行。
    """
    data = np.loadtxt(det_path, delimiter=',')

    # 防止只有一行的情况
    if data.ndim == 1:
        data = data[None, :]

    dets_by_frame = {}
    for idx, row in enumerate(data):
        frame = int(row[0])
        x, y, w, h = row[2], row[3], row[4], row[5]
        score = row[6]
        cls = row[7]
        det_info = [x, y, w, h, score, cls, idx]
        dets_by_frame.setdefault(frame, []).append(det_info)

    for f in dets_by_frame.keys():
        dets_by_frame[f] = np.array(dets_by_frame[f], dtype=np.float32)

    frames = sorted(dets_by_frame.keys())
    return frames, dets_by_frame, data


def temporal_nms_for_sequence(seq_name):
    raw_det_path = osp.join(RAW_DET_DIR, f"{seq_name}_det.txt")
    if not osp.exists(raw_det_path):
        print(f"[跳过] 找不到检测文件: {raw_det_path}")
        return

    frames, dets_by_frame, data_all = load_dets_by_frame(raw_det_path)
    num_frames = len(frames)

    keep_flags = np.zeros(len(data_all), dtype=bool)

    print(f"处理 {seq_name}: 共 {len(data_all)} 个检测，{num_frames} 帧有检测")

    # 遍历每一帧每一个检测
    for fi, frame in enumerate(frames):
        dets = dets_by_frame[frame]  # [N, 7]
        for det in dets:
            x, y, w, h, score, cls, global_idx = det
            x1, y1, x2, y2 = x, y, x + w, y + h

            # 高置信度，直接保留
            if score >= CONF_KEEP:
                keep_flags[int(global_idx)] = True
                continue

            # 低置信度 → 看时序支持度
            support = 0

            # 在 [frame-K, frame+K] 内查找邻居
            for dt in range(1, K + 1):
                for neigh_frame in (frame - dt, frame + dt):
                    if neigh_frame not in dets_by_frame:
                        continue
                    neigh_dets = dets_by_frame[neigh_frame]
                    # 判断这一帧是否至少有一个 IoU >= 阈值的框
                    has_match = False
                    for nd in neigh_dets:
                        nx, ny, nw, nh = nd[0], nd[1], nd[2], nd[3]
                        nx1, ny1, nx2, ny2 = nx, ny, nx + nw, ny + nh
                        iou = iou_xyxy([x1, y1, x2, y2], [nx1, ny1, nx2, ny2])
                        if iou >= IOU_THR:
                            has_match = True
                            break
                    if has_match:
                        support += 1
                # 已经足够了，可以提前结束
                if support >= MIN_SUPPORT:
                    break

            if support >= MIN_SUPPORT:
                keep_flags[int(global_idx)] = True
            else:
                # 时序孤立 + 低置信度 → 丢弃
                keep_flags[int(global_idx)] = False

    kept_data = data_all[keep_flags]
    print(f"{seq_name}: 原始 {len(data_all)} → 保留 {len(kept_data)} (删除 {len(data_all) - len(kept_data)})")

    # 保存到新文件
    save_path = osp.join(TEMP_DET_DIR, f"{seq_name}_det_temp.txt")
    np.savetxt(save_path, kept_data, fmt="%.2f,%.0f,%.2f,%.2f,%.2f,%.2f,%.4f,%.0f")
    print(f"{seq_name}: Temporal NMS 结果已保存到 {save_path}\n")


def main():
    for idx in range(START_IDX, END_IDX + 1):
        seq_name = f"BF_{idx:02d}"
        temporal_nms_for_sequence(seq_name)

    print("全部序列 Temporal NMS 处理完成！")


if __name__ == "__main__":
    main()
