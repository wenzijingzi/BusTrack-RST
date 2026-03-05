import os
import os.path as osp
from types import SimpleNamespace
import time
import csv

import cv2
import numpy as np
from loguru import logger

import torch
import sys

# 确保能 import yolox 和 tracker
sys.path.append('.')

from tracker.bot_sort import BoTSORT
from yolox.utils.visualize import plot_tracking


"""
使用已有 det.txt (YOLO-M2LA 导出的检测结果) 跑 BoT-SORT 追踪

输入：
    - 视频：E:\track\dataset\BusFrontMOT\videos\BF_01.mp4 ... BF_19.mp4
    - 检测：E:\track\dataset\BusFrontMOT\dets\BF_01_det.txt ... BF_19_det.txt

输出：
    - 轨迹 txt：./YOLOX_outputs/.../tracks/BF_01.txt ...
    - 可视化视频：./YOLOX_outputs/.../vis/BF_01.mp4 ...
    - timing.csv：./YOLOX_outputs/.../timing.csv   (用于评估 FPS)
"""

# ===================== 配置区域 ===================== #

VIDEO_ROOT = r"E:\track\dataset\BusFrontMOT\videos"
DETS_ROOT = r"E:\track\dataset\BusFrontMOT\dets_tnms_X"

# 轨迹与可视化保存目录（相对 BoT-SORT 主目录）
RESULTS_ROOT = r"E:/track/BoT-SORT-main/YOLOX_outputs/m2la_botsort_tempX_FPS"
TRACK_TXT_DIR = osp.join(RESULTS_ROOT, "tracks")
VIS_VIDEO_DIR = osp.join(RESULTS_ROOT, "vis")
TIMING_CSV_PATH = osp.join(RESULTS_ROOT, "timing.csv")

os.makedirs(TRACK_TXT_DIR, exist_ok=True)
os.makedirs(VIS_VIDEO_DIR, exist_ok=True)

# 序列范围 BF_01 ~ BF_19
START_IDX = 1
END_IDX = 19

# 是否保存可视化视频
SAVE_VIS = True

# ===================== 构造 BoT-SORT 参数 ===================== #

def build_tracker(frame_rate=30, seq_name="BF_01"):
    """
    手动构造一个 args，用于初始化 BoTSORT
    参数值基本照抄 tools/track.py 默认值
    """
    args = SimpleNamespace()

    # 这两个是 BoTSORT 里 GMC 会用到的
    args.name = seq_name
    args.ablation = False

    # ByteTrack / BoT-SORT tracking 参数
    args.track_high_thresh = 0.6
    args.track_low_thresh = 0.1
    args.new_track_thresh = 0.7
    args.track_buffer = 30
    args.match_thresh = 0.8
    args.aspect_ratio_thresh = 1.6
    args.min_box_area = 10
    args.mot20 = False

    # CMC（相机运动补偿）
    args.cmc_method = "busfront"   # 可改 "none"/"orb"/"ecc"/"sparseOptFlow"

    # ReID 相关
    args.with_reid = True
    args.fast_reid_config = "E:/track/BoT-SORT-main/fast_reid/configs/MOT17/sbs_S50.yml"
    args.fast_reid_weights = "E:/track/BoT-SORT-main/pretrained/mot17_sbs_S50.pth"
    args.proximity_thresh = 0.5
    args.appearance_thresh = 0.25

    # 其它占位参数
    args.device = "gpu"

    tracker = BoTSORT(args, frame_rate=frame_rate)
    return tracker


# ===================== 读取 det.txt ===================== #

def load_dets(det_txt_path):
    """
    读取 det.txt:
    frame,id,x,y,w,h,score,class

    返回：
        dets_by_frame: dict[int, np.ndarray(N, 6)]
        每一帧的检测为 [x1,y1,x2,y2,score,cls]
    """
    if not osp.exists(det_txt_path):
        logger.warning(f"检测文件不存在: {det_txt_path}")
        return {}

    data = np.loadtxt(det_txt_path, delimiter=',')
    if data.ndim == 1:
        data = data[None, :]

    dets_by_frame = {}
    for row in data:
        frame_id = int(row[0])
        x, y, w, h = row[2], row[3], row[4], row[5]
        score = row[6]
        cls = row[7]

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        det = [x1, y1, x2, y2, score, cls]
        dets_by_frame.setdefault(frame_id, []).append(det)

    for k in dets_by_frame.keys():
        dets_by_frame[k] = np.array(dets_by_frame[k], dtype=np.float32)

    logger.info(f"加载检测完成: {det_txt_path}，共 {len(data)} 个检测框，{len(dets_by_frame)} 帧有检测")
    return dets_by_frame


# ===================== 主追踪逻辑 ===================== #

def track_one_sequence(seq_name):
    """
    对单个视频序列 (BF_xx) 进行追踪

    返回：
        dict(seq, num_frames, seconds, fps)  用于汇总 timing.csv
        若跳过则返回 None
    """
    video_path = osp.join(VIDEO_ROOT, f"{seq_name}.mp4")
    det_txt_path = osp.join(DETS_ROOT, f"{seq_name}_det_temp.txt")

    if not osp.exists(video_path):
        logger.warning(f"视频不存在，跳过: {video_path}")
        return None

    if not osp.exists(det_txt_path):
        logger.warning(f"检测文件不存在，跳过: {det_txt_path}")
        return None

    logger.info(f"========== 开始追踪 {seq_name} ==========")
    logger.info(f"视频: {video_path}")
    logger.info(f"检测: {det_txt_path}")

    dets_by_frame = load_dets(det_txt_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return None

    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    if fps_vid <= 0:
        fps_vid = 30
    logger.info(f"视频 FPS = {fps_vid}")

    tracker = build_tracker(frame_rate=fps_vid, seq_name=seq_name)

    # 保存结果 txt（MOT 格式）
    save_txt_path = osp.join(TRACK_TXT_DIR, f"{seq_name}.txt")
    f_res = open(save_txt_path, "w", encoding="utf-8")

    # 视频保存
    if SAVE_VIS:
        save_vid_path = osp.join(VIS_VIDEO_DIR, f"{seq_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_vid_path, fourcc, fps_vid, (width, height))
        logger.info(f"可视化视频保存到: {save_vid_path}")
    else:
        vid_writer = None

    frame_id = 0

    # ====== timing start ======
    t0 = time.perf_counter()
    # =========================

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # 获取当前帧的检测
        dets = dets_by_frame.get(frame_id, None)
        if dets is None:
            dets = np.empty((0, 6), dtype=np.float32)

        # BoT-SORT 更新
        online_targets = tracker.update(dets, frame)

        online_tlwhs = []
        online_ids = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tscore = t.score

            if tlwh[2] * tlwh[3] <= 0:
                continue

            online_tlwhs.append(tlwh)
            online_ids.append(tid)

            # 保存到结果 txt:
            f_res.write(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                f"{tlwh[2]:.2f},{tlwh[3]:.2f},{tscore:.2f},-1,-1,-1\n"
            )

        # 可视化
        if SAVE_VIS:
            online_im = plot_tracking(
                frame, online_tlwhs, online_ids,
                frame_id=frame_id, fps=fps_vid
            )
            vid_writer.write(online_im)

        if frame_id % 50 == 0:
            logger.info(f"{seq_name}: 处理到第 {frame_id} 帧")

    # ====== timing end ======
    t1 = time.perf_counter()
    seconds = t1 - t0
    num_frames = frame_id
    fps_run = (num_frames / seconds) if seconds > 0 else float("nan")
    # ========================

    f_res.close()
    cap.release()
    if vid_writer is not None:
        vid_writer.release()

    logger.info(f"{seq_name} 追踪完成，结果保存至: {save_txt_path}")
    logger.info(f"[TIME] {seq_name}: frames={num_frames}, seconds={seconds:.6f}, fps={fps_run:.3f}")
    logger.info(f"=============================================\n")

    return {
        "seq": seq_name,
        "num_frames": int(num_frames),
        "seconds": float(seconds),
        "fps": float(fps_run),
    }


def main():
    logger.info("开始 BusFrontMOT 全部序列追踪（使用外部 det.txt + BoT-SORT）")

    timing_rows = []

    for idx in range(START_IDX, END_IDX + 1):
        seq_name = f"BF_{idx:02d}"
        rec = track_one_sequence(seq_name)
        if rec is not None:
            timing_rows.append(rec)

    # 写 timing.csv（放在 RESULTS_ROOT 下）
    if timing_rows:
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        with open(TIMING_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seq", "num_frames", "seconds", "fps"])
            for r in timing_rows:
                w.writerow([r["seq"], r["num_frames"], f"{r['seconds']:.6f}", f"{r['fps']:.6f}"])

        logger.info(f"[OK] timing.csv saved to: {TIMING_CSV_PATH}")

    logger.info("全部序列追踪完成！")


if __name__ == "__main__":
    main()
