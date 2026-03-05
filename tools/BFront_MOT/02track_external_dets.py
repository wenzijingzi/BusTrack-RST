import os
import os.path as osp
from types import SimpleNamespace

import cv2
import numpy as np
from loguru import logger

import torch
import sys

# 确保能 import yolox 和 tracker
sys.path.append('.')

from tracker.bot_sort_soft_oip_tcr_v1 import BoTSORT
from yolox.utils.visualize import plot_tracking


"""
使用已有 det.txt (YOLO-M2LA 导出的检测结果) 跑 BoT-SORT 追踪

输入：
    - 视频：E:\track\dataset\BusFrontMOT\videos\BF_01.mp4 ... BF_19.mp4
    - 检测：E:\track\dataset\BusFrontMOT\dets\BF_01_det.txt ... BF_19_det.txt

输出：
    - 轨迹 txt：./YOLOX_outputs/m2la_botsort/tracks/BF_01.txt ...
    - 可视化视频：./YOLOX_outputs/m2la_botsort/vis/BF_01.mp4 ...
"""

# ===================== 配置区域 ===================== #

VIDEO_ROOT = r"E:\track\dataset\BusFrontMOT\videos"
DETS_ROOT = r"E:\track\dataset\BusFrontMOT\dets"

# 轨迹与可视化保存目录（相对 BoT-SORT 主目录）
RESULTS_ROOT = r"E:/track/BoT-SORT-main/YOLOX_outputs/m2la_botsort"
TRACK_TXT_DIR = osp.join(RESULTS_ROOT, "tracks")
VIS_VIDEO_DIR = osp.join(RESULTS_ROOT, "vis")

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
    args.name = seq_name  # 序列名称，用于日志
    args.ablation = False  # 我们现在不做 ablation 实验

    # ByteTrack / BoT-SORT tracking 参数
    args.track_high_thresh = 0.6      # 高置信度阈值
    args.track_low_thresh = 0.1       # 最低跟踪阈值
    args.new_track_thresh = 0.7       # 新轨迹阈值
    args.track_buffer = 30            # 丢失保持帧数
    args.match_thresh = 0.8
    args.aspect_ratio_thresh = 1.6
    args.min_box_area = 10
    args.mot20 = False

    # CMC（相机运动补偿） — 先用 "none"，后面你可以改 "orb" / "ecc"
    args.cmc_method = "none"   # 或 "orb" / "ecc"

    # ReID 相关（你环境里已经下好了 MOT17 的权重）
    args.with_reid = True
    args.fast_reid_config = "E:/track/BoT-SORT-main/fast_reid/configs/MOT17/sbs_S50.yml"
    args.fast_reid_weights = "E:/track/BoT-SORT-main/pretrained/mot17_sbs_S50.pth"
    args.proximity_thresh = 0.5
    args.appearance_thresh = 0.25

    # 其它占位参数（BoTSORT 不一定用得到，但补上以防万一）
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
        data = data[None, :]   # 只有一行的情况

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

    # 转为 ndarray
    for k in dets_by_frame.keys():
        dets_by_frame[k] = np.array(dets_by_frame[k], dtype=np.float32)

    logger.info(f"加载检测完成: {det_txt_path}，共 {len(data)} 个检测框，{len(dets_by_frame)} 帧有检测")
    return dets_by_frame


# ===================== 主追踪逻辑 ===================== #

def track_one_sequence(seq_name):
    """
    对单个视频序列 (BF_xx) 进行追踪
    """
    video_path = osp.join(VIDEO_ROOT, f"{seq_name}.mp4")
    det_txt_path = osp.join(DETS_ROOT, f"{seq_name}_det.txt")

    if not osp.exists(video_path):
        logger.warning(f"视频不存在，跳过: {video_path}")
        return

    if not osp.exists(det_txt_path):
        logger.warning(f"检测文件不存在，跳过: {det_txt_path}")
        return

    logger.info(f"========== 开始追踪 {seq_name} ==========")
    logger.info(f"视频: {video_path}")
    logger.info(f"检测: {det_txt_path}")

    dets_by_frame = load_dets(det_txt_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    logger.info(f"视频 FPS = {fps}")

    tracker = build_tracker(frame_rate=fps, seq_name=seq_name)


    # 保存结果 txt（MOT 格式）
    save_txt_path = osp.join(TRACK_TXT_DIR, f"{seq_name}.txt")
    f_res = open(save_txt_path, "w")

    # 视频保存
    if SAVE_VIS:
        save_vid_path = osp.join(VIS_VIDEO_DIR, f"{seq_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_vid_path, fourcc, fps, (width, height))
        logger.info(f"可视化视频保存到: {save_vid_path}")
    else:
        vid_writer = None

    frame_id = 0

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
        online_scores = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tscore = t.score

            # 过滤掉太小目标（和 track.py 一致）
            if tlwh[2] * tlwh[3] <= 0:
                continue

            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(tscore)

            # 保存到结果 txt:
            # frame, id, x, y, w, h, score, -1, -1, -1
            f_res.write(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                f"{tlwh[2]:.2f},{tlwh[3]:.2f},{tscore:.2f},-1,-1,-1\n"
            )

        # 可视化
        if SAVE_VIS:
            online_im = plot_tracking(
                frame, online_tlwhs, online_ids,
                frame_id=frame_id, fps=fps
            )
            vid_writer.write(online_im)

        if frame_id % 50 == 0:
            logger.info(f"{seq_name}: 处理到第 {frame_id} 帧")

    f_res.close()
    cap.release()
    if vid_writer is not None:
        vid_writer.release()

    logger.info(f"{seq_name} 追踪完成，结果保存至: {save_txt_path}")
    logger.info(f"=============================================\n")


def main():
    logger.info("开始 BusFrontMOT 全部序列追踪（使用外部 det.txt + BoT-SORT）")

    for idx in range(START_IDX, END_IDX + 1):
        seq_name = f"BF_{idx:02d}"
        track_one_sequence(seq_name)

    logger.info("全部序列追踪完成！")


if __name__ == "__main__":
    main()
