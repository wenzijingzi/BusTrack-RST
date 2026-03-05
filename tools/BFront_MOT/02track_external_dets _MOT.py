import os
import os.path as osp
from types import SimpleNamespace
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from loguru import logger
import sys
import time
import csv

# 确保能 import tracker
sys.path.append(".")
from tracker.bot_sort import BoTSORT

"""
对 ST-NMS 后的 MOT17 detections 进行 BoT-SORT 追踪：
- 图像/seqinfo/gt 来自 MOT17 原始 train 目录（IMG_ROOT）
- det/det.txt 来自 ST-NMS 输出目录（DET_ROOT）

输入：
  IMG_ROOT/<seq>/img1/*.jpg
  IMG_ROOT/<seq>/seqinfo.ini   (可选但推荐)
  DET_ROOT/<seq>/det/det.txt   (ST-NMS 后)

输出：
  RESULTS_ROOT/tracks/<seq>.txt
  RESULTS_ROOT/vis/<seq>.mp4   (可选)
  RESULTS_ROOT/timing.csv
"""

# ===================== 配置区域 ===================== #

# 原始 MOT17 train（有 img1/ seqinfo.ini gt/）
IMG_ROOT = r"E:\track\dataset\MOT20\train"

# ST-NMS 输出（你现在只有 det/det.txt）
DET_ROOT = r"E:\track\dataset\MOT20\train_stnms"

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
RESULTS_ROOT = r"E:\track\dataset\MOT20\track_result_stnms_botsort"
TRACK_TXT_DIR = osp.join(RESULTS_ROOT, "tracks")
VIS_VIDEO_DIR = osp.join(RESULTS_ROOT, "vis")
os.makedirs(TRACK_TXT_DIR, exist_ok=True)
os.makedirs(VIS_VIDEO_DIR, exist_ok=True)

SAVE_VIS = True
SAVE_TIMING = True

# BoT-SORT 相机运动补偿
CMC_METHOD = "none"  # "none" / "orb" / "ecc"

# ===================== 可视化：按 track_id 固定颜色 ===================== #

CLASS_NAMES = {0: "Pedestrian"}

def color_by_id(tid: int):
    tid = int(tid)
    r = (tid * 37) % 255
    g = (tid * 17) % 255
    b = (tid * 29) % 255
    r = 64 + (r % 192)
    g = 64 + (g % 192)
    b = 64 + (b % 192)
    return (b, g, r)

# ===================== 构造 BoT-SORT 参数 ===================== #

def build_tracker(frame_rate=30, seq_name="MOT20-01"):
    args = SimpleNamespace()
    args.name = seq_name
    args.ablation = False

    args.track_high_thresh = 0.6
    args.track_low_thresh = 0.1
    args.new_track_thresh = 0.7
    args.track_buffer = 30
    args.match_thresh = 0.8
    args.aspect_ratio_thresh = 1.6
    args.min_box_area = 10
    args.mot20 = False

    # CMC
    args.cmc_method = CMC_METHOD

    # ReID
    args.with_reid = True
    args.fast_reid_config = "E:/track/BoT-SORT-main/fast_reid/configs/MOT17/sbs_S50.yml"
    args.fast_reid_weights = "E:/track/BoT-SORT-main/pretrained/mot17_sbs_S50.pth"
    args.proximity_thresh = 0.5
    args.appearance_thresh = 0.25

    args.device = "gpu"
    return BoTSORT(args, frame_rate=frame_rate)

# ===================== 读取 det/det.txt（兼容 8列或10列） ===================== #

def load_dets_auto(det_txt_path: str) -> Dict[int, np.ndarray]:
    """
    det.txt 兼容：
      - 8列：frame,id,x,y,w,h,score,cls
      - 10列：frame,id,x,y,w,h,score,x,y,z (MOT official det)

    返回：
      dets_by_frame[fid] = [N,6] -> [x1,y1,x2,y2,score,cls]
    cls 若不存在/为 MOT10，则默认 0（Pedestrian）
    """
    if not osp.exists(det_txt_path):
        logger.warning(f"det 不存在: {det_txt_path}")
        return {}

    data = np.loadtxt(det_txt_path, delimiter=",", dtype=np.float32)
    if data.size == 0:
        return {}
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 7:
        raise ValueError(f"det.txt 至少 7 列(frame,id,x,y,w,h,score)，但 got {data.shape[1]}: {det_txt_path}")

    ncol = data.shape[1]
    is_mot10 = (ncol >= 10)  # 10列时第8列不是 cls

    dets_by_frame: Dict[int, list] = {}

    for row in data:
        fid = int(row[0])
        x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        score = float(row[6])

        cls = 0.0
        if (not is_mot10) and ncol >= 8:
            cls = float(row[7])

        x1, y1, x2, y2 = x, y, x + w, y + h
        dets_by_frame.setdefault(fid, []).append([x1, y1, x2, y2, score, cls])

    for k in list(dets_by_frame.keys()):
        dets_by_frame[k] = np.asarray(dets_by_frame[k], dtype=np.float32)

    logger.info(f"det加载完成: {det_txt_path} | dets={data.shape[0]} | frames_with_det={len(dets_by_frame)}")
    return dets_by_frame

# ===================== 读 seqinfo.ini（fps/width/height） ===================== #

def read_seqinfo(seq_dir: str) -> Tuple[float, Optional[int], Optional[int]]:
    fps = 30.0
    w = h = None
    ini_path = osp.join(seq_dir, "seqinfo.ini")
    if not osp.exists(ini_path):
        return fps, w, h
    try:
        with open(ini_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("frameRate"):
                    fps = float(line.split("=")[1])
                elif line.startswith("imWidth"):
                    w = int(line.split("=")[1])
                elif line.startswith("imHeight"):
                    h = int(line.split("=")[1])
    except Exception as e:
        logger.warning(f"读 seqinfo.ini 失败，使用默认 fps=30: {e}")
    return fps, w, h

# ===================== 主追踪逻辑 ===================== #

def track_one_sequence(seq_name: str):
    """
    图像来自 IMG_ROOT，检测来自 DET_ROOT
    """
    img_seq_dir = osp.join(IMG_ROOT, seq_name)
    det_seq_dir = osp.join(DET_ROOT, seq_name)

    img_dir = osp.join(img_seq_dir, "img1")
    det_txt = osp.join(det_seq_dir, "det", "det.txt")

    if not osp.isdir(img_dir):
        logger.warning(f"img1 不存在，跳过: {img_dir}")
        return None

    if not osp.exists(det_txt):
        logger.warning(f"det 不存在，跳过: {det_txt}")
        return None

    fps, W, H = read_seqinfo(img_seq_dir)

    logger.info(f"========== {seq_name} ==========")
    logger.info(f"img1 : {img_dir}")
    logger.info(f"det  : {det_txt}")
    logger.info(f"fps  : {fps}")

    dets_by_frame = load_dets_auto(det_txt)

    tracker = build_tracker(frame_rate=fps, seq_name=seq_name)

    # 输出 tracking 文件
    out_txt = osp.join(TRACK_TXT_DIR, f"{seq_name}.txt")
    os.makedirs(osp.dirname(out_txt), exist_ok=True)

    # 可视化 writer
    vid_writer = None
    if SAVE_VIS:
        out_vid = osp.join(VIS_VIDEO_DIR, f"{seq_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 没读到宽高就读一张图
        if W is None or H is None:
            first = osp.join(img_dir, sorted(os.listdir(img_dir))[0])
            im0 = cv2.imread(first)
            if im0 is not None:
                H, W = im0.shape[:2]
            else:
                W, H = 1920, 1080

        vid_writer = cv2.VideoWriter(out_vid, fourcc, fps, (W, H))
        logger.info(f"vis  : {out_vid}")

    # 遍历帧
    frame_files = sorted([f for f in os.listdir(img_dir) if osp.splitext(f)[0].isdigit()])
    if not frame_files:
        logger.warning(f"{seq_name}: img1 为空")
        if vid_writer is not None:
            vid_writer.release()
        return None

    t0 = time.time()
    num_frames = 0

    with open(out_txt, "w", encoding="utf-8") as f_res:
        for fname in frame_files:
            fid = int(osp.splitext(fname)[0])
            img_path = osp.join(img_dir, fname)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            num_frames += 1
            dets = dets_by_frame.get(fid, np.empty((0, 6), dtype=np.float32))

            online_targets = tracker.update(dets, frame)

            if SAVE_VIS:
                online_im = frame.copy()

            for t in online_targets:
                tlwh = t.tlwh
                tid = int(t.track_id)
                tscore = float(t.score)

                if tlwh[2] * tlwh[3] <= 0:
                    continue

                # MOTChallenge: frame,id,x,y,w,h,score,-1,-1,-1
                f_res.write(
                    f"{fid},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                    f"{tlwh[2]:.2f},{tlwh[3]:.2f},{tscore:.2f},-1,-1,-1\n"
                )

                if SAVE_VIS:
                    x, y, w, h = tlwh
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    color = color_by_id(tid)
                    label = f"{tid}:Ped"
                    cv2.rectangle(online_im, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        online_im, label, (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
                    )

            if SAVE_VIS and vid_writer is not None:
                vid_writer.write(online_im)

            if num_frames % 200 == 0:
                logger.info(f"{seq_name}: processed {num_frames} frames (fid={fid})")

    t1 = time.time()
    seconds = t1 - t0
    fps_run = (num_frames / seconds) if seconds > 0 else float("nan")

    if vid_writer is not None:
        vid_writer.release()

    logger.info(f"[DONE] tracks -> {out_txt}")
    logger.info(f"[TIME] frames={num_frames}, seconds={seconds:.3f}, fps={fps_run:.2f}")
    logger.info("=============================================")

    return num_frames, seconds, fps_run


def main():
    logger.info("开始：使用 ST-NMS det + 原始 img1 进行 BoT-SORT（MOT17）")
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    timing_rows = []
    for seq in SEQS:
        out = track_one_sequence(seq)
        if out is None:
            continue
        timing_rows.append([seq, out[0], out[1], out[2]])

    if SAVE_TIMING:
        timing_path = osp.join(RESULTS_ROOT, "timing.csv")
        with open(timing_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seq", "num_frames", "seconds", "fps"])
            w.writerows(timing_rows)
        logger.info(f"[OK] Saved timing.csv: {timing_path}")

    logger.info("全部序列追踪完成！")


if __name__ == "__main__":
    main()
