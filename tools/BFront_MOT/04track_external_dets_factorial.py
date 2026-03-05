import os
import os.path as osp
from types import SimpleNamespace

import cv2
import numpy as np
from loguru import logger
import sys

# 确保能 import yolox 和 tracker
sys.path.append('.')

from tracker.bot_sort import BoTSORT

from yolox.utils.visualize import plot_tracking


"""
2×2 因子实验：Det (raw vs tempX) × CMC (none vs busfront)

- raw det:     {seq}_det.txt        in DETS_ROOT_RAW
- tempX det:   {seq}_det_temp.txt   in DETS_ROOT_TEMPX   (Temporal+Spatial NMS 后的 det)

输出：
  YOLOX_outputs/<exp_name>/
    tracks/BF_01.txt ...
    vis/BF_01.mp4 ...
"""

"""
在 BoT-SORT-main 根目录下（确保能 import tracker / yolox），依次运行：
# A1: baseline det (raw) + No-CMC
python tools/BFront_MOT/04track_external_dets_factorial.py --det raw --cmc none

# A2: baseline det (raw) + BusFront-GMC++
python tools/BFront_MOT/04track_external_dets_factorial.py --det raw --cmc busfront

# B1: tempX det (Temporal+Spatial NMS) + No-CMC
python tools/BFront_MOT/04track_external_dets_factorial.py --det tempX --cmc none

# B2: tempX det (Temporal+Spatial NMS) + BusFront-GMC++
python tools/BFront_MOT/04track_external_dets_factorial.py --det tempX --cmc busfront


输出会自动落在：

YOLOX_outputs/m2la_botsort_raw_nocmc/

YOLOX_outputs/m2la_botsort_raw_busfront/

YOLOX_outputs/m2la_botsort_tempX_nocmc/

YOLOX_outputs/m2la_botsort_tempX_busfront/
"""

# ===================== 你的数据路径（按需修改） =====================

VIDEO_ROOT = r"E:\track\dataset\BusFrontMOT\videos"

# raw det（baseline det）
DETS_ROOT_RAW = r"E:\track\dataset\BusFrontMOT\dets_all"

# tempX det（Temporal + Spatial NMS 后的 det）
DETS_ROOT_TEMPX = r"E:\track\dataset\BusFrontMOT\dets_tnms_X_all"  # 你现在的 tempX det 目录

# 输出根目录（统一放 YOLOX_outputs 下）
YOLOX_OUT_ROOT = r"E:\track\BoT-SORT-main\YOLOX_outputs\m2la_botsort_all"

# 序列范围 BF_01 ~ BF_19
START_IDX = 1
END_IDX = 19

# 是否保存可视化视频
SAVE_VIS = True

# ===================== 参数 & 工具函数 =====================

def build_tracker(frame_rate=30, seq_name="BF_01", cmc_method="none"):
    """
    手动构造 args 初始化 BoTSORT
    cmc_method: "none" | "orb" | "ecc" | "sparseOptFlow" | "busfront"
    """
    args = SimpleNamespace()

    # GMC 会用到
    args.name = seq_name
    args.ablation = False

    # BoT-SORT/ByteTrack 参数
    args.track_high_thresh = 0.6
    args.track_low_thresh = 0.1
    args.new_track_thresh = 0.7
    args.track_buffer = 30
    args.match_thresh = 0.8
    args.aspect_ratio_thresh = 1.6
    args.min_box_area = 10
    args.mot20 = False

    # 关键：CMC 方法
    args.cmc_method = cmc_method  # "none" or "busfront"

    # ReID
    args.with_reid = True
    args.fast_reid_config = "E:/track/BoT-SORT-main/fast_reid/configs/MOT17/sbs_S50.yml"
    args.fast_reid_weights = "E:/track/BoT-SORT-main/pretrained/mot17_sbs_S50.pth"
    args.proximity_thresh = 0.5
    args.appearance_thresh = 0.25

    # 占位
    args.device = "gpu"

    tracker = BoTSORT(args, frame_rate=frame_rate)
    return tracker


def load_dets(det_txt_path):
    """
    det.txt: frame,id,x,y,w,h,score,class
    返回 dets_by_frame[frame] = ndarray(N,6) [x1,y1,x2,y2,score,cls]
    """
    if not osp.exists(det_txt_path):
        logger.warning(f"检测文件不存在: {det_txt_path}")
        return {}

    data = np.loadtxt(det_txt_path, delimiter=',')
    if data.size == 0:
        return {}

    if data.ndim == 1:
        data = data[None, :]

    dets_by_frame = {}
    for row in data:
        frame_id = int(row[0])
        x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        score = float(row[6])
        cls = float(row[7])

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        dets_by_frame.setdefault(frame_id, []).append([x1, y1, x2, y2, score, cls])

    for k in list(dets_by_frame.keys()):
        dets_by_frame[k] = np.asarray(dets_by_frame[k], dtype=np.float32)

    logger.info(f"加载检测完成: {det_txt_path}，共 {len(data)} 个检测框，{len(dets_by_frame)} 帧有检测")
    return dets_by_frame


def get_det_path(seq_name: str, det_variant: str):
    """
    det_variant:
      - "raw"   -> dets/BF_xx_det.txt
      - "tempX" -> dets_tnms_X/BF_xx_det_temp.txt
    """
    if det_variant == "raw":
        det_root = DETS_ROOT_RAW
        det_file = f"{seq_name}_det.txt"
    elif det_variant == "tempX":
        det_root = DETS_ROOT_TEMPX
        det_file = f"{seq_name}_det_temp.txt"
    else:
        raise ValueError(f"Unknown det_variant={det_variant}, choose from [raw, tempX]")

    return osp.join(det_root, det_file)


def get_exp_name(det_variant: str, cmc_method: str):
    """
    统一命名：便于 eval_all_trackers.py 直接对比
    """
    # e.g. m2la_botsort_raw_nocmc / m2la_botsort_raw_busfront
    cmc_tag = "nocmc" if cmc_method == "none" else cmc_method
    return f"m2la_botsort_{det_variant}_{cmc_tag}"


def track_one_sequence(seq_name, det_variant, cmc_method):
    video_path = osp.join(VIDEO_ROOT, f"{seq_name}.mp4")
    det_txt_path = get_det_path(seq_name, det_variant)

    if not osp.exists(video_path):
        logger.warning(f"视频不存在，跳过: {video_path}")
        return

    if not osp.exists(det_txt_path):
        logger.warning(f"检测文件不存在，跳过: {det_txt_path}")
        return

    exp_name = get_exp_name(det_variant, cmc_method)
    results_root = osp.join(YOLOX_OUT_ROOT, exp_name)
    track_dir = osp.join(results_root, "tracks")
    vis_dir = osp.join(results_root, "vis")
    os.makedirs(track_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    logger.info(f"========== 开始追踪 {seq_name} ==========")
    logger.info(f"DET variant = {det_variant}, CMC = {cmc_method}")
    logger.info(f"视频: {video_path}")
    logger.info(f"检测: {det_txt_path}")
    logger.info(f"输出: {results_root}")

    dets_by_frame = load_dets(det_txt_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    logger.info(f"视频 FPS = {fps}")

    tracker = build_tracker(frame_rate=fps, seq_name=seq_name, cmc_method=cmc_method)

    # 输出 txt
    save_txt_path = osp.join(track_dir, f"{seq_name}.txt")
    f_res = open(save_txt_path, "w", encoding="utf-8")

    # 输出 vis
    if SAVE_VIS:
        save_vid_path = osp.join(vis_dir, f"{seq_name}.mp4")
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

        dets = dets_by_frame.get(frame_id, None)
        if dets is None:
            dets = np.empty((0, 6), dtype=np.float32)

        online_targets = tracker.update(dets, frame)

        online_tlwhs, online_ids = [], []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tscore = float(t.score)

            if tlwh[2] * tlwh[3] <= 0:
                continue

            online_tlwhs.append(tlwh)
            online_ids.append(tid)

            # MOT: frame,id,x,y,w,h,score,-1,-1,-1
            f_res.write(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                f"{tlwh[2]:.2f},{tlwh[3]:.2f},{tscore:.2f},-1,-1,-1\n"
            )

        if SAVE_VIS and vid_writer is not None:
            online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id, fps=fps)
            vid_writer.write(online_im)

        if frame_id % 200 == 0:
            logger.info(f"{seq_name}: 处理到第 {frame_id} 帧")

    f_res.close()
    cap.release()
    if vid_writer is not None:
        vid_writer.release()

    logger.info(f"{seq_name} 追踪完成，结果保存至: {save_txt_path}")
    logger.info("=============================================\n")


def run_all(det_variant: str, cmc_method: str):
    logger.info(f"开始 BusFrontMOT 全部序列追踪：det={det_variant}, cmc={cmc_method}")
    for idx in range(START_IDX, END_IDX + 1):
        seq_name = f"BF_{idx:02d}"
        track_one_sequence(seq_name, det_variant, cmc_method)
    logger.info("全部序列追踪完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--det", type=str, default="tempX", choices=["raw", "tempX"],
                        help="raw=baseline det, tempX=Temporal+Spatial NMS det")
    parser.add_argument("--cmc", type=str, default="none",
                        choices=["none", "busfront", "orb", "ecc", "sparseOptFlow"],
                        help="CMC method for BoT-SORT")
    args = parser.parse_args()

    run_all(det_variant=args.det, cmc_method=args.cmc)
