import os
import cv2
from ultralytics import YOLO

"""
YOLO-M2LA / YOLOv8/YOLOv9/YOLOv11 批量检测导出脚本
自动处理 BusFrontMOT 数据集中 BF_01 - BF_19 的所有视频

为每个视频生成对应的 BoT-SORT 检测文件：
BF_01_det.txt, BF_02_det.txt, ..., BF_19_det.txt

输出格式（MOT 标准）：
frame,id,x,y,w,h,score,class
"""


# ===================== 配置区域 ===================== #

# 你的视频目录
VIDEO_FOLDER = r"E:\track\dataset\BusFrontMOT\videos"

# 存储 det.txt 的目录
SAVE_FOLDER = r"E:\track\dataset\BusFrontMOT\dets"

# 加载 YOLO-M2LA 模型（你的 best.pt）
MODEL_PATH = r"E:\track\YOLOv11\runs\train\exp56\weights\best.pt"
model = YOLO(MODEL_PATH)

# 置信度阈值
CONF_THRES = 0.25

# 保留车辆类（YOLO-M2LA 类别定义：2-Car, 3-Truck, 4-Van）
VALID_CLASSES = [2, 3, 4]

# 强制统一成的类别 ID（建议 0 或 2，论文中可以写“all vehicles”）
UNIFIED_CLASS_ID = 2

# 批处理范围：BF_01 到 BF_19（可自行扩展）
START_IDX = 1
END_IDX = 19

# ==================================================== #


def export_one_video(video_path, save_path):
    """对单个 video 进行检测并导出 txt"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[跳过] 无法读取视频：{video_path}")
        return

    frame_id = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = model(frame)[0]

            for box in results.boxes:
                raw_cls = int(box.cls)   # 原始类别
                conf = float(box.conf)

                if conf < CONF_THRES:
                    continue
                if VALID_CLASSES and raw_cls not in VALID_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                w = x2 - x1
                h = y2 - y1

                # ❗关键：写入时统一成一个类别（避免 car/van/truck 抖动）
                cls_for_track = UNIFIED_CLASS_ID

                # 写入 BoT-SORT 格式：frame, id, x, y, w, h, score, class
                f.write(
                    f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{conf:.4f},{cls_for_track}\n"
                )

            print(f"{os.path.basename(video_path)} → Frame {frame_id}")

    cap.release()
    print(f"检测完成：{save_path}\n")


def batch_export():
    """批处理 BF_01 ~ BF_19"""

    for idx in range(START_IDX, END_IDX + 1):
        video_name = f"BF_{idx:02d}.mp4"
        video_path = os.path.join(VIDEO_FOLDER, video_name)
        save_path = os.path.join(SAVE_FOLDER, f"BF_{idx:02d}_det.txt")

        if not os.path.exists(video_path):
            print(f"[跳过] 找不到视频：{video_name}")
            continue

        print(f"\n========== 开始处理 {video_name} ==========")
        export_one_video(video_path, save_path)

    print("\n所有视频处理完成！")


if __name__ == "__main__":
    batch_export()
