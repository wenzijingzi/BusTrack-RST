import os
import numpy as np

gt_root = r"E:\track\dataset\BusFrontMOT\sequences\GT"

grand_total_boxes = 0
grand_total_tracks_no_merge = 0  # 轨迹数：各序列直接相加

files = [f for f in os.listdir(gt_root) if f.endswith(".txt") and f.startswith("BF_")]
files.sort()

print("Per-sequence stats:")
print("-" * 60)

for name in files:
    path = os.path.join(gt_root, name)

    try:
        data = np.loadtxt(path, delimiter=",")
    except Exception:
        data = np.genfromtxt(path, delimiter=",", dtype=float)

    # 空文件
    if data is None or (hasattr(data, "size") and data.size == 0):
        print(f"{name:10s} | boxes = {0:6d} | tracks = {0:6d}")
        continue

    if data.ndim == 1:
        data = data[None, :]

    boxes = int(data.shape[0])

    # 第2列是 track_id
    ids = data[:, 1].astype(int)

    # 如果你的GT里 -1 或 0 表示无效ID，可以过滤
    ids = ids[ids > 0]

    unique_ids = set(ids.tolist())
    num_tracks = len(unique_ids)

    print(f"{name:10s} | boxes = {boxes:6d} | tracks = {num_tracks:6d}")

    grand_total_boxes += boxes
    grand_total_tracks_no_merge += num_tracks

print("-" * 60)
print("Overall (dataset level):")
print("总标注框数 (all bounding boxes):", grand_total_boxes)
print("总轨迹数 (tracks, no cross-seq merge):", grand_total_tracks_no_merge)
