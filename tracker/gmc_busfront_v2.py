# tracker/gmc_busfront.py
# -*- coding: utf-8 -*-
"""
BusFront-GMC: camera motion compensation specialized for bus-front videos.

特点：
    1. 仅在“背景区域”提取特征点（利用 det 框构造前景 mask）
    2. 使用稀疏 LK 光流跟踪特征点
    3. 采用 RANSAC 估计 2D 刚体仿射（平移+微旋转）
    4. 带历史平滑与质量控制，避免错误 GMC 破坏轨迹

返回：
    2x3 仿射矩阵 H，满足 x' = H * [x, y, 1]^T
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BusFrontGMCConfig:
    # 下采样倍数（和原 GMC 一致）
    downscale: int = 2

    # 特征点参数（GoodFeaturesToTrack）
    max_corners: int = 800
    quality_level: float = 0.01
    min_distance: int = 4
    block_size: int = 5

    # 前景/边缘 mask 相关
    mask_margin_top: float = 0.03   # 顶部丢弃比例（天空/车顶）
    mask_margin_bottom: float = 0.02
    mask_margin_left: float = 0.02
    mask_margin_right: float = 0.02
    fg_dilate: int = 4              # 检测框向外膨胀的像素（在下采样后图像上）

    # RANSAC 与质量阈值
    ransac_thresh: float = 3.0      # 像素 reprojection threshold
    min_inliers: int = 25           # 最少内点数
    max_translation_ratio: float = 0.25  # 最大允许平移：0.25 * min(H, W)

    # 平滑参数
    smoothing_alpha: float = 0.25   # 越小越平滑

    # 调试
    debug: bool = False


class BusFrontGMC:
    def __init__(self, cfg: BusFrontGMCConfig | None = None):
        if cfg is None:
            cfg = BusFrontGMCConfig()
        self.cfg = cfg

        # GFTT 参数
        self.feature_params = dict(
            maxCorners=cfg.max_corners,
            qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance,
            blockSize=cfg.block_size,
            useHarrisDetector=False,
            k=0.04,
        )

        # LK 光流参数：使用默认即可
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # 内部状态
        self.prev_gray = None            # 上一帧灰度图（下采样后）
        self.prev_pts = None             # 上一帧特征点
        self.smooth_H = None             # 历史平滑后的仿射矩阵（2x3）
        self.initialized = False

    # --------- 公共接口：供 gmc.GMC 调用 --------- #
    def reset(self):
        """重新开始一段新序列时调用。"""
        self.prev_gray = None
        self.prev_pts = None
        self.smooth_H = None
        self.initialized = False

    def apply(self, frame_bgr, detections_tlbr: np.ndarray | None = None) -> np.ndarray:
        """
        计算当前帧相对于上一帧的 2x3 仿射变换矩阵。

        参数：
            frame_bgr : 当前帧 BGR 图像 (H, W, 3)
            detections_tlbr : 检测框 [N, 4]，格式为 [x1, y1, x2, y2]（原图坐标）
                              可以为 None（例如无检测）

        返回：
            H_smooth : 平滑后的 2x3 仿射矩阵（float32）
        """
        cfg = self.cfg
        h0, w0, _ = frame_bgr.shape
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 下采样
        ds = max(1, int(cfg.downscale))
        if ds > 1:
            gray_ds = cv2.resize(
                gray,
                (w0 // ds, h0 // ds),
                interpolation=cv2.INTER_AREA
            )
        else:
            gray_ds = gray
        h, w = gray_ds.shape

        # 构造 background mask：1=背景，0=前景或无效区域
        bg_mask = np.ones_like(gray_ds, dtype=np.uint8) * 255

        # 去掉图像四周一定比例（天空、仪表盘等）
        mt, mb = int(cfg.mask_margin_top * h), int(cfg.mask_margin_bottom * h)
        ml, mr = int(cfg.mask_margin_left * w), int(cfg.mask_margin_right * w)
        bg_mask[:mt, :] = 0
        bg_mask[h - mb:, :] = 0
        bg_mask[:, :ml] = 0
        bg_mask[:, w - mr:] = 0

        # 用检测框“挖掉”所有前景（车辆 / 行人 / 骑行者）
        if detections_tlbr is not None and len(detections_tlbr) > 0:
            dets = np.asarray(detections_tlbr, dtype=np.float32).copy()
            # 映射到下采样图像上
            dets /= float(ds)
            dil = int(cfg.fg_dilate)
            for x1, y1, x2, y2 in dets:
                x1i = max(0, int(x1) - dil)
                y1i = max(0, int(y1) - dil)
                x2i = min(w, int(x2) + dil)
                y2i = min(h, int(y2) + dil)
                bg_mask[y1i:y2i, x1i:x2i] = 0

        # 提取背景特征点
        pts = cv2.goodFeaturesToTrack(
            gray_ds, mask=bg_mask, **self.feature_params
        )

        if pts is None or len(pts) < cfg.min_inliers:
            # 背景点太少：本帧 GMC 不可靠
            if self.cfg.debug:
                print("[BusFrontGMC] too few bg points, use previous H.")
            return self._return_identity_or_smooth()

        # ------------ 第一帧：仅初始化，不做估计 ------------ #
        if not self.initialized:
            self.prev_gray = gray_ds
            self.prev_pts = copy.copy(pts)
            self.smooth_H = np.eye(2, 3, dtype=np.float32)
            self.initialized = True
            return self.smooth_H.copy()

        # ------------- 2. LK 光流跟踪特征点 ------------- #
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_ds, self.prev_pts, None, **self.lk_params
        )

        if next_pts is None or status is None:
            if self.cfg.debug:
                print("[BusFrontGMC] LK failed, use previous H.")
            return self._return_identity_or_smooth()

        status = status.reshape(-1).astype(bool)
        prev_good = self.prev_pts.reshape(-1, 2)[status]
        next_good = next_pts.reshape(-1, 2)[status]

        if len(prev_good) < cfg.min_inliers:
            if self.cfg.debug:
                print("[BusFrontGMC] not enough tracked points, use previous H.")
            # 更新上一帧，避免一直卡死在第一帧
            self.prev_gray = gray_ds
            self.prev_pts = pts
            return self._return_identity_or_smooth()

        # ------------- 3. 估计刚性仿射矩阵 H ------------- #
        # 使用部分仿射：只允许旋转+缩放+平移（不含投影 shear）
        H, inliers = cv2.estimateAffinePartial2D(
            prev_good, next_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.ransac_thresh,
            maxIters=2000,
            confidence=0.99,
            refineIters=10
        )

        if H is None:
            if self.cfg.debug:
                print("[BusFrontGMC] estimateAffinePartial2D failed, use previous H.")
            self.prev_gray = gray_ds
            self.prev_pts = pts
            return self._return_identity_or_smooth()

        # RANSAC 内点检查
        if inliers is not None:
            num_inliers = int(inliers.sum())
            if num_inliers < cfg.min_inliers:
                if self.cfg.debug:
                    print(f"[BusFrontGMC] inliers={num_inliers} < min_inliers, use previous H.")
                self.prev_gray = gray_ds
                self.prev_pts = pts
                return self._return_identity_or_smooth()

        # ------------- 4. 合理性检查（平移幅度） ------------- #
        # 将平移部分映射回原分辨率尺度
        tx = float(H[0, 2]) * ds
        ty = float(H[1, 2]) * ds
        max_trans = cfg.max_translation_ratio * min(h0, w0)

        if abs(tx) > max_trans or abs(ty) > max_trans:
            if self.cfg.debug:
                print(f"[BusFrontGMC] translation too large ({tx:.1f},{ty:.1f}), use previous H.")
            self.prev_gray = gray_ds
            self.prev_pts = pts
            return self._return_identity_or_smooth()

        # ------------- 5. 将 H 调整回原图尺度，并做历史平滑 ------------- #
        # 把平移调回原尺度
        H_full = H.copy()
        H_full[0, 2] *= ds
        H_full[1, 2] *= ds

        H_full = H_full.astype(np.float32)

        if self.smooth_H is None:
            self.smooth_H = H_full
        else:
            a = cfg.smoothing_alpha
            self.smooth_H = (a * H_full + (1.0 - a) * self.smooth_H).astype(np.float32)

        # ------------- 6. 更新内部状态并返回 ------------- #
        self.prev_gray = gray_ds
        self.prev_pts = pts

        return self.smooth_H.copy()

    # ----------------------------------------------------- #
    def _return_identity_or_smooth(self) -> np.ndarray:
        """
        当当前帧估计失败时：
            - 如果已有平滑历史 H，则返回该 H
            - 否则返回单位仿射
        """
        if self.smooth_H is not None:
            return self.smooth_H.copy()
        else:
            self.smooth_H = np.eye(2, 3, dtype=np.float32)
            return self.smooth_H.copy()
