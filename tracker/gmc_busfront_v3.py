# tracker/gmc_busfront.py
# -*- coding: utf-8 -*-
"""
BusFront-GMC++: camera motion compensation specialized for bus-front videos.

升级点（对应你的(1)(2)(4)）：
(1) 用可解释的“质量分数 quality_score”做门控，而不是只看 min_inliers / translation
(2) ROI 更贴合公交前视角：使用“路面梯形区域”作为背景优先区域，抑制天空/建筑边缘干扰
(4) 特征点策略：网格均匀采样 + 动态点数（随 ROI 可用面积自动调整）

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
    # 下采样倍数
    downscale: int = 2

    # ---------- (2) 路面 ROI：梯形区域 ----------
    # 以“下采样后图像”为基准，比例参数
    # ROI 顶边 y = road_top_ratio * H，底边为 H
    road_top_ratio: float = 0.40
    # 顶边左右收缩（越大越窄），底边左右收缩
    road_top_shrink: float = 0.18
    road_bottom_shrink: float = 0.02

    # 额外边缘 margin（避免四周畸变/遮挡）
    mask_margin_top: float = 0.02
    mask_margin_bottom: float = 0.02
    mask_margin_left: float = 0.02
    mask_margin_right: float = 0.02

    # 前景挖洞（检测框膨胀像素，基于下采样图）
    fg_dilate: int = 4

    # ---------- (4) 动态点数 + 网格均匀采样 ----------
    # 目标特征点“密度”：每 1e5 像素大约多少点（在 ROI 可用区域上估计）
    target_density_per_1e5: float = 220.0
    min_corners: int = 250
    max_corners: int = 1200

    # Shi-Tomasi 角点参数
    quality_level: float = 0.01
    min_distance: int = 4
    block_size: int = 5

    # 网格采样参数：rows x cols，每格最多保留 max_per_cell 个点
    grid_rows: int = 12
    grid_cols: int = 20
    max_per_cell: int = 8

    # ---------- RANSAC ----------
    ransac_thresh: float = 3.0
    max_iters: int = 2000
    confidence: float = 0.99
    refine_iters: int = 10

    # ---------- (1) 质量分数门控 ----------
    # 分数阈值：低于则拒绝本帧 GMC（回退到 smooth_H）
    min_quality: float = 0.35
    # 误差/平移归一化参考（越小越严格）
    err_ref_px: float = 3.0
    trans_ref_ratio: float = 0.08  # 0.08 * min(H,W)

    # 平滑（基础 alpha；可结合质量分数自适应）
    smoothing_alpha: float = 0.25
    adaptive_smoothing: bool = True
    alpha_min: float = 0.10
    alpha_max: float = 0.55

    debug: bool = False


class BusFrontGMC:
    def __init__(self, cfg: BusFrontGMCConfig | None = None):
        self.cfg = cfg if cfg is not None else BusFrontGMCConfig()

        # LK 光流参数
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # 内部状态
        self.prev_gray = None
        self.prev_pts = None
        self.smooth_H = None
        self.initialized = False

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.smooth_H = None
        self.initialized = False

    def apply(self, frame_bgr, detections_tlbr: np.ndarray | None = None) -> np.ndarray:
        cfg = self.cfg
        h0, w0, _ = frame_bgr.shape
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 下采样
        ds = max(1, int(cfg.downscale))
        if ds > 1:
            gray_ds = cv2.resize(gray, (w0 // ds, h0 // ds), interpolation=cv2.INTER_AREA)
        else:
            gray_ds = gray
        h, w = gray_ds.shape

        # ============== (2) 构造“路面 ROI 背景 mask” ==============
        bg_mask = self._build_road_roi_mask(h, w)  # 255=可用背景区域

        # 边缘 margin 再裁一次
        mt, mb = int(cfg.mask_margin_top * h), int(cfg.mask_margin_bottom * h)
        ml, mr = int(cfg.mask_margin_left * w), int(cfg.mask_margin_right * w)
        if mt > 0:
            bg_mask[:mt, :] = 0
        if mb > 0:
            bg_mask[h - mb :, :] = 0
        if ml > 0:
            bg_mask[:, :ml] = 0
        if mr > 0:
            bg_mask[:, w - mr :] = 0

        # 用 detections 挖掉前景（车辆/行人/骑行者等）
        if detections_tlbr is not None and len(detections_tlbr) > 0:
            dets = np.asarray(detections_tlbr, dtype=np.float32).copy()
            dets /= float(ds)
            dil = int(cfg.fg_dilate)
            for x1, y1, x2, y2 in dets:
                x1i = max(0, int(x1) - dil)
                y1i = max(0, int(y1) - dil)
                x2i = min(w, int(x2) + dil)
                y2i = min(h, int(y2) + dil)
                bg_mask[y1i:y2i, x1i:x2i] = 0

        # ============== (4) 动态点数：按 ROI 可用面积估计 maxCorners ==============
        valid_area = int((bg_mask > 0).sum())
        target = int(cfg.target_density_per_1e5 * (valid_area / 1e5))
        max_corners = int(np.clip(target, cfg.min_corners, cfg.max_corners))

        feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance,
            blockSize=cfg.block_size,
            useHarrisDetector=False,
            k=0.04,
        )

        # 提取角点（先多取一点，后续再网格均匀化）
        pts = cv2.goodFeaturesToTrack(gray_ds, mask=bg_mask, **feature_params)

        if pts is None or len(pts) < 12:
            if cfg.debug:
                print("[BusFrontGMC++] too few bg points -> fallback")
            # 更新 prev，避免卡死
            self._update_state(gray_ds, pts)
            return self._return_identity_or_smooth()

        # ============== (4) 网格均匀采样：抑制点聚集 ==============
        pts = self._grid_subsample(pts, h, w)

        # 第一帧：初始化
        if not self.initialized:
            self.prev_gray = gray_ds
            self.prev_pts = copy.copy(pts)
            self.smooth_H = np.eye(2, 3, dtype=np.float32)
            self.initialized = True
            return self.smooth_H.copy()

        # LK 跟踪
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_ds, self.prev_pts, None, **self.lk_params
        )
        if next_pts is None or status is None:
            if cfg.debug:
                print("[BusFrontGMC++] LK failed -> fallback")
            self._update_state(gray_ds, pts)
            return self._return_identity_or_smooth()

        status = status.reshape(-1).astype(bool)
        prev_good = self.prev_pts.reshape(-1, 2)[status]
        next_good = next_pts.reshape(-1, 2)[status]

        if len(prev_good) < 12:
            if cfg.debug:
                print("[BusFrontGMC++] too few tracked points -> fallback")
            self._update_state(gray_ds, pts)
            return self._return_identity_or_smooth()

        # 估计仿射（部分仿射：旋转/缩放/平移）
        H, inliers = cv2.estimateAffinePartial2D(
            prev_good,
            next_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.ransac_thresh,
            maxIters=cfg.max_iters,
            confidence=cfg.confidence,
            refineIters=cfg.refine_iters,
        )
        if H is None or inliers is None:
            if cfg.debug:
                print("[BusFrontGMC++] affine failed -> fallback")
            self._update_state(gray_ds, pts)
            return self._return_identity_or_smooth()

        inliers = inliers.reshape(-1).astype(bool)
        num_inliers = int(inliers.sum())
        if num_inliers < 8:
            if cfg.debug:
                print("[BusFrontGMC++] too few inliers -> fallback")
            self._update_state(gray_ds, pts)
            return self._return_identity_or_smooth()

        # ============== (1) 质量分数：inlier_ratio + median_reproj_err + translation ==============
        q, detail = self._quality_score(
            H=H,
            prev_pts=prev_good,
            next_pts=next_good,
            inliers=inliers,
            h0=h0,
            w0=w0,
            ds=ds,
        )

        if cfg.debug:
            print(f"[BusFrontGMC++] quality={q:.3f} detail={detail}")

        if q < cfg.min_quality:
            # 拒绝本帧 GMC：只更新状态，不更新 smooth_H
            if cfg.debug:
                print("[BusFrontGMC++] quality below threshold -> reject")
            self._update_state(gray_ds, pts)
            return self._return_identity_or_smooth()

        # 调整回原图尺度
        H_full = H.astype(np.float32).copy()
        H_full[0, 2] *= ds
        H_full[1, 2] *= ds

        # 平滑：可按质量分数自适应（质量高 -> alpha 大一点，更快响应；质量低 -> 更平滑）
        if self.smooth_H is None:
            self.smooth_H = H_full
        else:
            a = cfg.smoothing_alpha
            if cfg.adaptive_smoothing:
                # q∈[min_quality,1] 映射到 [alpha_min, alpha_max]
                qn = (q - cfg.min_quality) / max(1e-6, (1.0 - cfg.min_quality))
                a = cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * float(np.clip(qn, 0.0, 1.0))
            self.smooth_H = (a * H_full + (1.0 - a) * self.smooth_H).astype(np.float32)

        # 更新状态
        self._update_state(gray_ds, pts)
        return self.smooth_H.copy()

    # ======================== helper functions ========================

    def _update_state(self, gray_ds, pts):
        self.prev_gray = gray_ds
        # 如果 pts None，用上一帧的 prev_pts 也行，但通常会导致漂移；这里更安全是 None->保持原 prev_pts
        if pts is not None and len(pts) > 0:
            self.prev_pts = copy.copy(pts)
        # initialized 不变（已经初始化过就继续）

    def _build_road_roi_mask(self, h: int, w: int) -> np.ndarray:
        """
        (2) 路面 ROI：一个梯形区域
        """
        cfg = self.cfg
        mask = np.zeros((h, w), dtype=np.uint8)

        y_top = int(cfg.road_top_ratio * h)
        y_bot = h - 1

        # 顶边左右
        x_top_l = int(cfg.road_top_shrink * w)
        x_top_r = int(w - cfg.road_top_shrink * w)

        # 底边左右
        x_bot_l = int(cfg.road_bottom_shrink * w)
        x_bot_r = int(w - cfg.road_bottom_shrink * w)

        poly = np.array(
            [[x_top_l, y_top], [x_top_r, y_top], [x_bot_r, y_bot], [x_bot_l, y_bot]],
            dtype=np.int32
        )
        cv2.fillPoly(mask, [poly], 255)
        return mask

    def _grid_subsample(self, pts: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        (4) 网格均匀采样：把 pts 分配到网格中，每格最多保留 max_per_cell 个点
        说明：goodFeaturesToTrack 通常按强度排序，这里“先到先得”即可。
        """
        cfg = self.cfg
        pts2 = pts.reshape(-1, 2)

        gh, gw = cfg.grid_rows, cfg.grid_cols
        cell_h = max(1, h // gh)
        cell_w = max(1, w // gw)

        buckets = [[0] * (gh * gw)][0]  # 计数器
        keep = []

        for x, y in pts2:
            cx = int(x) // cell_w
            cy = int(y) // cell_h
            cx = min(gw - 1, max(0, cx))
            cy = min(gh - 1, max(0, cy))
            idx = cy * gw + cx
            if buckets[idx] < cfg.max_per_cell:
                keep.append([x, y])
                buckets[idx] += 1

        if len(keep) < 12:
            # 兜底：太少就不过滤
            return pts.astype(np.float32)
        return np.array(keep, dtype=np.float32).reshape(-1, 1, 2)

    def _quality_score(self, H, prev_pts, next_pts, inliers, h0, w0, ds):
        """
        (1) 可解释质量分数：
            - inlier_ratio 越大越好
            - median reprojection error 越小越好
            - translation 越小越好（但不是越小越绝对好，只是抖动补偿一般是小运动）
        返回：q∈(0,1]，以及可解释 detail dict
        """
        cfg = self.cfg

        n_all = max(1, len(prev_pts))
        n_in = int(inliers.sum())
        inlier_ratio = n_in / float(n_all)

        # 重投影误差（只在内点上）
        p = prev_pts[inliers]
        q = next_pts[inliers]
        # 预测：p' = A p + t
        A = H[:, :2]
        t = H[:, 2]
        pred = (p @ A.T) + t[None, :]
        err = np.linalg.norm(pred - q, axis=1)
        med_err = float(np.median(err)) if len(err) > 0 else 1e9

        # 平移幅度（映射回原图尺度）
        tx = float(H[0, 2]) * ds
        ty = float(H[1, 2]) * ds
        trans = float(np.hypot(tx, ty))
        trans_ref = cfg.trans_ref_ratio * float(min(h0, w0))

        # 三个子分数（都在 0~1）
        s_in = float(np.clip((inlier_ratio - 0.15) / (0.65 - 0.15), 0.0, 1.0))
        s_err = float(np.exp(-med_err / max(1e-6, cfg.err_ref_px)))
        s_tr = float(np.exp(-trans / max(1e-6, trans_ref)))

        # 组合（乘法更“短板效应”，更利于门控）
        quality = float(np.clip(s_in * s_err * s_tr, 0.0, 1.0))

        detail = dict(
            n_all=n_all,
            n_in=n_in,
            inlier_ratio=round(inlier_ratio, 4),
            med_reproj_err=round(med_err, 3),
            tx=round(tx, 2),
            ty=round(ty, 2),
            trans=round(trans, 2),
            s_in=round(s_in, 3),
            s_err=round(s_err, 3),
            s_tr=round(s_tr, 3),
        )
        return quality, detail

    def _return_identity_or_smooth(self) -> np.ndarray:
        if self.smooth_H is not None:
            return self.smooth_H.copy()
        self.smooth_H = np.eye(2, 3, dtype=np.float32)
        return self.smooth_H.copy()
