# tracker/gmc_busfront.py
"""
BusFront-GMC: Road-aware Camera Motion Compensation for BusFrontMOT.

核心思想:
- 只在“路面背景 + 非前景车辆”区域提特征点做匹配
- 用 RANSAC 拟合相机全局运动（单应/仿射）
- 通过 inlier ratio 估计 CMC 置信度，自适应减弱/关闭 CMC
- 在时间维度上对相机运动做平滑，避免 H 抖动

接口:
    gmc = BusFrontGMC(img_size=(H,W), downscale=2, ...)
    H = gmc.apply(cur_img, prev_img, dets=dets, tracks=tracks)
返回值:
    H: 3x3 单应矩阵 (numpy.float32)，用于将当前帧坐标 warp 回上一帧坐标系
"""

import cv2
import numpy as np


class BusFrontGMC:
    def __init__(
        self,
        img_size=None,
        downscale: int = 2,
        road_y_ratio: float = 0.45,
        max_corners: int = 1000,
        quality_level: float = 0.01,
        min_distance: int = 5,
        ransac_reproj_thr: float = 3.0,
        min_matches: int = 60,
        min_inlier_ratio: float = 0.4,
        alpha_smooth: float = 0.5,
        use_affine: bool = False,
    ):
        """
        img_size: (H, W) 原始图像尺寸，可选（没给则运行时自动从第一帧推断）
        downscale: 特征点检测时的下采样倍率 (2 表示 1/2 尺度)
        road_y_ratio: 假定地平线大致在 H * road_y_ratio 处，以下为路面 ROI
        max_corners: Shi-Tomasi 角点最大数量
        quality_level, min_distance: 角点检测参数
        ransac_reproj_thr: RANSAC 内点重投影误差阈值
        min_matches: 至少需要多少匹配点才尝试估计 H
        min_inlier_ratio: 低于该值则认为 H 不可靠，减弱/禁用 CMC
        alpha_smooth: 相机运动 EMA 平滑系数
        use_affine: True 则使用仿射估计，否则使用单应矩阵
        """
        self.img_size = img_size  # (H, W)
        self.downscale = downscale
        self.road_y_ratio = road_y_ratio

        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        self.ransac_reproj_thr = ransac_reproj_thr
        self.min_matches = min_matches
        self.min_inlier_ratio = min_inlier_ratio
        self.alpha_smooth = alpha_smooth
        self.use_affine = use_affine

        # 用于时间平滑的上一帧 H
        self.prev_H = None  # 3x3
        # 用于保存上一帧的灰度图 (下采样后)
        self.prev_gray_small = None

        # 初始化时先设为单位阵
        self.I = np.eye(3, dtype=np.float32)

    # ----------------- 主入口 ----------------- #
    def apply(self, cur_img, dets=None, tracks=None):
        """
        对当前帧估计一个相机运动补偿矩阵 H (3x3)，
        使得: p_{t-1} ≈ H * p_t   (齐次坐标)

        参数:
            cur_img: BGR (H, W, 3)，当前帧
            dets: (N, 5/6/7) 当前帧检测结果 (可选)，用于生成前景 mask
            tracks: 当前帧的跟踪框 (可选)

        返回:
            H_smooth: 3x3 float32 单应矩阵；如果无法估计，则返回 I
        """
        # 初始化图像尺寸
        if self.img_size is None:
            h, w = cur_img.shape[:2]
            self.img_size = (h, w)

        # 转灰度并下采样
        cur_gray_small = self._preprocess_gray(cur_img)

        # 第一帧没有 prev，不做 CMC，返回 I 并存缓存
        if self.prev_gray_small is None:
            self.prev_gray_small = cur_gray_small
            self.prev_H = self.I.copy()
            return self.I.copy()

        # 生成路面 ROI + 前景 mask
        mask_prev, mask_cur = self._build_background_masks(
            self.prev_gray_small, cur_gray_small, dets=dets, tracks=tracks
        )

        # 提取并匹配特征点
        pts_prev, pts_cur = self._detect_and_match(
            self.prev_gray_small, cur_gray_small,
            mask_prev, mask_cur
        )

        # 不足以估计 H，则直接使用历史 H 或 I
        if pts_prev.shape[0] < self.min_matches:
            H_raw = self.prev_H if self.prev_H is not None else self.I
            H_smooth = self._smooth_H(H_raw)
            # 更新缓存
            self.prev_gray_small = cur_gray_small
            self.prev_H = H_smooth
            return H_smooth

        # 估计 H (单应/仿射)
        H_raw, inlier_mask = self._estimate_motion(pts_prev, pts_cur)

        # 根据 inlier ratio 决定是否使用 H
        if H_raw is None or inlier_mask is None:
            H_use = self.prev_H if self.prev_H is not None else self.I
        else:
            inlier_ratio = float(inlier_mask.sum()) / float(len(inlier_mask))
            if inlier_ratio < self.min_inlier_ratio:
                # 置信度过低，退回历史 H
                H_use = self.prev_H if self.prev_H is not None else self.I
            else:
                H_use = H_raw

        # 对 H 做时间平滑
        H_smooth = self._smooth_H(H_use)

        # 更新缓存
        self.prev_gray_small = cur_gray_small
        self.prev_H = H_smooth

        return H_smooth

    # ----------------- 辅助函数 ----------------- #

    def _preprocess_gray(self, img_bgr):
        """转灰度 + 下采样."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            new_w = gray.shape[1] // self.downscale
            new_h = gray.shape[0] // self.downscale
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray

    def _build_background_masks(self, prev_gray, cur_gray, dets=None, tracks=None):
        """
        构造两个 mask:
            mask_prev, mask_cur: 1 表示背景可用区域，0 表示不使用（前景/天空等）

        简化实现:
            - 只取图像下部作为 road ROI
            - 根据 dets / tracks 在 road ROI 里涂黑前景框
        """
        h, w = prev_gray.shape[:2]
        # 初始全 0，然后填 1 的 road 区域
        mask_prev = np.zeros((h, w), dtype=np.uint8)
        mask_cur = np.zeros((h, w), dtype=np.uint8)

        # 路面 ROI：y > road_y_ratio * H
        y0 = int(h * self.road_y_ratio)
        mask_prev[y0:, :] = 255
        mask_cur[y0:, :] = 255

        # 前景遮挡：根据 dets / tracks 打 mask
        if dets is not None:
            self._erase_foreground(mask_prev, dets)
            self._erase_foreground(mask_cur, dets)
        if tracks is not None:
            self._erase_foreground(mask_prev, tracks)
            self._erase_foreground(mask_cur, tracks)

        return mask_prev, mask_cur

    def _erase_foreground(self, mask, bboxes, inflate: int = 4):
        """
        在 mask 上对前景 bbox 区域置 0.
        bboxes: 形如 (N, 4/5/6) 的数组或列表，支持 (x,y,w,h) 或 (x1,y1,x2,y2).

        为了简单，这里假设:
            - 若列数 >= 6，则前 4 列 (x,y,w,h) 或 (x1,y1,x2,y2) 手动调整即可。
        你可以根据自己 det.txt / track.txt 的格式微调这部分逻辑.
        """
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.size == 0:
            return

        # 简单假设: [x,y,w,h] (MOT 格式)
        if bboxes.shape[1] >= 6:
            x = bboxes[:, 2]
            y = bboxes[:, 3]
            w = bboxes[:, 4]
            h = bboxes[:, 5]
            x1 = (x / self.downscale).astype(int) - inflate
            y1 = (y / self.downscale).astype(int) - inflate
            x2 = ((x + w) / self.downscale).astype(int) + inflate
            y2 = ((y + h) / self.downscale).astype(int) + inflate
        elif bboxes.shape[1] >= 4:
            # 也可能是 [x,y,w,h] 直接传进来
            x = bboxes[:, 0]
            y = bboxes[:, 1]
            w = bboxes[:, 2]
            h = bboxes[:, 3]
            x1 = (x / self.downscale).astype(int) - inflate
            y1 = (y / self.downscale).astype(int) - inflate
            x2 = ((x + w) / self.downscale).astype(int) + inflate
            y2 = ((y + h) / self.downscale).astype(int) + inflate
        else:
            return

        h, w = mask.shape[:2]
        for i in range(len(x1)):
            xx1 = max(0, x1[i])
            yy1 = max(0, y1[i])
            xx2 = min(w - 1, x2[i])
            yy2 = min(h - 1, y2[i])
            if xx2 > xx1 and yy2 > yy1:
                mask[yy1:yy2, xx1:xx2] = 0

    def _detect_and_match(self, prev_gray, cur_gray, mask_prev, mask_cur):
        """
        在背景区域提角点并做匹配, 返回:
            pts_prev, pts_cur: (N, 2)
        简单版本: 使用 Shi-Tomasi + 光流跟踪 或 ORB 特征 + BFMatcher 二选一.
        这里先用 Shi-Tomasi + Lucas-Kanade 光流，比较稳也比较快.
        """
        # 1. 在 prev_gray 上检测角点
        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask_prev,
        )

        if pts_prev is None or len(pts_prev) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        # 2. 用 LK optical flow 跟踪到 cur_gray
        pts_cur, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            cur_gray,
            pts_prev,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        status = status.reshape(-1)
        pts_prev = pts_prev.reshape(-1, 2)
        pts_cur = pts_cur.reshape(-1, 2)

        # 3. 只保留成功跟踪 + 当前帧落在 mask_cur 背景区域的点
        h, w = cur_gray.shape[:2]
        good_prev = []
        good_cur = []

        for p0, p1, st in zip(pts_prev, pts_cur, status):
            if st != 1:
                continue
            x1, y1 = p1
            ix = int(round(x1))
            iy = int(round(y1))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            if mask_cur[iy, ix] == 0:
                continue
            good_prev.append(p0)
            good_cur.append(p1)

        if len(good_prev) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        pts_prev = np.asarray(good_prev, dtype=np.float32)
        pts_cur = np.asarray(good_cur, dtype=np.float32)
        return pts_prev, pts_cur

    def _estimate_motion(self, pts_prev, pts_cur):
        """
        利用 RANSAC 估计相机运动:
            - 若 use_affine=True: 估计仿射 (2x3)
            - 否则: 估计单应 (3x3)
        返回:
            H_raw: 3x3 float32
            inlier_mask: (N,) uint8 [0/1]
        """
        if pts_prev.shape[0] < 4:
            return None, None

        if self.use_affine:
            # 仿射变换: 2x3 => 转成 3x3
            M, inlier_mask = cv2.estimateAffine2D(
                pts_cur, pts_prev,  # 当前 -> 上一帧
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_thr,
                maxIters=2000,
                confidence=0.99,
            )
            if M is None:
                return None, None
            H_raw = np.eye(3, dtype=np.float32)
            H_raw[:2, :] = M
        else:
            # 单应矩阵: 3x3
            H_raw, inlier_mask = cv2.findHomography(
                pts_cur, pts_prev,  # 当前 -> 上一帧
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_thr,
                maxIters=2000,
                confidence=0.99,
            )
            if H_raw is None:
                return None, None
            H_raw = H_raw.astype(np.float32)

        inlier_mask = inlier_mask.reshape(-1).astype(np.uint8)
        return H_raw, inlier_mask

    def _smooth_H(self, H_new):
        """
        对 H 做简单的 EMA 平滑:
            H_smooth = alpha * H_new + (1-alpha) * H_prev
        注意:
            严格来讲单应矩阵不适合线性插值，但在小运动假设下近似可行。
            如有需要，可以后续扩展为对平移/旋转参数的分解再平滑.
        """
        if self.prev_H is None:
            return H_new.copy()
        return (self.alpha_smooth * H_new +
                (1.0 - self.alpha_smooth) * self.prev_H).astype(np.float32)
