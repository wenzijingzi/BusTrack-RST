# tracker/gmc.py
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

from numpy import int_, float_

from .gmc_busfront_v2 import BusFrontGMC, BusFrontGMCConfig


class GMC:
    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        """
        method:
            'none'          : 不做 CMC，返回单位仿射
            'sparseOptFlow' : 原版稀疏 LK 光流 GMC
            'ecc'           : ECC 算法
            'orb' / 'sift'  : 特征点 + 描述子匹配
            'file'          : 从预计算 GMC 文件读取（MOT17 用）
            'busfront'      : BusFront-GMC（为 BusFrontMOT 场景定制）
        """
        super(GMC, self).__init__()

        self.method = method
        self.downscale = max(1, int(downscale))

        # BusFront-GMC 实例占位
        self.busfront = None

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(
                nOctaveLayers=3,
                contrastThreshold=0.02,
                edgeThreshold=20
            )
            self.extractor = cv2.SIFT_create(
                nOctaveLayers=3,
                contrastThreshold=0.02,
                edgeThreshold=20
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,
                termination_eps,
            )

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(
                maxCorners=1000,
                qualityLevel=0.01,
                minDistance=1,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04,
            )
            # self.gmc_file = open('GMC_results.txt', 'w')

        elif self.method == 'file' or self.method == 'files':
            # 读取预计算 GMC（原本用于 MOTChallenge）
            seqName = verbose[0]
            ablation = verbose[1]
            if ablation:
                filePath = r'tracker/GMC_files/MOT17_ablation'
            else:
                filePath = r'tracker/GMC_files/MOTChallenge'

            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]

            self.gmcFile = open(filePath + "/GMC-" + seqName + ".txt", 'r')

            if self.gmcFile is None:
                raise ValueError(
                    "Error: Unable to open GMC file in directory:" + filePath
                )

        elif self.method == 'busfront':
            """
            BusFront-GMC: 基于背景区域 + 前景掩膜 + 稀疏 LK 光流的
            相机运动补偿模块，为 BusFrontMOT/公交前视场景定制。
            """
            cfg = BusFrontGMCConfig(
                downscale=self.downscale,
                max_corners=800,
                quality_level=0.01,
                min_distance=4,
                block_size=5,
                fg_dilate=4,
                smoothing_alpha=0.25,
                debug=False,
            )
            self.busfront = BusFrontGMC(cfg)

        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'

        else:
            raise ValueError("Error: Unknown CMC method:" + method)

        # 以下变量只在 orb / sift / ecc / sparseOptFlow 模式下使用
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    # ----------------------------------------------------------- #
    # 主入口
    # ----------------------------------------------------------- #
    def apply(self, raw_frame, detections=None):
        """
        raw_frame: 当前帧 BGR 图像
        detections: 当前帧检测框（通常是 [N, >=4]，前4维为 tlbr 或 tlwh）
        返回: 2x3 仿射矩阵 H (float32)
        """
        if self.method == 'orb' or self.method == 'sift':
            return self.applyFeaures(raw_frame, detections)

        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)

        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)

        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)

        elif self.method == 'busfront':
            # BusFront-GMC: 内部接受 tlbr 检测框（前4维），返回 2x3 H
            # 这里简单把 detections 原样传入，由 BusFrontGMC 自己做解析
            if self.busfront is None:
                return np.eye(2, 3, dtype=np.float32)
            H2 = self.busfront.apply(raw_frame, detections_tlbr=detections)
            if H2 is None:
                return np.eye(2, 3, dtype=np.float32)
            if H2.shape == (2, 3):
                return H2.astype(np.float32)
            elif H2.shape == (3, 3):
                return H2[:2, :].astype(np.float32)
            else:
                return np.eye(2, 3, dtype=np.float32)

        elif self.method == 'none':
            return np.eye(2, 3, dtype=np.float32)

        else:
            return np.eye(2, 3, dtype=np.float32)

    # ----------------------------------------------------------- #
    # ECC
    # ----------------------------------------------------------- #
    def applyEcc(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(
                frame, (width // self.downscale, height // self.downscale)
            )
            width = width // self.downscale
            height = height // self.downscale

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Run the ECC algorithm. The results are stored in warp_matrix.
        try:
            (cc, H) = cv2.findTransformECC(
                self.prevFrame,
                frame,
                H,
                self.warp_mode,
                self.criteria,
                None,
                1,
            )
        except Exception as e:
            print('Warning: findTransformECC failed, set warp as identity:', e)
            H = np.eye(2, 3, dtype=np.float32)

        # 更新 prevFrame
        self.prevFrame = frame.copy()

        return H

    # ----------------------------------------------------------- #
    # 特征点 (ORB / SIFT)
    # ----------------------------------------------------------- #
    def applyFeaures(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            frame = cv2.resize(
                frame, (width // self.downscale, height // self.downscale)
            )
            width = width // self.downscale
            height = height // self.downscale

        # find the keypoints
        mask = np.zeros_like(frame)
        mask[int(0.02 * height): int(0.98 * height),
             int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                # 假定 det[:4] 是 tlbr（x1,y1,x2,y2）
                tlbr = (det[:4] / self.downscale).astype(int_)
                x1, y1, x2, y2 = tlbr
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))
                mask[y1:y2, x1:x2] = 0

        keypoints = self.detector.detect(frame, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (
                    prevKeyPointLocation[0] - currKeyPointLocation[0],
                    prevKeyPointLocation[1] - currKeyPointLocation[1],
                )

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                   (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        if len(spatialDistances) == 0:
            # 没有有效匹配
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        spatialDistances = np.array(spatialDistances)
        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0) + 1e-6

        inliers = (np.abs(spatialDistances - meanSpatialDistances)
                   < 2.5 * stdSpatialDistances)

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliers[i, 0] and inliers[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Debug 可视化（关闭）
        if 0:
            matches_img = np.hstack((self.prevFrame, frame))
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.prevFrame, 1)
            for m in goodMatches:
                prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))

                matches_img = cv2.line(matches_img, prev_pt, curr_pt, color, 1, cv2.LINE_AA)
                matches_img = cv2.circle(matches_img, prev_pt, 2, color, -1)
                matches_img = cv2.circle(matches_img, curr_pt, 2, color, -1)

            plt.figure()
            plt.imshow(matches_img)
            plt.show()

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(currPoints, 0)):
            H, inliers = cv2.estimateAffinePartial2D(
                prevPoints, currPoints, cv2.RANSAC
            )

            # Handle downscale
            if self.downscale > 1.0 and H is not None:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        if H is None:
            H = np.eye(2, 3, dtype=np.float32)

        return H

    # ----------------------------------------------------------- #
    # Sparse Optical Flow
    # ----------------------------------------------------------- #
    def applySparseOptFlow(self, raw_frame, detections=None):

        t0 = time.time()

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image
        if self.downscale > 1.0:
            frame = cv2.resize(
                frame, (width // self.downscale, height // self.downscale)
            )

        # find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(
            self.prevFrame, frame, self.prevKeyPoints, None
        )

        # leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(currPoints, 0)):
            H, inliers = cv2.estimateAffinePartial2D(
                prevPoints, currPoints, cv2.RANSAC
            )

            # Handle downscale
            if self.downscale > 1.0 and H is not None:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        t1 = time.time()
        # 如需统计耗时，可在此写日志

        if H is None:
            H = np.eye(2, 3, dtype=np.float32)

        return H

    # ----------------------------------------------------------- #
    # 从文件读取预计算 GMC
    # ----------------------------------------------------------- #
    def applyFile(self, raw_frame, detections=None):
        line = self.gmcFile.readline()
        tokens = line.split("\t")
        H = np.eye(2, 3, dtype=float_)
        H[0, 0] = float(tokens[1])
        H[0, 1] = float(tokens[2])
        H[0, 2] = float(tokens[3])
        H[1, 0] = float(tokens[4])
        H[1, 1] = float(tokens[5])
        H[1, 2] = float(tokens[6])

        return H
