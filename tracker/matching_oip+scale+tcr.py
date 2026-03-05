import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)

    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

# ===================== BusFront: OIP (soft) + TCR (soft) =====================
def _tlbr_from_det_row(det_row):
    # det_row: [x1,y1,x2,y2,score,cls] or [x1,y1,x2,y2,score]
    x1, y1, x2, y2 = det_row[:4]
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def _tlbr_from_track(track):
    # Robustly fetch tlbr from STrack-like objects
    if hasattr(track, "tlbr"):
        tlbr = track.tlbr
        return np.asarray(tlbr, dtype=np.float32)
    if hasattr(track, "tlwh"):
        x, y, w, h = track.tlwh
        return np.asarray([x, y, x + w, y + h], dtype=np.float32)
    if hasattr(track, "_tlwh"):
        x, y, w, h = track._tlwh
        return np.asarray([x, y, x + w, y + h], dtype=np.float32)
    # fallback: assume first 4 entries are tlbr
    return np.asarray(track[:4], dtype=np.float32)

def _bbox_iou_tlbr(a, b):
    # a: (4,), b:(4,)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)

def _pairwise_iou_tlbr(boxes):
    n = len(boxes)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            v = _bbox_iou_tlbr(boxes[i], boxes[j])
            out[i, j] = v
            out[j, i] = v
    return out

def _occlusion_prob_from_dets(dets_tlbr, iou_thr=0.3):
    # dets_tlbr: (M,4)
    m = len(dets_tlbr)
    if m <= 1:
        return np.zeros((m,), dtype=np.float32)
    ious = _pairwise_iou_tlbr(dets_tlbr)
    ious[np.eye(m, dtype=bool)] = 0.0
    max_iou = ious.max(axis=0)  # per-detection
    # Soft occlusion probability: 0 below iou_thr, grows to 1
    occ = np.clip((max_iou - float(iou_thr)) / max(1e-6, 1.0 - float(iou_thr)), 0.0, 1.0)
    return occ.astype(np.float32)

def _center_from_tlbr(tlbr):
    x1, y1, x2, y2 = tlbr
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

def apply_oip_tcr_soft(
    cost_matrix,
    tracks,
    detections,
    alpha=0.15,          # OIP weight
    occ_iou_thr=0.3,     # OIP threshold
    beta=0.10,           # scale penalty weight
    scale_tau=0.7,       # tolerate ratio in [tau, 1/tau]
    gamma=0.08,          # TCR reward weight (subtract from cost)
    sigma=80.0,          # pixels, controls how fast reward decays
):
    '''
    Soft, differentiable-like post-adjustment on the association cost matrix.

    Inputs:
      cost_matrix: (N_tracks, M_dets), usually 1-IoU fused with appearance.
      tracks: list[STrack]
      detections: list[STrack] OR ndarray(M,6) where [:4]=tlbr

    Returns:
      adjusted_cost: same shape as cost_matrix.
    '''
    C = np.asarray(cost_matrix, dtype=np.float32)
    if C.size == 0:
        return C

    n, m = C.shape

    # ---- build det tlbr list ----
    det_tlbr = []
    if isinstance(detections, np.ndarray):
        for j in range(min(m, len(detections))):
            det_tlbr.append(_tlbr_from_det_row(detections[j]))
    else:
        # list of detection STrack (as used in BoT-SORT)
        for j in range(min(m, len(detections))):
            det_tlbr.append(_tlbr_from_track(detections[j]))
    det_tlbr = np.stack(det_tlbr, axis=0).astype(np.float32) if len(det_tlbr) else np.zeros((m, 4), np.float32)

    # ---- OIP: penalize matching to heavily-overlapping detections (likely occluded / duplicate) ----
    occ = _occlusion_prob_from_dets(det_tlbr, iou_thr=occ_iou_thr)  # (m,)
    if alpha and m:
        C = C + float(alpha) * occ[None, :]  # broadcast over tracks

    # ---- Scale soft penalty: discourage extreme area ratio mismatches ----
    if beta and n and m:
        det_wh = np.clip(det_tlbr[:, 2:] - det_tlbr[:, :2], 1.0, None)
        det_area = det_wh[:, 0] * det_wh[:, 1]  # (m,)
        for i in range(n):
            t_tlbr = _tlbr_from_track(tracks[i])
            t_wh = np.clip(t_tlbr[2:] - t_tlbr[:2], 1.0, None)
            t_area = float(t_wh[0] * t_wh[1])
            ratio = det_area / max(t_area, 1e-6)  # (m,)
            # penalty outside [tau, 1/tau]
            tau = float(scale_tau)
            lo = tau
            hi = 1.0 / max(tau, 1e-6)
            bad = (ratio < lo) | (ratio > hi)
            # soft magnitude based on log-ratio distance
            mag = np.abs(np.log(np.clip(ratio, 1e-6, 1e6)))  # (m,)
            C[i, bad] = C[i, bad] + float(beta) * mag[bad]

    # ---- TCR: reward temporal continuity (motion-consistent matches) ----
    if gamma and n and m:
        for i in range(n):
            trk = tracks[i]
            # need at least 2 historical observations for velocity
            if not hasattr(trk, "history") or len(getattr(trk, "history")) < 2:
                continue
            # history is expected to store tlwh or tlbr; we handle both
            h1 = trk.history[-1]
            h0 = trk.history[-2]
            if len(h1) == 4:
                # assume tlwh
                x, y, w, h = h1
                tlbr1 = np.array([x, y, x + w, y + h], dtype=np.float32)
            else:
                tlbr1 = np.asarray(h1[:4], dtype=np.float32)
            if len(h0) == 4:
                x, y, w, h = h0
                tlbr0 = np.array([x, y, x + w, y + h], dtype=np.float32)
            else:
                tlbr0 = np.asarray(h0[:4], dtype=np.float32)

            c1 = _center_from_tlbr(tlbr1)
            c0 = _center_from_tlbr(tlbr0)
            v = c1 - c0
            pred = c1 + v  # constant velocity prediction

            det_centers = (det_tlbr[:, :2] + det_tlbr[:, 2:]) * 0.5  # (m,2)
            dev = np.linalg.norm(det_centers - pred[None, :], axis=1)  # (m,)
            reward = float(gamma) * np.exp(-dev / max(float(sigma), 1e-6))
            # subtract reward (smaller cost -> preferred)
            C[i, :] = C[i, :] - reward.astype(np.float32)

    return C
