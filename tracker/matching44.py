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

# ===========================
# Soft OIP + Soft TCR helpers
# ===========================

def _sigmoid(x: np.ndarray, k: float = 10.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))

def apply_soft_oip_tcr(cost_matrix: np.ndarray,
                       tracks,
                       detections,
                       img_wh=(1920, 1080),
                       frame_id: int = 0,
                       alpha: float = 0.25,
                       beta: float = 0.20,
                       trig_n: int = 6,
                       trig_iou: float = 0.35,
                       small_area: float = 0.010,
                       tcr_max_dt: int = 2) -> np.ndarray:
    """Apply *soft* occlusion-informed penalty (OIP) and temporal continuity reward (TCR).

    - Does NOT change the assignment algorithm; only adjusts the existing cost matrix.
    - Safe fallback: if inputs are empty or shapes mismatch, return original cost_matrix.

    Args:
        cost_matrix: (N,M) cost (lower is better).
        tracks: list of STrack (or compatible) with .tlbr and .time_since_update/.tracklet_len.
        detections: list of STrack (or compatible) with .tlbr and .score (optional).
        img_wh: (W,H) used for relative area.
        alpha: OIP penalty weight.
        beta:  TCR reward weight.
        trig_n, trig_iou, small_area: adaptive trigger & small-object emphasis.
        tcr_max_dt: only reward tracks updated within this delta frames.

    Returns:
        adjusted cost matrix (same shape).
    """
    C = cost_matrix
    try:
        if C is None:
            return C
        C = np.asarray(C, dtype=float)
        if C.ndim != 2:
            return cost_matrix
        n, m = C.shape
        if n == 0 or m == 0:
            return cost_matrix
        if detections is None or len(detections) != m:
            return cost_matrix
        if tracks is None or len(tracks) != n:
            return cost_matrix

        W, H = float(img_wh[0]), float(img_wh[1])
        eps = 1e-6

        det_tlbr = np.asarray([d.tlbr for d in detections], dtype=float)
        # -------- Adaptive trigger: only activate when occlusion risk is high --------
        if m < trig_n:
            return cost_matrix

        dd_iou = ious(det_tlbr, det_tlbr)  # (m,m)
        np.fill_diagonal(dd_iou, 0.0)
        max_iou = dd_iou.max(axis=1)  # (m,)
        mean_max_iou = float(np.mean(max_iou)) if m > 0 else 0.0

        # ratio of small detections
        det_wh = np.clip(det_tlbr[:, 2:4] - det_tlbr[:, 0:2], a_min=0.0, a_max=None)
        det_area_rel = (det_wh[:, 0] * det_wh[:, 1]) / max(W * H, eps)
        small_ratio = float(np.mean(det_area_rel < small_area))

        if not (mean_max_iou >= (0.8 * trig_iou) or small_ratio >= 0.5):
            return cost_matrix

        # ---------------- OIP: penalize ambiguous / occluded detections ----------------
        # occlusion likelihood increases with overlap to other detections and small size
        occ_overlap = _sigmoid(max_iou - trig_iou, k=12.0)  # (m,)
        small_factor = np.clip((small_area - det_area_rel) / max(small_area, eps), 0.0, 1.0)
        occ_prob = occ_overlap * (0.5 + 0.5 * small_factor)  # (m,)

        C = C + float(alpha) * occ_prob[None, :]

        # ---------------- TCR: reward temporal continuity (IoU + scale consistency) ----------------
        trk_tlbr = np.asarray([t.tlbr for t in tracks], dtype=float)  # (n,4)
        td_iou = ious(trk_tlbr, det_tlbr)  # (n,m)

        trk_wh = np.clip(trk_tlbr[:, 2:4] - trk_tlbr[:, 0:2], a_min=0.0, a_max=None)
        trk_area = trk_wh[:, 0] * trk_wh[:, 1] + eps
        det_area = det_wh[:, 0] * det_wh[:, 1] + eps
        area_ratio = (trk_area[:, None] / det_area[None, :])
        scale_sim = np.exp(-np.abs(np.log(area_ratio)) / 0.7)  # (n,m) in (0,1]

        # only apply to recently-updated tracks (stable history)
        tsu = np.asarray([getattr(t, 'time_since_update', 0) for t in tracks], dtype=int)
        tlen = np.asarray([getattr(t, 'tracklet_len', 0) for t in tracks], dtype=int)
        stable = (tsu <= int(tcr_max_dt)) & (tlen >= 2)
        if np.any(stable):
            reward = float(beta) * (td_iou * scale_sim)
            C[stable, :] = C[stable, :] - reward[stable, :]

        return C
    except Exception:
        return cost_matrix
