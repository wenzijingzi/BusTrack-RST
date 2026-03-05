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



def apply_oip_scale_gating(tracks, detections, cost_matrix,
                           scale_log_thr=np.log(2.0),
                           occ_iou_thr=0.20,
                           occ_big_ratio=1.80,
                           set_to=1.0):
    """Apply OIP (Occlusion Interaction Penalty) + scale gating on association cost.

    This is a *training-free* post-cost gating designed for moving on-board bus cameras with:
    - large scale variation (far small vs near large)
    - frequent partial occlusion, causing ID transfer from occluded small target to occluder.

    Args:
        tracks: list[STrack]
        detections: list[STrack]
        cost_matrix: (N,M) cost, smaller is better (BoT-SORT uses IoU/appearance fused costs)
        scale_log_thr: threshold on |log(area_t / area_d)|
        occ_iou_thr: track is considered occluded if it overlaps other tracks above this IoU
        occ_big_ratio: if occluded track matches a detection whose area is >= ratio * track area, penalize
        set_to: value to set for invalid pairs (1.0 works for [0,1] costs)

    Returns:
        gated_cost: (N,M) numpy array
    """
    if cost_matrix is None:
        return cost_matrix
    if len(tracks) == 0 or len(detections) == 0:
        return cost_matrix

    gated = cost_matrix.copy()

    # --- precompute track areas and occlusion score ---
    t_tlbr = np.asarray([t.tlbr for t in tracks], dtype=np.float32)  # (N,4)
    t_w = np.clip(t_tlbr[:, 2] - t_tlbr[:, 0], 1e-6, None)
    t_h = np.clip(t_tlbr[:, 3] - t_tlbr[:, 1], 1e-6, None)
    t_area = t_w * t_h

    d_tlbr = np.asarray([d.tlbr for d in detections], dtype=np.float32)  # (M,4)
    d_w = np.clip(d_tlbr[:, 2] - d_tlbr[:, 0], 1e-6, None)
    d_h = np.clip(d_tlbr[:, 3] - d_tlbr[:, 1], 1e-6, None)
    d_area = d_w * d_h

    # pairwise IoU among tracks for occlusion estimation
    # reuse existing ious() if present; otherwise compute here
    try:
        iou_tt = ious(t_tlbr, t_tlbr)  # (N,N)
    except Exception:
        iou_tt = _ious_tlbr(t_tlbr, t_tlbr)

    np.fill_diagonal(iou_tt, 0.0)
    occ_score = iou_tt.max(axis=1)  # (N,)

    # --- 1) scale gating: block abnormal scale jump ---
    # condition: |log(area_t/area_d)| > thr
    log_ratio = np.abs(np.log((t_area[:, None] + 1e-6) / (d_area[None, :] + 1e-6)))
    bad_scale = log_ratio > float(scale_log_thr)
    gated[bad_scale] = set_to

    # --- 2) OIP: for occluded tracks, block matches to much larger detections ---
    # if track is occluded (high overlap with other tracks), it should not "jump" to occluder
    occ_mask = occ_score >= float(occ_iou_thr)
    if occ_mask.any():
        big_jump = (d_area[None, :] / (t_area[:, None] + 1e-6)) >= float(occ_big_ratio)
        bad = occ_mask[:, None] & big_jump
        gated[bad] = set_to

    return gated


def _ious_tlbr(atlbrs, btlbrs):
    """Fallback IoU computation for tlbr arrays (N,4) and (M,4)."""
    N = len(atlbrs)
    M = len(btlbrs)
    out = np.zeros((N, M), dtype=np.float32)
    if N == 0 or M == 0:
        return out
    a = np.asarray(atlbrs, dtype=np.float32)
    b = np.asarray(btlbrs, dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1, bx2, by2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    out = inter / np.clip(union, 1e-6, None)
    return out

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