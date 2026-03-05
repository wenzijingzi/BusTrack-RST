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



def apply_oip_scale_gating(*args, **kwargs):
    """
    Backward-compatible OIP + Scale gating.

    Supports BOTH calling styles:
    1) New style:
        new_cost = apply_oip_scale_gating(cost_matrix, tracks, detections, iou_matrix=..., ...)
    2) Legacy style (used in some patched BoT-SORT forks):
        new_cost = apply_oip_scale_gating(tracks, detections, cost_matrix,
                                          scale_log_thr=..., occ_iou_thr=..., occ_big_ratio=..., set_to=...)

    Returns:
        gated_cost_matrix (same shape as input cost_matrix)
    """
    import numpy as np

    # -------- Parse positional args --------
    if len(args) < 3:
        raise TypeError("apply_oip_scale_gating expects at least 3 positional arguments")

    # Detect which style:
    # - If first arg is an ndarray -> new style
    # - Else assume legacy: (tracks, dets, cost)
    if isinstance(args[0], np.ndarray):
        cost_matrix, tracks, detections = args[0], args[1], args[2]
        iou_matrix = kwargs.pop("iou_matrix", None)
    else:
        tracks, detections, cost_matrix = args[0], args[1], args[2]
        iou_matrix = kwargs.pop("iou_matrix", None)

    # -------- Map legacy parameter names --------
    # legacy: scale_log_thr = log(ratio_thr)
    scale_log_thr = kwargs.pop("scale_log_thr", None)
    if scale_log_thr is not None:
        try:
            scale_ratio_thr = float(np.exp(scale_log_thr))
        except Exception:
            scale_ratio_thr = 3.0
    else:
        scale_ratio_thr = float(kwargs.pop("scale_ratio_thr", 3.0))

    occ_iou_thr = float(kwargs.pop("occ_iou_thr", kwargs.pop("oip_occ_iou_thr", 0.15)))
    occ_big_ratio = float(kwargs.pop("occ_big_ratio", kwargs.pop("oip_big_ratio", 1.8)))

    # hard gate value
    set_to = kwargs.pop("set_to", None)
    hard_set_to = kwargs.pop("hard_set_to", None)
    if set_to is None:
        set_to = hard_set_to
    if set_to is None:
        set_to = 1.0

    # soft weights
    alpha_oip = float(kwargs.pop("alpha_oip", kwargs.pop("oip_alpha", 0.25)))
    alpha_scale = float(kwargs.pop("alpha_scale", kwargs.pop("scale_alpha", 0.15)))

    enable_conditional = bool(kwargs.pop("enable_conditional", kwargs.pop("oip_conditional", True)))
    hard_gate = bool(kwargs.pop("hard_gate", kwargs.pop("oip_hard_gate", False)))
    iou_floor = float(kwargs.pop("iou_floor", kwargs.pop("oip_iou_floor", 0.01)))

    # ignore unknown kwargs safely (important for different forks)
    # kwargs is intentionally not used afterwards.

    return _apply_oip_scale_gating_core(
        cost_matrix=cost_matrix,
        tracks=tracks,
        detections=detections,
        iou_matrix=iou_matrix,
        alpha_oip=alpha_oip,
        alpha_scale=alpha_scale,
        enable_conditional=enable_conditional,
        occ_iou_thr=occ_iou_thr,
        occ_big_ratio=occ_big_ratio,
        hard_gate=hard_gate,
        set_to=set_to,
        scale_ratio_thr=scale_ratio_thr,
        iou_floor=iou_floor,
    )


def _apply_oip_scale_gating_core(
    cost_matrix,
    tracks,
    detections,
    iou_matrix=None,
    *,
    alpha_oip=0.25,
    alpha_scale=0.15,
    enable_conditional=True,
    occ_iou_thr=0.15,
    occ_big_ratio=1.8,
    hard_gate=False,
    set_to=1.0,
    scale_ratio_thr=3.0,
    iou_floor=0.01,
):
    """
    Core implementation (soft + conditional trigger; optional hard gate).
    """
    import numpy as np

    if cost_matrix is None or getattr(cost_matrix, "size", 0) == 0:
        return cost_matrix
    new_cost = cost_matrix.astype(np.float32, copy=True)

    N, M = new_cost.shape

    # ---- IoU matrix ----
    if iou_matrix is None:
        # compute from tlbr
        atlbrs = np.asarray([t.tlbr for t in tracks], dtype=np.float32) if len(tracks) else np.zeros((0, 4), np.float32)
        btlbrs = np.asarray([d.tlbr for d in detections], dtype=np.float32) if len(detections) else np.zeros((0, 4), np.float32)
        if len(atlbrs) == 0 or len(btlbrs) == 0:
            return new_cost
        tl = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
        br = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])
        wh = np.clip(br - tl, 0, None)
        inter = wh[..., 0] * wh[..., 1]
        area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
        area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        iou_matrix = inter / np.clip(union, 1e-6, None)
    else:
        iou_matrix = iou_matrix.astype(np.float32, copy=False)

    # ---- areas for scale ratio ----
    def _area(obj):
        if hasattr(obj, "tlwh"):
            w, h = float(obj.tlwh[2]), float(obj.tlwh[3])
            return max(w * h, 1e-6)
        if hasattr(obj, "tlbr"):
            x1, y1, x2, y2 = obj.tlbr
            return max(float(x2 - x1) * float(y2 - y1), 1e-6)
        return 1.0

    t_area = np.asarray([_area(t) for t in tracks], dtype=np.float32)[:, None]  # (N,1)
    d_area = np.asarray([_area(d) for d in detections], dtype=np.float32)[None, :]  # (1,M)

    ratio = np.maximum(t_area / np.clip(d_area, 1e-6, None), d_area / np.clip(t_area, 1e-6, None))  # (N,M)
    scale_pen = np.log(ratio + 1e-6) / np.log(scale_ratio_thr + 1e-6)
    scale_pen = np.clip(scale_pen, 0.0, 1.0).astype(np.float32)

    # ---- OIP penalty ----
    oip_pen = np.zeros((N, M), dtype=np.float32)

    if enable_conditional:
        overlap_counts = (iou_matrix >= occ_iou_thr).sum(axis=0)  # (M,)
        occ_det_mask = overlap_counts >= 2
    else:
        occ_det_mask = np.ones((M,), dtype=bool)

    if occ_det_mask.any():
        best_track_for_det = np.argmax(iou_matrix, axis=0)
        for j in np.where(occ_det_mask)[0]:
            i_best = int(best_track_for_det[j])
            iou_best = float(iou_matrix[i_best, j])
            denom = max(iou_best, 1e-6)
            rel = np.clip(iou_matrix[:, j] / denom, 0.0, 1.0)
            oip_pen[:, j] = rel
            oip_pen[i_best, j] = 0.0

            # extra protection for BusFront small targets:
            # if a big track competes for a small det, penalize more (helps "ID stolen" during occlusion)
            big_vs_small = (t_area[:, 0] / np.clip(d_area[0, j], 1e-6, None)) > occ_big_ratio
            oip_pen[big_vs_small, j] = np.maximum(oip_pen[big_vs_small, j], 1.0)

    if enable_conditional:
        oip_pen[:, ~occ_det_mask] = 0.0
        # down-weight scale penalty when IoU already good
        scale_pen = scale_pen * (1.0 - np.clip(iou_matrix, 0.0, 1.0))

    # soft fusion
    new_cost = new_cost + alpha_oip * np.clip(oip_pen, 0.0, 1.0) + alpha_scale * scale_pen

    # optional hard gate
    if hard_gate:
        bad = np.zeros((N, M), dtype=bool)
        if enable_conditional:
            bad[:, occ_det_mask] |= (iou_matrix[:, occ_det_mask] < iou_floor)
        else:
            bad |= (iou_matrix < iou_floor)
        bad |= (ratio > scale_ratio_thr)
        if bad.any():
            new_cost[bad] = float(set_to)

    return new_cost

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


# ================== BusFront: AAR helpers ==================

def aar_build_cost(tracks, detections, kf=None, lambda_emb=0.3, use_fuse_score=False):
    # Backward-compat: allow caller to pass fuse_score=... as an alias of use_fuse_score
    if fuse_score is not None:
        use_fuse_score = bool(fuse_score)
    """Build a relaxed association cost for AAR (Adaptive Association Relaxation).

    The default BoT-SORT association may become too strict under heavy occlusion/small objects.
    This helper down-weights appearance and emphasizes IoU+motion (Kalman).
    Cost is in [0, +inf), lower is better.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    # IoU cost (1 - IoU)
    d_iou = iou_distance(tracks, detections)

    # Appearance cost (cosine distance) if features exist
    try:
        d_emb = embedding_distance(tracks, detections)
    except Exception:
        d_emb = None

    if d_emb is None or not np.isfinite(d_emb).any():
        d = d_iou
    else:
        lambda_emb = float(np.clip(lambda_emb, 0.0, 1.0))
        d = (1.0 - lambda_emb) * d_iou + lambda_emb * d_emb

    if kf is not None:
        # motion gating/fusion
        d = fuse_motion(kf, d, tracks, detections)

    if use_fuse_score:
        # encourage high-score detections
        d = fuse_score(d, detections)

    return d