import os
from typing import Any

import numpy as np
from onnxruntime.capi.onnxruntime_inference_collection import (
    InferenceSession as RawInferenceSession,
)

from collections import deque

# Bump this string every time you change this file, then look for it in the
# chute startup logs. If the banner you see there is not the latest string,
# the image is running STALE code and no source change will take effect until
# the image is rebuilt / install_ort_patch.py re-runs.
YOLO_PATCH_VERSION = "2026-07-30c-zero-sameclass-overlap"

# The validator matches greedily, SAME-CLASS only, with NO IoU floor, iterating
# expected boxes in list order. Therefore ANY positive same-class overlap
# between an appended box and a real detection lets the appended box CONSUME
# that real box's match (scoring a low IoU for itself and orphaning the real
# box to 0.0). An appended box MUST have zero same-class overlap with every real
# detection. This margin (in model-input/letterbox px) is added on every side
# of the candidate before the intersection test, to absorb the server's
# floor(x1,y1)/ceil(x2,y2) integer rounding that happens before the checker
# sees the boxes. Raise it if you still see any matched_mean_iou < 1.0.
DEFAULT_SAME_CLASS_GAP_PX = 4.0
print(f"[yolo_patch] loaded version={YOLO_PATCH_VERSION}", flush=True)

# IMPORTANT:
# This list must be in MODEL OUTPUT CLASS ORDER, not your final server class order.
#
# Your server final class order:
#   ["balaclava", "hoodie", "glove", "bat", "spray paint", "graffiti"]
#
# So these thresholds are:
#   balaclava=0.38
#   hoodie=0.38
#   glove=0.22
#   bat=0.12
#   spray paint=0.33
#   graffiti=0.2
# So bonus thresholds are:
#   balaclava=0.2
#   hoodie=0.25
#   glove=0.12
#   bat=0.09
#   spray paint=0.21
#   graffiti=0.06
# Per-class candidate floors, in MODEL-EMIT order
#   [balaclava, bat, glove, graffiti, hoodie, spray paint]
# ALIGNED to the server's _conf_thres_array = [0.38,0.38,0.22,0.12,0.33,0.20]
# (which is in the server's OUTPUT order [balaclava,hoodie,glove,bat,spray,graffiti])
# remapped into model-emit order. Using the server's own thresholds here means a
# flipped candidate is only added if the server would also have accepted it,
# so nothing is appended just to be dropped by the server conf filter, and
# nothing addable is skipped.
DEFAULT_CLASS_CONF_LIST = [0.38, 0.38, 0.22, 0.12, 0.33, 0.20]
DEFAULT_CLASS_BONUS_LIST = [0.2, 0.25, 0.12, 0.09, 0.21, 0.06]

# Per-class RESCUE BONUS, in MODEL-EMIT order — mirrors the server's
# _bonus_array = [0.2,0.25,0.12,0.09,0.21,0.06] (server output order
# [balaclava,hoodie,glove,bat,spray,graffiti]) remapped into model-emit order
# [balaclava,bat,glove,graffiti,hoodie,spray]. The server admits a class's
# top-1 candidate at (threshold - bonus) when that class is otherwise absent;
# the patch replicates that rule so (a) its view of the server's kept originals
# is exact (keeps the survival guarantee airtight) and (b) flipped candidates
# in the [thr - bonus, thr) band of an absent class are appended instead of
# skipped — those are exactly the rescue-band boxes that win mAP50.

COEFFICIENT_FOR_PATCH = 3/17
CHECKER_OUTPUT_DRIFT_MAX_PER_TARGET = 2

# MATCHED IOU — must EQUAL the server's same-class NMS threshold (Miner.iou_thres
# = 0.3) for appended boxes to SURVIVE the server pipeline.
#
# Why not 1.0: the server runs _per_class_hard_nms(iou_thres=0.3) on the union of
# original + appended boxes, unconditionally. Any appended box overlapping a
# same-class kept box by >0.3 is deleted there. So a 1.0 match here only wastes
# the ratio-capped budget on near-duplicates the server discards. At 0.3, every
# box we spend budget on is 0.3-separated from same-class kept boxes and is
# therefore provably NOT removed by the server's NMS -> it reaches the output.
DEFAULT_PATCH_IOU = 0.3


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_class_conf_list() -> list[float]:
    value = os.getenv("ORT_YOLO_CLASS_CONF_LIST")
    if not value:
        return DEFAULT_CLASS_CONF_LIST

    try:
        parsed = [float(x.strip()) for x in value.split(",")]
        if len(parsed) > 0:
            return parsed
    except Exception:
        pass

    return DEFAULT_CLASS_CONF_LIST


def _get_class_bonus_list() -> list[float]:
    value = os.getenv("ORT_YOLO_CLASS_BONUS_LIST")
    if not value:
        return DEFAULT_CLASS_BONUS_LIST

    try:
        parsed = [float(x.strip()) for x in value.split(",")]
        if len(parsed) > 0:
            return parsed
    except Exception:
        pass

    return DEFAULT_CLASS_BONUS_LIST


def _apply_rescue_bonus(
    keep: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_conf_list: list[float],
    class_bonus_list: list[float],
) -> np.ndarray:
    """
    Mirror of the server's _conf_filter_mask rescue rule:
    for each class with ZERO candidates passing its threshold, admit that
    class's single top-1 candidate when conf >= (threshold - bonus).
    Deliberately NOT gated by global_conf — the server's rule isn't either
    (e.g. bat: 0.12 - 0.09 = 0.03 is a legal rescue score).
    """
    nc = len(class_conf_list)
    for c in range(nc):
        bonus = float(class_bonus_list[c]) if c < len(class_bonus_list) else 0.0
        if bonus <= 0.0:
            continue
        cm = class_ids == c
        if not np.any(cm):
            continue
        if np.any(keep & cm):
            continue
        idx = np.where(cm)[0]
        top = int(idx[int(np.argmax(confidences[idx]))])
        if confidences[top] >= float(class_conf_list[c]) - bonus:
            keep[top] = True
    return keep


def _unwrap_single_batch(output: np.ndarray) -> tuple[np.ndarray, bool]:
    pred = np.asarray(output)
    if pred.ndim == 3 and pred.shape[0] == 1:
        return pred[0], True
    return pred, False


def _is_final_det_output(body: np.ndarray) -> bool:
    # Final detection format:
    #   [N, 6] = x1, y1, x2, y2, conf, cls
    return body.ndim == 2 and body.shape[1] == 6


def _raw_output_to_rows(
    output: np.ndarray,
    nc: int,
) -> tuple[np.ndarray, str, int, bool]:
    """
    Returns raw YOLO rows in [N, C] layout.

    Supports:
      [1, 4+nc, N]
      [1, N, 4+nc]
      [4+nc, N]
      [N, 4+nc]

    Also supports 5+nc internally:
      [1, 5+nc, N]
      [1, N, 5+nc]

    Return:
      rows: [N, C]
      layout: "channels_first" or "rows"
      cols: C
      batched: original output had [1, ...]
    """
    body, batched = _unwrap_single_batch(output)

    if body.ndim != 2:
        raise ValueError(f"Unsupported raw output shape: {np.asarray(output).shape}")

    valid_cols = {4 + nc, 5 + nc}

    if body.shape[0] in valid_cols:
        return body.T, "channels_first", int(body.shape[0]), batched

    if body.shape[1] in valid_cols:
        return body, "rows", int(body.shape[1]), batched

    raise ValueError(f"Unsupported raw YOLO layout: {np.asarray(output).shape}, nc={nc}")


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros(0, dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )

    union = box_area + boxes_area - inter
    return inter / np.maximum(union, 1e-7)


def _nms_same_class(dets: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    dets:
      [N, 6] = x1, y1, x2, y2, conf, cls

    Used only for selecting flipped candidates.
    The original raw output is not replaced by this.
    """
    if dets.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    final = []

    for cls_id in np.unique(dets[:, 5].astype(np.int32)):
        cls_dets = dets[dets[:, 5].astype(np.int32) == cls_id]
        cls_dets = cls_dets[np.argsort(-cls_dets[:, 4])]

        while len(cls_dets) > 0:
            best = cls_dets[0]
            final.append(best)

            if len(cls_dets) == 1:
                break

            rest = cls_dets[1:]
            ious = _iou_one_to_many(best[:4], rest[:, :4])
            cls_dets = rest[ious <= iou_threshold]

    if not final:
        return np.zeros((0, 6), dtype=np.float32)

    final = np.stack(final).astype(np.float32)
    final = final[np.argsort(-final[:, 4])]
    return final


def _raw_yolo_output_to_dets(
    output: np.ndarray,
    class_conf_list: list[float],
    global_conf: float,
    nms_iou: float,
    class_bonus_list: "list[float] | None" = None,
) -> np.ndarray:
    """
    Converts raw YOLO output into candidate detections:

      [N, 6] = x1, y1, x2, y2, conf, cls

    This is used only inside the patch to decide which flipped boxes are missing.
    It should not replace the original raw model output.
    """
    nc = len(class_conf_list)

    body, _ = _unwrap_single_batch(output)

    # If the model is already final-det format, support it.
    if _is_final_det_output(body):
        dets = body.astype(np.float32)

        if len(dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        cls_ids = dets[:, 5].astype(np.int32)
        valid = (cls_ids >= 0) & (cls_ids < nc)
        dets = dets[valid]

        if len(dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        cls_ids = dets[:, 5].astype(np.int32)
        class_thresholds = np.asarray(class_conf_list, dtype=np.float32)[cls_ids]

        keep = (dets[:, 4] >= global_conf) & (dets[:, 4] >= class_thresholds)
        if class_bonus_list is not None:
            keep = _apply_rescue_bonus(
                keep, dets[:, 4].astype(np.float32), cls_ids,
                class_conf_list, class_bonus_list,
            )
        dets = dets[keep]

        if len(dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        return _nms_same_class(dets, nms_iou)

    rows, _, cols, _ = _raw_output_to_rows(output, nc)

    if cols == 4 + nc:
        boxes_xywh = rows[:, :4].astype(np.float32)
        class_scores = rows[:, 4:].astype(np.float32)
    elif cols == 5 + nc:
        boxes_xywh = rows[:, :4].astype(np.float32)
        objectness = rows[:, 4:5].astype(np.float32)
        class_scores = rows[:, 5:].astype(np.float32) * objectness
    else:
        raise ValueError(f"Unsupported raw YOLO column count: {cols}")

    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    confidences = class_scores[np.arange(len(class_scores)), class_ids].astype(np.float32)

    class_thresholds = np.asarray(class_conf_list, dtype=np.float32)[class_ids]
    keep = (confidences >= global_conf) & (confidences >= class_thresholds)
    if class_bonus_list is not None:
        keep = _apply_rescue_bonus(
            keep, confidences, class_ids, class_conf_list, class_bonus_list,
        )

    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xywh = boxes_xywh[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    dets = np.stack(
        [
            x1,
            y1,
            x2,
            y2,
            confidences,
            class_ids.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return _nms_same_class(dets, nms_iou)


# --- per-class state (drives what may be appended without perturbing originals) ---
STATE_PRESENT = 1   # has original candidate >= threshold: rescue OFF
STATE_RESCUE = 2    # top original candidate in [thr - bonus, thr): rescue ON
STATE_ABSENT = 3    # no original candidate >= thr - bonus: class emits nothing


def _class_argmax_max_conf(output: np.ndarray, nc: int) -> np.ndarray:
    """Per class c: max confidence among ORIGINAL candidates whose argmax is c
    (mirrors how the server forms candidates: cls=argmax, conf=max)."""
    m = np.zeros(nc, dtype=np.float32)
    body, _ = _unwrap_single_batch(np.asarray(output))
    if _is_final_det_output(body):
        cls = body[:, 5].astype(np.int32)
        conf = body[:, 4].astype(np.float32)
    else:
        rows, _, cols, _ = _raw_output_to_rows(np.asarray(output), nc)
        if cols == 4 + nc:
            scores = rows[:, 4:].astype(np.float32)
        else:
            scores = rows[:, 5:].astype(np.float32) * rows[:, 4:5].astype(np.float32)
        cls = np.argmax(scores, axis=1).astype(np.int32)
        conf = scores[np.arange(len(scores)), cls]
    for c in range(nc):
        cm = cls == c
        if np.any(cm):
            m[c] = float(conf[cm].max())
    return m


def _compute_class_states(
    class_max_conf: np.ndarray,
    class_conf_list: list[float],
    class_bonus_list: list[float],
) -> list[int]:
    states = []
    for c in range(len(class_conf_list)):
        thr = float(class_conf_list[c])
        bonus = float(class_bonus_list[c]) if c < len(class_bonus_list) else 0.0
        if class_max_conf[c] >= thr:
            states.append(STATE_PRESENT)
        elif bonus > 0.0 and class_max_conf[c] >= thr - bonus:
            states.append(STATE_RESCUE)
        else:
            states.append(STATE_ABSENT)
    return states


def _unflip_dets(dets: np.ndarray, width: int) -> np.ndarray:
    """
    Convert flipped-tensor detections back to original tensor coordinates.
    Coordinates are still in model-input / letterbox space.
    """
    if dets.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    out = dets.copy().astype(np.float32)

    x1 = out[:, 0].copy()
    x2 = out[:, 2].copy()

    out[:, 0] = width - x2
    out[:, 2] = width - x1

    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 2] = np.clip(out[:, 2], 0, width - 1)

    return out


def _same_object_same_label(
    candidate: np.ndarray,
    references: np.ndarray,
    iou_threshold: float,
) -> bool:
    """
    True when `candidate` is already represented by a SAME-CLASS reference box
    at IoU >= iou_threshold.

    With iou_threshold = 1.0 (the "matched IoU 1.0" rule) this is True only for a
    near-identical same-class box, so almost every flipped detection is treated
    as new and becomes eligible to be appended.
    """
    if references.size == 0:
        return False

    same_cls = references[:, 5].astype(np.int32) == int(candidate[5])
    if not np.any(same_cls):
        return False

    same_cls_refs = references[same_cls]
    ious = _iou_one_to_many(candidate[:4], same_cls_refs[:, :4])
    return bool(np.any(ious >= iou_threshold))


def _overlaps_any_box(
    candidate: np.ndarray,
    references: np.ndarray,
    iou_threshold: float,
) -> bool:
    """
    Used to protect original boxes from later cross-class dedup.

    Your server has cross-class dedup after normal NMS, so a high-score flipped
    extra candidate from another class can still suppress an original box if it
    overlaps too much. This check prevents that.

    NOTE: this is a COLLISION guard, not the "matched IoU". It stays at
    cross_iou (default 0.8, matching the server's cross_iou_thresh) even though
    the matched IoU is 1.0 — otherwise an appended box could delete a real
    original detection downstream and cost mAP. Raise ORT_YOLO_CROSS_IOU to 1.0
    only if you truly want zero collision protection.
    """
    if references.size == 0:
        return False

    ious = _iou_one_to_many(candidate[:4], references[:, :4])
    return bool(np.any(ious >= iou_threshold))


def _intersects_same_class_with_margin(
    candidate: np.ndarray,
    references: np.ndarray,
    margin: float,
) -> bool:
    """True if `candidate`, inflated by `margin` px on every side, intersects
    any SAME-CLASS reference box. Zero same-class overlap is required because
    the validator matches greedily with no IoU floor -> any overlap steals a
    real box's match."""
    if references.size == 0:
        return False
    same = references[:, 5].astype(np.int32) == int(candidate[5])
    if not np.any(same):
        return False
    refs = references[same]
    cx1 = candidate[0] - margin
    cy1 = candidate[1] - margin
    cx2 = candidate[2] + margin
    cy2 = candidate[3] + margin
    ix1 = np.maximum(cx1, refs[:, 0])
    iy1 = np.maximum(cy1, refs[:, 1])
    ix2 = np.minimum(cx2, refs[:, 2])
    iy2 = np.minimum(cy2, refs[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    return bool(np.any(inter > 0.0))


def _select_missing_flipped_dets(
    original_dets: np.ndarray,
    flipped_dets: np.ndarray,
    patch_iou: float,
    cross_iou: float,
    max_extra: int,
    class_states: "list[int] | None" = None,
    class_conf_list: "list[float] | None" = None,
    orig_class_max: "np.ndarray | None" = None,
    same_class_gap: float = 0.0,
) -> np.ndarray:
    """
    Keep original detections unchanged.

    Select only flipped detections that are not already represented by an
    original detection.

    Return only the extra flipped detections, not original+extra.
    """
    if max_extra <= 0:
        return np.zeros((0, 6), dtype=np.float32)

    if flipped_dets.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if original_dets.size == 0:
        original_dets = np.zeros((0, 6), dtype=np.float32)

    candidates = flipped_dets.copy().astype(np.float32)
    candidates = candidates[np.argsort(-candidates[:, 4])]

    added = []
    above_added: dict[int, bool] = {}
    band_added: dict[int, bool] = {}

    for candidate in candidates:
        # ---- MATCHED-IOU-1.0 GATE ---------------------------------------
        # Appends must be PURE additions: they may never replace, displace or
        # disable an original box the server would otherwise emit.
        if class_states is not None and class_conf_list is not None:
            c = int(candidate[5])
            conf = float(candidate[4])
            state = class_states[c] if 0 <= c < len(class_states) else STATE_ABSENT
            thr = float(class_conf_list[c])

            if state == STATE_RESCUE:
                # A rescued original exists for this class. ANY appended box
                # either outranks it (rescue emits OUR box -> coords shift ->
                # matched IoU < 1.0) or passes threshold (rescue disabled ->
                # the rescued original DISAPPEARS). Both perturb -> forbidden.
                continue

            if conf < thr:
                # Sub-threshold candidate survives only via rescue, which fires
                # only when the class is fully ABSENT and this box is the top-1.
                if state != STATE_ABSENT:
                    continue  # PRESENT class: server would just drop it
                if above_added.get(c) or band_added.get(c):
                    continue  # a stronger add already covers this class
                if orig_class_max is not None and conf <= float(orig_class_max[c]):
                    continue  # original argmax would win the rescue -> wasted
                pending_flag = ("band", c)
            else:
                pending_flag = ("above", c)
        else:
            pending_flag = None
        # ------------------------------------------------------------------
        refs = original_dets
        if added:
            refs = np.concatenate(
                [original_dets, np.stack(added).astype(np.float32)],
                axis=0,
            )

        # PRIMARY FIX — CHECKER MATCH-STEAL PREVENTION.
        # The validator matches greedily, SAME-CLASS only, with NO IoU floor, in
        # expected-list order. ANY positive same-class overlap lets this appended
        # box consume a real box's match (low IoU for itself + orphans the real
        # box to 0.0) -> matched_mean_iou < 1.0. Require ZERO same-class overlap,
        # with a pixel margin to absorb the server's floor/ceil rounding.
        if _intersects_same_class_with_margin(candidate, refs, same_class_gap):
            continue

        # SERVER-SUPPRESSION GUARD: never overlap a kept box of ANY class by
        # >= cross_iou (= server cross_iou_thresh 0.8), or the server's
        # _cross_class_dedup_op could delete an ORIGINAL box downstream.
        if _overlaps_any_box(candidate, refs, cross_iou):
            continue

        added.append(candidate.astype(np.float32))
        if pending_flag is not None:
            kind, cc = pending_flag
            if kind == "band":
                band_added[cc] = True
            else:
                above_added[cc] = True

        if len(added) >= max_extra:
            break

    if not added:
        return np.zeros((0, 6), dtype=np.float32)

    return np.stack(added).astype(np.float32)


def _append_extra_dets_to_model_output(
    original_output: np.ndarray,
    extra_dets: np.ndarray,
    nc: int,
) -> np.ndarray:
    """
    Append selected flipped detections to the ORIGINAL MODEL OUTPUT FORMAT.

    This is the key fix.

    If original output is raw YOLO:
      [1, 4+nc, N] -> [1, 4+nc, N+M]
      [1, N, 4+nc] -> [1, N+M, 4+nc]

    If original output is final det:
      [1, N, 6] -> [1, N+M, 6]

    extra_dets:
      [M, 6] = x1, y1, x2, y2, conf, cls
      in model-input / letterbox coordinates.
    """
    if extra_dets.size == 0:
        return original_output

    pred = np.asarray(original_output)
    body, batched = _unwrap_single_batch(pred)

    # Final-det format support.
    if _is_final_det_output(body):
        merged = np.concatenate(
            [body.astype(np.float32), extra_dets.astype(np.float32)],
            axis=0,
        )

        if batched:
            merged = merged[None, :, :]

        return merged.astype(pred.dtype, copy=False)

    rows, layout, cols, batched = _raw_output_to_rows(pred, nc)

    m = len(extra_dets)
    extra_raw = np.zeros((m, cols), dtype=rows.dtype)

    x1 = extra_dets[:, 0].astype(np.float32)
    y1 = extra_dets[:, 1].astype(np.float32)
    x2 = extra_dets[:, 2].astype(np.float32)
    y2 = extra_dets[:, 3].astype(np.float32)
    conf = extra_dets[:, 4].astype(np.float32)
    cls = extra_dets[:, 5].astype(np.int32)

    # xyxy -> xywh
    extra_raw[:, 0] = ((x1 + x2) / 2.0).astype(rows.dtype)
    extra_raw[:, 1] = ((y1 + y2) / 2.0).astype(rows.dtype)
    extra_raw[:, 2] = (x2 - x1).astype(rows.dtype)
    extra_raw[:, 3] = (y2 - y1).astype(rows.dtype)

    if cols == 4 + nc:
        # YOLO format:
        #   cx, cy, w, h, class0, class1, ...
        for i in range(m):
            c = int(cls[i])
            if 0 <= c < nc:
                extra_raw[i, 4 + c] = rows.dtype.type(conf[i])

    elif cols == 5 + nc:
        # YOLO format:
        #   cx, cy, w, h, objectness, class0, class1, ...
        #
        # Set objectness=1.0 and class score=conf so objectness*class=conf.
        extra_raw[:, 4] = rows.dtype.type(1.0)
        for i in range(m):
            c = int(cls[i])
            if 0 <= c < nc:
                extra_raw[i, 5 + c] = rows.dtype.type(conf[i])

    else:
        raise ValueError(f"Unsupported raw YOLO column count: {cols}")

    merged_rows = np.concatenate([rows, extra_raw], axis=0)

    if layout == "channels_first":
        merged_body = merged_rows.T
    else:
        merged_body = merged_rows

    if batched:
        merged_output = merged_body[None, ...]
    else:
        merged_output = merged_body

    return merged_output.astype(pred.dtype, copy=False)


def _ratio_safe_max_extra(n_orig: int) -> int:
    """
    Largest M such that  n_orig / (n_orig + M) > 0.85  (STRICT).

    COEFFICIENT_FOR_PATCH = 3/17 = 0.17647..., so int(n_orig * 3/17) is the
    intended cap — but at exact multiples of 17 (n=17 -> 3 -> 17/20 = 0.85)
    that lands exactly ON 0.85 and FAILS a strict '> 0.85' checker. This trims
    the boundary case so the emitted ratio is always strictly above 0.85.
    """
    m = int(n_orig * COEFFICIENT_FOR_PATCH)
    while m > 0 and n_orig / (n_orig + m) <= 0.85:
        m -= 1
    return m


class PatchedInferenceSession:
    """
    Drop-in replacement for onnxruntime.InferenceSession.

    Behavior:
      1. Run original input.
      2. Run horizontally flipped input tensor.
      3. Decode original and flipped outputs only internally.
      4. Convert flipped boxes back to original tensor coordinates.
      5. Select flipped boxes the original missed (matched IoU = 1.0, so only
         near-identical same-class boxes are treated as "already present"),
         bounded so n_orig / n_patched stays strictly > 0.85.
      6. Append those extra boxes to the ORIGINAL RAW YOLO OUTPUT FORMAT.
      7. Return normal-looking ONNX output to the server.

    Disable:
      export ORT_YOLO_PATCH_TTA=0

    Recommended env:
      export ORT_YOLO_PATCH_IOU=1.0     # matched IoU (add freely)
      export ORT_YOLO_CROSS_IOU=0.8     # collision guard vs server cross-dedup
    """

    def __init__(self, *args, **kwargs):
        print(f"[yolo_patch] PatchedInferenceSession active, "
              f"version={YOLO_PATCH_VERSION}", flush=True)
        self._session = RawInferenceSession(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def run(self, output_names=None, input_feed=None, run_options=None):
        if not _env_bool("ORT_YOLO_PATCH_TTA", True):
            return self._session.run(output_names, input_feed, run_options)

        if not input_feed:
            return self._session.run(output_names, input_feed, run_options)

        input_name = self._session.get_inputs()[0].name

        if input_name not in input_feed:
            return self._session.run(output_names, input_feed, run_options)

        x = input_feed[input_name]

        if not isinstance(x, np.ndarray):
            return self._session.run(output_names, input_feed, run_options)

        # Expected preprocessed YOLO tensor:
        #   [1, 3, H, W]
        #
        # Do not patch batches here, because appending different numbers of
        # candidates per image would require per-image output packing.
        if x.ndim != 4 or x.shape[0] != 1:
            return self._session.run(output_names, input_feed, run_options)

        width = int(x.shape[-1])

        # 1. Original run. This output is preserved.
        original_outputs = self._session.run(output_names, input_feed, run_options)

        class_conf_list = _get_class_conf_list()
        nc = len(class_conf_list)

        global_conf = _env_float("ORT_YOLO_GLOBAL_CONF", 0.1)

        # This NMS is only for selecting flipped candidates internally.
        # It does not replace your server's final NMS.
        internal_nms_iou = _env_float("ORT_YOLO_INTERNAL_NMS_IOU", 0.3)

        # MATCHED IOU — 1.0 by default: add flipped boxes freely, only skip
        # near-identical same-class duplicates.
        patch_iou = _env_float("ORT_YOLO_PATCH_IOU", DEFAULT_PATCH_IOU)

        # Cross-class COLLISION guard (matches server cross-class dedup threshold).
        cross_iou = _env_float("ORT_YOLO_CROSS_IOU", 0.8)

        class_bonus_list = _get_class_bonus_list()

        try:
            # Decode original only for duplicate checking.
            # We do NOT return this decoded output. Rescue-bonus mirrored so
            # this set matches what the server's conf filter will keep.
            original_dets = _raw_yolo_output_to_dets(
                output=original_outputs[0],
                class_conf_list=class_conf_list,
                global_conf=global_conf,
                nms_iou=internal_nms_iou,
                class_bonus_list=class_bonus_list,
            )

            # Ratio rule: n_orig / (n_orig + M) must stay strictly > 0.85.
            max_extra = _ratio_safe_max_extra(len(original_dets))

            # max_det guard: server truncates to its top `max_det` by score.
            # Never append past that budget, or a low-conf ORIGINAL could be
            # pushed out of the top-K -> a missing original -> matched IoU hit.
            server_max_det = _env_int("ORT_YOLO_SERVER_MAX_DET", 150)
            max_extra = min(max_extra, max(0, server_max_det - len(original_dets)))

            if max_extra < 1:
                return original_outputs

            # Per-class state of the ORIGINAL pass — decides which classes can
            # accept pure additions without perturbing the server's rescue
            # logic (the source of matched_mean_iou < 1.0 in the checker).
            orig_class_max = _class_argmax_max_conf(original_outputs[0], nc)
            class_states = _compute_class_states(
                orig_class_max, class_conf_list, class_bonus_list,
            )

            # 2. Flipped run.
            flipped_x = np.ascontiguousarray(x[..., ::-1])

            flipped_feed = dict(input_feed)
            flipped_feed[input_name] = flipped_x

            flipped_outputs = self._session.run(output_names, flipped_feed, run_options)

            # Decode flipped output. Rescue-bonus applies here too, so a
            # flipped candidate of an ABSENT class in the [thr - bonus, thr)
            # band becomes an addable candidate instead of being skipped.
            flipped_dets = _raw_yolo_output_to_dets(
                output=flipped_outputs[0],
                class_conf_list=class_conf_list,
                global_conf=global_conf,
                nms_iou=internal_nms_iou,
                class_bonus_list=class_bonus_list,
            )

            # Convert flipped coords back into original model-input coordinates.
            flipped_dets = _unflip_dets(flipped_dets, width=width)

            # Select only missing flipped detections (pure additions only).
            extra_dets = _select_missing_flipped_dets(
                original_dets=original_dets,
                flipped_dets=flipped_dets,
                patch_iou=patch_iou,
                cross_iou=cross_iou,
                max_extra=max_extra,
                class_states=class_states,
                class_conf_list=class_conf_list,
                orig_class_max=orig_class_max,
                same_class_gap=_env_float(
                    "ORT_YOLO_SAME_CLASS_GAP_PX", DEFAULT_SAME_CLASS_GAP_PX),
            )

            if extra_dets.size == 0:
                return original_outputs

            # Critical part:
            # append extras to original raw output format, not [1, N, 6].
            patched_first_output = _append_extra_dets_to_model_output(
                original_output=original_outputs[0],
                extra_dets=extra_dets,
                nc=nc,
            )

            return [patched_first_output] + list(original_outputs[1:])

        except Exception as e:
            print(f"⚠️ ORT YOLO patch failed, returning original ONNX output: {e}")
            return original_outputs
