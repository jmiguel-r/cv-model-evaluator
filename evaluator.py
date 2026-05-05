"""
evaluator.py
------------
Core evaluation module for computer vision models.
Computes mAP, IoU, per-class metrics, and confusion matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou


@dataclass
class Detection:
    """Single model detection or ground-truth annotation."""
    bbox: List[float]       # [x1, y1, x2, y2] absolute pixels
    class_id: int
    confidence: float = 1.0
    image_id: str = ""


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics of a single run."""
    map50: float = 0.0
    map50_95: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)
    total_images: int = 0
    total_gt_boxes: int = 0
    total_pred_boxes: int = 0


def compute_iou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between predicted and ground-truth boxes.

    Args:
        pred_boxes: (N, 4) tensor in xyxy format
        gt_boxes:   (M, 4) tensor in xyxy format

    Returns:
        (N, M) IoU matrix
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((len(pred_boxes), len(gt_boxes)))
    return box_iou(pred_boxes, gt_boxes)


def match_detections(
    predictions: List[Detection],
    ground_truths: List[Detection],
    iou_threshold: float = 0.5,
) -> Tuple[List[bool], List[bool]]:
    """
    Match predictions to ground truths using greedy IoU matching.

    Returns:
        tp_flags: bool list (len = predictions) — True if TP
        fn_flags: bool list (len = ground_truths) — True if missed
    """
    if not predictions or not ground_truths:
        return [False] * len(predictions), [True] * len(ground_truths)

    pred_boxes = torch.tensor([d.bbox for d in predictions], dtype=torch.float32)
    gt_boxes   = torch.tensor([d.bbox for d in ground_truths], dtype=torch.float32)
    iou_mat    = compute_iou_matrix(pred_boxes, gt_boxes)

    matched_gt = set()
    tp_flags   = []

    # Sort predictions by confidence (descending)
    order = sorted(range(len(predictions)), key=lambda i: predictions[i].confidence, reverse=True)

    tp_map: Dict[int, bool] = {}
    for i in order:
        if iou_mat.shape[1] == 0:
            tp_map[i] = False
            continue
        best_iou, best_j = iou_mat[i].max(0)
        best_j = best_j.item()
        if (
            best_iou.item() >= iou_threshold
            and predictions[i].class_id == ground_truths[best_j].class_id
            and best_j not in matched_gt
        ):
            tp_map[i] = True
            matched_gt.add(best_j)
        else:
            tp_map[i] = False

    tp_flags = [tp_map[i] for i in range(len(predictions))]
    fn_flags = [j not in matched_gt for j in range(len(ground_truths))]
    return tp_flags, fn_flags


def compute_average_precision(
    tp_list: List[bool],
    confidences: List[float],
    n_gt: int,
) -> float:
    """Compute AP using 11-point interpolation."""
    if n_gt == 0:
        return 0.0

    order = np.argsort(-np.array(confidences))
    tp = np.array(tp_list)[order].astype(float)

    cumtp = np.cumsum(tp)
    cumfp = np.cumsum(1 - tp)

    precision = cumtp / (cumtp + cumfp + 1e-9)
    recall    = cumtp / (n_gt + 1e-9)

    # 11-point interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        mask = recall >= thr
        ap += precision[mask].max() if mask.any() else 0.0
    return ap / 11.0


def evaluate(
    all_predictions: Dict[str, List[Detection]],
    all_ground_truths: Dict[str, List[Detection]],
    class_names: List[str],
    iou_thresholds: Optional[List[float]] = None,
) -> EvaluationResult:
    """
    Full evaluation pipeline.

    Args:
        all_predictions:  {image_id: [Detection, ...]}
        all_ground_truths: {image_id: [Detection, ...]}
        class_names: ordered list of class name strings
        iou_thresholds: defaults to [0.5] and 0.5:0.95:0.05

    Returns:
        EvaluationResult with all metrics populated
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]

    n_classes = len(class_names)
    image_ids = set(all_ground_truths.keys()) | set(all_predictions.keys())

    # Per-class AP storage: {class_id: {iou_thr: (tp_list, conf_list, n_gt)}}
    class_data: Dict[int, Dict[float, dict]] = {
        c: {t: {"tp": [], "conf": [], "n_gt": 0} for t in iou_thresholds}
        for c in range(n_classes)
    }

    conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    total_gt   = 0
    total_pred = 0

    for img_id in image_ids:
        preds = all_predictions.get(img_id, [])
        gts   = all_ground_truths.get(img_id, [])
        total_gt   += len(gts)
        total_pred += len(preds)

        for iou_thr in iou_thresholds:
            tp_flags, fn_flags = match_detections(preds, gts, iou_thr)
            for i, (pred, is_tp) in enumerate(zip(preds, tp_flags)):
                cid = pred.class_id
                if 0 <= cid < n_classes:
                    class_data[cid][iou_thr]["tp"].append(is_tp)
                    class_data[cid][iou_thr]["conf"].append(pred.confidence)

            for gt in gts:
                cid = gt.class_id
                if 0 <= cid < n_classes:
                    class_data[cid][iou_thr]["n_gt"] += 1

        # Confusion matrix at IoU=0.5
        tp50, _ = match_detections(preds, gts, 0.5)
        for pred, is_tp in zip(preds, tp50):
            pc = pred.class_id if 0 <= pred.class_id < n_classes else n_classes
            if is_tp:
                conf_matrix[pc][pc] += 1
            else:
                conf_matrix[pc][n_classes] += 1  # FP

        _, fn50 = match_detections(preds, gts, 0.5)
        for gt, is_fn in zip(gts, fn50):
            gc = gt.class_id if 0 <= gt.class_id < n_classes else n_classes
            if is_fn:
                conf_matrix[n_classes][gc] += 1  # FN

    # Compute per-class AP
    aps_50:    Dict[str, float] = {}
    aps_50_95: Dict[str, float] = {}

    for cid, name in enumerate(class_names):
        ap_at_thrs = []
        for t in iou_thresholds:
            d = class_data[cid][t]
            ap = compute_average_precision(d["tp"], d["conf"], d["n_gt"])
            ap_at_thrs.append(ap)
            if t == 0.5:
                aps_50[name] = ap
        aps_50_95[name] = float(np.mean(ap_at_thrs))

    map50    = float(np.mean(list(aps_50.values()))) if aps_50 else 0.0
    map50_95 = float(np.mean(list(aps_50_95.values()))) if aps_50_95 else 0.0

    # Global precision / recall / F1 at IoU=0.5
    all_tp   = sum(sum(class_data[c][0.5]["tp"]) for c in range(n_classes))
    all_pred = sum(len(class_data[c][0.5]["tp"]) for c in range(n_classes))
    all_gt   = sum(class_data[c][0.5]["n_gt"] for c in range(n_classes))

    prec = all_tp / (all_pred + 1e-9)
    rec  = all_tp / (all_gt  + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    return EvaluationResult(
        map50=map50,
        map50_95=map50_95,
        per_class_ap=aps_50,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        confusion_matrix=conf_matrix,
        class_names=class_names,
        total_images=len(image_ids),
        total_gt_boxes=total_gt,
        total_pred_boxes=total_pred,
    )


def save_results(result: EvaluationResult, output_path: str | Path) -> None:
    """Serialize EvaluationResult to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "map50":          result.map50,
        "map50_95":       result.map50_95,
        "per_class_ap":   result.per_class_ap,
        "precision":      result.precision,
        "recall":         result.recall,
        "f1":             result.f1,
        "total_images":   result.total_images,
        "total_gt_boxes": result.total_gt_boxes,
        "total_pred_boxes": result.total_pred_boxes,
        "class_names":    result.class_names,
        "confusion_matrix": result.confusion_matrix.tolist() if result.confusion_matrix is not None else None,
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_path}")
