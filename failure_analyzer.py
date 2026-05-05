"""
failure_analyzer.py
-------------------
Categorizes model prediction errors into interpretable failure modes.
Helps identify systematic weaknesses in CV models.

Failure categories:
  - class_confusion   : correct box, wrong class label
  - missed_small      : ground-truth box with area < threshold not detected
  - missed_occluded   : GT box with high overlap with another GT (proxy for occlusion)
  - false_positive_bg : prediction with low overlap with any GT (background noise)
  - localization_error: correct class, but IoU between 0.1 and threshold (bad bbox)
  - duplicate_detection: multiple high-conf predictions for the same GT box
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from evaluator import Detection, compute_iou_matrix

SMALL_AREA_THRESHOLD  = 32 * 32   # pixels² — COCO "small" object definition
OCCLUSION_IOU_PROXY   = 0.3       # IoU between two GT boxes → probable occlusion
LOCALIZATION_IOU_LOW  = 0.1       # IoU above this = localization error (not pure FP)
MATCH_IOU_THRESHOLD   = 0.5


@dataclass
class FailureRecord:
    image_id:      str
    failure_type:  str
    pred_bbox:     List[float] | None
    gt_bbox:       List[float] | None
    pred_class:    int | None
    gt_class:      int | None
    confidence:    float | None
    iou:           float | None
    details:       str = ""


@dataclass
class FailureReport:
    records:   List[FailureRecord] = field(default_factory=list)
    summary:   Dict[str, int]      = field(default_factory=dict)

    def add(self, record: FailureRecord) -> None:
        self.records.append(record)
        self.summary[record.failure_type] = self.summary.get(record.failure_type, 0) + 1

    def top_failures(self, n: int = 5) -> List[Tuple[str, int]]:
        return sorted(self.summary.items(), key=lambda x: x[1], reverse=True)[:n]

    def failure_rate(self, failure_type: str, total_preds: int) -> float:
        return self.summary.get(failure_type, 0) / max(total_preds, 1)


def _bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def analyze_image(
    image_id: str,
    predictions: List[Detection],
    ground_truths: List[Detection],
    iou_threshold: float = MATCH_IOU_THRESHOLD,
) -> List[FailureRecord]:
    """
    Analyze a single image and return all failure records found.
    """
    failures: List[FailureRecord] = []

    if not predictions and not ground_truths:
        return failures

    # ── Case: no predictions at all ──────────────────────────────────────────
    if not predictions:
        for gt in ground_truths:
            area = _bbox_area(gt.bbox)
            ftype = "missed_small" if area < SMALL_AREA_THRESHOLD else "missed_detection"
            failures.append(FailureRecord(
                image_id=image_id, failure_type=ftype,
                pred_bbox=None, gt_bbox=gt.bbox,
                pred_class=None, gt_class=gt.class_id,
                confidence=None, iou=0.0,
                details=f"area={area:.0f}px²",
            ))
        return failures

    pred_boxes = torch.tensor([d.bbox for d in predictions], dtype=torch.float32)
    gt_boxes   = torch.tensor([d.bbox for d in ground_truths], dtype=torch.float32) if ground_truths else torch.zeros((0, 4))

    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)  # (N_pred, N_gt)

    matched_gt:   set[int] = set()
    matched_pred: set[int] = set()

    # Sort by confidence
    order = sorted(range(len(predictions)), key=lambda i: predictions[i].confidence, reverse=True)

    for i in order:
        pred = predictions[i]
        if iou_mat.shape[1] == 0:
            # Pure false positive — no GTs at all
            failures.append(FailureRecord(
                image_id=image_id, failure_type="false_positive_bg",
                pred_bbox=pred.bbox, gt_bbox=None,
                pred_class=pred.class_id, gt_class=None,
                confidence=pred.confidence, iou=0.0,
                details="No ground truth in image",
            ))
            matched_pred.add(i)
            continue

        ious_row  = iou_mat[i]
        best_iou  = ious_row.max().item()
        best_j    = ious_row.argmax().item()
        gt        = ground_truths[best_j] if ground_truths else None

        if best_iou >= iou_threshold and best_j not in matched_gt:
            if gt and pred.class_id == gt.class_id:
                # True positive — no failure
                matched_gt.add(best_j)
                matched_pred.add(i)
            elif gt:
                # Class confusion
                failures.append(FailureRecord(
                    image_id=image_id, failure_type="class_confusion",
                    pred_bbox=pred.bbox, gt_bbox=gt.bbox,
                    pred_class=pred.class_id, gt_class=gt.class_id,
                    confidence=pred.confidence, iou=best_iou,
                    details=f"predicted class {pred.class_id}, actual class {gt.class_id}",
                ))
                matched_gt.add(best_j)
                matched_pred.add(i)

        elif best_iou >= LOCALIZATION_IOU_LOW and gt and pred.class_id == gt.class_id:
            # Correct class, poor localization
            failures.append(FailureRecord(
                image_id=image_id, failure_type="localization_error",
                pred_bbox=pred.bbox, gt_bbox=gt.bbox,
                pred_class=pred.class_id, gt_class=gt.class_id,
                confidence=pred.confidence, iou=best_iou,
                details=f"IoU={best_iou:.3f} below threshold {iou_threshold}",
            ))
            matched_pred.add(i)

        elif best_j in matched_gt and best_iou >= iou_threshold:
            # Already matched GT — duplicate detection
            failures.append(FailureRecord(
                image_id=image_id, failure_type="duplicate_detection",
                pred_bbox=pred.bbox, gt_bbox=gt.bbox if gt else None,
                pred_class=pred.class_id, gt_class=gt.class_id if gt else None,
                confidence=pred.confidence, iou=best_iou,
                details="GT already matched by higher-confidence prediction",
            ))
            matched_pred.add(i)

        else:
            # Background false positive
            failures.append(FailureRecord(
                image_id=image_id, failure_type="false_positive_bg",
                pred_bbox=pred.bbox, gt_bbox=gt.bbox if gt else None,
                pred_class=pred.class_id, gt_class=gt.class_id if gt else None,
                confidence=pred.confidence, iou=best_iou,
                details=f"best IoU={best_iou:.3f} with any GT",
            ))
            matched_pred.add(i)

    # Unmatched ground truths → missed detections
    for j, gt in enumerate(ground_truths):
        if j not in matched_gt:
            area = _bbox_area(gt.bbox)

            # Check occlusion proxy: does this GT heavily overlap another GT?
            is_occluded = False
            if len(ground_truths) > 1:
                other_gts = [g for k, g in enumerate(ground_truths) if k != j]
                other_boxes = torch.tensor([g.bbox for g in other_gts], dtype=torch.float32)
                this_box    = torch.tensor([gt.bbox], dtype=torch.float32)
                occ_ious    = compute_iou_matrix(this_box, other_boxes)
                is_occluded = bool(occ_ious.max().item() >= OCCLUSION_IOU_PROXY)

            if area < SMALL_AREA_THRESHOLD:
                ftype = "missed_small"
                detail = f"area={area:.0f}px² < {SMALL_AREA_THRESHOLD}px²"
            elif is_occluded:
                ftype = "missed_occluded"
                detail = f"GT overlaps another GT (IoU≥{OCCLUSION_IOU_PROXY})"
            else:
                ftype = "missed_detection"
                detail = f"area={area:.0f}px²"

            failures.append(FailureRecord(
                image_id=image_id, failure_type=ftype,
                pred_bbox=None, gt_bbox=gt.bbox,
                pred_class=None, gt_class=gt.class_id,
                confidence=None, iou=0.0,
                details=detail,
            ))

    return failures


def analyze_dataset(
    all_predictions:   Dict[str, List[Detection]],
    all_ground_truths: Dict[str, List[Detection]],
    iou_threshold: float = MATCH_IOU_THRESHOLD,
) -> FailureReport:
    """
    Run failure analysis across the entire dataset.

    Returns a FailureReport with all records and a summary dict.
    """
    report = FailureReport()
    all_ids = set(all_predictions.keys()) | set(all_ground_truths.keys())

    for img_id in all_ids:
        preds = all_predictions.get(img_id, [])
        gts   = all_ground_truths.get(img_id, [])
        for record in analyze_image(img_id, preds, gts, iou_threshold):
            report.add(record)

    return report


def print_summary(report: FailureReport) -> None:
    """Pretty-print a failure analysis summary to stdout."""
    total = sum(report.summary.values())
    print("\n── Failure Mode Analysis ──────────────────────────────")
    print(f"{'Failure Type':<30} {'Count':>8}  {'Share':>8}")
    print("─" * 52)
    for ftype, count in sorted(report.summary.items(), key=lambda x: x[1], reverse=True):
        share = count / max(total, 1) * 100
        print(f"{ftype:<30} {count:>8}  {share:>7.1f}%")
    print("─" * 52)
    print(f"{'TOTAL':<30} {total:>8}")
    print()
