"""
annotation_qa.py
----------------
Quality assurance tools for CV training data annotations.

Checks:
  - Class distribution balance
  - Bounding box sanity (out-of-bounds, zero-area, extreme aspect ratios)
  - Duplicate / near-duplicate annotations (high IoU between same-class boxes)
  - Inter-annotator agreement (Cohen's kappa proxy for multi-annotator sets)
  - Small / large object frequency analysis
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from evaluator import Detection, compute_iou_matrix

DUPLICATE_IOU_THRESHOLD    = 0.85   # same-class boxes with IoU > this → duplicate
EXTREME_ASPECT_RATIO_LIMIT = 10.0   # width/height or height/width above this → flag
SMALL_AREA_THRESHOLD       = 32 * 32
LARGE_AREA_THRESHOLD       = 300 * 300


@dataclass
class AnnotationIssue:
    image_id:   str
    issue_type: str
    class_id:   Optional[int]
    bbox:       Optional[List[float]]
    details:    str = ""


@dataclass
class QAReport:
    issues:            List[AnnotationIssue]      = field(default_factory=list)
    class_distribution: Dict[str, int]            = field(default_factory=dict)
    issue_summary:     Dict[str, int]             = field(default_factory=dict)
    total_annotations: int                        = 0
    total_images:      int                        = 0

    def add_issue(self, issue: AnnotationIssue) -> None:
        self.issues.append(issue)
        self.issue_summary[issue.issue_type] = (
            self.issue_summary.get(issue.issue_type, 0) + 1
        )

    def issue_rate(self) -> float:
        return len(self.issues) / max(self.total_annotations, 1)

    def print_summary(self) -> None:
        print("\n── Annotation QA Report ────────────────────────────────────")
        print(f"Images    : {self.total_images}")
        print(f"Annotations: {self.total_annotations}")
        print(f"Issue rate: {self.issue_rate()*100:.1f}%  ({len(self.issues)} issues)")
        print()
        print("Issue breakdown:")
        for itype, count in sorted(self.issue_summary.items(), key=lambda x: x[1], reverse=True):
            pct = count / max(self.total_annotations, 1) * 100
            print(f"  {itype:<35} {count:>6}  ({pct:.1f}%)")
        print()
        print("Class distribution:")
        total_ann = sum(self.class_distribution.values())
        for cls, count in sorted(self.class_distribution.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(count / max(total_ann, 1) * 40)
            bar = "█" * bar_len
            print(f"  {cls:<25} {count:>6}  {bar}")
        print()


# ── Per-image checks ──────────────────────────────────────────────────────────

def _check_bbox_sanity(
    ann: Detection, image_w: int, image_h: int, image_id: str
) -> Optional[AnnotationIssue]:
    x1, y1, x2, y2 = ann.bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return AnnotationIssue(
            image_id=image_id, issue_type="zero_or_negative_area",
            class_id=ann.class_id, bbox=ann.bbox,
            details=f"w={w:.1f}, h={h:.1f}",
        )

    if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h:
        return AnnotationIssue(
            image_id=image_id, issue_type="out_of_bounds",
            class_id=ann.class_id, bbox=ann.bbox,
            details=f"image={image_w}×{image_h}, box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]",
        )

    aspect = max(w / max(h, 1), h / max(w, 1))
    if aspect > EXTREME_ASPECT_RATIO_LIMIT:
        return AnnotationIssue(
            image_id=image_id, issue_type="extreme_aspect_ratio",
            class_id=ann.class_id, bbox=ann.bbox,
            details=f"aspect_ratio={aspect:.1f}",
        )

    return None


def _check_duplicates(
    annotations: List[Detection], image_id: str
) -> List[AnnotationIssue]:
    issues: List[AnnotationIssue] = []
    if len(annotations) < 2:
        return issues

    boxes = torch.tensor([a.bbox for a in annotations], dtype=torch.float32)
    iou_mat = compute_iou_matrix(boxes, boxes)

    reported: set[Tuple[int, int]] = set()
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            if (
                iou_mat[i, j].item() > DUPLICATE_IOU_THRESHOLD
                and annotations[i].class_id == annotations[j].class_id
                and (i, j) not in reported
            ):
                reported.add((i, j))
                issues.append(AnnotationIssue(
                    image_id=image_id, issue_type="duplicate_annotation",
                    class_id=annotations[i].class_id,
                    bbox=annotations[i].bbox,
                    details=f"IoU={iou_mat[i,j].item():.3f} with annotation {j}",
                ))
    return issues


# ── Dataset-level QA ──────────────────────────────────────────────────────────

def run_qa(
    all_annotations: Dict[str, List[Detection]],
    class_names: List[str],
    image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
) -> QAReport:
    """
    Run all QA checks on a dataset.

    Args:
        all_annotations: {image_id: [Detection, ...]}
        class_names: ordered list of class name strings
        image_sizes: optional {image_id: (width, height)} for bounds checks

    Returns:
        QAReport with all findings
    """
    report = QAReport()
    report.total_images = len(all_annotations)

    class_counter: Counter = Counter()

    for img_id, anns in all_annotations.items():
        report.total_annotations += len(anns)

        img_w, img_h = (image_sizes or {}).get(img_id, (10_000, 10_000))

        for ann in anns:
            # Class distribution
            cname = class_names[ann.class_id] if 0 <= ann.class_id < len(class_names) else f"class_{ann.class_id}"
            class_counter[cname] += 1

            # Bbox sanity
            issue = _check_bbox_sanity(ann, img_w, img_h, img_id)
            if issue:
                report.add_issue(issue)

            # Small / large object flags (informational)
            area = max(0, ann.bbox[2] - ann.bbox[0]) * max(0, ann.bbox[3] - ann.bbox[1])
            if area < SMALL_AREA_THRESHOLD:
                report.add_issue(AnnotationIssue(
                    image_id=img_id, issue_type="very_small_object",
                    class_id=ann.class_id, bbox=ann.bbox,
                    details=f"area={area:.0f}px²",
                ))
            elif area > LARGE_AREA_THRESHOLD:
                report.add_issue(AnnotationIssue(
                    image_id=img_id, issue_type="very_large_object",
                    class_id=ann.class_id, bbox=ann.bbox,
                    details=f"area={area:.0f}px²",
                ))

        # Duplicate check
        for dup_issue in _check_duplicates(anns, img_id):
            report.add_issue(dup_issue)

    report.class_distribution = dict(class_counter)

    # Class imbalance check
    if class_counter:
        max_count = max(class_counter.values())
        min_count = min(class_counter.values())
        if max_count > 0 and min_count / max_count < 0.1:
            print(
                f"[QA WARNING] Severe class imbalance detected: "
                f"max={max_count} vs min={min_count} "
                f"(ratio={min_count/max_count:.3f})"
            )

    return report


def compute_inter_annotator_agreement(
    annotations_a: Dict[str, List[Detection]],
    annotations_b: Dict[str, List[Detection]],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute a simple inter-annotator agreement score (IoU-based).

    Measures what fraction of annotations from annotator A are matched
    by annotator B at the given IoU threshold.

    Returns:
        agreement ∈ [0, 1]
    """
    total_a = 0
    matched = 0

    all_ids = set(annotations_a.keys()) | set(annotations_b.keys())
    for img_id in all_ids:
        a_anns = annotations_a.get(img_id, [])
        b_anns = annotations_b.get(img_id, [])
        total_a += len(a_anns)

        if not a_anns or not b_anns:
            continue

        boxes_a = torch.tensor([d.bbox for d in a_anns], dtype=torch.float32)
        boxes_b = torch.tensor([d.bbox for d in b_anns], dtype=torch.float32)
        iou_mat = compute_iou_matrix(boxes_a, boxes_b)

        matched_b: set[int] = set()
        for i, ann_a in enumerate(a_anns):
            row = iou_mat[i]
            best_iou, best_j = row.max(0)
            best_j = best_j.item()
            if (
                best_iou.item() >= iou_threshold
                and ann_a.class_id == b_anns[best_j].class_id
                and best_j not in matched_b
            ):
                matched += 1
                matched_b.add(best_j)

    return matched / max(total_a, 1)


def save_qa_report(report: QAReport, output_path: str | Path) -> None:
    """Save QA report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_images":      report.total_images,
        "total_annotations": report.total_annotations,
        "issue_rate":        report.issue_rate(),
        "issue_summary":     report.issue_summary,
        "class_distribution": report.class_distribution,
        "issues": [
            {
                "image_id":  r.image_id,
                "issue_type": r.issue_type,
                "class_id":  r.class_id,
                "bbox":      r.bbox,
                "details":   r.details,
            }
            for r in report.issues
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"QA report saved to {output_path}")
