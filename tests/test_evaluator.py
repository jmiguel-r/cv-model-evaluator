"""
Tests for the CV Model Evaluator core modules.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from annotation_qa import compute_inter_annotator_agreement, run_qa
from evaluator import (
    Detection,
    compute_average_precision,
    compute_iou_matrix,
    evaluate,
    match_detections,
)
from failure_analyzer import analyze_image

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_det(bbox, class_id=0, confidence=1.0, image_id="img_001"):
    return Detection(bbox=bbox, class_id=class_id, confidence=confidence, image_id=image_id)


# ── evaluator.py ──────────────────────────────────────────────────────────────

class TestComputeIouMatrix:
    def test_perfect_overlap(self):
        box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        iou = compute_iou_matrix(box, box)
        assert iou.shape == (1, 1)
        assert abs(iou[0, 0].item() - 1.0) < 1e-5

    def test_no_overlap(self):
        pred = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        gt   = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        iou  = compute_iou_matrix(pred, gt)
        assert iou[0, 0].item() == 0.0

    def test_partial_overlap(self):
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gt   = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
        iou  = compute_iou_matrix(pred, gt)
        # intersection = 5×10=50; union = 10×10 + 10×10 - 50 = 150
        assert abs(iou[0, 0].item() - 50 / 150) < 1e-4

    def test_empty_predictions(self):
        pred = torch.zeros((0, 4))
        gt   = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        iou  = compute_iou_matrix(pred, gt)
        assert iou.shape == (0, 1)

    def test_multiple_boxes(self):
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        gt   = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        iou  = compute_iou_matrix(pred, gt)
        assert abs(iou[0, 0].item() - 1.0) < 1e-5
        assert iou[1, 0].item() == 0.0


class TestMatchDetections:
    def test_perfect_tp(self):
        pred = [make_det([0, 0, 10, 10], confidence=0.9)]
        gt   = [make_det([0, 0, 10, 10])]
        tp, fn = match_detections(pred, gt, iou_threshold=0.5)
        assert tp == [True]
        assert fn == [False]

    def test_missed_detection(self):
        pred = []
        gt   = [make_det([0, 0, 10, 10])]
        tp, fn = match_detections(pred, gt)
        assert fn == [True]

    def test_false_positive(self):
        pred = [make_det([50, 50, 60, 60])]
        gt   = [make_det([0, 0, 10, 10])]
        tp, fn = match_detections(pred, gt)
        assert tp == [False]
        assert fn == [True]

    def test_class_mismatch(self):
        pred = [make_det([0, 0, 10, 10], class_id=1)]
        gt   = [make_det([0, 0, 10, 10], class_id=0)]
        tp, fn = match_detections(pred, gt, iou_threshold=0.5)
        assert tp == [False]

    def test_duplicate_only_first_is_tp(self):
        pred = [
            make_det([0, 0, 10, 10], confidence=0.9),
            make_det([0, 0, 10, 10], confidence=0.8),
        ]
        gt = [make_det([0, 0, 10, 10])]
        tp, fn = match_detections(pred, gt)
        assert tp.count(True) == 1
        assert tp.count(False) == 1


class TestComputeAP:
    def test_perfect_ap(self):
        tp_list     = [True, True, True]
        confidences = [0.9, 0.8, 0.7]
        ap = compute_average_precision(tp_list, confidences, n_gt=3)
        assert ap > 0.85

    def test_zero_ap_all_fp(self):
        tp_list     = [False, False, False]
        confidences = [0.9, 0.8, 0.7]
        ap = compute_average_precision(tp_list, confidences, n_gt=3)
        assert ap == 0.0

    def test_no_gt(self):
        ap = compute_average_precision([], [], n_gt=0)
        assert ap == 0.0


class TestEvaluate:
    def test_perfect_predictions(self):
        preds = {"img1": [make_det([0, 0, 10, 10], confidence=0.9, image_id="img1")]}
        gts   = {"img1": [make_det([0, 0, 10, 10], image_id="img1")]}
        result = evaluate(preds, gts, class_names=["cat"])
        assert result.map50 > 0.85
        assert result.recall > 0.99

    def test_no_predictions(self):
        preds  = {}
        gts    = {"img1": [make_det([0, 0, 10, 10], image_id="img1")]}
        result = evaluate(preds, gts, class_names=["cat"])
        assert result.map50 == 0.0
        assert result.recall == 0.0

    def test_multiple_classes(self):
        preds = {
            "img1": [
                make_det([0, 0, 10, 10], class_id=0, confidence=0.9, image_id="img1"),
                make_det([20, 20, 30, 30], class_id=1, confidence=0.8, image_id="img1"),
            ]
        }
        gts = {
            "img1": [
                make_det([0, 0, 10, 10], class_id=0, image_id="img1"),
                make_det([20, 20, 30, 30], class_id=1, image_id="img1"),
            ]
        }
        result = evaluate(preds, gts, class_names=["cat", "dog"])
        assert "cat" in result.per_class_ap
        assert "dog" in result.per_class_ap
        assert result.per_class_ap["cat"] > 0.85


# ── failure_analyzer.py ───────────────────────────────────────────────────────

class TestAnalyzeImage:
    def test_true_positive_no_failures(self):
        pred = [make_det([0, 0, 10, 10], confidence=0.9)]
        gt   = [make_det([0, 0, 10, 10])]
        failures = analyze_image("img1", pred, gt)
        assert all(f.failure_type not in ("false_positive_bg", "missed_detection") for f in failures)

    def test_class_confusion_detected(self):
        pred = [make_det([0, 0, 10, 10], class_id=1, confidence=0.9)]
        gt   = [make_det([0, 0, 10, 10], class_id=0)]
        failures = analyze_image("img1", pred, gt)
        types = [f.failure_type for f in failures]
        assert "class_confusion" in types

    def test_missed_small_object(self):
        pred = []
        gt   = [make_det([0, 0, 5, 5])]   # area = 25px² < 32×32
        failures = analyze_image("img1", pred, gt)
        types = [f.failure_type for f in failures]
        assert "missed_small" in types

    def test_false_positive_bg(self):
        pred = [make_det([100, 100, 120, 120], confidence=0.9)]
        gt   = [make_det([0, 0, 10, 10])]
        failures = analyze_image("img1", pred, gt)
        types = [f.failure_type for f in failures]
        assert "false_positive_bg" in types

    def test_localization_error(self):
        pred = [make_det([3, 0, 18, 10], class_id=0, confidence=0.9)]  # IoU ~0.47, localization error
        gt   = [make_det([0, 0, 10, 10], class_id=0)]
        failures = analyze_image("img1", pred, gt)
        types = [f.failure_type for f in failures]
        assert "localization_error" in types


# ── annotation_qa.py ──────────────────────────────────────────────────────────

class TestAnnotationQA:
    def test_valid_annotations_no_issues(self):
        anns = {"img1": [make_det([10, 10, 100, 100])]}
        report = run_qa(anns, class_names=["cat"], image_sizes={"img1": (640, 480)})
        critical = [i for i in report.issues if i.issue_type in ("zero_or_negative_area", "out_of_bounds")]
        assert len(critical) == 0

    def test_out_of_bounds_detected(self):
        anns = {"img1": [make_det([0, 0, 700, 700])]}
        report = run_qa(anns, class_names=["cat"], image_sizes={"img1": (640, 480)})
        types = [i.issue_type for i in report.issues]
        assert "out_of_bounds" in types

    def test_zero_area_detected(self):
        anns = {"img1": [make_det([10, 10, 10, 10])]}
        report = run_qa(anns, class_names=["cat"], image_sizes={"img1": (640, 480)})
        types = [i.issue_type for i in report.issues]
        assert "zero_or_negative_area" in types

    def test_duplicate_detected(self):
        anns = {"img1": [
            make_det([10, 10, 100, 100], class_id=0),
            make_det([11, 11, 101, 101], class_id=0),  # near-identical → duplicate
        ]}
        report = run_qa(anns, class_names=["cat"])
        types = [i.issue_type for i in report.issues]
        assert "duplicate_annotation" in types

    def test_class_distribution(self):
        anns = {
            "img1": [make_det([0, 0, 100, 100], class_id=0)],
            "img2": [make_det([0, 0, 100, 100], class_id=1)],
        }
        report = run_qa(anns, class_names=["cat", "dog"])
        assert report.class_distribution.get("cat", 0) == 1
        assert report.class_distribution.get("dog", 0) == 1


class TestInterAnnotatorAgreement:
    def test_perfect_agreement(self):
        anns = {"img1": [make_det([0, 0, 100, 100])]}
        score = compute_inter_annotator_agreement(anns, anns)
        assert score > 0.99

    def test_zero_agreement(self):
        a = {"img1": [make_det([0, 0, 10, 10])]}
        b = {"img1": [make_det([500, 500, 600, 600])]}
        score = compute_inter_annotator_agreement(a, b)
        assert score == 0.0

    def test_partial_agreement(self):
        a = {"img1": [make_det([0, 0, 10, 10]), make_det([100, 100, 200, 200])]}
        b = {"img1": [make_det([0, 0, 10, 10])]}   # only matches first
        score = compute_inter_annotator_agreement(a, b)
        assert 0.4 < score < 0.6
