"""
benchmark.py
------------
Compare multiple CV models on the same dataset.
Runs evaluation + failure analysis for each model and produces
a side-by-side comparison table and per-model reports.

Supported model backends (via a simple adapter interface):
  - Ultralytics YOLOv8 / YOLO11
  - HuggingFace transformers (DETR, RT-DETR, ViT-Det)
  - Any custom callable that returns List[Detection] per image
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from evaluator import Detection, EvaluationResult, evaluate, save_results
from failure_analyzer import FailureReport, analyze_dataset

# ── Model adapter protocol ────────────────────────────────────────────────────

ModelFn = Callable[[str], List[Detection]]
"""A callable that takes an image path and returns a list of Detections."""


def make_ultralytics_adapter(model_name: str, conf: float = 0.25) -> ModelFn:
    """
    Build a ModelFn adapter for any Ultralytics YOLO model.

    Args:
        model_name: e.g. 'yolov8n.pt', 'yolo11s.pt'
        conf: confidence threshold for filtering predictions

    Returns:
        Callable(image_path) → List[Detection]
    """
    from ultralytics import YOLO  # type: ignore
    model = YOLO(model_name)

    def _predict(image_path: str) -> List[Detection]:
        results = model(image_path, conf=conf, verbose=False)
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy  = box.xyxy[0].tolist()
                cls   = int(box.cls[0].item())
                score = float(box.conf[0].item())
                detections.append(Detection(
                    bbox=xyxy, class_id=cls, confidence=score, image_id=image_path
                ))
        return detections

    return _predict


def make_hf_detr_adapter(model_id: str = "facebook/detr-resnet-50", threshold: float = 0.5) -> ModelFn:
    """
    Build a ModelFn adapter for HuggingFace DETR models.

    Args:
        model_id: HuggingFace model identifier
        threshold: score threshold

    Returns:
        Callable(image_path) → List[Detection]
    """
    from PIL import Image  # type: ignore
    from transformers import AutoImageProcessor, AutoModelForObjectDetection  # type: ignore

    processor = AutoImageProcessor.from_pretrained(model_id)
    model     = AutoModelForObjectDetection.from_pretrained(model_id)
    model.eval()

    def _predict(image_path: str) -> List[Detection]:
        image  = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]])
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        detections: List[Detection] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append(Detection(
                bbox=box.tolist(),
                class_id=int(label.item()),
                confidence=float(score.item()),
                image_id=image_path,
            ))
        return detections

    return _predict


# ── Benchmark runner ──────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model_name:     str
    eval_result:    EvaluationResult
    failure_report: FailureReport
    inference_time_ms: float   # avg ms / image


@dataclass
class BenchmarkReport:
    model_results: List[ModelResult] = field(default_factory=list)

    def summary_table(self) -> str:
        """Return a markdown table comparing all models."""
        header = (
            "| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 | "
            "Avg Latency (ms) | FP (bg) | Missed (small) |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        rows = []
        for mr in self.model_results:
            e = mr.eval_result
            fr = mr.failure_report
            fp_bg    = fr.summary.get("false_positive_bg", 0)
            missed_s = fr.summary.get("missed_small", 0)
            rows.append(
                f"| {mr.model_name} "
                f"| {e.map50:.3f} "
                f"| {e.map50_95:.3f} "
                f"| {e.precision:.3f} "
                f"| {e.recall:.3f} "
                f"| {e.f1:.3f} "
                f"| {mr.inference_time_ms:.1f} "
                f"| {fp_bg} "
                f"| {missed_s} |"
            )
        return header + "\n".join(rows)

    def best_model(self, metric: str = "map50") -> Optional[ModelResult]:
        if not self.model_results:
            return None
        return max(
            self.model_results,
            key=lambda mr: getattr(mr.eval_result, metric, 0.0),
        )


def run_benchmark(
    models:            Dict[str, ModelFn],
    image_paths:       List[str],
    ground_truths:     Dict[str, List[Detection]],
    class_names:       List[str],
    output_dir:        str | Path = "outputs/benchmark",
    iou_threshold:     float = 0.5,
) -> BenchmarkReport:
    """
    Run all models on the same image set and compare results.

    Args:
        models:        {model_name: ModelFn}
        image_paths:   list of absolute image paths
        ground_truths: {image_path: [Detection, ...]}
        class_names:   ordered class name list
        output_dir:    where to save per-model JSON reports
        iou_threshold: matching IoU threshold

    Returns:
        BenchmarkReport with per-model results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = BenchmarkReport()

    for model_name, predict_fn in models.items():
        print(f"\n[Benchmark] Running: {model_name}")

        all_predictions: Dict[str, List[Detection]] = {}
        total_time = 0.0

        for img_path in image_paths:
            t0 = time.perf_counter()
            preds = predict_fn(img_path)
            total_time += (time.perf_counter() - t0) * 1000  # → ms

            for d in preds:
                d.image_id = img_path
            all_predictions[img_path] = preds

        avg_ms = total_time / max(len(image_paths), 1)
        print(f"  Inference: {avg_ms:.1f} ms/image")

        eval_result    = evaluate(all_predictions, ground_truths, class_names)
        failure_report = analyze_dataset(all_predictions, ground_truths, iou_threshold)

        print(f"  mAP@50={eval_result.map50:.3f}  mAP@50:95={eval_result.map50_95:.3f}")

        # Save JSON
        save_results(eval_result, output_dir / f"{model_name}_eval.json")

        report.model_results.append(ModelResult(
            model_name=model_name,
            eval_result=eval_result,
            failure_report=failure_report,
            inference_time_ms=avg_ms,
        ))

    # Print summary table
    print("\n── Benchmark Summary ─────────────────────────────────────────")
    print(report.summary_table())

    best = report.best_model("map50")
    if best:
        print(f"\nBest model by mAP@50: {best.model_name} ({best.eval_result.map50:.3f})")

    return report
