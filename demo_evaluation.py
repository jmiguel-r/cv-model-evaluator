import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluator import Detection, evaluate
from failure_analyzer import analyze_dataset, print_summary

print("=" * 60)
print("DEMO: CV Model Evaluator")
print("=" * 60)

predictions = {
    "img_001": [
        Detection(bbox=[10, 10, 100, 100], class_id=0, confidence=0.95, image_id="img_001"),
        Detection(bbox=[150, 150, 250, 250], class_id=1, confidence=0.87, image_id="img_001"),
    ]
}

ground_truths = {
    "img_001": [
        Detection(bbox=[10, 10, 100, 100], class_id=0, image_id="img_001"),
        Detection(bbox=[150, 150, 250, 250], class_id=1, image_id="img_001"),
    ]
}

class_names = ["person", "car"]
result = evaluate(predictions, ground_truths, class_names)

print(f"\nmAP@50:     {result.map50:.3f}")
print(f"Precision:  {result.precision:.3f}")
print(f"Recall:     {result.recall:.3f}")
print(f"F1:         {result.f1:.3f}")

failure_report = analyze_dataset(predictions, ground_truths)
print_summary(failure_report)

from evaluator import save_results
Path("outputs").mkdir(exist_ok=True)
save_results(result, "outputs/demo_eval.json")
print("\n✅ Results saved to outputs/demo_eval.json")
