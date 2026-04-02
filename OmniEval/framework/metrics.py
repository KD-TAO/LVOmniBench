# OmniEval/framework/metrics.py
from collections import defaultdict
from typing import List, Dict

def calculate_accuracy_metrics(results: List[Dict]) -> Dict:
    """
    Calculates overall and fine-grained accuracy metrics from evaluation results.
    """
    valid_results = [r for r in results if not r.get("skipped", False) and r.get("model_prediction") is not None]
    skipped_results = [r for r in results if r.get("skipped", False) or r.get("model_prediction") is None]
    
    total_samples = len(valid_results)
    correct_predictions = sum(1 for r in valid_results if r.get("is_correct", False))
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    metrics = {
        "overall_accuracy": overall_accuracy,
        "total_correct": correct_predictions,
        "total_samples": total_samples,
        "total_skipped": len(skipped_results),
        "fine_grained": {}
    }

    # Define the dimensions we want to analyze
    dimensions = ["question_type", "audio_type", "difficulty", "video_category"]
    
    for dim in dimensions:
        dim_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for result in valid_results:
            category = result.get(dim, "unknown")
            dim_stats[category]["total"] += 1
            if result.get("is_correct", False):
                dim_stats[category]["correct"] += 1
                
        # Calculate percentages
        dim_accuracy = {}
        for category, stats in dim_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            dim_accuracy[category] = {
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"]
            }
        metrics["fine_grained"][dim] = dim_accuracy

    return metrics