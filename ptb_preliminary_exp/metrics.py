from __future__ import annotations

from typing import Dict, List


def compute_option_summary(option_scores: Dict[str, float], ground_truth: str) -> Dict:
    ranked = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
    pred_label, pred_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else pred_score
    gt_score = float(option_scores[ground_truth])
    best_other_score = max(float(v) for k, v in option_scores.items() if k != ground_truth)
    return {
        "prediction_label": pred_label,
        "prediction_score": float(pred_score),
        "ground_truth": ground_truth,
        "ground_truth_score": gt_score,
        "is_correct": pred_label == ground_truth,
        "prediction_margin": float(pred_score - second_score),
        "gt_margin": float(gt_score - best_other_score),
        "ranked_options": [{"label": k, "score": float(v)} for k, v in ranked],
    }


def compute_margin_drop(reference: Dict, other: Dict) -> Dict[str, float]:
    return {
        "prediction_margin_drop": float(reference["prediction_margin"] - other["prediction_margin"]),
        "gt_margin_drop": float(reference["gt_margin"] - other["gt_margin"]),
        "prediction_score_drop": float(reference["prediction_score"] - other["prediction_score"]),
        "gt_score_drop": float(reference["ground_truth_score"] - other["ground_truth_score"]),
    }


def compute_margin_gain(reference: Dict, other: Dict) -> Dict[str, float]:
    return {
        "prediction_margin_gain": float(other["prediction_margin"] - reference["prediction_margin"]),
        "gt_margin_gain": float(other["gt_margin"] - reference["gt_margin"]),
        "prediction_score_gain": float(other["prediction_score"] - reference["prediction_score"]),
        "gt_score_gain": float(other["ground_truth_score"] - reference["ground_truth_score"]),
    }


def compute_selective_reliance_gap(reference: Dict, targeted: Dict, random_cond: Dict) -> Dict[str, float]:
    targeted_drop = compute_margin_drop(reference, targeted)
    random_drop = compute_margin_drop(reference, random_cond)
    return {
        "targeted_prediction_margin_drop": targeted_drop["prediction_margin_drop"],
        "random_prediction_margin_drop": random_drop["prediction_margin_drop"],
        "targeted_gt_margin_drop": targeted_drop["gt_margin_drop"],
        "random_gt_margin_drop": random_drop["gt_margin_drop"],
        "selective_reliance_gap_prediction_margin": float(
            targeted_drop["prediction_margin_drop"] - random_drop["prediction_margin_drop"]
        ),
        "selective_reliance_gap_gt_margin": float(
            targeted_drop["gt_margin_drop"] - random_drop["gt_margin_drop"]
        ),
    }


def compute_selective_evidence_gain(reference: Dict, targeted: Dict, random_cond: Dict) -> Dict[str, float]:
    targeted_gain = compute_margin_gain(reference, targeted)
    random_gain = compute_margin_gain(reference, random_cond)
    return {
        "targeted_prediction_margin_gain": targeted_gain["prediction_margin_gain"],
        "random_prediction_margin_gain": random_gain["prediction_margin_gain"],
        "targeted_gt_margin_gain": targeted_gain["gt_margin_gain"],
        "random_gt_margin_gain": random_gain["gt_margin_gain"],
        "selective_evidence_gain_prediction_margin": float(
            targeted_gain["prediction_margin_gain"] - random_gain["prediction_margin_gain"]
        ),
        "selective_evidence_gain_gt_margin": float(
            targeted_gain["gt_margin_gain"] - random_gain["gt_margin_gain"]
        ),
    }


def summarize_boolean_rate(items: List[bool]) -> Dict[str, float]:
    total = len(items)
    positive = int(sum(bool(x) for x in items))
    return {
        "count": total,
        "positive": positive,
        "rate": float(positive / total) if total > 0 else 0.0,
    }
