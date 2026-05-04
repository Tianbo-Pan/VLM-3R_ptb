#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def build_key(item: dict) -> Tuple:
    doc = item.get("doc", {})
    return (
        doc.get("id"),
        doc.get("dataset"),
        doc.get("scene_name"),
        doc.get("question_type"),
        doc.get("question"),
    )


def get_prediction(item: dict):
    doc = item.get("doc", {})
    if "prediction" in doc:
        return doc.get("prediction")
    filtered = item.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return filtered[0]
    return None


def get_sample_score(item: dict) -> Tuple[Optional[float], Optional[str]]:
    candidate_sources = [item.get("vsibench_score"), item.get("doc")]
    scored = []
    for source in candidate_sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            priority = 1 if ("mra" in key.lower() or "accuracy" in key.lower() or "score" in key.lower()) else 0
            scored.append((priority, key, float(value)))
    if not scored:
        return None, None
    scored.sort(reverse=True)
    _, key, value = scored[0]
    return value, key


def load_logs(path: Path) -> Dict[Tuple, dict]:
    data = json.loads(path.read_text())
    return {build_key(item): item for item in data.get("logs", [])}


def main():
    parser = argparse.ArgumentParser(description="Compare two VSIBench lmms-eval JSON logs.")
    parser.add_argument("--baseline_json", type=Path, required=True)
    parser.add_argument("--current_json", type=Path, required=True)
    args = parser.parse_args()

    baseline_logs = load_logs(args.baseline_json)
    current_logs = load_logs(args.current_json)

    shared_keys = sorted(set(baseline_logs) & set(current_logs))
    if not shared_keys:
        raise SystemExit("No overlapping VSIBench samples found between baseline and current logs.")

    same_prediction = 0
    changed_prediction = 0
    improved = 0
    worsened = 0
    equal_score = 0
    changed_and_improved = 0
    changed_and_worsened = 0
    changed_and_equal = 0
    metric_name = None

    for key in shared_keys:
        baseline_item = baseline_logs[key]
        current_item = current_logs[key]

        baseline_pred = get_prediction(baseline_item)
        current_pred = get_prediction(current_item)
        if baseline_pred == current_pred:
            same_prediction += 1
        else:
            changed_prediction += 1

        baseline_score, metric_name_candidate = get_sample_score(baseline_item)
        current_score, _ = get_sample_score(current_item)
        if metric_name is None and metric_name_candidate is not None:
            metric_name = metric_name_candidate

        if baseline_score is None or current_score is None:
            continue

        if current_score > baseline_score:
            improved += 1
            if baseline_pred != current_pred:
                changed_and_improved += 1
        elif current_score < baseline_score:
            worsened += 1
            if baseline_pred != current_pred:
                changed_and_worsened += 1
        else:
            equal_score += 1
            if baseline_pred != current_pred:
                changed_and_equal += 1

    print("=== VSIBench comparison vs baseline ===")
    print(f"baseline_json: {args.baseline_json}")
    print(f"current_json : {args.current_json}")
    print(f"shared_samples: {len(shared_keys)}")
    if metric_name is not None:
        print(f"sample_metric: {metric_name}")
    print(f"same_prediction_count   : {same_prediction}")
    print(f"changed_prediction_count: {changed_prediction}")
    print(f"improved_count          : {improved}")
    print(f"worsened_count          : {worsened}")
    print(f"equal_score_count       : {equal_score}")
    print(f"changed_and_improved    : {changed_and_improved}")
    print(f"changed_and_worsened    : {changed_and_worsened}")
    print(f"changed_and_equal_score : {changed_and_equal}")


if __name__ == "__main__":
    main()
