#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
    parser = argparse.ArgumentParser(description="Summarize VSIBench changes by question type.")
    parser.add_argument("--baseline_json", type=Path, required=True)
    parser.add_argument("--current_json", type=Path, required=True)
    args = parser.parse_args()

    baseline_logs = load_logs(args.baseline_json)
    current_logs = load_logs(args.current_json)
    shared_keys = sorted(set(baseline_logs) & set(current_logs))
    if not shared_keys:
        raise SystemExit("No overlapping VSIBench samples found between baseline and current logs.")

    per_type = defaultdict(Counter)
    metric_name = None

    for key in shared_keys:
        baseline_item = baseline_logs[key]
        current_item = current_logs[key]
        doc = current_item.get("doc", {}) or baseline_item.get("doc", {})
        question_type = doc.get("question_type", "UNKNOWN")

        per_type[question_type]["total"] += 1

        baseline_pred = get_prediction(baseline_item)
        current_pred = get_prediction(current_item)
        if baseline_pred == current_pred:
            per_type[question_type]["same_prediction"] += 1
        else:
            per_type[question_type]["changed_prediction"] += 1

        baseline_score, metric_name_candidate = get_sample_score(baseline_item)
        current_score, _ = get_sample_score(current_item)
        if metric_name is None and metric_name_candidate is not None:
            metric_name = metric_name_candidate

        if baseline_score is None or current_score is None:
            per_type[question_type]["missing_score"] += 1
            continue

        if current_score > baseline_score:
            per_type[question_type]["improved"] += 1
        elif current_score < baseline_score:
            per_type[question_type]["worsened"] += 1
        else:
            per_type[question_type]["equal_score"] += 1

    print("=== VSIBench change summary by question type ===")
    print(f"baseline_json: {args.baseline_json}")
    print(f"current_json : {args.current_json}")
    print(f"shared_samples: {len(shared_keys)}")
    if metric_name is not None:
        print(f"sample_metric: {metric_name}")
    print()
    print("question_type\ttotal\tchanged\timproved\tworsened\tnet\tequal_score")
    for question_type in sorted(per_type.keys()):
        stats = per_type[question_type]
        improved = stats["improved"]
        worsened = stats["worsened"]
        print(
            f"{question_type}\t{stats['total']}\t{stats['changed_prediction']}\t"
            f"{improved}\t{worsened}\t{improved - worsened}\t{stats['equal_score']}"
        )


if __name__ == "__main__":
    main()
