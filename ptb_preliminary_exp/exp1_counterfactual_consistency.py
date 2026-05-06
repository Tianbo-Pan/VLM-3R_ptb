#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import torch

from .case_io import DEFAULT_MANIFEST_PATH, build_case_records, filter_case_records, find_mismatched_case, load_manifest, sample_case_frame_paths
from .common import ensure_dir, save_json
from .metrics import compute_option_summary, summarize_boolean_rate
from .model_runner import build_original_video_features, load_model_bundle, option_metrics_to_scores, prepare_video_inputs, score_options_with_video_features
from .perturbation_utils import pack_encoded_video_features
from .plotting import save_counterfactual_barplot


DEFAULT_OUTPUT_ROOT = Path("/local_home/pantianbo/projects/vision_reasoning/VLM-3R/ptb_preliminary_exp/outputs/counterfactual_consistency")


def parse_args():
    parser = argparse.ArgumentParser(description="Preliminary Experiment 1: counterfactual consistency on ptb_test cases.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--qa-id", type=str, default=None)
    parser.add_argument("--scene-name", type=str, default=None)
    parser.add_argument("--flat-index", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--mm-spatial-pool-mode", type=str, default="average")
    parser.add_argument("--mm-newline-position", type=str, default="grid")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--score-key", type=str, default="avg_logprob", choices=["avg_logprob", "sequence_logprob", "first_token_logprob"])
    parser.add_argument("--scoring-mode", type=str, default="fusion_2d3d")
    parser.add_argument("--fusion-2d-weight", type=float, default=1.0)
    parser.add_argument("--fusion-3d-weight", type=float, default=1.0)
    parser.add_argument("--no-vision-mode", type=str, default="zero", choices=["zero", "frame_mean"])
    return parser.parse_args()


def choose_mismatched_frames(reference_record, all_records, num_frames: int) -> List[Path]:
    mismatched = find_mismatched_case(reference_record, all_records)
    if mismatched is None:
        raise RuntimeError("Failed to find a mismatched case for counterfactual evaluation.")
    return sample_case_frame_paths(mismatched.case, num_frames)


def frame_shuffle_paths(frame_paths: List[Path]) -> List[Path]:
    ordered = list(frame_paths)
    if len(ordered) <= 1:
        return ordered
    return list(reversed(ordered))


def summarize_case(
    tokenizer,
    model,
    image_processor,
    record,
    args,
    all_records,
):
    case = record.case
    parsed = case.question_entry["parsed_question"]
    option_labels = list((parsed.get("options_map") or {}).keys())
    gt_label = parsed["answer_value"]
    case_dir = ensure_dir(args.output_root / case.scene_name / case.qa_id)

    original_frames = sample_case_frame_paths(case, args.num_frames)
    mismatched_frames = choose_mismatched_frames(record, all_records, args.num_frames)
    shuffled_frames = frame_shuffle_paths(original_frames)
    condition_frames = {
        "original": original_frames,
        "mismatched_video": mismatched_frames,
        "frame_shuffle": shuffled_frames,
    }

    condition_results: Dict[str, Dict] = {}
    reference_encoded = None
    reference_scores = None
    prompt_prefix = None
    for condition_name, frame_paths in condition_frames.items():
        video, prompt_prefix, input_ids, attention_mask = prepare_video_inputs(
            model, tokenizer, image_processor, frame_paths, case.prompt, args.conv_mode, args.device
        )
        encoded_video_features, patch_scores, video_features, _, _ = build_original_video_features(
            tokenizer,
            model,
            video,
            input_ids,
            attention_mask,
            scoring_mode=args.scoring_mode,
            fusion_2d_weight=args.fusion_2d_weight,
            fusion_3d_weight=args.fusion_3d_weight,
        )
        option_metrics = score_options_with_video_features(tokenizer, model, prompt_prefix, option_labels, video_features)
        option_scores = option_metrics_to_scores(option_metrics, score_key=args.score_key)
        summary = compute_option_summary(option_scores, gt_label)
        summary.update(
            {
                "condition": condition_name,
                "option_scores": option_scores,
                "frame_paths": [str(x) for x in frame_paths],
            }
        )
        condition_results[condition_name] = summary
        if condition_name == "original":
            reference_encoded = encoded_video_features
            reference_scores = patch_scores

    assert reference_encoded is not None
    if args.no_vision_mode == "zero":
        no_vision_encoded = torch.zeros_like(reference_encoded)
    else:
        no_vision_encoded = reference_encoded.mean(dim=1, keepdim=True).expand_as(reference_encoded).clone()
    no_vision_video_features = pack_encoded_video_features(model, no_vision_encoded)
    option_metrics = score_options_with_video_features(tokenizer, model, prompt_prefix, option_labels, no_vision_video_features)
    option_scores = option_metrics_to_scores(option_metrics, score_key=args.score_key)
    no_vision_summary = compute_option_summary(option_scores, gt_label)
    no_vision_summary.update({"condition": "no_vision", "option_scores": option_scores, "frame_paths": []})
    condition_results["no_vision"] = no_vision_summary

    original_pred = condition_results["original"]["prediction_label"]
    consistency = {
        "same_as_original_no_vision": condition_results["no_vision"]["prediction_label"] == original_pred,
        "same_as_original_mismatched_video": condition_results["mismatched_video"]["prediction_label"] == original_pred,
        "same_as_original_frame_shuffle": condition_results["frame_shuffle"]["prediction_label"] == original_pred,
        "same_as_original_all_counterfactuals": all(
            condition_results[key]["prediction_label"] == original_pred
            for key in ["no_vision", "mismatched_video", "frame_shuffle"]
        ),
    }

    result = {
        "qa_id": case.qa_id,
        "scene_name": case.scene_name,
        "question_type": parsed.get("question_type"),
        "question_text": parsed.get("question_text"),
        "options_map": parsed.get("options_map"),
        "ground_truth": gt_label,
        "conditions": condition_results,
        "consistency": consistency,
    }
    save_json(result, case_dir / "case_result.json")
    return result


def aggregate_results(results: List[Dict]) -> Dict:
    keys = [
        "same_as_original_no_vision",
        "same_as_original_mismatched_video",
        "same_as_original_frame_shuffle",
        "same_as_original_all_counterfactuals",
    ]
    overall = {key: summarize_boolean_rate([x["consistency"][key] for x in results]) for key in keys}
    per_type: Dict[str, Dict] = {}
    qtypes = sorted(set(x["question_type"] for x in results))
    for qtype in qtypes:
        subset = [x for x in results if x["question_type"] == qtype]
        per_type[qtype] = {key: summarize_boolean_rate([x["consistency"][key] for x in subset]) for key in keys}
    hallucination_candidates = [
        x for x in results
        if x["consistency"]["same_as_original_no_vision"] or x["consistency"]["same_as_original_mismatched_video"]
    ]
    return {
        "num_cases": len(results),
        "overall": overall,
        "per_question_type": per_type,
        "hallucination_candidate_rate": float(len(hallucination_candidates) / len(results)) if results else 0.0,
    }


def main():
    args = parse_args()
    ensure_dir(args.output_root)
    manifest = load_manifest(args.manifest_path)
    all_records = build_case_records(manifest, require_options=True)
    selected_records = filter_case_records(
        all_records,
        qa_id=args.qa_id,
        scene_name=args.scene_name,
        flat_index=args.flat_index,
        max_cases=args.max_cases,
    )
    tokenizer, model, image_processor = load_model_bundle(args)

    results = []
    for idx, record in enumerate(selected_records):
        print(f"[{idx + 1}/{len(selected_records)}] {record.case.scene_name} / {record.case.qa_id}")
        results.append(summarize_case(tokenizer, model, image_processor, record, args, all_records))

    aggregate = aggregate_results(results)
    save_json(
        {
            "config": vars(args),
            "aggregate": aggregate,
            "case_summaries": [
                {
                    "qa_id": x["qa_id"],
                    "scene_name": x["scene_name"],
                    "question_type": x["question_type"],
                    "original_pred": x["conditions"]["original"]["prediction_label"],
                    "gt": x["ground_truth"],
                    **x["consistency"],
                }
                for x in results
            ],
        },
        args.output_root / "aggregate_counterfactual_consistency.json",
    )
    save_counterfactual_barplot(
        {
            "no_vision": aggregate["overall"]["same_as_original_no_vision"]["rate"],
            "mismatched": aggregate["overall"]["same_as_original_mismatched_video"]["rate"],
            "shuffle": aggregate["overall"]["same_as_original_frame_shuffle"]["rate"],
        },
        args.output_root / "counterfactual_consistency_overall.png",
    )
    print(f"Saved aggregate summary to: {args.output_root / 'aggregate_counterfactual_consistency.json'}")


if __name__ == "__main__":
    main()
