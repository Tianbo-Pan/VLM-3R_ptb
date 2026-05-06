#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch
from PIL import Image, ImageDraw

from .case_io import DEFAULT_MANIFEST_PATH, build_case_records, filter_case_records, load_manifest, sample_case_frame_paths
from .common import ensure_dir, save_json
from .metrics import compute_margin_drop, compute_option_summary, compute_selective_reliance_gap, summarize_boolean_rate
from .model_runner import build_original_video_features, load_model_bundle, option_metrics_to_scores, prepare_video_inputs, score_options_with_video_features
from .perturbation_utils import (
    apply_patch_warp_ablation,
    apply_patch_warp_ablation_to_patch_tokens,
    apply_patch_zero_ablation,
    apply_patch_zero_ablation_to_patch_tokens,
    pack_encoded_video_features,
    repack_from_perturbed_patch_tokens,
    resolve_patch_scores_for_target,
    select_patch_indices_from_scores,
)
from .plotting import save_margin_boxplot, save_selective_gap_histogram


DEFAULT_OUTPUT_ROOT = Path("/local_home/pantianbo/projects/vision_reasoning/VLM-3R/ptb_preliminary_exp/outputs/targeted_vs_random_ablation")


def parse_args():
    parser = argparse.ArgumentParser(description="Preliminary Experiment 2: targeted vs random patch ablation.")
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
    parser.add_argument("--patch-ratio", type=float, default=0.06)
    parser.add_argument("--ablation-mode", type=str, default="zero", choices=["zero", "warp"])
    parser.add_argument("--perturbation-target", type=str, default="encoded", choices=["encoded", "prefusion_3d_patch"])
    parser.add_argument("--warp-shift-size", type=int, default=2)
    parser.add_argument("--warp-mix-ratio", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--save-case-visuals", action="store_true")
    return parser.parse_args()


def patch_index_to_xyxy(patch_idx: int, grid_size: int, width: int, height: int):
    row = int(patch_idx) // grid_size
    col = int(patch_idx) % grid_size
    x1 = int(round(col * width / grid_size))
    y1 = int(round(row * height / grid_size))
    x2 = int(round((col + 1) * width / grid_size))
    y2 = int(round((row + 1) * height / grid_size))
    return x1, y1, x2, y2


def draw_patch_boxes(base_frame_path: Path, selected_indices: np.ndarray, out_path: Path, title: str, grid_size: int, color=(255, 0, 0)):
    image = Image.open(base_frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle([0, 0, min(image.width, 420), 34], fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    for patch_idx in selected_indices.tolist():
        x1, y1, x2, y2 = patch_index_to_xyxy(int(patch_idx), grid_size, width, height)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    image.save(out_path)


def apply_ablation(model, encoded_video_features, aux, selected_indices, args):
    if args.perturbation_target == "encoded":
        if args.ablation_mode == "zero":
            degraded_encoded = apply_patch_zero_ablation(encoded_video_features, selected_indices)
        else:
            degraded_encoded = apply_patch_warp_ablation(
                encoded_video_features,
                selected_indices,
                shift_size=args.warp_shift_size,
                mix_ratio=args.warp_mix_ratio,
            )
        return degraded_encoded, pack_encoded_video_features(model, degraded_encoded)

    if args.perturbation_target == "prefusion_3d_patch":
        patch_tokens = aux.get("branch_features", {}).get("patch_tokens")
        if patch_tokens is None:
            raise ValueError("prefusion_3d_patch perturbation requires patch_tokens from the spatial tower.")
        if args.ablation_mode == "zero":
            perturbed_patch_tokens = apply_patch_zero_ablation_to_patch_tokens(patch_tokens, selected_indices)
        else:
            perturbed_patch_tokens = apply_patch_warp_ablation_to_patch_tokens(
                patch_tokens,
                selected_indices,
                shift_size=args.warp_shift_size,
                mix_ratio=args.warp_mix_ratio,
            )
        return repack_from_perturbed_patch_tokens(model, aux, perturbed_patch_tokens)

    raise ValueError(f"Unsupported perturbation_target: {args.perturbation_target}")


def summarize_case(tokenizer, model, image_processor, record, args):
    case = record.case
    parsed = case.question_entry["parsed_question"]
    option_labels = list((parsed.get("options_map") or {}).keys())
    gt_label = parsed["answer_value"]
    case_dir = ensure_dir(args.output_root / case.scene_name / case.qa_id)
    frame_paths = sample_case_frame_paths(case, args.num_frames)
    video, prompt_prefix, input_ids, attention_mask = prepare_video_inputs(
        model, tokenizer, image_processor, frame_paths, case.prompt, args.conv_mode, args.device
    )
    encoded_video_features, patch_scores, video_features, metadata, aux = build_original_video_features(
        tokenizer,
        model,
        video,
        input_ids,
        attention_mask,
        scoring_mode=args.scoring_mode,
        fusion_2d_weight=args.fusion_2d_weight,
        fusion_3d_weight=args.fusion_3d_weight,
    )
    base_metrics = score_options_with_video_features(tokenizer, model, prompt_prefix, option_labels, video_features)
    base_scores = option_metrics_to_scores(base_metrics, score_key=args.score_key)
    base_summary = compute_option_summary(base_scores, gt_label)
    base_summary.update({"option_scores": base_scores})

    selection_patch_scores = resolve_patch_scores_for_target(
        patch_scores,
        aux,
        perturbation_target=args.perturbation_target,
    )

    random_generator = torch.Generator(device=selection_patch_scores.device)
    random_generator.manual_seed(args.random_seed + record.flat_index)
    targeted_idx, targeted_vals = select_patch_indices_from_scores(selection_patch_scores, args.patch_ratio, mode="targeted")
    random_idx, random_vals = select_patch_indices_from_scores(
        selection_patch_scores,
        args.patch_ratio,
        mode="random",
        generator=random_generator,
    )
    low_idx, low_vals = select_patch_indices_from_scores(selection_patch_scores, args.patch_ratio, mode="low_score")

    conditions = {
        "targeted_ablation": (targeted_idx, targeted_vals),
        "random_ablation": (random_idx, random_vals),
        "low_score_ablation": (low_idx, low_vals),
    }
    result_conditions: Dict[str, Dict] = {"original": base_summary}

    for name, (indices, values) in conditions.items():
        degraded_encoded, degraded_video_features = apply_ablation(model, encoded_video_features, aux, indices, args)
        option_metrics = score_options_with_video_features(tokenizer, model, prompt_prefix, option_labels, degraded_video_features)
        option_scores = option_metrics_to_scores(option_metrics, score_key=args.score_key)
        summary = compute_option_summary(option_scores, gt_label)
        summary.update(
            {
                "option_scores": option_scores,
                "selected_indices": indices.detach().cpu().tolist(),
                "selected_scores": values.detach().cpu().tolist(),
                "margin_drop_vs_original": compute_margin_drop(base_summary, summary),
            }
        )
        result_conditions[name] = summary

    selective_gap = compute_selective_reliance_gap(
        base_summary,
        result_conditions["targeted_ablation"],
        result_conditions["random_ablation"],
    )

    if args.save_case_visuals:
        best_frame_path = case.best_frame_path
        best_frame_rank = 0
        for idx, frame_path in enumerate(frame_paths):
            if frame_path.name == best_frame_path.name:
                best_frame_rank = idx
                break
        target_indices = np.array(result_conditions["targeted_ablation"]["selected_indices"][best_frame_rank], dtype=int)
        random_indices = np.array(result_conditions["random_ablation"]["selected_indices"][best_frame_rank], dtype=int)
        num_tokens = int(selection_patch_scores.shape[1])
        grid_size = int(round(num_tokens ** 0.5))
        draw_patch_boxes(best_frame_path, target_indices, case_dir / "targeted_patches.jpg", "targeted patches", grid_size=grid_size, color=(255, 0, 0))
        draw_patch_boxes(best_frame_path, random_indices, case_dir / "random_patches.jpg", "random patches", grid_size=grid_size, color=(255, 215, 0))

    result = {
        "qa_id": case.qa_id,
        "scene_name": case.scene_name,
        "question_type": parsed.get("question_type"),
        "question_text": parsed.get("question_text"),
        "ground_truth": gt_label,
        "patch_ratio": args.patch_ratio,
        "ablation_mode": args.ablation_mode,
        "perturbation_target": args.perturbation_target,
        "conditions": result_conditions,
        "summary": {
            **selective_gap,
            "targeted_flip": result_conditions["targeted_ablation"]["prediction_label"] != base_summary["prediction_label"],
            "random_flip": result_conditions["random_ablation"]["prediction_label"] != base_summary["prediction_label"],
            "low_score_flip": result_conditions["low_score_ablation"]["prediction_label"] != base_summary["prediction_label"],
        },
    }
    save_json(result, case_dir / "case_result.json")
    return result


def aggregate_results(results: List[Dict]) -> Dict:
    targeted_gt = [x["summary"]["targeted_gt_margin_drop"] for x in results]
    random_gt = [x["summary"]["random_gt_margin_drop"] for x in results]
    low_gt = [x["conditions"]["low_score_ablation"]["margin_drop_vs_original"]["gt_margin_drop"] for x in results]
    srg_gt = [x["summary"]["selective_reliance_gap_gt_margin"] for x in results]
    overall = {
        "num_cases": len(results),
        "mean_targeted_gt_margin_drop": float(np.mean(targeted_gt)) if targeted_gt else 0.0,
        "mean_random_gt_margin_drop": float(np.mean(random_gt)) if random_gt else 0.0,
        "mean_low_score_gt_margin_drop": float(np.mean(low_gt)) if low_gt else 0.0,
        "mean_selective_reliance_gap_gt_margin": float(np.mean(srg_gt)) if srg_gt else 0.0,
        "targeted_flip_rate": summarize_boolean_rate([x["summary"]["targeted_flip"] for x in results]),
        "random_flip_rate": summarize_boolean_rate([x["summary"]["random_flip"] for x in results]),
        "low_score_flip_rate": summarize_boolean_rate([x["summary"]["low_score_flip"] for x in results]),
        "weak_visual_reliance_rate": summarize_boolean_rate([x["summary"]["selective_reliance_gap_gt_margin"] <= 0.0 for x in results]),
    }
    per_type = {}
    qtypes = sorted(set(x["question_type"] for x in results))
    for qtype in qtypes:
        subset = [x for x in results if x["question_type"] == qtype]
        per_type[qtype] = {
            "num_cases": len(subset),
            "mean_targeted_gt_margin_drop": float(np.mean([x["summary"]["targeted_gt_margin_drop"] for x in subset])) if subset else 0.0,
            "mean_random_gt_margin_drop": float(np.mean([x["summary"]["random_gt_margin_drop"] for x in subset])) if subset else 0.0,
            "mean_selective_reliance_gap_gt_margin": float(np.mean([x["summary"]["selective_reliance_gap_gt_margin"] for x in subset])) if subset else 0.0,
        }
    return {"overall": overall, "per_question_type": per_type}


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
        results.append(summarize_case(tokenizer, model, image_processor, record, args))

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
                    "ground_truth": x["ground_truth"],
                    "original_pred": x["conditions"]["original"]["prediction_label"],
                    **x["summary"],
                }
                for x in results
            ],
        },
        args.output_root / "aggregate_targeted_vs_random_ablation.json",
    )
    save_margin_boxplot(
        [x["summary"]["targeted_gt_margin_drop"] for x in results],
        [x["summary"]["random_gt_margin_drop"] for x in results],
        [x["conditions"]["low_score_ablation"]["margin_drop_vs_original"]["gt_margin_drop"] for x in results],
        args.output_root / "gt_margin_drop_boxplot.png",
    )
    save_selective_gap_histogram(
        [x["summary"]["selective_reliance_gap_gt_margin"] for x in results],
        args.output_root / "selective_reliance_gap_histogram.png",
    )
    print(f"Saved aggregate summary to: {args.output_root / 'aggregate_targeted_vs_random_ablation.json'}")


if __name__ == "__main__":
    main()
