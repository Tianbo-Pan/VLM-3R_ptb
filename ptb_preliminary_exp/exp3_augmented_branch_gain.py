#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import math
from typing import Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import numpy as np
import torch

from .case_io import DEFAULT_MANIFEST_PATH, build_case_records, filter_case_records, load_manifest, sample_case_frame_paths
from .common import ensure_dir, save_json
from .metrics import compute_margin_gain, compute_option_summary, compute_selective_evidence_gain, summarize_boolean_rate
from .model_runner import build_original_video_features, load_model_bundle, option_metrics_to_scores, prepare_video_inputs, score_options_with_video_features
from .perturbation_utils import select_patch_indices_from_scores
from ptb_patch_selection.selective_pooling import (
    pack_already_pooled_tokens,
    pack_coarse_tokens,
    pack_contextual_coarse_tokens,
    pack_fine_selected_tokens,
)


DEFAULT_OUTPUT_ROOT = Path("/local_home/pantianbo/projects/vision_reasoning/VLM-3R/ptb_preliminary_exp/outputs/augmented_branch_gain")


def parse_args():
    parser = argparse.ArgumentParser(description="Preliminary Experiment 3: augmented branch gain with selected fine tokens.")
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
    parser.add_argument("--scoring-mode", type=str, default="question_cosine")
    parser.add_argument("--fusion-2d-weight", type=float, default=1.0)
    parser.add_argument("--fusion-3d-weight", type=float, default=1.0)
    parser.add_argument("--patch-ratio", type=float, default=0.3)
    parser.add_argument("--selection-scope", type=str, default="per_frame", choices=["per_frame", "global"])
    parser.add_argument("--fine-scale", type=float, default=1.0)
    parser.add_argument(
        "--injection-mode",
        type=str,
        default="tail_append",
        choices=[
            "tail_append",
            "fine_first",
            "coarse_replace",
            "coarse_replace_tail",
            "coarse_residual",
            "coarse_residual_tail",
            "contextual_coarse_tail",
            "contextual_coarse_tail_fine",
            "inplace_boost_coarse",
            "inplace_boost_coarse_tail",
        ],
    )
    parser.add_argument("--append-newline", action="store_true")
    parser.add_argument("--exclude-coarse", action="store_true")
    parser.add_argument("--coarse-alpha", type=float, default=0.5)
    parser.add_argument("--coarse-agg-mode", type=str, default="score_weighted", choices=["mean", "score_weighted"])
    parser.add_argument("--preserve-coarse-norm", action="store_true")
    parser.add_argument("--context-radius", type=int, default=0)
    parser.add_argument("--context-topk", type=int, default=4)
    parser.add_argument("--context-scale", type=float, default=1.0)
    parser.add_argument("--boost-factor", type=float, default=1.5)
    parser.add_argument("--background-decay", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def _infer_square_grid_size(num_tokens: int) -> int:
    grid_size = int(round(math.sqrt(num_tokens)))
    if grid_size * grid_size != int(num_tokens):
        raise ValueError(f"Expected square token grid, got num_tokens={num_tokens}.")
    return grid_size


def build_augmented_coarse_frame_features(
    model,
    encoded_video_features: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_scores: torch.Tensor,
    coarse_alpha: float,
    coarse_agg_mode: str,
    preserve_coarse_norm: bool,
    residual_mode: bool,
) -> Tuple[torch.Tensor, Dict]:
    coarse_frame_features = model.get_2dPool(encoded_video_features)
    fine_grid_size = _infer_square_grid_size(encoded_video_features.shape[1])
    coarse_grid_size = _infer_square_grid_size(coarse_frame_features.shape[1])

    aggregated = torch.zeros_like(coarse_frame_features)
    weights = torch.zeros(
        coarse_frame_features.shape[:2],
        dtype=coarse_frame_features.dtype,
        device=coarse_frame_features.device,
    )

    for frame_idx in range(encoded_video_features.shape[0]):
        frame_indices = selected_indices[frame_idx]
        frame_scores = selected_scores[frame_idx]
        frame_indices = frame_indices[frame_indices >= 0]
        if frame_indices.numel() == 0:
            continue
        valid_count = frame_indices.numel()
        frame_scores = frame_scores[:valid_count]
        if coarse_agg_mode == "score_weighted":
            token_weights = torch.softmax(frame_scores.float(), dim=0).to(dtype=coarse_frame_features.dtype)
        else:
            token_weights = torch.full(
                (valid_count,),
                1.0 / max(valid_count, 1),
                dtype=coarse_frame_features.dtype,
                device=coarse_frame_features.device,
            )
        for local_rank, fine_idx_tensor in enumerate(frame_indices):
            fine_idx = int(fine_idx_tensor.item())
            fine_row = fine_idx // fine_grid_size
            fine_col = fine_idx % fine_grid_size
            coarse_row = min(coarse_grid_size - 1, int(math.floor((fine_row / max(fine_grid_size, 1)) * coarse_grid_size)))
            coarse_col = min(coarse_grid_size - 1, int(math.floor((fine_col / max(fine_grid_size, 1)) * coarse_grid_size)))
            coarse_idx = coarse_row * coarse_grid_size + coarse_col
            weight = token_weights[local_rank]
            aggregated[frame_idx, coarse_idx] += weight * encoded_video_features[frame_idx, fine_idx]
            weights[frame_idx, coarse_idx] += weight

    mask = weights > 0
    if mask.any():
        normalized_aggregated = aggregated.clone()
        normalized_aggregated[mask] = normalized_aggregated[mask] / weights[mask].unsqueeze(-1).clamp_min(1e-6)
    else:
        normalized_aggregated = aggregated

    augmented = coarse_frame_features.clone()
    if mask.any():
        original = coarse_frame_features[mask]
        evidence = normalized_aggregated[mask]
        if residual_mode:
            updated = original + float(coarse_alpha) * evidence
        else:
            updated = (1.0 - float(coarse_alpha)) * original + float(coarse_alpha) * evidence
        if preserve_coarse_norm:
            original_norm = original.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            updated_norm = updated.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            updated = updated * (original_norm / updated_norm)
        augmented[mask] = updated

    metadata = {
        "coarse_alpha": float(coarse_alpha),
        "coarse_agg_mode": coarse_agg_mode,
        "preserve_coarse_norm": bool(preserve_coarse_norm),
        "augmented_coarse_positions": int(mask.sum().item()),
    }
    return augmented, metadata


def build_augmented_video_features(
    model,
    encoded_video_features: torch.Tensor,
    patch_scores: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_scores: torch.Tensor,
    include_coarse: bool,
    append_newline: bool,
    fine_scale: float,
    injection_mode: str,
    coarse_alpha: float,
    coarse_agg_mode: str,
    preserve_coarse_norm: bool,
    context_radius: int,
    context_topk: int,
    context_scale: float,
    boost_factor: float,
    background_decay: float,
) -> torch.Tensor:
    coarse_tokens = pack_coarse_tokens(model, encoded_video_features) if include_coarse else None
    fine_tokens = pack_fine_selected_tokens(
        model,
        encoded_video_features,
        selected_indices,
        append_newline=append_newline,
        fine_scale=fine_scale,
    )
    if injection_mode in {"coarse_replace", "coarse_replace_tail", "coarse_residual", "coarse_residual_tail"}:
        if not include_coarse:
            raise ValueError(f"{injection_mode} requires coarse tokens.")
        coarse_frame_features, _ = build_augmented_coarse_frame_features(
            model,
            encoded_video_features,
            selected_indices,
            selected_scores,
            coarse_alpha=coarse_alpha,
            coarse_agg_mode=coarse_agg_mode,
            preserve_coarse_norm=preserve_coarse_norm,
            residual_mode=injection_mode in {"coarse_residual", "coarse_residual_tail"},
        )
        coarse_tokens = pack_already_pooled_tokens(model, coarse_frame_features)
        if injection_mode in {"coarse_replace", "coarse_residual"}:
            return coarse_tokens
        if fine_tokens.numel() == 0:
            return coarse_tokens
        return torch.cat((coarse_tokens, fine_tokens), dim=0)
    if injection_mode in {"contextual_coarse_tail", "contextual_coarse_tail_fine"}:
        if not include_coarse:
            raise ValueError(f"{injection_mode} requires coarse tokens.")
        contextual_tokens, _ = pack_contextual_coarse_tokens(
            model,
            encoded_video_features,
            selected_indices,
            patch_scores=patch_scores,
            append_newline=append_newline,
            coarse_scale=context_scale,
            context_radius=context_radius,
            context_topk=context_topk,
        )
        if contextual_tokens.numel() == 0:
            return coarse_tokens
        if injection_mode == "contextual_coarse_tail":
            return torch.cat((coarse_tokens, contextual_tokens), dim=0)
        if fine_tokens.numel() == 0:
            return torch.cat((coarse_tokens, contextual_tokens), dim=0)
        return torch.cat((coarse_tokens, contextual_tokens, fine_tokens), dim=0)
    if injection_mode in {"inplace_boost_coarse", "inplace_boost_coarse_tail"}:
        boosted_encoded = encoded_video_features.clone()
        if background_decay != 1.0:
            boosted_encoded = boosted_encoded * float(background_decay)
        for frame_idx in range(boosted_encoded.shape[0]):
            frame_indices = selected_indices[frame_idx]
            frame_indices = frame_indices[frame_indices >= 0]
            if frame_indices.numel() == 0:
                continue
            boosted_encoded[frame_idx, frame_indices.long()] = encoded_video_features[frame_idx, frame_indices.long()] * float(boost_factor)
        coarse_tokens = pack_coarse_tokens(model, boosted_encoded) if include_coarse else None
        if injection_mode == "inplace_boost_coarse":
            if coarse_tokens is None:
                return pack_fine_selected_tokens(
                    model,
                    boosted_encoded,
                    selected_indices,
                    append_newline=append_newline,
                    fine_scale=fine_scale,
                )
            return coarse_tokens
        boosted_fine_tokens = pack_fine_selected_tokens(
            model,
            boosted_encoded,
            selected_indices,
            append_newline=append_newline,
            fine_scale=fine_scale,
        )
        if coarse_tokens is None:
            return boosted_fine_tokens
        if boosted_fine_tokens.numel() == 0:
            return coarse_tokens
        return torch.cat((coarse_tokens, boosted_fine_tokens), dim=0)
    if coarse_tokens is None:
        return fine_tokens
    if fine_tokens.numel() == 0:
        return coarse_tokens
    if injection_mode == "fine_first":
        return torch.cat((fine_tokens, coarse_tokens), dim=0)
    return torch.cat((coarse_tokens, fine_tokens), dim=0)


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
    encoded_video_features, patch_scores, base_video_features, metadata, aux = build_original_video_features(
        tokenizer,
        model,
        video,
        input_ids,
        attention_mask,
        scoring_mode=args.scoring_mode,
        fusion_2d_weight=args.fusion_2d_weight,
        fusion_3d_weight=args.fusion_3d_weight,
    )

    base_metrics = score_options_with_video_features(tokenizer, model, prompt_prefix, option_labels, base_video_features)
    base_scores = option_metrics_to_scores(base_metrics, score_key=args.score_key)
    base_summary = compute_option_summary(base_scores, gt_label)
    base_summary.update({"option_scores": base_scores})

    random_generator = torch.Generator(device=patch_scores.device)
    random_generator.manual_seed(args.random_seed + record.flat_index)

    targeted_idx, targeted_vals = select_patch_indices_from_scores(
        patch_scores,
        args.patch_ratio,
        mode="targeted",
        selection_scope=args.selection_scope,
    )
    random_idx, random_vals = select_patch_indices_from_scores(
        patch_scores,
        args.patch_ratio,
        mode="random",
        selection_scope=args.selection_scope,
        generator=random_generator,
    )
    low_idx, low_vals = select_patch_indices_from_scores(
        patch_scores,
        args.patch_ratio,
        mode="low_score",
        selection_scope=args.selection_scope,
    )

    conditions = {
        "targeted_augmented": (targeted_idx, targeted_vals),
        "random_augmented": (random_idx, random_vals),
        "low_score_augmented": (low_idx, low_vals),
    }
    result_conditions: Dict[str, Dict] = {"base_only": base_summary}
    include_coarse = not args.exclude_coarse

    for name, (indices, values) in conditions.items():
        augmented_video_features = build_augmented_video_features(
            model,
            encoded_video_features,
            patch_scores,
            indices,
            values,
            include_coarse=include_coarse,
            append_newline=args.append_newline,
            fine_scale=args.fine_scale,
            injection_mode=args.injection_mode,
            coarse_alpha=args.coarse_alpha,
            coarse_agg_mode=args.coarse_agg_mode,
            preserve_coarse_norm=args.preserve_coarse_norm,
            context_radius=args.context_radius,
            context_topk=args.context_topk,
            context_scale=args.context_scale,
            boost_factor=args.boost_factor,
            background_decay=args.background_decay,
        )
        option_metrics = score_options_with_video_features(tokenizer, model, prompt_prefix, option_labels, augmented_video_features)
        option_scores = option_metrics_to_scores(option_metrics, score_key=args.score_key)
        summary = compute_option_summary(option_scores, gt_label)
        summary.update(
            {
                "option_scores": option_scores,
                "selected_indices": indices.detach().cpu().tolist(),
                "selected_scores": values.detach().cpu().tolist(),
                "margin_gain_vs_base": compute_margin_gain(base_summary, summary),
                "video_token_count": int(augmented_video_features.shape[0]),
            }
        )
        result_conditions[name] = summary

    selective_gain = compute_selective_evidence_gain(
        base_summary,
        result_conditions["targeted_augmented"],
        result_conditions["random_augmented"],
    )

    result = {
        "qa_id": case.qa_id,
        "scene_name": case.scene_name,
        "question_type": parsed.get("question_type"),
        "question_text": parsed.get("question_text"),
        "ground_truth": gt_label,
        "patch_ratio": args.patch_ratio,
        "selection_scope": args.selection_scope,
        "fine_scale": args.fine_scale,
        "injection_mode": args.injection_mode,
        "coarse_alpha": args.coarse_alpha,
        "coarse_agg_mode": args.coarse_agg_mode,
        "preserve_coarse_norm": args.preserve_coarse_norm,
        "context_radius": args.context_radius,
        "context_topk": args.context_topk,
        "context_scale": args.context_scale,
        "boost_factor": args.boost_factor,
        "background_decay": args.background_decay,
        "include_coarse": include_coarse,
        "append_newline": args.append_newline,
        "conditions": result_conditions,
        "summary": {
            **selective_gain,
            "targeted_fix": (not base_summary["is_correct"]) and result_conditions["targeted_augmented"]["is_correct"],
            "random_fix": (not base_summary["is_correct"]) and result_conditions["random_augmented"]["is_correct"],
            "low_score_fix": (not base_summary["is_correct"]) and result_conditions["low_score_augmented"]["is_correct"],
            "targeted_hurt": base_summary["is_correct"] and (not result_conditions["targeted_augmented"]["is_correct"]),
            "random_hurt": base_summary["is_correct"] and (not result_conditions["random_augmented"]["is_correct"]),
            "low_score_hurt": base_summary["is_correct"] and (not result_conditions["low_score_augmented"]["is_correct"]),
        },
    }
    save_json(result, case_dir / "case_result.json")
    return result


def aggregate_results(results: List[Dict]) -> Dict:
    targeted_gt = [x["summary"]["targeted_gt_margin_gain"] for x in results]
    random_gt = [x["summary"]["random_gt_margin_gain"] for x in results]
    low_gt = [x["conditions"]["low_score_augmented"]["margin_gain_vs_base"]["gt_margin_gain"] for x in results]
    seg_gt = [x["summary"]["selective_evidence_gain_gt_margin"] for x in results]
    overall = {
        "num_cases": len(results),
        "mean_targeted_gt_margin_gain": float(np.mean(targeted_gt)) if targeted_gt else 0.0,
        "mean_random_gt_margin_gain": float(np.mean(random_gt)) if random_gt else 0.0,
        "mean_low_score_gt_margin_gain": float(np.mean(low_gt)) if low_gt else 0.0,
        "mean_selective_evidence_gain_gt_margin": float(np.mean(seg_gt)) if seg_gt else 0.0,
        "targeted_fix_rate": summarize_boolean_rate([x["summary"]["targeted_fix"] for x in results]),
        "random_fix_rate": summarize_boolean_rate([x["summary"]["random_fix"] for x in results]),
        "low_score_fix_rate": summarize_boolean_rate([x["summary"]["low_score_fix"] for x in results]),
        "targeted_hurt_rate": summarize_boolean_rate([x["summary"]["targeted_hurt"] for x in results]),
        "random_hurt_rate": summarize_boolean_rate([x["summary"]["random_hurt"] for x in results]),
        "low_score_hurt_rate": summarize_boolean_rate([x["summary"]["low_score_hurt"] for x in results]),
        "negative_selective_evidence_rate": summarize_boolean_rate([x["summary"]["selective_evidence_gain_gt_margin"] <= 0.0 for x in results]),
    }
    per_type = {}
    qtypes = sorted(set(x["question_type"] for x in results))
    for qtype in qtypes:
        subset = [x for x in results if x["question_type"] == qtype]
        per_type[qtype] = {
            "num_cases": len(subset),
            "mean_targeted_gt_margin_gain": float(np.mean([x["summary"]["targeted_gt_margin_gain"] for x in subset])) if subset else 0.0,
            "mean_random_gt_margin_gain": float(np.mean([x["summary"]["random_gt_margin_gain"] for x in subset])) if subset else 0.0,
            "mean_selective_evidence_gain_gt_margin": float(np.mean([x["summary"]["selective_evidence_gain_gt_margin"] for x in subset])) if subset else 0.0,
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
                    "base_pred": x["conditions"]["base_only"]["prediction_label"],
                    **x["summary"],
                }
                for x in results
            ],
        },
        args.output_root / "aggregate_augmented_branch_gain.json",
    )
    print(f"Saved aggregate summary to: {args.output_root / 'aggregate_augmented_branch_gain.json'}")


if __name__ == "__main__":
    main()
