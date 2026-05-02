from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from llava.model.feature_cd.common import compose_encoded_features, extract_video_branch_features
from ptb_patch_selection.fusion_guided_selection import (
    build_fusion_guided_patch_scores,
    compose_encoded_features_with_fusion_details,
)
from ptb_patch_selection.selective_pooling import (
    build_coarse_and_fine_video_features,
    build_coarse_and_fine_video_features_from_patch_scores,
    build_question_embedding,
    pack_coarse_tokens,
    select_topk_patch_indices,
    score_patch_tokens,
)


@torch.no_grad()
def _extract_encoded_video_features(model, video):
    branch_features = extract_video_branch_features(model, video)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]
    encoded_video_features = compose_encoded_features(model, orig_visual, camera_tokens, patch_tokens)
    return encoded_video_features, branch_features


@torch.no_grad()
def _normalize_per_frame(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    min_scores = scores.amin(dim=-1, keepdim=True)
    max_scores = scores.amax(dim=-1, keepdim=True)
    return (scores - min_scores) / (max_scores - min_scores).clamp_min(1e-6)


@torch.no_grad()
def _corrupt_patch_tokens(
    encoded_video_features: torch.Tensor,
    selected_indices: torch.Tensor,
    corrupt_mode: str = "frame_mean",
) -> torch.Tensor:
    if encoded_video_features.ndim != 3:
        raise ValueError(
            f"Expected encoded video features with shape [frames, tokens, dim], got {tuple(encoded_video_features.shape)}."
        )
    if selected_indices.ndim != 2:
        raise ValueError(f"Expected selected_indices with shape [frames, k], got {tuple(selected_indices.shape)}.")

    degraded = encoded_video_features.clone()
    if selected_indices.numel() == 0:
        return degraded

    if corrupt_mode == "zero":
        replacement = None
    elif corrupt_mode == "frame_mean":
        replacement = encoded_video_features.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported semantic negative corrupt mode: {corrupt_mode}")

    for frame_idx in range(degraded.shape[0]):
        frame_indices = selected_indices[frame_idx]
        if frame_indices.numel() == 0:
            continue
        if corrupt_mode == "zero":
            degraded[frame_idx, frame_indices.long()] = 0
        else:
            degraded[frame_idx, frame_indices.long()] = replacement[frame_idx].expand(frame_indices.shape[0], -1)
    return degraded


@torch.no_grad()
def build_stage0_coarse_only_branch(args, model, video, input_ids=None, attention_mask=None):
    encoded_video_features, _ = _extract_encoded_video_features(model, video)
    video_features = pack_coarse_tokens(model, encoded_video_features)
    return {
        "video_features": video_features,
        "metadata": {
            "stage_name": "stage0_coarse_only",
            "coarse_token_count": int(video_features.shape[0]),
            "fine_token_count": 0,
            "combined_token_count": int(video_features.shape[0]),
        },
    }


@torch.no_grad()
def build_stage0_semantic_negative_coarse_branch(args, model, video, input_ids, attention_mask):
    encoded_video_features, _ = _extract_encoded_video_features(model, video)
    question_embedding = build_question_embedding(model, input_ids, attention_mask)
    patch_scores = score_patch_tokens(
        encoded_video_features,
        question_embedding,
        mode=args.stage1_scoring_mode,
    )
    num_tokens = encoded_video_features.shape[1]
    semantic_neg_ratio = float(args.semantic_neg_ratio)
    if semantic_neg_ratio <= 0 or semantic_neg_ratio > 1:
        raise ValueError(f"semantic_neg_ratio must be in (0, 1], got {semantic_neg_ratio}.")
    semantic_neg_topk = min(num_tokens, max(1, int(math.ceil(num_tokens * semantic_neg_ratio))))
    selected_indices, selected_scores = select_topk_patch_indices(patch_scores, semantic_neg_topk)
    degraded_encoded_video_features = _corrupt_patch_tokens(
        encoded_video_features,
        selected_indices,
        corrupt_mode=args.semantic_neg_corrupt_mode,
    )
    video_features = pack_coarse_tokens(model, degraded_encoded_video_features)
    return {
        "video_features": video_features,
        "metadata": {
            "stage_name": "stage0_semantic_negative_coarse",
            "semantic_neg_ratio": float(semantic_neg_ratio),
            "semantic_neg_topk": int(semantic_neg_topk),
            "semantic_neg_corrupt_mode": args.semantic_neg_corrupt_mode,
            "semantic_neg_scoring_mode": args.stage1_scoring_mode,
            "selected_indices": selected_indices.detach().cpu(),
            "selected_scores": selected_scores.detach().cpu(),
            "patch_scores": patch_scores.detach().cpu(),
            "coarse_token_count": int(video_features.shape[0]),
            "fine_token_count": 0,
            "combined_token_count": int(video_features.shape[0]),
        },
    }


@torch.no_grad()
def build_stage1_semantic_branch(args, model, video, input_ids, attention_mask):
    encoded_video_features, _ = _extract_encoded_video_features(model, video)
    video_features, metadata = build_coarse_and_fine_video_features(
        model,
        encoded_video_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        fine_topk=args.stage1_topk,
        scoring_mode=args.stage1_scoring_mode,
        fine_scale=args.stage1_fine_scale,
        include_coarse=True,
        append_newline=args.append_newline,
    )
    metadata.update(
        {
            "stage_name": "stage1_semantic_fine",
            "stage_topk": int(args.stage1_topk),
            "stage_scoring_mode": args.stage1_scoring_mode,
        }
    )
    return {"video_features": video_features, "metadata": metadata}


@torch.no_grad()
def build_stage2_fusion_guided_branch(args, model, video, input_ids, attention_mask):
    branch_features = extract_video_branch_features(model, video)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    encoded_video_features, fusion_details = compose_encoded_features_with_fusion_details(
        model,
        orig_visual,
        camera_tokens,
        patch_tokens,
    )
    patch_scores, fusion_metadata = build_fusion_guided_patch_scores(
        encoded_video_features,
        fusion_details,
        scoring_mode=args.stage2_scoring_mode,
        fusion_2d_weight=args.stage2_fusion_2d_weight,
        fusion_3d_weight=args.stage2_fusion_3d_weight,
    )
    video_features, metadata = build_coarse_and_fine_video_features_from_patch_scores(
        model,
        encoded_video_features,
        patch_scores=patch_scores,
        fine_topk=args.stage2_topk,
        fine_scale=args.stage2_fine_scale,
        include_coarse=True,
        append_newline=args.append_newline,
        metadata_extra=fusion_metadata,
    )
    metadata.update(
        {
            "stage_name": "stage2_fusion_guided_fine",
            "stage_topk": int(args.stage2_topk),
            "stage_scoring_mode": args.stage2_scoring_mode,
        }
    )
    return {"video_features": video_features, "metadata": metadata}


@torch.no_grad()
def build_stage3_joint_branch(args, model, video, input_ids, attention_mask):
    branch_features = extract_video_branch_features(model, video)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    encoded_video_features, fusion_details = compose_encoded_features_with_fusion_details(
        model,
        orig_visual,
        camera_tokens,
        patch_tokens,
    )

    question_embedding = build_question_embedding(model, input_ids, attention_mask)
    semantic_scores = score_patch_tokens(
        encoded_video_features,
        question_embedding,
        mode=args.stage1_scoring_mode,
    )
    semantic_scores = _normalize_per_frame(semantic_scores)

    fusion_scores, fusion_metadata = build_fusion_guided_patch_scores(
        encoded_video_features,
        fusion_details,
        scoring_mode=args.stage2_scoring_mode,
        fusion_2d_weight=args.stage2_fusion_2d_weight,
        fusion_3d_weight=args.stage2_fusion_3d_weight,
    )
    fusion_scores = _normalize_per_frame(fusion_scores)

    joint_scores = args.stage3_semantic_weight * semantic_scores + args.stage3_fusion_weight * fusion_scores
    joint_scores = joint_scores / max(args.stage3_semantic_weight + args.stage3_fusion_weight, 1e-6)

    video_features, metadata = build_coarse_and_fine_video_features_from_patch_scores(
        model,
        encoded_video_features,
        patch_scores=joint_scores,
        fine_topk=args.stage3_topk,
        fine_scale=args.stage3_fine_scale,
        include_coarse=True,
        append_newline=args.append_newline,
        metadata_extra={
            **fusion_metadata,
            "semantic_scores": semantic_scores,
            "joint_scores": joint_scores,
        },
    )
    metadata.update(
        {
            "stage_name": "stage3_joint_semantic_fusion",
            "stage_topk": int(args.stage3_topk),
            "stage_scoring_mode": "joint_semantic_fusion",
            "stage3_semantic_weight": float(args.stage3_semantic_weight),
            "stage3_fusion_weight": float(args.stage3_fusion_weight),
        }
    )
    return {"video_features": video_features, "metadata": metadata}


@torch.no_grad()
def build_stage_setting_bundles(args, model, video, input_ids, attention_mask) -> Dict[str, Dict]:
    return {
        "stage0_coarse_only": build_stage0_coarse_only_branch(args, model, video, input_ids, attention_mask),
        "stage0_semantic_negative_coarse": build_stage0_semantic_negative_coarse_branch(args, model, video, input_ids, attention_mask),
        "stage1_semantic_fine": build_stage1_semantic_branch(args, model, video, input_ids, attention_mask),
        "stage2_fusion_guided_fine": build_stage2_fusion_guided_branch(args, model, video, input_ids, attention_mask),
        "stage3_joint_semantic_fusion": build_stage3_joint_branch(args, model, video, input_ids, attention_mask),
    }
