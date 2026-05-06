from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

from llava.model.feature_cd.common import extract_video_branch_features
from ptb_patch_selection.fusion_guided_selection import (
    FUSION_GUIDED_SCORING_MODES,
    build_fusion_guided_patch_scores,
    compose_encoded_features_with_fusion_details,
)
from ptb_patch_selection.selective_pooling import (
    build_question_embedding,
    pack_coarse_tokens,
    score_patch_tokens,
    select_topk_patch_indices,
)
from vcd.vcd_vision_token.methods import build_stage0_coarse_only_branch


QUERY_GUIDED_SCORING_MODES = {
    "question_cosine",
    "feature_norm",
    "question_cosine_x_norm",
}


def _extract_encoded_video_features_with_details(model, video) -> Tuple[torch.Tensor, Dict]:
    branch_features = extract_video_branch_features(model, video)
    encoded_video_features, fusion_details = compose_encoded_features_with_fusion_details(
        model,
        branch_features["visual_features"],
        branch_features["camera_tokens"],
        branch_features["patch_tokens"],
    )
    return encoded_video_features, fusion_details


def _score_patches_for_selection(
    args: SimpleNamespace,
    model,
    video,
    input_ids,
    attention_mask,
) -> Tuple[torch.Tensor, Dict]:
    encoded_video_features, fusion_details = _extract_encoded_video_features_with_details(model, video)
    scoring_mode = str(args.patch_warp_selection_mode)

    if scoring_mode in QUERY_GUIDED_SCORING_MODES:
        question_embedding = build_question_embedding(model, input_ids, attention_mask)
        patch_scores = score_patch_tokens(
            encoded_video_features,
            question_embedding,
            mode=scoring_mode,
        )
        metadata = {
            "selection_mode": scoring_mode,
            "selection_source": "text_query",
            "fusion_metadata": None,
        }
        return encoded_video_features, {
            "patch_scores": patch_scores,
            "selection_metadata": metadata,
            "fusion_details": fusion_details,
        }

    if scoring_mode in FUSION_GUIDED_SCORING_MODES:
        patch_scores, fusion_metadata = build_fusion_guided_patch_scores(
            encoded_video_features,
            fusion_details,
            scoring_mode=scoring_mode,
            fusion_2d_weight=float(args.patch_warp_fusion_2d_weight),
            fusion_3d_weight=float(args.patch_warp_fusion_3d_weight),
        )
        metadata = {
            "selection_mode": scoring_mode,
            "selection_source": "fusion_attention",
            "fusion_metadata": fusion_metadata,
        }
        return encoded_video_features, {
            "patch_scores": patch_scores,
            "selection_metadata": metadata,
            "fusion_details": fusion_details,
        }

    supported_modes = sorted(QUERY_GUIDED_SCORING_MODES | set(FUSION_GUIDED_SCORING_MODES))
    raise ValueError(f"Unsupported patch_warp_selection_mode: {scoring_mode}. Supported: {supported_modes}")


def _resolve_shift_target(
    row: int,
    col: int,
    grid_h: int,
    grid_w: int,
    rank: int,
    shift_size: int,
) -> Tuple[int, int]:
    direction = rank % 4
    if direction == 0:  # left
        return row, max(0, col - shift_size)
    if direction == 1:  # right
        return row, min(grid_w - 1, col + shift_size)
    if direction == 2:  # up
        return max(0, row - shift_size), col
    return min(grid_h - 1, row + shift_size), col  # down


def _apply_local_patch_shift(
    encoded_video_features: torch.Tensor,
    selected_indices: torch.Tensor,
    shift_size: int = 1,
    mix_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if encoded_video_features.ndim != 3:
        raise ValueError(
            f"Expected encoded video features with shape [frames, tokens, dim], got {tuple(encoded_video_features.shape)}."
        )
    if selected_indices.ndim != 2:
        raise ValueError(f"Expected selected_indices with shape [frames, k], got {tuple(selected_indices.shape)}.")

    num_frames, num_tokens, _ = encoded_video_features.shape
    if num_tokens <= 1 or selected_indices.numel() == 0:
        return encoded_video_features.clone(), torch.empty_like(selected_indices)

    mix_ratio = float(mix_ratio)
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"patch_warp_mix_ratio must be in [0, 1], got {mix_ratio}.")

    shift_size = max(1, int(shift_size))
    grid_h = int(round(math.sqrt(num_tokens)))
    grid_w = grid_h
    use_grid = grid_h * grid_w == num_tokens

    degraded = encoded_video_features.clone()
    source_indices = torch.empty_like(selected_indices)

    for frame_idx in range(num_frames):
        orig_frame = encoded_video_features[frame_idx]
        degraded_frame = degraded[frame_idx]
        for rank, patch_idx_tensor in enumerate(selected_indices[frame_idx]):
            patch_idx = int(patch_idx_tensor.item())
            if use_grid:
                row, col = divmod(patch_idx, grid_w)
                src_row, src_col = _resolve_shift_target(row, col, grid_h, grid_w, rank, shift_size)
                src_idx = src_row * grid_w + src_col
            else:
                step = shift_size if rank % 2 == 0 else -shift_size
                src_idx = min(max(patch_idx + step, 0), num_tokens - 1)
            source_indices[frame_idx, rank] = src_idx
            degraded_frame[patch_idx] = mix_ratio * orig_frame[patch_idx] + (1.0 - mix_ratio) * orig_frame[src_idx]
        degraded[frame_idx] = degraded_frame
    return degraded, source_indices


@torch.no_grad()
def build_stage0_local_patch_shift_negative_coarse_branch(args, model, video, input_ids, attention_mask):
    encoded_video_features, scoring_payload = _score_patches_for_selection(
        args,
        model,
        video,
        input_ids,
        attention_mask,
    )
    patch_scores = scoring_payload["patch_scores"]

    num_tokens = encoded_video_features.shape[1]
    patch_warp_ratio = float(args.patch_warp_ratio)
    if patch_warp_ratio <= 0 or patch_warp_ratio > 1:
        raise ValueError(f"patch_warp_ratio must be in (0, 1], got {patch_warp_ratio}.")

    patch_warp_topk = min(num_tokens, max(1, int(math.ceil(num_tokens * patch_warp_ratio))))
    selected_indices, selected_scores = select_topk_patch_indices(patch_scores, patch_warp_topk)
    degraded_encoded_video_features, source_indices = _apply_local_patch_shift(
        encoded_video_features,
        selected_indices,
        shift_size=int(args.patch_warp_shift_size),
        mix_ratio=float(args.patch_warp_mix_ratio),
    )
    video_features = pack_coarse_tokens(model, degraded_encoded_video_features)
    selection_metadata = scoring_payload["selection_metadata"]
    return {
        "video_features": video_features,
        "metadata": {
            "stage_name": "stage0_local_patch_shift_negative_coarse",
            "patch_warp_ratio": patch_warp_ratio,
            "patch_warp_topk": int(patch_warp_topk),
            "patch_warp_selection_mode": selection_metadata["selection_mode"],
            "patch_warp_selection_source": selection_metadata["selection_source"],
            "patch_warp_shift_size": int(args.patch_warp_shift_size),
            "patch_warp_mix_ratio": float(args.patch_warp_mix_ratio),
            "selected_indices": selected_indices.detach().cpu(),
            "selected_scores": selected_scores.detach().cpu(),
            "source_indices": source_indices.detach().cpu(),
            "patch_scores": patch_scores.detach().cpu(),
            "fusion_metadata": selection_metadata["fusion_metadata"],
            "coarse_token_count": int(video_features.shape[0]),
            "fine_token_count": 0,
            "combined_token_count": int(video_features.shape[0]),
        },
    }


@torch.no_grad()
def build_generation_patch_warp_branch_bundle(
    args,
    model,
    video,
    input_ids,
    attention_mask,
    branch_mode: str = "pairwise",
) -> Dict[str, object]:
    branch_mode = str(branch_mode).lower()
    if branch_mode != "pairwise":
        raise ValueError(f"Unsupported branch_mode: {branch_mode}. Only `pairwise` is kept in patch-warp VCD.")

    bundles = {
        "stage0_local_patch_shift_negative_coarse": build_stage0_local_patch_shift_negative_coarse_branch(
            args,
            model,
            video,
            input_ids,
            attention_mask,
        ),
        "stage0_coarse_only": build_stage0_coarse_only_branch(
            args,
            model,
            video,
            input_ids,
            attention_mask,
        ),
    }
    branch_names: List[str] = [
        "stage0_local_patch_shift_negative_coarse",
        "stage0_coarse_only",
    ]
    branches = [
        {
            "name": name,
            "video_features": bundles[name]["video_features"],
            "metadata": bundles[name].get("metadata"),
        }
        for name in branch_names
    ]
    return {
        "branch_mode": branch_mode,
        "branches": branches,
        "all_stage_bundles": bundles,
    }
