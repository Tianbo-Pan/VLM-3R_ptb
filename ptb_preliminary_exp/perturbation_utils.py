from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from llava.model.feature_cd.common import compose_encoded_features, extract_video_branch_features
from ptb_patch_selection.fusion_guided_selection import build_fusion_guided_patch_scores, compose_encoded_features_with_fusion_details
from ptb_patch_selection.selective_pooling import (
    build_question_embedding,
    build_question_token_embeddings,
    pack_coarse_tokens,
    score_patch_tokens,
    select_topk_patch_indices,
)


QUERY_SCORING_MODES = {"question_cosine", "feature_norm", "question_cosine_x_norm", "question_token_max", "question_token_topk_mean"}
FUSION_SCORING_MODES = {"fusion_2d3d", "fusion_2d", "fusion_3d"}


@torch.no_grad()
def build_encoded_video_and_patch_scores(
    model,
    tokenizer,
    video,
    input_ids,
    attention_mask,
    scoring_mode: str = "fusion_2d3d",
    fusion_2d_weight: float = 1.0,
    fusion_3d_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    branch_features = extract_video_branch_features(model, video)
    visual_features = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    if scoring_mode in FUSION_SCORING_MODES:
        encoded_video_features, fusion_details = compose_encoded_features_with_fusion_details(
            model,
            visual_features,
            camera_tokens,
            patch_tokens,
        )
        patch_scores, fusion_metadata = build_fusion_guided_patch_scores(
            encoded_video_features,
            fusion_details,
            scoring_mode=scoring_mode,
            fusion_2d_weight=fusion_2d_weight,
            fusion_3d_weight=fusion_3d_weight,
        )
        fusion_metadata = dict(fusion_metadata)
        fusion_metadata["branch_features"] = {
            "visual_features": visual_features,
            "camera_tokens": camera_tokens,
            "patch_tokens": patch_tokens,
        }
        return encoded_video_features, patch_scores, fusion_metadata

    encoded_video_features = compose_encoded_features(model, visual_features, camera_tokens, patch_tokens)
    if scoring_mode in {"question_token_max", "question_token_topk_mean"}:
        question_embedding = build_question_token_embeddings(
            model,
            input_ids,
            attention_mask,
            tokenizer=tokenizer,
            content_token_filter=True,
        )
    else:
        question_embedding = build_question_embedding(model, input_ids, attention_mask, tokenizer=tokenizer)
    patch_scores = score_patch_tokens(encoded_video_features, question_embedding, mode=scoring_mode)
    return encoded_video_features, patch_scores, {
        "scoring_mode": scoring_mode,
        "branch_features": {
            "visual_features": visual_features,
            "camera_tokens": camera_tokens,
            "patch_tokens": patch_tokens,
        },
    }


def _resize_token_scores(token_scores: torch.Tensor, target_num_tokens: int) -> torch.Tensor:
    if token_scores.shape[-1] == target_num_tokens:
        return token_scores

    src_tokens = token_scores.shape[-1]
    src_side = int(round(math.sqrt(src_tokens)))
    tgt_side = int(round(math.sqrt(target_num_tokens)))
    if src_side * src_side == src_tokens and tgt_side * tgt_side == target_num_tokens:
        resized = torch.nn.functional.interpolate(
            token_scores.view(token_scores.shape[0], 1, src_side, src_side),
            size=(tgt_side, tgt_side),
            mode="bilinear",
            align_corners=False,
        )
        return resized.view(token_scores.shape[0], target_num_tokens)

    resized = torch.nn.functional.interpolate(
        token_scores.unsqueeze(1),
        size=target_num_tokens,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(1)


@torch.no_grad()
def resolve_patch_scores_for_target(
    patch_scores: torch.Tensor,
    metadata: Dict,
    perturbation_target: str = "encoded",
) -> torch.Tensor:
    perturbation_target = str(perturbation_target).lower()
    if perturbation_target == "encoded":
        return patch_scores

    if perturbation_target == "prefusion_3d_patch":
        patch_tokens = metadata.get("branch_features", {}).get("patch_tokens")
        if patch_tokens is None:
            raise ValueError("prefusion_3d_patch perturbation requires patch_tokens from the spatial tower.")
        return _resize_token_scores(patch_scores, patch_tokens.shape[1])

    raise ValueError(f"Unsupported perturbation_target: {perturbation_target}")


@torch.no_grad()
def select_patch_indices_from_scores(
    patch_scores: torch.Tensor,
    ratio: float,
    mode: str = "targeted",
    selection_scope: str = "per_frame",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if patch_scores.ndim != 2:
        raise ValueError(f"Expected patch_scores with shape [frames, tokens], got {tuple(patch_scores.shape)}")
    num_frames, num_tokens = patch_scores.shape
    topk = min(num_tokens, max(1, int(math.ceil(num_tokens * float(ratio)))))
    if mode == "targeted":
        selected_indices, selected_scores, _, _ = select_topk_patch_indices(
            patch_scores,
            topk,
            selection_scope=selection_scope,
        )
        return selected_indices, selected_scores
    if mode == "low_score":
        low_scores, low_indices = torch.topk(patch_scores, k=topk, dim=-1, largest=False)
        return low_indices, low_scores
    if mode == "random":
        if generator is None:
            generator = torch.Generator(device=patch_scores.device)
            generator.manual_seed(0)
        indices = []
        values = []
        for frame_idx in range(num_frames):
            perm = torch.randperm(num_tokens, generator=generator, device=patch_scores.device)[:topk]
            indices.append(perm)
            values.append(patch_scores[frame_idx].index_select(0, perm))
        return torch.stack(indices, dim=0), torch.stack(values, dim=0)
    raise ValueError(f"Unsupported patch selection mode: {mode}")


@torch.no_grad()
def apply_patch_zero_ablation(encoded_video_features: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
    degraded = encoded_video_features.clone()
    for frame_idx in range(degraded.shape[0]):
        degraded[frame_idx, selected_indices[frame_idx].long()] = 0
    return degraded


@torch.no_grad()
def apply_patch_zero_ablation_to_patch_tokens(patch_tokens: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
    degraded = patch_tokens.clone()
    for frame_idx in range(degraded.shape[0]):
        valid_indices = selected_indices[frame_idx]
        valid_indices = valid_indices[valid_indices >= 0]
        if valid_indices.numel() == 0:
            continue
        degraded[frame_idx, valid_indices.long()] = 0
    return degraded


def _resolve_shift_target(row: int, col: int, grid_h: int, grid_w: int, rank: int, shift_size: int):
    direction = rank % 4
    if direction == 0:
        return row, max(0, col - shift_size)
    if direction == 1:
        return row, min(grid_w - 1, col + shift_size)
    if direction == 2:
        return max(0, row - shift_size), col
    return min(grid_h - 1, row + shift_size), col


@torch.no_grad()
def apply_patch_warp_ablation(
    encoded_video_features: torch.Tensor,
    selected_indices: torch.Tensor,
    shift_size: int = 1,
    mix_ratio: float = 0.5,
) -> torch.Tensor:
    degraded = encoded_video_features.clone()
    num_frames, num_tokens, _ = degraded.shape
    grid_h = int(round(math.sqrt(num_tokens)))
    grid_w = grid_h
    use_grid = grid_h * grid_w == num_tokens
    for frame_idx in range(num_frames):
        source = encoded_video_features[frame_idx]
        for rank, patch_idx_tensor in enumerate(selected_indices[frame_idx]):
            patch_idx = int(patch_idx_tensor.item())
            if use_grid:
                row, col = divmod(patch_idx, grid_w)
                src_row, src_col = _resolve_shift_target(row, col, grid_h, grid_w, rank, shift_size)
                src_idx = src_row * grid_w + src_col
            else:
                offset = shift_size if rank % 2 == 0 else -shift_size
                src_idx = min(max(patch_idx + offset, 0), num_tokens - 1)
            degraded[frame_idx, patch_idx] = mix_ratio * source[patch_idx] + (1.0 - mix_ratio) * source[src_idx]
    return degraded


@torch.no_grad()
def apply_patch_warp_ablation_to_patch_tokens(
    patch_tokens: torch.Tensor,
    selected_indices: torch.Tensor,
    shift_size: int = 1,
    mix_ratio: float = 0.5,
) -> torch.Tensor:
    degraded = patch_tokens.clone()
    num_frames, num_tokens, _ = degraded.shape
    grid_h = int(round(math.sqrt(num_tokens)))
    grid_w = grid_h
    use_grid = grid_h * grid_w == num_tokens
    for frame_idx in range(num_frames):
        source = patch_tokens[frame_idx]
        for rank, patch_idx_tensor in enumerate(selected_indices[frame_idx]):
            patch_idx = int(patch_idx_tensor.item())
            if patch_idx < 0:
                continue
            if use_grid:
                row, col = divmod(patch_idx, grid_w)
                src_row, src_col = _resolve_shift_target(row, col, grid_h, grid_w, rank, shift_size)
                src_idx = src_row * grid_w + src_col
            else:
                offset = shift_size if rank % 2 == 0 else -shift_size
                src_idx = min(max(patch_idx + offset, 0), num_tokens - 1)
            degraded[frame_idx, patch_idx] = mix_ratio * source[patch_idx] + (1.0 - mix_ratio) * source[src_idx]
    return degraded


@torch.no_grad()
def pack_encoded_video_features(model, encoded_video_features: torch.Tensor) -> torch.Tensor:
    return pack_coarse_tokens(model, encoded_video_features)


@torch.no_grad()
def repack_from_perturbed_patch_tokens(
    model,
    metadata: Dict,
    perturbed_patch_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    visual_features = metadata.get("branch_features", {}).get("visual_features")
    camera_tokens = metadata.get("branch_features", {}).get("camera_tokens")
    if visual_features is None:
        raise ValueError("Missing visual_features in metadata for pre-fusion patch-token perturbation.")
    degraded_encoded_video_features = compose_encoded_features(
        model,
        visual_features,
        camera_tokens,
        perturbed_patch_tokens,
    )
    degraded_video_features = pack_coarse_tokens(model, degraded_encoded_video_features)
    return degraded_encoded_video_features, degraded_video_features
