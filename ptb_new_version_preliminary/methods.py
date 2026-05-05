from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from llava.model.feature_cd.common import (
    compose_encoded_features,
    compose_visual_only_encoded_features,
    extract_video_branch_features,
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

PATCH_SELECTION_SCOPES = {
    "per_frame",
    "global",
}


def _resize_token_scores(token_scores: torch.Tensor, target_num_tokens: int) -> torch.Tensor:
    if token_scores.shape[-1] == target_num_tokens:
        return token_scores

    src_tokens = token_scores.shape[-1]
    src_side = int(round(math.sqrt(src_tokens)))
    tgt_side = int(round(math.sqrt(target_num_tokens)))
    if src_side * src_side == src_tokens and tgt_side * tgt_side == target_num_tokens:
        resized = F.interpolate(
            token_scores.view(token_scores.shape[0], 1, src_side, src_side),
            size=(tgt_side, tgt_side),
            mode="bilinear",
            align_corners=False,
        )
        return resized.view(token_scores.shape[0], target_num_tokens)

    resized = F.interpolate(
        token_scores.unsqueeze(1),
        size=target_num_tokens,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(1)


def _resolve_shift_target(
    row: int,
    col: int,
    grid_h: int,
    grid_w: int,
    rank: int,
    shift_size: int,
) -> Tuple[int, int]:
    direction = rank % 4
    if direction == 0:
        return row, max(0, col - shift_size)
    if direction == 1:
        return row, min(grid_w - 1, col + shift_size)
    if direction == 2:
        return max(0, row - shift_size), col
    return min(grid_h - 1, row + shift_size), col


def _apply_local_patch_shift(
    patch_tokens: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_mask: Optional[torch.Tensor] = None,
    shift_size: int = 1,
    mix_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch_tokens with shape [frames, tokens, dim], got {tuple(patch_tokens.shape)}.")
    if selected_indices.ndim != 2:
        raise ValueError(f"Expected selected_indices with shape [frames, k], got {tuple(selected_indices.shape)}.")
    if selected_mask is not None and selected_mask.shape != selected_indices.shape:
        raise ValueError(
            f"Expected selected_mask with shape {tuple(selected_indices.shape)}, got {tuple(selected_mask.shape)}."
        )

    num_frames, num_tokens, _ = patch_tokens.shape
    if num_tokens <= 1 or selected_indices.numel() == 0:
        return patch_tokens.clone(), torch.empty_like(selected_indices)

    mix_ratio = float(mix_ratio)
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"ptb_preliminary_mix_ratio must be in [0, 1], got {mix_ratio}.")

    shift_size = max(1, int(shift_size))
    grid_h = int(round(math.sqrt(num_tokens)))
    grid_w = grid_h
    use_grid = grid_h * grid_w == num_tokens

    degraded = patch_tokens.clone()
    source_indices = torch.full_like(selected_indices, fill_value=-1)

    for frame_idx in range(num_frames):
        orig_frame = patch_tokens[frame_idx]
        degraded_frame = degraded[frame_idx]
        for rank, patch_idx_tensor in enumerate(selected_indices[frame_idx]):
            if selected_mask is not None and not bool(selected_mask[frame_idx, rank].item()):
                continue
            patch_idx = int(patch_idx_tensor.item())
            if patch_idx < 0:
                continue
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


def _select_patch_indices_by_scope(
    patch_scores: torch.Tensor,
    patch_ratio: float,
    selection_scope: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    if patch_scores.ndim != 2:
        raise ValueError(f"Expected patch_scores with shape [frames, tokens], got {tuple(patch_scores.shape)}.")

    num_frames, num_tokens = patch_scores.shape
    selection_scope = str(selection_scope).lower()
    if selection_scope not in PATCH_SELECTION_SCOPES:
        raise ValueError(
            f"Unsupported ptb_preliminary_selection_scope: {selection_scope}. Supported: {sorted(PATCH_SELECTION_SCOPES)}"
        )

    if selection_scope == "per_frame":
        topk = min(num_tokens, max(1, int(math.ceil(num_tokens * patch_ratio))))
        selected_indices, selected_scores = select_topk_patch_indices(patch_scores, topk)
        selected_mask = torch.ones_like(selected_indices, dtype=torch.bool)
        return selected_indices, selected_scores, selected_mask, {
            "selected_topk_value": int(topk),
            "selected_total_count": int(num_frames * topk),
            "selected_per_frame": [int(topk)] * num_frames,
        }

    total_candidates = num_frames * num_tokens
    total_selected = min(total_candidates, max(1, int(math.ceil(total_candidates * patch_ratio))))
    flat_scores = patch_scores.reshape(-1)
    top_scores, top_flat_indices = torch.topk(flat_scores, k=total_selected, dim=0)

    frame_indices = torch.div(top_flat_indices, num_tokens, rounding_mode="floor")
    token_indices = torch.remainder(top_flat_indices, num_tokens)

    per_frame_pairs: List[List[Tuple[float, int]]] = [[] for _ in range(num_frames)]
    for score_tensor, frame_tensor, token_tensor in zip(top_scores, frame_indices, token_indices):
        per_frame_pairs[int(frame_tensor.item())].append((float(score_tensor.item()), int(token_tensor.item())))

    max_selected_in_frame = max((len(items) for items in per_frame_pairs), default=0)
    selected_indices = torch.full(
        (num_frames, max_selected_in_frame),
        fill_value=-1,
        dtype=torch.long,
        device=patch_scores.device,
    )
    selected_scores = torch.full(
        (num_frames, max_selected_in_frame),
        fill_value=float("-inf"),
        dtype=patch_scores.dtype,
        device=patch_scores.device,
    )
    selected_mask = torch.zeros((num_frames, max_selected_in_frame), dtype=torch.bool, device=patch_scores.device)

    selected_per_frame: List[int] = []
    for frame_idx, items in enumerate(per_frame_pairs):
        items.sort(key=lambda x: x[0], reverse=True)
        selected_per_frame.append(len(items))
        for rank, (score_value, token_idx) in enumerate(items):
            selected_indices[frame_idx, rank] = token_idx
            selected_scores[frame_idx, rank] = score_value
            selected_mask[frame_idx, rank] = True

    return selected_indices, selected_scores, selected_mask, {
        "selected_topk_value": int(total_selected),
        "selected_total_count": int(total_selected),
        "selected_per_frame": selected_per_frame,
    }


@torch.no_grad()
def build_stage0_question_query_3d_patch_warp_negative_coarse_branch(
    args: SimpleNamespace,
    model,
    video,
    input_ids,
    attention_mask,
):
    branch_features = extract_video_branch_features(model, video)
    visual_features = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    if patch_tokens is None:
        raise ValueError("ptb_new_version_preliminary requires patch_tokens from the spatial tower.")

    query_mode = str(args.ptb_preliminary_query_mode)
    if query_mode not in QUERY_GUIDED_SCORING_MODES:
        raise ValueError(f"Unsupported ptb_preliminary_query_mode: {query_mode}. Supported: {sorted(QUERY_GUIDED_SCORING_MODES)}")

    visual_only_encoded = compose_visual_only_encoded_features(model, visual_features)
    question_embedding = build_question_embedding(model, input_ids, attention_mask)
    visual_patch_scores = score_patch_tokens(
        visual_only_encoded,
        question_embedding,
        mode=query_mode,
    )
    patch_token_scores = _resize_token_scores(visual_patch_scores, patch_tokens.shape[1])

    patch_ratio = float(args.ptb_preliminary_ratio)
    if patch_ratio <= 0 or patch_ratio > 1:
        raise ValueError(f"ptb_preliminary_ratio must be in (0, 1], got {patch_ratio}.")

    selection_scope = str(getattr(args, "ptb_preliminary_selection_scope", "per_frame")).lower()
    selected_indices, selected_scores, selected_mask, selection_scope_metadata = _select_patch_indices_by_scope(
        patch_token_scores,
        patch_ratio=patch_ratio,
        selection_scope=selection_scope,
    )
    degraded_patch_tokens, source_indices = _apply_local_patch_shift(
        patch_tokens,
        selected_indices,
        selected_mask=selected_mask,
        shift_size=int(args.ptb_preliminary_shift_size),
        mix_ratio=float(args.ptb_preliminary_mix_ratio),
    )
    degraded_encoded_video_features = compose_encoded_features(
        model,
        visual_features,
        camera_tokens,
        degraded_patch_tokens,
    )
    video_features = pack_coarse_tokens(model, degraded_encoded_video_features)
    return {
        "video_features": video_features,
        "metadata": {
            "stage_name": "stage0_question_query_3d_patch_warp_negative_coarse",
            "ptb_preliminary_ratio": patch_ratio,
            "ptb_preliminary_query_mode": query_mode,
            "ptb_preliminary_selection_scope": selection_scope,
            "ptb_preliminary_shift_size": int(args.ptb_preliminary_shift_size),
            "ptb_preliminary_mix_ratio": float(args.ptb_preliminary_mix_ratio),
            "query_feature_source": "visual_only_projected_2d",
            "perturb_feature_source": "patch_tokens_3d_pre_fusion",
            "visual_token_count": int(visual_only_encoded.shape[1]),
            "patch_token_count": int(patch_tokens.shape[1]),
            "selected_topk_value": int(selection_scope_metadata["selected_topk_value"]),
            "selected_total_count": int(selection_scope_metadata["selected_total_count"]),
            "selected_per_frame": selection_scope_metadata["selected_per_frame"],
            "visual_patch_scores": visual_patch_scores.detach().cpu(),
            "patch_token_scores": patch_token_scores.detach().cpu(),
            "selected_indices": selected_indices.detach().cpu(),
            "selected_scores": selected_scores.detach().cpu(),
            "selected_mask": selected_mask.detach().cpu(),
            "source_indices": source_indices.detach().cpu(),
            "coarse_token_count": int(video_features.shape[0]),
            "fine_token_count": 0,
            "combined_token_count": int(video_features.shape[0]),
        },
    }


@torch.no_grad()
def build_generation_ptb_preliminary_branch_bundle(
    args,
    model,
    video,
    input_ids,
    attention_mask,
    branch_mode: str = "pairwise",
) -> Dict[str, object]:
    branch_mode = str(branch_mode).lower()
    if branch_mode != "pairwise":
        raise ValueError(f"Unsupported branch_mode: {branch_mode}. Only `pairwise` is kept in ptb_new_version_preliminary.")

    bundles = {
        "stage0_question_query_3d_patch_warp_negative_coarse": build_stage0_question_query_3d_patch_warp_negative_coarse_branch(
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
        "stage0_question_query_3d_patch_warp_negative_coarse",
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
