import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX


def _to_batch_first_ids(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


@torch.no_grad()
def build_question_embedding(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    input_ids, attention_mask = _to_batch_first_ids(input_ids, attention_mask)
    if input_ids.shape[0] != 1:
        raise ValueError(f"Selective patch pooling currently expects batch size 1, got {input_ids.shape[0]}.")

    cur_ids = input_ids[0]
    if attention_mask is not None:
        cur_ids = cur_ids[attention_mask[0].bool()]

    cur_ids = cur_ids[cur_ids != IMAGE_TOKEN_INDEX]
    if cur_ids.numel() == 0:
        raise ValueError("No text tokens remain after filtering IMAGE_TOKEN_INDEX.")

    text_embeds = model.get_model().embed_tokens(cur_ids)
    return text_embeds.mean(dim=0)


@torch.no_grad()
def score_patch_tokens(
    encoded_video_features: torch.Tensor,
    question_embedding: torch.Tensor,
    mode: str = "question_cosine",
) -> torch.Tensor:
    if encoded_video_features.ndim != 3:
        raise ValueError(
            f"Expected encoded video features with shape [frames, tokens, dim], got {tuple(encoded_video_features.shape)}."
        )

    if mode == "question_cosine":
        norm_video = F.normalize(encoded_video_features.float(), dim=-1)
        norm_question = F.normalize(question_embedding.float(), dim=-1)
        return torch.einsum("ftd,d->ft", norm_video, norm_question)

    if mode == "feature_norm":
        return encoded_video_features.float().norm(dim=-1)

    if mode == "question_cosine_x_norm":
        norm_video = F.normalize(encoded_video_features.float(), dim=-1)
        norm_question = F.normalize(question_embedding.float(), dim=-1)
        cosine = torch.einsum("ftd,d->ft", norm_video, norm_question)
        magnitude = encoded_video_features.float().norm(dim=-1)
        magnitude = magnitude / magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return cosine * magnitude

    raise ValueError(f"Unsupported scoring mode: {mode}")


@torch.no_grad()
def select_topk_patch_indices(
    patch_scores: torch.Tensor,
    fine_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if patch_scores.ndim != 2:
        raise ValueError(f"Expected patch_scores with shape [frames, tokens], got {tuple(patch_scores.shape)}.")

    num_frames, num_tokens = patch_scores.shape
    if fine_topk <= 0:
        empty_idx = torch.empty((num_frames, 0), dtype=torch.long, device=patch_scores.device)
        empty_scores = torch.empty((num_frames, 0), dtype=patch_scores.dtype, device=patch_scores.device)
        return empty_idx, empty_scores

    topk = min(fine_topk, num_tokens)
    top_scores, top_indices = torch.topk(patch_scores, k=topk, dim=-1)
    return top_indices, top_scores


def _get_newline_token(model, ref_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if not hasattr(model, "model") or not hasattr(model.model, "image_newline"):
        return None
    return model.model.image_newline.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


@torch.no_grad()
def pack_coarse_tokens(
    model,
    encoded_video_features: torch.Tensor,
) -> torch.Tensor:
    coarse_tokens = model.get_2dPool(encoded_video_features)
    mm_patch_merge_type = getattr(model.config, "mm_patch_merge_type", "flat")
    mm_newline_position = getattr(model.config, "mm_newline_position", "one_token")
    if mm_patch_merge_type == "flat":
        return coarse_tokens.flatten(0, 1)
    if not mm_patch_merge_type.startswith("spatial"):
        raise ValueError(f"Unsupported mm_patch_merge_type for selective patch pooling: {mm_patch_merge_type}")
    if mm_newline_position == "grid":
        return model.add_token_per_grid(coarse_tokens)
    if mm_newline_position == "frame":
        return model.add_token_per_frame(coarse_tokens).flatten(0, 1)
    if mm_newline_position == "one_token":
        coarse_tokens = coarse_tokens.flatten(0, 1)
        if "unpad" in mm_patch_merge_type:
            newline_token = _get_newline_token(model, coarse_tokens)
            if newline_token is not None:
                coarse_tokens = torch.cat((coarse_tokens, newline_token.unsqueeze(0)), dim=0)
        return coarse_tokens
    if mm_newline_position == "no_token":
        return coarse_tokens.flatten(0, 1)
    raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")


def _detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach_to_cpu(v) for v in obj)
    return obj


@torch.no_grad()
def pack_fine_selected_tokens(
    model,
    encoded_video_features: torch.Tensor,
    selected_indices: torch.Tensor,
    append_newline: bool = True,
    fine_scale: float = 1.0,
) -> torch.Tensor:
    if selected_indices.ndim != 2:
        raise ValueError(f"Expected selected_indices with shape [frames, k], got {tuple(selected_indices.shape)}.")

    newline_token = _get_newline_token(model, encoded_video_features)
    blocks = []
    for frame_idx in range(encoded_video_features.shape[0]):
        frame_indices = selected_indices[frame_idx]
        if frame_indices.numel() == 0:
            continue
        frame_tokens = encoded_video_features[frame_idx].index_select(0, frame_indices.long())
        if fine_scale != 1.0:
            frame_tokens = frame_tokens * fine_scale
        blocks.append(frame_tokens)
        if append_newline and newline_token is not None:
            blocks.append(newline_token.unsqueeze(0))

    if not blocks:
        return encoded_video_features.new_zeros((0, encoded_video_features.shape[-1]))
    return torch.cat(blocks, dim=0)


@torch.no_grad()
def build_coarse_and_fine_video_features_from_patch_scores(
    model,
    encoded_video_features: torch.Tensor,
    patch_scores: torch.Tensor,
    fine_topk: int = 16,
    fine_scale: float = 1.0,
    include_coarse: bool = True,
    append_newline: bool = True,
    metadata_extra: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict]:
    if encoded_video_features.ndim != 3:
        raise ValueError(
            f"Expected encoded video features with shape [frames, tokens, dim], got {tuple(encoded_video_features.shape)}."
        )
    if patch_scores.ndim != 2:
        raise ValueError(f"Expected patch_scores with shape [frames, tokens], got {tuple(patch_scores.shape)}.")
    if patch_scores.shape[:2] != encoded_video_features.shape[:2]:
        raise ValueError(
            "patch_scores must align with encoded_video_features on [frames, tokens], "
            f"got {tuple(patch_scores.shape)} vs {tuple(encoded_video_features.shape)}."
        )

    coarse_tokens = None
    if include_coarse:
        coarse_tokens = pack_coarse_tokens(model, encoded_video_features)

    selected_indices, selected_scores = select_topk_patch_indices(patch_scores, fine_topk)
    fine_tokens = pack_fine_selected_tokens(
        model,
        encoded_video_features,
        selected_indices,
        append_newline=append_newline,
        fine_scale=fine_scale,
    )

    if coarse_tokens is None:
        combined = fine_tokens
    elif fine_tokens.numel() == 0:
        combined = coarse_tokens
    else:
        combined = torch.cat((coarse_tokens, fine_tokens), dim=0)

    num_raw_tokens = encoded_video_features.shape[1]
    grid_size = int(round(math.sqrt(num_raw_tokens)))
    metadata = {
        "fine_topk": int(fine_topk),
        "fine_scale": float(fine_scale),
        "include_coarse": bool(include_coarse),
        "append_newline": bool(append_newline),
        "grid_size": grid_size,
        "selected_indices": selected_indices.detach().cpu(),
        "selected_scores": selected_scores.detach().cpu(),
        "patch_scores": patch_scores.detach().cpu(),
        "coarse_token_count": 0 if coarse_tokens is None else int(coarse_tokens.shape[0]),
        "fine_token_count": int(fine_tokens.shape[0]),
        "combined_token_count": int(combined.shape[0]),
    }
    if metadata_extra:
        metadata.update(_detach_to_cpu(metadata_extra))
    return combined, metadata


@torch.no_grad()
def build_coarse_and_fine_video_features(
    model,
    encoded_video_features: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    fine_topk: int = 16,
    scoring_mode: str = "question_cosine",
    fine_scale: float = 1.0,
    include_coarse: bool = True,
    append_newline: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    question_embedding = build_question_embedding(model, input_ids, attention_mask)
    patch_scores = score_patch_tokens(encoded_video_features, question_embedding, mode=scoring_mode)
    return build_coarse_and_fine_video_features_from_patch_scores(
        model,
        encoded_video_features,
        patch_scores,
        fine_topk=fine_topk,
        fine_scale=fine_scale,
        include_coarse=include_coarse,
        append_newline=append_newline,
        metadata_extra={"scoring_mode": scoring_mode},
    )
