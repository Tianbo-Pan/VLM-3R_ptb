import math
import re
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX


QUESTION_TOKEN_TOPK = 3

CONTENT_TOKEN_STOPWORDS = {
    "a", "an", "and", "answer", "are", "as", "at", "be", "by", "choose", "choice", "choices",
    "directly", "do", "either", "fill", "following", "for", "from", "given", "go", "if", "in",
    "into", "is", "it", "letter", "me", "multiple", "my", "of", "on", "option", "or", "out",
    "perform", "phrase", "please", "question", "refer", "response", "single", "shown", "space",
    "standing", "term", "text", "than", "that", "the", "these", "this", "to", "until", "using",
    "video", "what", "which", "will", "with", "word", "you", "your",
}

CONTENT_TOKEN_PRIORITY_WORDS = {
    "above", "after", "appearance", "back", "back-left", "back-right", "before", "behind", "below",
    "bigger", "biggest", "bottom", "centimeter", "centimeters", "close", "closer", "closest",
    "combined", "count", "counterclockwise", "depth", "diagonal", "direction", "distance", "earlier",
    "far", "farther", "farthest", "first", "forward", "front", "front-left", "front-right",
    "height", "how", "largest", "last", "left", "length", "meter", "meters", "more", "nearest",
    "next", "number", "order", "right", "robot", "room", "second", "size", "smallest", "square",
    "third", "toward", "turn", "under", "view", "width",
}


def _to_batch_first_ids(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


def _normalize_decoded_token(text: str) -> str:
    text = text.replace("Ġ", " ").replace("▁", " ").replace("Ċ", " ").replace("\n", " ")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    return text


def _is_content_token(token_text: str) -> bool:
    if not token_text:
        return False
    if token_text in CONTENT_TOKEN_PRIORITY_WORDS:
        return True
    if token_text in CONTENT_TOKEN_STOPWORDS:
        return False
    if token_text in {"a", "b", "c", "d", "e"}:
        return False
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token_text):
        return True
    if not re.search(r"[a-z0-9]", token_text):
        return False
    return len(token_text) > 1


@torch.no_grad()
def build_question_token_embeddings(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    content_token_filter: bool = False,
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

    if tokenizer is not None:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        keep_mask = torch.tensor(
            [int(token_id) not in special_ids for token_id in cur_ids.tolist()],
            device=cur_ids.device,
            dtype=torch.bool,
        )
        if keep_mask.any():
            cur_ids = cur_ids[keep_mask]

    if content_token_filter and tokenizer is not None:
        selected_ids = []
        for token_id in cur_ids.tolist():
            decoded = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            normalized = _normalize_decoded_token(decoded)
            if _is_content_token(normalized):
                selected_ids.append(token_id)
        if selected_ids:
            cur_ids = torch.tensor(selected_ids, device=input_ids.device, dtype=input_ids.dtype)

    if cur_ids.numel() == 0:
        raise ValueError("No text tokens remain after content-token filtering.")

    return model.get_model().embed_tokens(cur_ids)


@torch.no_grad()
def build_question_embedding(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> torch.Tensor:
    text_embeds = build_question_token_embeddings(
        model,
        input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        content_token_filter=False,
    )
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

    if mode in {"question_token_max", "question_token_topk_mean"}:
        if question_embedding.ndim == 1:
            question_embedding = question_embedding.unsqueeze(0)
        if question_embedding.ndim != 2:
            raise ValueError(
                f"Expected token-level question embeddings with shape [num_text_tokens, dim], got {tuple(question_embedding.shape)}."
            )
        norm_video = F.normalize(encoded_video_features.float(), dim=-1)
        norm_question = F.normalize(question_embedding.float(), dim=-1)
        pairwise_scores = torch.einsum("ftd,qd->ftq", norm_video, norm_question)
        if mode == "question_token_max":
            return pairwise_scores.amax(dim=-1)
        topk = min(QUESTION_TOKEN_TOPK, pairwise_scores.shape[-1])
        topk_scores = torch.topk(pairwise_scores, k=topk, dim=-1).values
        return topk_scores.mean(dim=-1)

    raise ValueError(f"Unsupported scoring mode: {mode}")


def _normalize_selection_scope(selection_scope: str) -> str:
    normalized = str(selection_scope).strip().lower()
    aliases = {
        "frame": "per_frame",
        "perframe": "per_frame",
        "per_frame": "per_frame",
        "global": "global",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported selection_scope: {selection_scope}. Expected 'per_frame' or 'global'.")
    return aliases[normalized]


def _parse_optional_ratio(fine_ratio: Optional[float]) -> Optional[float]:
    if fine_ratio is None:
        return None
    ratio = float(fine_ratio)
    if ratio <= 0:
        return None
    if ratio > 1:
        raise ValueError(f"fine_ratio must be in (0, 1], got {fine_ratio}. Use 0.3 for top 30%.")
    return ratio


@torch.no_grad()
def select_topk_patch_indices(
    patch_scores: torch.Tensor,
    fine_topk: int,
    fine_ratio: Optional[float] = None,
    selection_scope: str = "per_frame",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    if patch_scores.ndim != 2:
        raise ValueError(f"Expected patch_scores with shape [frames, tokens], got {tuple(patch_scores.shape)}.")

    num_frames, num_tokens = patch_scores.shape
    selection_scope = _normalize_selection_scope(selection_scope)
    ratio = _parse_optional_ratio(fine_ratio)
    selection_mode = "ratio" if ratio is not None else "topk"

    if selection_mode == "ratio":
        requested_count_per_frame = max(0, math.ceil(num_tokens * ratio))
    else:
        requested_count_per_frame = max(0, int(fine_topk))

    if selection_scope == "per_frame":
        topk = min(requested_count_per_frame, num_tokens)
        if topk <= 0:
            empty_idx = torch.empty((num_frames, 0), dtype=torch.long, device=patch_scores.device)
            empty_scores = torch.empty((num_frames, 0), dtype=patch_scores.dtype, device=patch_scores.device)
            empty_counts = torch.zeros((num_frames,), dtype=torch.long, device=patch_scores.device)
            return empty_idx, empty_scores, empty_counts, {
                "selection_mode": selection_mode,
                "selection_scope": selection_scope,
                "requested_count_per_frame": requested_count_per_frame,
                "requested_total_selected": 0,
                "effective_total_selected": 0,
            }
        top_scores, top_indices = torch.topk(patch_scores, k=topk, dim=-1)
        selected_counts = torch.full((num_frames,), topk, dtype=torch.long, device=patch_scores.device)
        return top_indices, top_scores, selected_counts, {
            "selection_mode": selection_mode,
            "selection_scope": selection_scope,
            "requested_count_per_frame": requested_count_per_frame,
            "requested_total_selected": int(num_frames * requested_count_per_frame),
            "effective_total_selected": int(selected_counts.sum().item()),
        }

    if selection_mode == "ratio":
        total_requested = max(0, math.ceil(num_frames * num_tokens * ratio))
    else:
        total_requested = max(0, int(fine_topk))
    total_requested = min(total_requested, num_frames * num_tokens)

    if total_requested <= 0:
        empty_idx = torch.empty((num_frames, 0), dtype=torch.long, device=patch_scores.device)
        empty_scores = torch.empty((num_frames, 0), dtype=patch_scores.dtype, device=patch_scores.device)
        empty_counts = torch.zeros((num_frames,), dtype=torch.long, device=patch_scores.device)
        return empty_idx, empty_scores, empty_counts, {
            "selection_mode": selection_mode,
            "selection_scope": selection_scope,
            "requested_count_per_frame": requested_count_per_frame,
            "requested_total_selected": 0,
            "effective_total_selected": 0,
        }

    flat_scores = patch_scores.reshape(-1)
    _, flat_top_indices = torch.topk(flat_scores, k=total_requested, dim=0)
    selected_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    selected_mask[flat_top_indices] = True
    selected_mask = selected_mask.view(num_frames, num_tokens)
    selected_counts = selected_mask.sum(dim=-1).to(torch.long)
    max_selected = int(selected_counts.max().item())

    padded_indices = torch.full(
        (num_frames, max_selected),
        fill_value=-1,
        dtype=torch.long,
        device=patch_scores.device,
    )
    padded_scores = torch.full(
        (num_frames, max_selected),
        fill_value=float("-inf"),
        dtype=patch_scores.dtype,
        device=patch_scores.device,
    )

    for frame_idx in range(num_frames):
        frame_indices = torch.nonzero(selected_mask[frame_idx], as_tuple=False).flatten()
        if frame_indices.numel() == 0:
            continue
        frame_scores = patch_scores[frame_idx].index_select(0, frame_indices)
        frame_scores, order = torch.sort(frame_scores, descending=True)
        frame_indices = frame_indices.index_select(0, order)
        padded_indices[frame_idx, : frame_indices.numel()] = frame_indices
        padded_scores[frame_idx, : frame_scores.numel()] = frame_scores

    return padded_indices, padded_scores, selected_counts, {
        "selection_mode": selection_mode,
        "selection_scope": selection_scope,
        "requested_count_per_frame": requested_count_per_frame,
        "requested_total_selected": int(total_requested),
        "effective_total_selected": int(selected_counts.sum().item()),
    }


def _get_newline_token(model, ref_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if not hasattr(model, "model") or not hasattr(model.model, "image_newline"):
        return None
    return model.model.image_newline.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def _infer_square_grid_size(num_tokens: int) -> int:
    grid_size = int(round(math.sqrt(num_tokens)))
    if grid_size * grid_size != int(num_tokens):
        raise ValueError(f"Expected a square token grid, got num_tokens={num_tokens}.")
    return grid_size


@torch.no_grad()
def pack_already_pooled_tokens(
    model,
    coarse_tokens: torch.Tensor,
) -> torch.Tensor:
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


@torch.no_grad()
def pack_coarse_tokens(
    model,
    encoded_video_features: torch.Tensor,
) -> torch.Tensor:
    coarse_tokens = model.get_2dPool(encoded_video_features)
    return pack_already_pooled_tokens(model, coarse_tokens)


def _detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach_to_cpu(v) for v in obj)
    return obj


@torch.no_grad()
def _pack_selected_frame_tokens(
    selected_frame_features,
    append_newline: bool = True,
    token_scale: float = 1.0,
    newline_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    blocks = []
    ref_tensor = None
    for frame_tokens in selected_frame_features:
        if ref_tensor is None:
            ref_tensor = frame_tokens
        if frame_tokens.numel() == 0:
            continue
        if token_scale != 1.0:
            frame_tokens = frame_tokens * token_scale
        blocks.append(frame_tokens)
        if append_newline and newline_token is not None:
            blocks.append(newline_token.unsqueeze(0))

    if not blocks:
        if ref_tensor is None:
            raise ValueError("selected_frame_features must contain at least one tensor.")
        return ref_tensor.new_zeros((0, ref_tensor.shape[-1]))
    return torch.cat(blocks, dim=0)


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
    selected_frame_features = []
    for frame_idx in range(encoded_video_features.shape[0]):
        frame_indices = selected_indices[frame_idx]
        if frame_indices.numel() == 0:
            selected_frame_features.append(encoded_video_features.new_zeros((0, encoded_video_features.shape[-1])))
            continue
        frame_indices = frame_indices[frame_indices >= 0]
        if frame_indices.numel() == 0:
            selected_frame_features.append(encoded_video_features.new_zeros((0, encoded_video_features.shape[-1])))
            continue
        frame_tokens = encoded_video_features[frame_idx].index_select(0, frame_indices.long())
        selected_frame_features.append(frame_tokens)

    return _pack_selected_frame_tokens(
        selected_frame_features,
        append_newline=append_newline,
        token_scale=fine_scale,
        newline_token=newline_token,
    )


@torch.no_grad()
def compute_contextual_coarse_indices(
    model,
    encoded_video_features: torch.Tensor,
    selected_fine_indices: torch.Tensor,
    patch_scores: Optional[torch.Tensor] = None,
    context_radius: int = 0,
    context_topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    coarse_frame_features = model.get_2dPool(encoded_video_features)
    num_frames, num_coarse_tokens, _ = coarse_frame_features.shape
    fine_grid_size = _infer_square_grid_size(encoded_video_features.shape[1])
    coarse_grid_size = _infer_square_grid_size(num_coarse_tokens)
    coarse_scale_factor = float(fine_grid_size) / float(coarse_grid_size)

    contextual_mask = torch.zeros(
        (num_frames, num_coarse_tokens),
        dtype=torch.bool,
        device=encoded_video_features.device,
    )
    score_tensor = coarse_frame_features.new_zeros((num_frames, num_coarse_tokens), dtype=torch.float32)

    if patch_scores is not None:
        coarse_scores = patch_scores.float().view(num_frames, fine_grid_size, fine_grid_size)
        coarse_scores = torch.nn.functional.interpolate(
            coarse_scores.unsqueeze(1),
            size=(coarse_grid_size, coarse_grid_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        score_tensor = coarse_scores.reshape(num_frames, -1).to(device=encoded_video_features.device)

    for frame_idx in range(num_frames):
        frame_indices = selected_fine_indices[frame_idx]
        frame_indices = frame_indices[frame_indices >= 0]
        if frame_indices.numel() == 0:
            continue
        for fine_idx in frame_indices.tolist():
            fine_row = int(fine_idx) // fine_grid_size
            fine_col = int(fine_idx) % fine_grid_size
            coarse_row = min(
                coarse_grid_size - 1,
                int(math.floor((fine_row / max(fine_grid_size, 1)) * coarse_grid_size)),
            )
            coarse_col = min(
                coarse_grid_size - 1,
                int(math.floor((fine_col / max(fine_grid_size, 1)) * coarse_grid_size)),
            )
            row_start = max(0, coarse_row - int(context_radius))
            row_end = min(coarse_grid_size - 1, coarse_row + int(context_radius))
            col_start = max(0, coarse_col - int(context_radius))
            col_end = min(coarse_grid_size - 1, coarse_col + int(context_radius))
            for rr in range(row_start, row_end + 1):
                for cc in range(col_start, col_end + 1):
                    contextual_mask[frame_idx, rr * coarse_grid_size + cc] = True

    if context_topk is not None and context_topk > 0:
        limited_mask = torch.zeros_like(contextual_mask)
        for frame_idx in range(num_frames):
            candidate_indices = torch.nonzero(contextual_mask[frame_idx], as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                continue
            candidate_scores = score_tensor[frame_idx].index_select(0, candidate_indices)
            topk = min(int(context_topk), candidate_indices.numel())
            top_order = torch.topk(candidate_scores, k=topk, largest=True).indices
            kept = candidate_indices.index_select(0, top_order)
            limited_mask[frame_idx, kept] = True
        contextual_mask = limited_mask

    selected_counts = contextual_mask.sum(dim=-1)
    max_count = int(selected_counts.max().item()) if selected_counts.numel() > 0 else 0
    padded_indices = torch.full(
        (num_frames, max_count),
        -1,
        dtype=torch.long,
        device=encoded_video_features.device,
    )
    padded_scores = coarse_frame_features.new_full((num_frames, max_count), float("-inf"), dtype=torch.float32)

    for frame_idx in range(num_frames):
        frame_indices = torch.nonzero(contextual_mask[frame_idx], as_tuple=False).flatten()
        if frame_indices.numel() == 0:
            continue
        frame_scores = score_tensor[frame_idx].index_select(0, frame_indices)
        frame_scores, order = torch.sort(frame_scores, descending=True)
        frame_indices = frame_indices.index_select(0, order)
        padded_indices[frame_idx, : frame_indices.numel()] = frame_indices
        padded_scores[frame_idx, : frame_scores.numel()] = frame_scores

    metadata = {
        "coarse_grid_size": int(coarse_grid_size),
        "coarse_scale_factor": float(coarse_scale_factor),
        "context_radius": int(context_radius),
        "context_topk": None if context_topk is None else int(context_topk),
        "contextual_selected_counts_per_frame": selected_counts.detach().cpu(),
    }
    return padded_indices, padded_scores, selected_counts, metadata


@torch.no_grad()
def pack_contextual_coarse_tokens(
    model,
    encoded_video_features: torch.Tensor,
    selected_fine_indices: torch.Tensor,
    patch_scores: Optional[torch.Tensor] = None,
    append_newline: bool = True,
    coarse_scale: float = 1.0,
    context_radius: int = 0,
    context_topk: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    coarse_frame_features = model.get_2dPool(encoded_video_features)
    contextual_indices, contextual_scores, contextual_counts, metadata = compute_contextual_coarse_indices(
        model,
        encoded_video_features,
        selected_fine_indices,
        patch_scores=patch_scores,
        context_radius=context_radius,
        context_topk=context_topk,
    )
    newline_token = _get_newline_token(model, coarse_frame_features)
    selected_frame_features = []
    for frame_idx in range(coarse_frame_features.shape[0]):
        frame_indices = contextual_indices[frame_idx]
        frame_indices = frame_indices[frame_indices >= 0]
        if frame_indices.numel() == 0:
            selected_frame_features.append(coarse_frame_features.new_zeros((0, coarse_frame_features.shape[-1])))
            continue
        selected_frame_features.append(coarse_frame_features[frame_idx].index_select(0, frame_indices.long()))

    if not selected_frame_features:
        packed = coarse_frame_features.new_zeros((0, coarse_frame_features.shape[-1]))
    else:
        packed = _pack_selected_frame_tokens(
            selected_frame_features,
            append_newline=append_newline,
            token_scale=coarse_scale,
            newline_token=newline_token,
        )
    metadata.update(
        {
            "contextual_selected_indices": contextual_indices.detach().cpu(),
            "contextual_selected_scores": contextual_scores.detach().cpu(),
            "contextual_coarse_token_count": int(packed.shape[0]),
            "contextual_selected_count_total": int(contextual_counts.sum().item()),
            "contextual_coarse_scale": float(coarse_scale),
        }
    )
    return packed, metadata


@torch.no_grad()
def build_coarse_and_fine_video_features_from_patch_scores(
    model,
    encoded_video_features: torch.Tensor,
    patch_scores: torch.Tensor,
    fine_topk: int = 16,
    fine_ratio: Optional[float] = None,
    selection_scope: str = "per_frame",
    fine_scale: float = 1.0,
    include_coarse: bool = True,
    append_newline: bool = True,
    coarse_mode: str = "full",
    coarse_context_radius: int = 0,
    coarse_context_topk: Optional[int] = None,
    coarse_context_scale: float = 1.0,
    contextual_coarse_first: bool = True,
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

    selected_indices, selected_scores, selected_counts, selection_metadata = select_topk_patch_indices(
        patch_scores,
        fine_topk,
        fine_ratio=fine_ratio,
        selection_scope=selection_scope,
    )
    fine_tokens = pack_fine_selected_tokens(
        model,
        encoded_video_features,
        selected_indices,
        append_newline=append_newline,
        fine_scale=fine_scale,
    )

    coarse_tokens = None
    contextual_metadata = {}
    normalized_coarse_mode = str(coarse_mode).lower()
    if include_coarse:
        if normalized_coarse_mode == "full":
            coarse_tokens = pack_coarse_tokens(model, encoded_video_features)
        elif normalized_coarse_mode == "contextual":
            coarse_tokens, contextual_metadata = pack_contextual_coarse_tokens(
                model,
                encoded_video_features,
                selected_indices,
                patch_scores=patch_scores,
                append_newline=append_newline,
                coarse_scale=coarse_context_scale,
                context_radius=coarse_context_radius,
                context_topk=coarse_context_topk,
            )
        elif normalized_coarse_mode == "full_plus_contextual":
            full_coarse_tokens = pack_coarse_tokens(model, encoded_video_features)
            contextual_tokens, contextual_metadata = pack_contextual_coarse_tokens(
                model,
                encoded_video_features,
                selected_indices,
                patch_scores=patch_scores,
                append_newline=append_newline,
                coarse_scale=coarse_context_scale,
                context_radius=coarse_context_radius,
                context_topk=coarse_context_topk,
            )
            if full_coarse_tokens.numel() == 0:
                coarse_tokens = contextual_tokens
            elif contextual_tokens.numel() == 0:
                coarse_tokens = full_coarse_tokens
            elif contextual_coarse_first:
                coarse_tokens = torch.cat((contextual_tokens, full_coarse_tokens), dim=0)
            else:
                coarse_tokens = torch.cat((full_coarse_tokens, contextual_tokens), dim=0)
        else:
            raise ValueError(f"Unsupported coarse_mode: {coarse_mode}")

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
        "fine_ratio": None if fine_ratio is None else float(fine_ratio),
        "fine_scale": float(fine_scale),
        "include_coarse": bool(include_coarse),
        "append_newline": bool(append_newline),
        "coarse_mode": normalized_coarse_mode,
        "coarse_context_radius": int(coarse_context_radius),
        "coarse_context_topk": None if coarse_context_topk is None else int(coarse_context_topk),
        "coarse_context_scale": float(coarse_context_scale),
        "contextual_coarse_first": bool(contextual_coarse_first),
        "grid_size": grid_size,
        "selected_indices": selected_indices.detach().cpu(),
        "selected_scores": selected_scores.detach().cpu(),
        "selected_counts_per_frame": selected_counts.detach().cpu(),
        "patch_scores": patch_scores.detach().cpu(),
        "coarse_token_count": 0 if coarse_tokens is None else int(coarse_tokens.shape[0]),
        "fine_token_count": int(fine_tokens.shape[0]),
        "combined_token_count": int(combined.shape[0]),
    }
    metadata.update(selection_metadata)
    metadata.update(_detach_to_cpu(contextual_metadata))
    if metadata_extra:
        metadata.update(_detach_to_cpu(metadata_extra))
    return combined, metadata


@torch.no_grad()
def build_coarse_and_fine_video_features(
    model,
    encoded_video_features: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    fine_topk: int = 16,
    fine_ratio: Optional[float] = None,
    selection_scope: str = "per_frame",
    scoring_mode: str = "question_cosine",
    fine_scale: float = 1.0,
    include_coarse: bool = True,
    append_newline: bool = True,
    coarse_mode: str = "full",
    coarse_context_radius: int = 0,
    coarse_context_topk: Optional[int] = None,
    coarse_context_scale: float = 1.0,
    contextual_coarse_first: bool = True,
) -> Tuple[torch.Tensor, Dict]:
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
    return build_coarse_and_fine_video_features_from_patch_scores(
        model,
        encoded_video_features,
        patch_scores,
        fine_topk=fine_topk,
        fine_ratio=fine_ratio,
        selection_scope=selection_scope,
        fine_scale=fine_scale,
        include_coarse=include_coarse,
        append_newline=append_newline,
        coarse_mode=coarse_mode,
        coarse_context_radius=coarse_context_radius,
        coarse_context_topk=coarse_context_topk,
        coarse_context_scale=coarse_context_scale,
        contextual_coarse_first=contextual_coarse_first,
        metadata_extra={"scoring_mode": scoring_mode},
    )
