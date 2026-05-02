import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


FUSION_GUIDED_SCORING_MODES = {
    "fusion_2d3d",
    "fusion_2d",
    "fusion_3d",
}


def is_fusion_guided_scoring_mode(mode: str) -> bool:
    return mode in FUSION_GUIDED_SCORING_MODES


def _normalize_spatial_feature_names(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        return ["patch_tokens"]
    normalized = []
    for name in str(raw_value).split(","):
        name = name.strip()
        if not name:
            continue
        if name == "all_tokens":
            name = "all"
        normalized.append(name)
    return normalized or ["patch_tokens"]


def _collect_spatial_features(
    model,
    camera_tokens: Optional[torch.Tensor],
    patch_tokens: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, slice]]:
    feature_names = _normalize_spatial_feature_names(
        getattr(model.config, "spatial_tower_select_feature", "patch_tokens")
    )
    selected_features = []
    slices: Dict[str, slice] = {}
    cursor = 0

    def append_feature(name: str, value: Optional[torch.Tensor]):
        nonlocal cursor
        if value is None:
            return
        selected_features.append(value)
        slices[name] = slice(cursor, cursor + value.shape[1])
        cursor += value.shape[1]

    for feature_name in feature_names:
        if feature_name == "camera_tokens":
            append_feature("camera_tokens", camera_tokens)
        elif feature_name == "patch_tokens":
            append_feature("patch_tokens", patch_tokens)
        elif feature_name == "all":
            append_feature("camera_tokens", camera_tokens)
            append_feature("patch_tokens", patch_tokens)
        else:
            raise ValueError(f"Unexpected spatial_tower_select_feature: {feature_name}")

    if not selected_features:
        raise ValueError("No spatial features selected for fusion-guided patch selection.")
    return torch.cat(selected_features, dim=1).to(model.dtype), slices


@torch.no_grad()
def compose_encoded_features_with_fusion_details(
    model,
    visual_features: torch.Tensor,
    camera_tokens: Optional[torch.Tensor],
    patch_tokens: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict]:
    mm_projector = model.get_model().mm_projector
    fusion_block = model.get_model().get_fusion_block()
    spatial_encoder_type = getattr(model.get_model().config, "spatial_tower", "")
    fusion_block_type = getattr(model.get_model().config, "fusion_block", None)

    visual_only_encoded = mm_projector(visual_features)
    fusion_details: Dict = {
        "fusion_block_type": fusion_block_type,
        "visual_only_encoded": visual_only_encoded,
        "camera_tokens": camera_tokens,
        "patch_tokens": patch_tokens,
        "attn_weights": None,
        "spatial_feature_slices": {},
        "spatial_importance_source": None,
    }

    if camera_tokens is None and patch_tokens is None:
        return visual_only_encoded, fusion_details

    if spatial_encoder_type.endswith("points"):
        raise NotImplementedError("Fusion-guided patch selection does not yet support point-based spatial towers.")

    if fusion_block_type == "cross_attention":
        final_image_features, feature_slices = _collect_spatial_features(model, camera_tokens, patch_tokens)
        fused_features, attn_weights = fusion_block(visual_features, final_image_features)
        fusion_details["attn_weights"] = attn_weights
        fusion_details["spatial_feature_slices"] = feature_slices
        fusion_details["spatial_importance_source"] = "cross_attention_received"
        return mm_projector(fused_features), fusion_details

    if fusion_block_type == "cross_attention_with_mlp":
        if patch_tokens is None:
            raise ValueError("cross_attention_with_mlp requires patch_tokens for fusion-guided patch selection.")
        fused_features, attn_weights = fusion_block(visual_features, patch_tokens)
        fusion_details["attn_weights"] = attn_weights
        fusion_details["spatial_feature_slices"] = {"patch_tokens": slice(0, patch_tokens.shape[1])}
        fusion_details["spatial_importance_source"] = "cross_attention_received"
        return mm_projector(fused_features), fusion_details

    if fusion_block_type == "transformer":
        final_image_features, feature_slices = _collect_spatial_features(model, camera_tokens, patch_tokens)
        fused_features = fusion_block(visual_features, final_image_features)
        fusion_details["spatial_feature_slices"] = feature_slices
        fusion_details["spatial_importance_source"] = "patch_token_norm"
        return mm_projector(fused_features), fusion_details

    if fusion_block_type in ["mlp_after_clip_proj", "concat_mlp", "concat_self_attention"]:
        if patch_tokens is None:
            raise ValueError(f"{fusion_block_type} requires patch_tokens for fusion-guided patch selection.")
        fusion_details["spatial_feature_slices"] = {"patch_tokens": slice(0, patch_tokens.shape[1])}
        fusion_details["spatial_importance_source"] = "patch_token_norm"
        return fusion_block(visual_only_encoded, patch_tokens), fusion_details

    if fusion_block is None:
        return visual_only_encoded, fusion_details

    raise ValueError(f"Unsupported fusion_block type for fusion-guided patch selection: {fusion_block_type}")


def _normalize_per_frame(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    min_scores = scores.amin(dim=-1, keepdim=True)
    max_scores = scores.amax(dim=-1, keepdim=True)
    return (scores - min_scores) / (max_scores - min_scores).clamp_min(1e-6)


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


@torch.no_grad()
def build_fusion_guided_patch_scores(
    encoded_video_features: torch.Tensor,
    fusion_details: Dict,
    scoring_mode: str = "fusion_2d3d",
    fusion_2d_weight: float = 1.0,
    fusion_3d_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    if scoring_mode not in FUSION_GUIDED_SCORING_MODES:
        raise ValueError(f"Unsupported fusion-guided scoring mode: {scoring_mode}")

    num_frames, num_tokens, _ = encoded_video_features.shape
    visual_only_encoded = fusion_details.get("visual_only_encoded")
    patch_tokens = fusion_details.get("patch_tokens")
    attn_weights = fusion_details.get("attn_weights")
    patch_slice = fusion_details.get("spatial_feature_slices", {}).get("patch_tokens")

    delta_scores = None
    if visual_only_encoded is not None and tuple(visual_only_encoded.shape) == tuple(encoded_video_features.shape):
        delta_scores = (encoded_video_features.float() - visual_only_encoded.float()).norm(dim=-1)

    spatial_scores = None
    spatial_score_source = None
    if attn_weights is not None and patch_slice is not None:
        spatial_scores = attn_weights.float().mean(dim=1)[:, patch_slice]
        spatial_scores = _resize_token_scores(spatial_scores, num_tokens)
        spatial_score_source = "attn_received"
    elif patch_tokens is not None:
        spatial_scores = patch_tokens.float().norm(dim=-1)
        spatial_scores = _resize_token_scores(spatial_scores, num_tokens)
        spatial_score_source = "patch_norm"

    normalized_delta = _normalize_per_frame(delta_scores) if delta_scores is not None else None
    normalized_spatial = _normalize_per_frame(spatial_scores) if spatial_scores is not None else None

    score_terms = []
    if scoring_mode in {"fusion_2d3d", "fusion_2d"} and normalized_delta is not None:
        score_terms.append((float(fusion_2d_weight), normalized_delta))
    if scoring_mode in {"fusion_2d3d", "fusion_3d"} and normalized_spatial is not None:
        score_terms.append((float(fusion_3d_weight), normalized_spatial))

    if not score_terms:
        combined_scores = encoded_video_features.float().norm(dim=-1)
        fallback_source = "encoded_feature_norm_fallback"
    else:
        total_weight = sum(weight for weight, _ in score_terms)
        combined_scores = sum(weight * values for weight, values in score_terms) / max(total_weight, 1e-6)
        fallback_source = None

    metadata = {
        "scoring_mode": scoring_mode,
        "fusion_2d_weight": float(fusion_2d_weight),
        "fusion_3d_weight": float(fusion_3d_weight),
        "fusion_delta_scores": normalized_delta,
        "fusion_spatial_scores": normalized_spatial,
        "fusion_spatial_score_source": spatial_score_source,
        "fusion_has_attention": bool(attn_weights is not None),
        "fusion_score_fallback_source": fallback_source,
        "fusion_patch_token_count": 0 if patch_tokens is None else int(patch_tokens.shape[1]),
        "fusion_frame_count": int(num_frames),
    }
    return combined_scores, metadata
