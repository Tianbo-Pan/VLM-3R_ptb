import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX


TensorPair = Tuple[torch.Tensor, torch.Tensor]


def _ensure_video_tensor(images):
    if isinstance(images, list):
        if len(images) != 1:
            raise ValueError(f"Feature-level CD helpers currently expect a single video sample, got {len(images)} samples.")
        return images[0]
    if images.ndim == 5:
        if images.shape[0] != 1:
            raise ValueError(f"Feature-level CD helpers currently expect batch size 1, got {images.shape[0]}.")
        return images[0]
    if images.ndim == 4:
        return images
    raise ValueError(f"Unexpected video tensor shape: {tuple(images.shape)}")


@torch.no_grad()
def extract_video_branch_features(model, images: Sequence[torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
    video = _ensure_video_tensor(images)
    visual_features = model.get_model().get_vision_tower()(video)

    spatial_tower = model.get_model().get_spatial_tower()
    if spatial_tower is None:
        return {
            "video": video,
            "visual_features": visual_features,
            "camera_tokens": None,
            "patch_tokens": None,
        }

    spatial_encoder_type = getattr(model.get_model().config, "spatial_tower", "")
    if spatial_encoder_type.endswith("points"):
        raise NotImplementedError("Feature-level CD helpers do not yet support point-based spatial towers.")

    camera_tokens, patch_tokens = spatial_tower(video)
    return {
        "video": video,
        "visual_features": visual_features,
        "camera_tokens": camera_tokens,
        "patch_tokens": patch_tokens,
    }


def apply_token_dropout_and_noise(
    features: Optional[torch.Tensor],
    drop_rate: float = 0.0,
    noise_std: float = 0.0,
) -> Optional[torch.Tensor]:
    if features is None:
        return None

    out = features.clone()
    if drop_rate > 0:
        keep_mask = (torch.rand(out.shape[:-1], device=out.device) > drop_rate).to(out.dtype).unsqueeze(-1)
        out = out * keep_mask
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out


@torch.no_grad()
def compose_encoded_features(
    model,
    visual_features: torch.Tensor,
    camera_tokens: Optional[torch.Tensor],
    patch_tokens: Optional[torch.Tensor],
) -> torch.Tensor:
    mm_projector = model.get_model().mm_projector
    fusion_block = model.get_model().get_fusion_block()
    spatial_encoder_type = getattr(model.get_model().config, "spatial_tower", "")
    fusion_block_type = getattr(model.get_model().config, "fusion_block", None)

    if camera_tokens is None and patch_tokens is None:
        return mm_projector(visual_features)

    if spatial_encoder_type.endswith("points"):
        raise NotImplementedError("Point-based spatial towers are not supported in the feature-level CD helpers.")

    if fusion_block_type == "cross_attention":
        spatial_tower_select_feature = getattr(model.config, "spatial_tower_select_feature", "patch_tokens")
        spatial_tower_select_feature_list = spatial_tower_select_feature.split(",")
        final_image_features = []
        for feature_name in spatial_tower_select_feature_list:
            if feature_name == "camera_tokens":
                final_image_features.append(camera_tokens)
            elif feature_name == "patch_tokens":
                final_image_features.append(patch_tokens)
            elif feature_name == "all":
                final_image_features = [camera_tokens, patch_tokens]
                break
            else:
                raise ValueError(f"Unexpected spatial_tower_select_feature: {feature_name}")
        final_image_features = torch.cat(final_image_features, dim=1).to(model.dtype)
        fused_features, _ = fusion_block(visual_features, final_image_features)
        return mm_projector(fused_features)

    if fusion_block_type == "cross_attention_with_mlp":
        fused_features, _ = fusion_block(visual_features, patch_tokens)
        return mm_projector(fused_features)

    if fusion_block_type == "transformer":
        spatial_tower_select_feature = getattr(model.config, "spatial_tower_select_feature", "patch_tokens")
        if spatial_tower_select_feature != "all":
            raise ValueError("The transformer fusion block helper currently expects spatial_tower_select_feature='all'.")
        final_image_features = torch.cat((camera_tokens, patch_tokens), dim=1).to(model.dtype)
        fused_features = fusion_block(visual_features, final_image_features)
        return mm_projector(fused_features)

    if fusion_block_type in ["mlp_after_clip_proj", "concat_mlp", "concat_self_attention"]:
        projected_features = mm_projector(visual_features)
        return fusion_block(projected_features, patch_tokens)

    if fusion_block is None:
        return mm_projector(visual_features)

    raise ValueError(f"Unsupported fusion_block type for feature-level CD: {fusion_block_type}")


@torch.no_grad()
def compose_visual_only_encoded_features(model, visual_features: torch.Tensor) -> torch.Tensor:
    return model.get_model().mm_projector(visual_features)


@torch.no_grad()
def postprocess_video_encoded_features(model, encoded_video_features: torch.Tensor) -> torch.Tensor:
    if encoded_video_features.ndim != 3:
        raise ValueError(f"Expected encoded video features with shape [frames, tokens, dim], got {tuple(encoded_video_features.shape)}")

    image_features = model.get_2dPool(encoded_video_features)
    mm_patch_merge_type = getattr(model.config, "mm_patch_merge_type", "flat")
    mm_newline_position = getattr(model.config, "mm_newline_position", "one_token")

    if mm_patch_merge_type == "flat":
        return image_features.flatten(0, 1)

    if not mm_patch_merge_type.startswith("spatial"):
        raise ValueError(f"Unsupported mm_patch_merge_type for feature-level CD helpers: {mm_patch_merge_type}")

    if mm_newline_position == "grid":
        return model.add_token_per_grid(image_features)
    if mm_newline_position == "frame":
        return model.add_token_per_frame(image_features).flatten(0, 1)
    if mm_newline_position == "one_token":
        image_features = image_features.flatten(0, 1)
        if "unpad" in mm_patch_merge_type:
            image_features = torch.cat((image_features, model.model.image_newline[None].to(image_features.device)), dim=0)
        return image_features
    if mm_newline_position == "no_token":
        return image_features.flatten(0, 1)

    raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")


@torch.no_grad()
def build_inputs_embeds_from_video_features(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    video_features: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)

    original_attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()

    filtered_input_ids = [cur_ids[cur_mask] for cur_ids, cur_mask in zip(input_ids, attention_mask)]
    new_input_embeds = []

    for cur_input_ids in filtered_input_ids:
        image_token_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        if not image_token_positions:
            new_input_embeds.append(model.get_model().embed_tokens(cur_input_ids))
            continue

        image_token_indices = [-1] + image_token_positions + [cur_input_ids.shape[0]]
        cur_input_ids_noim = [
            cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            for i in range(len(image_token_indices) - 1)
        ]
        split_sizes = [segment.shape[0] for segment in cur_input_ids_noim]
        text_embeds = model.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        text_embeds = torch.split(text_embeds, split_sizes, dim=0)

        cur_new_embeds = []
        for idx in range(len(text_embeds)):
            cur_new_embeds.append(text_embeds[idx])
            if idx < len(text_embeds) - 1:
                cur_new_embeds.append(video_features)

        new_input_embeds.append(torch.cat(cur_new_embeds, dim=0))

    tokenizer_model_max_length = getattr(model.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [embed[:tokenizer_model_max_length] for embed in new_input_embeds]

    max_len = max(embed.shape[0] for embed in new_input_embeds)
    batch_size = len(new_input_embeds)

    padded_embeds = []
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=input_ids.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_ids.device)

    padding_side = getattr(model.config, "tokenizer_padding_side", "right")
    for i, cur_embed in enumerate(new_input_embeds):
        cur_len = cur_embed.shape[0]
        if padding_side == "left":
            pad = torch.zeros((max_len - cur_len, cur_embed.shape[1]), dtype=cur_embed.dtype, device=cur_embed.device)
            padded_embeds.append(torch.cat((pad, cur_embed), dim=0))
            padded_attention_mask[i, -cur_len:] = True
            position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=torch.long, device=cur_embed.device)
        else:
            pad = torch.zeros((max_len - cur_len, cur_embed.shape[1]), dtype=cur_embed.dtype, device=cur_embed.device)
            padded_embeds.append(torch.cat((cur_embed, pad), dim=0))
            padded_attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=cur_embed.device)

    inputs_embeds = torch.stack(padded_embeds, dim=0)
    if original_attention_mask is None:
        final_attention_mask = None
        final_position_ids = None
    else:
        final_attention_mask = padded_attention_mask.to(dtype=original_attention_mask.dtype)
        final_position_ids = position_ids
    return final_position_ids, final_attention_mask, inputs_embeds


@torch.no_grad()
def generate_from_inputs_embeds(model, position_ids, attention_mask, inputs_embeds, **generate_kwargs):
    return super(type(model), model).generate(
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **generate_kwargs,
    )
