from typing import Dict, Optional, Sequence, Tuple

import torch

from llava.model.feature_cd.common import (
    build_inputs_embeds_from_video_features,
    compose_encoded_features,
    extract_video_branch_features,
    generate_from_inputs_embeds,
)

from .selective_pooling import build_coarse_and_fine_video_features


@torch.no_grad()
def build_selective_patch_video_features(
    model,
    input_ids: torch.Tensor,
    images: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    modalities: str = "video",
    fine_topk: int = 16,
    scoring_mode: str = "question_cosine",
    fine_scale: float = 1.0,
    include_coarse: bool = True,
    append_newline: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    if modalities != "video":
        raise ValueError("Selective patch pooling currently supports only video modality.")

    branch_features = extract_video_branch_features(model, images)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    encoded_video_features = compose_encoded_features(model, orig_visual, camera_tokens, patch_tokens)
    return build_coarse_and_fine_video_features(
        model,
        encoded_video_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        fine_topk=fine_topk,
        scoring_mode=scoring_mode,
        fine_scale=fine_scale,
        include_coarse=include_coarse,
        append_newline=append_newline,
    )


@torch.no_grad()
def generate_with_selective_patch_pooling(
    model,
    input_ids,
    images,
    attention_mask=None,
    modalities="video",
    fine_topk=16,
    scoring_mode="question_cosine",
    fine_scale=1.0,
    include_coarse=True,
    append_newline=True,
    return_metadata=False,
    **generate_kwargs,
):
    video_features, metadata = build_selective_patch_video_features(
        model=model,
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        modalities=modalities,
        fine_topk=fine_topk,
        scoring_mode=scoring_mode,
        fine_scale=fine_scale,
        include_coarse=include_coarse,
        append_newline=append_newline,
    )

    position_ids, final_attention_mask, inputs_embeds = build_inputs_embeds_from_video_features(
        model,
        input_ids,
        attention_mask,
        video_features,
    )
    output_ids = generate_from_inputs_embeds(
        model,
        position_ids,
        final_attention_mask,
        inputs_embeds,
        **generate_kwargs,
    )
    if return_metadata:
        return output_ids, metadata
    return output_ids
