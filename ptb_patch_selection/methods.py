from typing import Dict, Optional, Sequence, Tuple

import torch

from llava.model.feature_cd.common import (
    build_inputs_embeds_from_video_features,
    compose_encoded_features,
    extract_video_branch_features,
    generate_from_inputs_embeds,
)

from .fusion_guided_selection import (
    build_fusion_guided_patch_scores,
    compose_encoded_features_with_fusion_details,
    is_fusion_guided_scoring_mode,
)
from .selective_pooling import (
    build_question_embedding,
    build_question_token_embeddings,
    build_coarse_and_fine_video_features,
    build_coarse_and_fine_video_features_from_patch_scores,
    pack_coarse_tokens,
    pack_fine_selected_tokens,
    score_patch_tokens,
    select_topk_patch_indices,
)


@torch.no_grad()
def build_selective_patch_video_features(
    model,
    input_ids: torch.Tensor,
    images: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    modalities: str = "video",
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
    fusion_2d_weight: float = 1.0,
    fusion_3d_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    if modalities != "video":
        raise ValueError("Selective patch pooling currently supports only video modality.")

    branch_features = extract_video_branch_features(model, images)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    if is_fusion_guided_scoring_mode(scoring_mode):
        encoded_video_features, fusion_details = compose_encoded_features_with_fusion_details(
            model,
            orig_visual,
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
        return build_coarse_and_fine_video_features_from_patch_scores(
            model,
            encoded_video_features,
            patch_scores=patch_scores,
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
            metadata_extra=fusion_metadata,
        )

    encoded_video_features = compose_encoded_features(model, orig_visual, camera_tokens, patch_tokens)
    return build_coarse_and_fine_video_features(
        model,
        encoded_video_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        fine_topk=fine_topk,
        fine_ratio=fine_ratio,
        selection_scope=selection_scope,
        scoring_mode=scoring_mode,
        fine_scale=fine_scale,
        include_coarse=include_coarse,
        append_newline=append_newline,
        coarse_mode=coarse_mode,
        coarse_context_radius=coarse_context_radius,
        coarse_context_topk=coarse_context_topk,
        coarse_context_scale=coarse_context_scale,
        contextual_coarse_first=contextual_coarse_first,
    )


@torch.no_grad()
def generate_with_selective_patch_pooling(
    model,
    input_ids,
    images,
    attention_mask=None,
    tokenizer=None,
    modalities="video",
    fine_topk=16,
    fine_ratio=None,
    selection_scope="per_frame",
    scoring_mode="question_cosine",
    fine_scale=1.0,
    include_coarse=True,
    append_newline=True,
    coarse_mode="full",
    coarse_context_radius=0,
    coarse_context_topk=None,
    coarse_context_scale=1.0,
    contextual_coarse_first=True,
    fusion_2d_weight=1.0,
    fusion_3d_weight=1.0,
    return_metadata=False,
    **generate_kwargs,
):
    video_features, metadata = build_selective_patch_video_features(
        model=model,
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        modalities=modalities,
        fine_topk=fine_topk,
        fine_ratio=fine_ratio,
        selection_scope=selection_scope,
        scoring_mode=scoring_mode,
        fine_scale=fine_scale,
        include_coarse=include_coarse,
        append_newline=append_newline,
        coarse_mode=coarse_mode,
        coarse_context_radius=coarse_context_radius,
        coarse_context_topk=coarse_context_topk,
        coarse_context_scale=coarse_context_scale,
        contextual_coarse_first=contextual_coarse_first,
        fusion_2d_weight=fusion_2d_weight,
        fusion_3d_weight=fusion_3d_weight,
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


@torch.no_grad()
def build_fusion_guided_patch_video_features(
    model,
    input_ids: torch.Tensor,
    images: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    modalities: str = "video",
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
    fusion_2d_weight: float = 1.0,
    fusion_3d_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    return build_selective_patch_video_features(
        model=model,
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        modalities=modalities,
        fine_topk=fine_topk,
        fine_ratio=fine_ratio,
        selection_scope=selection_scope,
        scoring_mode="fusion_2d3d",
        fine_scale=fine_scale,
        include_coarse=include_coarse,
        append_newline=append_newline,
        coarse_mode=coarse_mode,
        coarse_context_radius=coarse_context_radius,
        coarse_context_topk=coarse_context_topk,
        coarse_context_scale=coarse_context_scale,
        contextual_coarse_first=contextual_coarse_first,
        fusion_2d_weight=fusion_2d_weight,
        fusion_3d_weight=fusion_3d_weight,
    )


@torch.no_grad()
def generate_with_fusion_guided_patch_pooling(
    model,
    input_ids,
    images,
    attention_mask=None,
    tokenizer=None,
    modalities="video",
    fine_topk=16,
    fine_ratio=None,
    selection_scope="per_frame",
    fine_scale=1.0,
    include_coarse=True,
    append_newline=True,
    coarse_mode="full",
    coarse_context_radius=0,
    coarse_context_topk=None,
    coarse_context_scale=1.0,
    contextual_coarse_first=True,
    fusion_2d_weight=1.0,
    fusion_3d_weight=1.0,
    return_metadata=False,
    **generate_kwargs,
):
    return generate_with_selective_patch_pooling(
        model=model,
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        modalities=modalities,
        fine_topk=fine_topk,
        fine_ratio=fine_ratio,
        selection_scope=selection_scope,
        scoring_mode="fusion_2d3d",
        fine_scale=fine_scale,
        include_coarse=include_coarse,
        append_newline=append_newline,
        coarse_mode=coarse_mode,
        coarse_context_radius=coarse_context_radius,
        coarse_context_topk=coarse_context_topk,
        coarse_context_scale=coarse_context_scale,
        contextual_coarse_first=contextual_coarse_first,
        fusion_2d_weight=fusion_2d_weight,
        fusion_3d_weight=fusion_3d_weight,
        return_metadata=return_metadata,
        **generate_kwargs,
    )


@torch.no_grad()
def build_augmented_patch_video_features(
    model,
    input_ids: torch.Tensor,
    images: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    modalities: str = "video",
    patch_topk: int = 16,
    patch_ratio: Optional[float] = None,
    selection_scope: str = "global",
    scoring_mode: str = "question_cosine",
    injection_mode: str = "inplace_boost_coarse",
    boost_factor: float = 2.0,
    background_decay: float = 0.98,
    fine_scale: float = 1.0,
    append_newline: bool = True,
    include_coarse: bool = True,
    fusion_2d_weight: float = 1.0,
    fusion_3d_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    if modalities != "video":
        raise ValueError("Augmented patch branch currently supports only video modality.")
    if injection_mode not in {"inplace_boost_coarse", "inplace_boost_coarse_tail"}:
        raise ValueError(f"Unsupported augmentation injection_mode: {injection_mode}")

    branch_features = extract_video_branch_features(model, images)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    if is_fusion_guided_scoring_mode(scoring_mode):
        encoded_video_features, fusion_details = compose_encoded_features_with_fusion_details(
            model,
            orig_visual,
            camera_tokens,
            patch_tokens,
        )
        patch_scores, extra_metadata = build_fusion_guided_patch_scores(
            encoded_video_features,
            fusion_details,
            scoring_mode=scoring_mode,
            fusion_2d_weight=fusion_2d_weight,
            fusion_3d_weight=fusion_3d_weight,
        )
    else:
        encoded_video_features = compose_encoded_features(model, orig_visual, camera_tokens, patch_tokens)
        if scoring_mode in {"question_token_max", "question_token_topk_mean"}:
            question_embedding = build_question_token_embeddings(
                model,
                input_ids,
                attention_mask,
                tokenizer=tokenizer,
                content_token_filter=True,
            )
        else:
            question_embedding = build_question_embedding(
                model,
                input_ids,
                attention_mask,
                tokenizer=tokenizer,
            )
        patch_scores = score_patch_tokens(encoded_video_features, question_embedding, mode=scoring_mode)
        extra_metadata = {"scoring_mode": scoring_mode}

    selected_indices, selected_scores, selected_counts, selection_metadata = select_topk_patch_indices(
        patch_scores,
        fine_topk=patch_topk,
        fine_ratio=patch_ratio,
        selection_scope=selection_scope,
    )

    boosted_encoded = encoded_video_features.clone()
    if background_decay != 1.0:
        boosted_encoded = boosted_encoded * float(background_decay)
    for frame_idx in range(boosted_encoded.shape[0]):
        frame_indices = selected_indices[frame_idx]
        frame_indices = frame_indices[frame_indices >= 0]
        if frame_indices.numel() == 0:
            continue
        boosted_encoded[frame_idx, frame_indices.long()] = (
            encoded_video_features[frame_idx, frame_indices.long()] * float(boost_factor)
        )

    coarse_tokens = pack_coarse_tokens(model, boosted_encoded) if include_coarse else None
    if injection_mode == "inplace_boost_coarse":
        if coarse_tokens is None:
            video_features = pack_fine_selected_tokens(
                model,
                boosted_encoded,
                selected_indices,
                append_newline=append_newline,
                fine_scale=fine_scale,
            )
        else:
            video_features = coarse_tokens
    else:
        boosted_fine_tokens = pack_fine_selected_tokens(
            model,
            boosted_encoded,
            selected_indices,
            append_newline=append_newline,
            fine_scale=fine_scale,
        )
        if coarse_tokens is None:
            video_features = boosted_fine_tokens
        elif boosted_fine_tokens.numel() == 0:
            video_features = coarse_tokens
        else:
            video_features = torch.cat((coarse_tokens, boosted_fine_tokens), dim=0)

    metadata = {
        "augmentation_mode": injection_mode,
        "patch_topk": int(patch_topk),
        "patch_ratio": None if patch_ratio is None else float(patch_ratio),
        "selection_scope": selection_scope,
        "scoring_mode": scoring_mode,
        "boost_factor": float(boost_factor),
        "background_decay": float(background_decay),
        "fine_scale": float(fine_scale),
        "include_coarse": bool(include_coarse),
        "append_newline": bool(append_newline),
        "video_token_count": int(video_features.shape[0]),
        "selected_indices": selected_indices.detach().cpu(),
        "selected_scores": selected_scores.detach().cpu(),
        "selected_counts_per_frame": selected_counts.detach().cpu(),
        "patch_scores": patch_scores.detach().cpu(),
        "branch_features": {
            "visual_features_shape": list(orig_visual.shape),
            "camera_tokens_shape": list(camera_tokens.shape),
            "patch_tokens_shape": list(patch_tokens.shape),
            "encoded_video_features_shape": list(encoded_video_features.shape),
        },
    }
    metadata.update(selection_metadata)
    metadata.update(extra_metadata)
    return video_features, metadata


@torch.no_grad()
def generate_with_augmented_patch_pooling(
    model,
    input_ids,
    images,
    attention_mask=None,
    tokenizer=None,
    modalities="video",
    patch_topk=16,
    patch_ratio=None,
    selection_scope="global",
    scoring_mode="question_cosine",
    injection_mode="inplace_boost_coarse",
    boost_factor=2.0,
    background_decay=0.98,
    fine_scale=1.0,
    append_newline=True,
    include_coarse=True,
    fusion_2d_weight=1.0,
    fusion_3d_weight=1.0,
    return_metadata=False,
    **generate_kwargs,
):
    video_features, metadata = build_augmented_patch_video_features(
        model=model,
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        modalities=modalities,
        patch_topk=patch_topk,
        patch_ratio=patch_ratio,
        selection_scope=selection_scope,
        scoring_mode=scoring_mode,
        injection_mode=injection_mode,
        boost_factor=boost_factor,
        background_decay=background_decay,
        fine_scale=fine_scale,
        append_newline=append_newline,
        include_coarse=include_coarse,
        fusion_2d_weight=fusion_2d_weight,
        fusion_3d_weight=fusion_3d_weight,
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
