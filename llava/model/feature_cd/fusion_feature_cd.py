import torch

from .common import (
    apply_token_dropout_and_noise,
    build_inputs_embeds_from_video_features,
    compose_encoded_features,
    compose_visual_only_encoded_features,
    extract_video_branch_features,
    generate_from_inputs_embeds,
    postprocess_video_encoded_features,
)


@torch.no_grad()
def generate_with_fusion_feature_cd(
    model,
    input_ids,
    images,
    attention_mask=None,
    modalities="video",
    cd_guidance_scale=1.0,
    cd_fusion_weak_ratio=0.35,
    cd_fusion_drop_rate=0.05,
    cd_fusion_noise_std=0.005,
    **generate_kwargs,
):
    if modalities != "video":
        raise ValueError("The fusion feature-level CD prototype currently supports only video modality.")

    branch_features = extract_video_branch_features(model, images)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    orig_encoded = compose_encoded_features(model, orig_visual, camera_tokens, patch_tokens)
    visual_only_encoded = compose_visual_only_encoded_features(model, orig_visual)

    fusion_delta = orig_encoded - visual_only_encoded
    weakened_fusion_delta = fusion_delta * cd_fusion_weak_ratio
    neg_encoded = visual_only_encoded + weakened_fusion_delta
    neg_encoded = apply_token_dropout_and_noise(
        neg_encoded,
        drop_rate=cd_fusion_drop_rate,
        noise_std=cd_fusion_noise_std,
    )

    orig_video_features = postprocess_video_encoded_features(model, orig_encoded)
    neg_video_features = postprocess_video_encoded_features(model, neg_encoded)
    guided_video_features = orig_video_features + cd_guidance_scale * (orig_video_features - neg_video_features)

    position_ids, final_attention_mask, inputs_embeds = build_inputs_embeds_from_video_features(
        model,
        input_ids,
        attention_mask,
        guided_video_features,
    )

    return generate_from_inputs_embeds(
        model,
        position_ids,
        final_attention_mask,
        inputs_embeds,
        **generate_kwargs,
    )
