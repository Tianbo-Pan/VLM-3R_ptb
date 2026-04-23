import torch

from .common import (
    apply_token_dropout_and_noise,
    build_inputs_embeds_from_video_features,
    compose_encoded_features,
    extract_video_branch_features,
    generate_from_inputs_embeds,
    postprocess_video_encoded_features,
)


@torch.no_grad()
def generate_with_2d_feature_cd(
    model,
    input_ids,
    images,
    attention_mask=None,
    modalities="video",
    cd_guidance_scale=1.0,
    cd_visual_drop_rate=0.10,
    cd_visual_noise_std=0.01,
    **generate_kwargs,
):
    if modalities != "video":
        raise ValueError("The 2D feature-level CD prototype currently supports only video modality.")

    branch_features = extract_video_branch_features(model, images)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    orig_encoded = compose_encoded_features(model, orig_visual, camera_tokens, patch_tokens)

    neg_visual = apply_token_dropout_and_noise(
        orig_visual,
        drop_rate=cd_visual_drop_rate,
        noise_std=cd_visual_noise_std,
    )
    neg_encoded = compose_encoded_features(model, neg_visual, camera_tokens, patch_tokens)

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
