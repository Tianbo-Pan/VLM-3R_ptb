from .option_demo_utils import (
    apply_gaussian_noise_to_video,
    apply_patch_token_noise,
    extract_spatial_features,
)
from llava.model.feature_cd.common import (
    apply_token_dropout_and_noise,
    compose_encoded_features,
    compose_visual_only_encoded_features,
    extract_video_branch_features,
    postprocess_video_encoded_features,
)


def build_2d_feature_branch(args, model, video):
    branch_features = extract_video_branch_features(model, video)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    orig_encoded = compose_encoded_features(model, orig_visual, camera_tokens, patch_tokens)
    degraded_visual = apply_token_dropout_and_noise(
        orig_visual,
        drop_rate=args.visual_drop_rate,
        noise_std=args.visual_noise_std,
    )
    degraded_encoded = compose_encoded_features(model, degraded_visual, camera_tokens, patch_tokens)

    return {
        "orig_video_features": postprocess_video_encoded_features(model, orig_encoded),
        "degraded_video_features": postprocess_video_encoded_features(model, degraded_encoded),
        "metadata": {
            "visual_drop_rate": args.visual_drop_rate,
            "visual_noise_std": args.visual_noise_std,
        },
    }


def build_weak_fusion_branch(args, model, video):
    branch_features = extract_video_branch_features(model, video)
    orig_visual = branch_features["visual_features"]
    camera_tokens = branch_features["camera_tokens"]
    patch_tokens = branch_features["patch_tokens"]

    if camera_tokens is None and patch_tokens is None:
        raise ValueError("Weak-fusion branch requires spatial features, but the current model returned no camera/patch tokens.")

    orig_encoded = compose_encoded_features(model, orig_visual, camera_tokens, patch_tokens)
    visual_only_encoded = compose_visual_only_encoded_features(model, orig_visual)

    fusion_delta = orig_encoded - visual_only_encoded
    weakened_fusion_delta = fusion_delta * args.fusion_weak_ratio
    weak_fusion_encoded = visual_only_encoded + weakened_fusion_delta
    weak_fusion_encoded = apply_token_dropout_and_noise(
        weak_fusion_encoded,
        drop_rate=args.fusion_drop_rate,
        noise_std=args.fusion_noise_std,
    )

    return {
        "orig_video_features": postprocess_video_encoded_features(model, orig_encoded),
        "degraded_video_features": postprocess_video_encoded_features(model, weak_fusion_encoded),
        "metadata": {
            "fusion_weak_ratio": args.fusion_weak_ratio,
            "fusion_drop_rate": args.fusion_drop_rate,
            "fusion_noise_std": args.fusion_noise_std,
        },
    }


def build_rgb_noise_branch(args, model, video):
    if hasattr(args, "visual_drop_rate") or hasattr(args, "visual_noise_std"):
        return build_2d_feature_branch(args, model, video)
    return build_pixel_noise_branch(args, model, video)


def build_pixel_noise_branch(args, model, video):
    degraded_video = apply_gaussian_noise_to_video(video, args.rgb_noise_std)
    return {
        "degraded_images": degraded_video,
        "degraded_spatial_features": None,
        "metadata": {"rgb_noise_std": args.rgb_noise_std},
    }


def build_spatial_noise_branch(args, model, video):
    original_spatial = extract_spatial_features(model, video)
    degraded_spatial = apply_patch_token_noise(
        original_spatial,
        drop_rate=args.spatial_drop_rate,
        noise_std=args.spatial_noise_std,
    )
    return {
        "orig_images": video,
        "orig_spatial_features": original_spatial,
        "degraded_images": video,
        "degraded_spatial_features": degraded_spatial,
        "metadata": {
            "spatial_drop_rate": args.spatial_drop_rate,
            "spatial_noise_std": args.spatial_noise_std,
        },
    }
