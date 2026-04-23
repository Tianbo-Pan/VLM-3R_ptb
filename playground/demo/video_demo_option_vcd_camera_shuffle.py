import argparse

from phase1_option_vcd_utils import add_common_arguments, apply_view_shuffle, extract_spatial_features, inject_default_demo_argv, run_option_demo


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-1 option-level VCD demo with camera/view-order shuffle degradation.")
    add_common_arguments(parser)
    return parser.parse_args()


def build_camera_shuffle_branch(args, model, video):
    original_spatial = extract_spatial_features(model, video)
    degraded_spatial, permutation = apply_view_shuffle(original_spatial)
    return {
        "orig_images": video,
        "orig_spatial_features": original_spatial,
        "degraded_images": video,
        "degraded_spatial_features": degraded_spatial,
        "metadata": {"frame_permutation": permutation},
    }


if __name__ == "__main__":
    inject_default_demo_argv(
        [
            "--output_dir",
            "./work_dirs/phase1_option_demo/camera_shuffle_vcd",
            "--output_name",
            "camera_shuffle_vcd",
            "--alpha",
            "1.0",
        ]
    )
    args = parse_args()
    run_option_demo(
        args,
        setting_name="phase1_option_vcd_camera_shuffle",
        degraded_branch_builder=build_camera_shuffle_branch,
    )
