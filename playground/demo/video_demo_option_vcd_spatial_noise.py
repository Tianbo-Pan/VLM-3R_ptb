from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from vcd.vcd_feature_degradation.methods import build_spatial_noise_branch
from vcd.vcd_feature_degradation.option_demo_utils import add_common_arguments, inject_default_demo_argv, run_option_demo


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-1 option-level VCD demo with spatial-token degradation.")
    add_common_arguments(parser)
    parser.add_argument("--spatial_drop_rate", type=float, default=0.15, help="Drop rate for patch tokens.")
    parser.add_argument("--spatial_noise_std", type=float, default=0.10, help="Gaussian noise std for patch tokens.")
    return parser.parse_args()


if __name__ == "__main__":
    inject_default_demo_argv(
        [
            "--output_dir",
            "./work_dirs/phase1_option_demo/spatial_noise_vcd",
            "--output_name",
            "spatial_noise_vcd",
            "--alpha",
            "1.0",
            "--spatial_drop_rate",
            "0.15",
            "--spatial_noise_std",
            "0.10",
        ]
    )
    args = parse_args()
    run_option_demo(
        args,
        setting_name="phase1_option_vcd_spatial_noise",
        degraded_branch_builder=build_spatial_noise_branch,
    )
