from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from vcd.methods import build_rgb_noise_branch
from vcd.option_demo_utils import add_common_arguments, inject_default_demo_argv, run_option_demo


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-1 option-level VCD demo with RGB Gaussian-noise degradation.")
    add_common_arguments(parser)
    parser.add_argument("--rgb_noise_std", type=float, default=0.15, help="Gaussian noise std applied to video pixels.")
    return parser.parse_args()


if __name__ == "__main__":
    inject_default_demo_argv(
        [
            "--output_dir",
            "./work_dirs/phase1_option_demo/rgb_noise_vcd",
            "--output_name",
            "rgb_noise_vcd",
            "--alpha",
            "1.0",
            "--rgb_noise_std",
            "0.15",
        ]
    )
    args = parse_args()
    run_option_demo(args, setting_name="phase1_option_vcd_rgb_noise", degraded_branch_builder=build_rgb_noise_branch)
