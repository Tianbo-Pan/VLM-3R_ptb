import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from phase1_option_vcd_utils import add_common_arguments, inject_default_demo_argv, run_option_demo


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-1 option-level baseline scoring demo for VLM-3R.")
    add_common_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    inject_default_demo_argv(
        [
            "--output_dir",
            "./work_dirs/phase1_option_demo/baseline",
            "--output_name",
            "baseline",
        ]
    )
    args = parse_args()
    run_option_demo(args, setting_name="phase1_option_baseline")
