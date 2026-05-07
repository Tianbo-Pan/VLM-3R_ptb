#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_SCRIPT="${SCRIPT_DIR}/eval_vlm_3r_gen_vcd_tri_branch_vsibench_15_per_type.sh"

declare -a CONFIGS=(
  "tri_cons_alpha10_beta5e3|CONTRAST_MODE=tri_rectified CONTRAST_ALPHAS=1.0:1.0 BETA=0.005 AUG_BOOST_FACTOR=1.5 PATCH_WARP_RATIO=0.2"
  "tri_cons_alpha08_beta3e3|CONTRAST_MODE=tri_rectified CONTRAST_ALPHAS=0.8:0.8 BETA=0.003 AUG_BOOST_FACTOR=1.5 PATCH_WARP_RATIO=0.2"
  "tri_deg_stronger|CONTRAST_MODE=tri_rectified CONTRAST_ALPHAS=0.8:1.2 BETA=0.005 AUG_BOOST_FACTOR=1.5 PATCH_WARP_RATIO=0.2"
  "tri_aug_stronger|CONTRAST_MODE=tri_rectified CONTRAST_ALPHAS=1.2:0.8 BETA=0.005 AUG_BOOST_FACTOR=2.0 PATCH_WARP_RATIO=0.15"
)

echo "Launching tri-branch 15-per-type sweep with ${#CONFIGS[@]} configs"

for config in "${CONFIGS[@]}"; do
    name="${config%%|*}"
    kvs="${config#*|}"
    echo
    echo "=============================="
    echo "Running config: ${name}"
    echo "Overrides: ${kvs}"
    echo "=============================="

    # shellcheck disable=SC2086
    env RUN_SUFFIX="${name}" OUTPUT_ROOT="logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_gen_vcd_tri_branch_sweep/${name}" ${kvs} bash "${RUN_SCRIPT}"
done
