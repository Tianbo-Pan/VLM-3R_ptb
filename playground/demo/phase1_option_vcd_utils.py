from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""Backward-compatible shim.

The VCD demo/eval implementation now lives under `vcd/option_demo_utils.py` so the
modified VCD code is centralized in the repo-level `vcd/` directory.
"""

from vcd.vcd_feature_degradation.option_demo_utils import *  # noqa: F401,F403
