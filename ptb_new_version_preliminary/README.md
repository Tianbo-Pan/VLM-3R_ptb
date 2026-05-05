# ptb_new_version_preliminary

Preliminary generation-time VCD variant for VLM-3R:

- use the **question** to query **projected 2D visual-only tokens**
- map the selected locations onto **3D patch tokens before fusion**
- perturb the selected **3D patch tokens**
- re-run normal 2D/3D fusion to build the negative branch

This is intended as a cleaner “semantic localization first, spatial corruption second”
baseline than the previous fusion-after-warp implementation.
