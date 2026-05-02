# VCD new generator

This directory evaluates the **general generation-time VCD** decoder on VSIBench.

## Default question types

The default shell script now uses `QUESTION_TYPES=all`, i.e. all question types found in the loaded VSIBench source, with `PER_TYPE_LIMIT=5` by default.

For the May 1, 2026 VSIBench log this means:

- `obj_appearance_order`
- `object_abs_distance`
- `object_counting`
- `object_rel_direction_easy`
- `object_rel_direction_hard`
- `object_rel_direction_medium`
- `object_rel_distance`
- `object_size_estimation`
- `room_size_estimation`
- `route_planning`

## Compared settings

- `baseline_generate`: standard model `generate(...)`
- `pairwise_gen_vcd`: pairwise generation-time VCD with a coarse positive branch and a semantic-negative branch

## Run

```bash
bash vcd/vcd_new_generator/run_vsibench_gen_vcd_5_types.sh
```

Example with a fixed historical VSIBench sample source:

```bash
CUDA_VISIBLE_DEVICES=4 \
SOURCE_LOG_JSON=/local_home/pantianbo/projects/vision_reasoning/VLM-3R/logs/20260501/vsibench/0501_1035_vlm_3r_7b_qwen2_lora_vlm_3r_model_args_70e1b2/vsibench.json \
PER_TYPE_LIMIT=5 \
bash vcd/vcd_new_generator/run_vsibench_gen_vcd_5_types.sh
```

## Outputs

- `results/<run_name>/results.json`
- `results/<run_name>/summary.json`
- `results/<run_name>/summary.csv`
