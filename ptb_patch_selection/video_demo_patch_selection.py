import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from ptb_patch_selection import generate_with_selective_patch_pooling


DEFAULT_DEMO_ARGV = [
    "--model-path", "Journey9ni/vlm-3r-llava-qwen2-lora",
    "--model-base", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2",
    "--video_path", "playground/demo/47334096.mp4",
    "--output_dir", "./work_dirs/video_demo/ptb_patch_selection_demo",
    "--output_name", "pred",
    "--overwrite", "True",
    "--mm_spatial_pool_stride", "2",
    "--for_get_frames_num", "32",
    "--conv-mode", "qwen_1_5",
    "--mm_spatial_pool_mode", "average",
    "--mm_newline_position", "grid",
    "--fine_topk", "16",
    "--scoring_mode", "question_cosine",
    "--prompt", "If I am standing by the stool and facing the stove, is the sofa to my left, right, or back?\nAn object is to my back if I would have to turn at least 135 degrees in order to face it.",
]


def inject_default_demo_argv():
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_DEMO_ARGV)
        print("No CLI arguments detected. Using built-in demo arguments.")


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-selection demo for VLM-3R scheme A.")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="pred")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--overwrite", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--load_8bit", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mm_newline_position", type=str, default="grid")
    parser.add_argument("--force_sample", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--add_time_instruction", type=lambda x: str(x).lower() == "true", default=False)

    parser.add_argument("--fine_topk", type=int, default=16)
    parser.add_argument("--scoring_mode", type=str, default="question_cosine")
    parser.add_argument("--fine_scale", type=float, default=1.0)
    parser.add_argument("--include_coarse", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--append_newline", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--save_patch_visualization", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--patch_box_thickness", type=int, default=3)
    parser.add_argument("--patch_score_precision", type=int, default=3)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


def load_video(video_path, args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "0.00s", 0.0

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    video_time = total_frame_num / vr.get_avg_fps()

    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    frame_time_text = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time_text, video_time


def build_overwrite_config(args):
    overwrite_config = {
        "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
        "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
        "mm_newline_position": args.mm_newline_position,
    }
    cfg_pretrained = AutoConfig.from_pretrained(args.model_path)
    if "qwen" not in args.model_path.lower():
        patches_per_side = 16 if "224" in getattr(cfg_pretrained, "mm_vision_tower", "") else 24
        least_token_number = args.for_get_frames_num * (patches_per_side // args.mm_spatial_pool_stride) ** 2 + 1000
        scaling_factor = math.ceil(least_token_number / 4096)
        if scaling_factor >= 2:
            if "vicuna" in cfg_pretrained._name_or_path.lower():
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
    return overwrite_config, cfg_pretrained


def _color_lerp(low_bgr, high_bgr, t: float):
    t = float(max(0.0, min(1.0, t)))
    return tuple(int(round(low_bgr[i] * (1.0 - t) + high_bgr[i] * t)) for i in range(3))


def save_patch_selection_visualization(
    frames_np: np.ndarray,
    metadata: dict,
    output_dir: str,
    patch_box_thickness: int = 3,
    patch_score_precision: int = 3,
):
    selected_indices = metadata["selected_indices"].numpy()
    selected_scores = metadata["selected_scores"].numpy()
    patch_scores = metadata["patch_scores"].numpy()
    grid_size = int(metadata["grid_size"])
    fine_topk = int(metadata["fine_topk"])

    vis_dir = os.path.join(output_dir, "patch_selection_vis")
    os.makedirs(vis_dir, exist_ok=True)

    manifest = []
    low_color = (255, 140, 0)   # orange in BGR
    high_color = (0, 0, 255)    # red in BGR
    text_color = (255, 255, 255)

    for frame_idx, frame in enumerate(frames_np):
        frame_bgr = frame[:, :, ::-1].copy()
        height, width = frame_bgr.shape[:2]
        patch_h = height / grid_size
        patch_w = width / grid_size

        cur_indices = selected_indices[frame_idx].tolist()
        cur_scores = selected_scores[frame_idx].tolist()
        cur_patch_scores = patch_scores[frame_idx]

        max_score = max(cur_scores) if cur_scores else 1.0
        min_score = min(cur_scores) if cur_scores else 0.0
        denom = max(max_score - min_score, 1e-6)

        selected_entries = []
        for rank, (patch_idx, score) in enumerate(zip(cur_indices, cur_scores), start=1):
            row = int(patch_idx) // grid_size
            col = int(patch_idx) % grid_size
            x1 = int(round(col * patch_w))
            y1 = int(round(row * patch_h))
            x2 = int(round((col + 1) * patch_w))
            y2 = int(round((row + 1) * patch_h))
            color_t = (score - min_score) / denom
            color = _color_lerp(low_color, high_color, color_t)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness=patch_box_thickness)

            label = f"{rank}:{score:.{patch_score_precision}f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_x = x1
            text_y = max(y1 + text_h + 2, text_h + 2)
            cv2.rectangle(
                frame_bgr,
                (text_x, max(0, text_y - text_h - baseline - 2)),
                (min(width - 1, text_x + text_w + 4), min(height - 1, text_y + baseline)),
                color,
                thickness=-1,
            )
            cv2.putText(
                frame_bgr,
                label,
                (text_x + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA,
            )

            selected_entries.append(
                {
                    "rank": rank,
                    "patch_idx": int(patch_idx),
                    "row": row,
                    "col": col,
                    "score": float(score),
                    "raw_patch_score": float(cur_patch_scores[int(patch_idx)]),
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )

        frame_label = f"frame {frame_idx:02d} | topk={fine_topk} | grid={grid_size}x{grid_size}"
        cv2.putText(
            frame_bgr,
            frame_label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        save_path = os.path.join(vis_dir, f"frame_{frame_idx:04d}_topk.png")
        cv2.imwrite(save_path, frame_bgr)
        manifest.append(
            {
                "frame_idx": frame_idx,
                "save_path": save_path,
                "selected_patches": selected_entries,
            }
        )

    manifest_path = os.path.join(vis_dir, "patch_selection_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "grid_size": grid_size,
                "fine_topk": fine_topk,
                "frames": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return vis_dir, manifest_path


def main():
    inject_default_demo_argv()
    args = parse_args()

    model_name = get_model_name_from_path(args.model_path)
    overwrite_config, cfg_pretrained = build_overwrite_config(args)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        overwrite_config=overwrite_config if args.overwrite else None,
    )
    model.to("cuda")

    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction

    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")

    raw_video_frames, frame_time, video_time = load_video(args.video_path, args)
    video = image_processor.preprocess(raw_video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    video = [video]

    question = args.prompt
    qs = question
    if args.add_time_instruction:
        qs = (
            f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}.Please answer the following questions related to this video.\n{qs}"
        )
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 151643
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids, metadata = generate_with_selective_patch_pooling(
            model,
            input_ids=input_ids,
            images=video,
            attention_mask=attention_masks,
            modalities="video",
            fine_topk=args.fine_topk,
            scoring_mode=args.scoring_mode,
            fine_scale=args.fine_scale,
            include_coarse=args.include_coarse,
            append_newline=args.append_newline,
            return_metadata=True,
            do_sample=False,
            temperature=0.0,
            top_p=0.1,
            num_beams=1,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=[stopping_criteria] if "mistral" not in cfg_pretrained._name_or_path.lower() else None,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if "mistral" not in cfg_pretrained._name_or_path.lower() and outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    print(f"Question: {question}\n")
    print(f"Response: {outputs}\n")
    print(
        "Patch-selection metadata:",
        json.dumps(
            {
                "fine_topk": metadata["fine_topk"],
                "coarse_token_count": metadata["coarse_token_count"],
                "fine_token_count": metadata["fine_token_count"],
                "combined_token_count": metadata["combined_token_count"],
                "scoring_mode": metadata["scoring_mode"],
            },
            ensure_ascii=False,
        ),
    )

    vis_dir = None
    manifest_path = None
    if args.save_patch_visualization:
        vis_dir, manifest_path = save_patch_selection_visualization(
            raw_video_frames,
            metadata,
            output_dir=args.output_dir,
            patch_box_thickness=args.patch_box_thickness,
            patch_score_precision=args.patch_score_precision,
        )
        print(f"Saved patch-selection visualization to: {vis_dir}")

    result = {
        "Q": question,
        "video_name": args.video_path,
        "pred": outputs,
        "patch_selection": {
            "fine_topk": metadata["fine_topk"],
            "coarse_token_count": metadata["coarse_token_count"],
            "fine_token_count": metadata["fine_token_count"],
            "combined_token_count": metadata["combined_token_count"],
            "scoring_mode": metadata["scoring_mode"],
            "selected_indices": metadata["selected_indices"].tolist(),
            "selected_scores": metadata["selected_scores"].tolist(),
            "patch_visualization_dir": vis_dir,
            "patch_visualization_manifest": manifest_path,
        },
    }
    with open(answers_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
