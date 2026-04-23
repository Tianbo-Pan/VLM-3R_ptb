#!/usr/bin/env python3
"""Experiment 1: simplest attention grounding test on ptb_test cases.

This script builds on experiment 0:
1. Run the same VLM-3R inference + fusion-attention extraction.
2. Recover the attention map for the manifest-selected best frame.
3. Compare the best-frame attention against GT object regions using:
   - instance masks (preferred)
   - bbox fallback metrics
4. Save quantitative grounding metrics and a best-frame visualization.

Default evaluation scope:
- best frame only (because the current ptb_test manifest stores rendered instance
  annotations for the selected best frame)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

import experiment0_attention_rollout as exp0


DEFAULT_OUTPUT_ROOT = Path(
    "/local_home/pantianbo/projects/vision_reasoning/VLM-3R/ptb_test/experiment1_outputs"
)

PALETTE = [
    (231, 76, 60),
    (52, 152, 219),
    (155, 89, 182),
    (26, 188, 156),
    (243, 156, 18),
    (46, 204, 113),
    (230, 126, 34),
    (241, 196, 15),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: simplest attention grounding metrics.")
    parser.add_argument("--manifest-path", type=Path, default=exp0.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--qa-id", type=str, default=None)
    parser.add_argument("--scene-name", type=str, default=None)
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--flat-index", type=int, default=None)

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--mm-spatial-pool-mode", type=str, default="average")
    parser.add_argument("--mm-newline-position", type=str, default="grid")

    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)

    parser.add_argument("--threshold-quantile", type=float, default=0.85, help="Quantile for binarizing attention heatmap.")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--font-size", type=int, default=18)
    return parser.parse_args()


def resize_attention_map(frame_map: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    resized = cv2.resize(frame_map.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    return resized


def normalize_unit_interval(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)


def normalize_to_probability(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    total = float(arr.sum())
    if total <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / total


def bbox_to_mask(bbox: Sequence[int], shape_hw: Tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def load_instance_map(path: Path) -> np.ndarray:
    instance_img = Image.open(path)
    instance_map = np.array(instance_img)
    if instance_map.ndim == 3:
        instance_map = instance_map[..., 0]
    return instance_map


def mask_from_instance_id(instance_map: np.ndarray, instance_id: int) -> np.ndarray:
    return instance_map == int(instance_id)


def binary_attention_mask(attn_unit: np.ndarray, quantile: float) -> np.ndarray:
    quantile = min(max(float(quantile), 0.0), 1.0)
    thresh = float(np.quantile(attn_unit, quantile))
    return attn_unit >= thresh


def contour_overlay(base_rgb: np.ndarray, mask: np.ndarray, color_rgb: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    overlay = base_rgb.copy()
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    cv2.drawContours(overlay, contours, -1, color_bgr, thickness=thickness)
    return overlay


def compute_region_metrics(
    attn_unit: np.ndarray,
    attn_prob: np.ndarray,
    attn_binary: np.ndarray,
    region_mask: np.ndarray,
) -> Dict[str, float]:
    region_mask = region_mask.astype(bool)
    total_pixels = int(region_mask.size)
    region_pixels = int(region_mask.sum())
    area_fraction = float(region_pixels / max(total_pixels, 1))

    mass = float(attn_prob[region_mask].sum()) if region_pixels > 0 else 0.0
    density = float(mass / max(area_fraction, 1e-12)) if region_pixels > 0 else 0.0

    inter_binary = np.logical_and(attn_binary, region_mask).sum()
    union_binary = np.logical_or(attn_binary, region_mask).sum()
    iou = float(inter_binary / union_binary) if union_binary > 0 else 0.0

    soft_inter = float((attn_unit * region_mask.astype(np.float32)).sum())
    soft_dice = float(
        2.0 * soft_inter / max(float(attn_unit.sum()) + float(region_mask.sum()), 1e-12)
    )

    if attn_unit.size == 0:
        pointing_hit = 0.0
        peak_xy = [-1, -1]
    else:
        peak_flat = int(np.argmax(attn_unit))
        peak_y, peak_x = np.unravel_index(peak_flat, attn_unit.shape)
        pointing_hit = float(region_mask[peak_y, peak_x])
        peak_xy = [int(peak_x), int(peak_y)]

    return {
        "area_pixels": region_pixels,
        "area_fraction": area_fraction,
        "attention_mass": mass,
        "attention_density": density,
        "soft_dice": soft_dice,
        "binary_iou": iou,
        "pointing_hit": pointing_hit,
        "peak_xy": peak_xy,
    }


def group_instances_by_object(drawn_instances: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in drawn_instances:
        grouped.setdefault(item["object_name"], []).append(item)
    return grouped


def evaluate_best_frame_grounding(
    case,
    best_frame_map: np.ndarray,
    threshold_quantile: float,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Image.Image]:
    render_info = case.question_entry["render_info"]
    frame_path = Path(render_info["frame_color_path"])
    instance_map_path = Path(render_info["frame_instance_path"])
    drawn_instances = render_info.get("drawn_instances", [])

    frame_img = Image.open(frame_path).convert("RGB")
    frame_np = np.array(frame_img)
    height, width = frame_np.shape[:2]

    attn_resized = resize_attention_map(best_frame_map, (width, height))
    attn_unit = normalize_unit_interval(attn_resized)
    attn_prob = normalize_to_probability(attn_unit)
    attn_binary = binary_attention_mask(attn_unit, threshold_quantile)

    instance_map = load_instance_map(instance_map_path)
    if instance_map.shape[:2] != (height, width):
        raise ValueError(
            f"Instance map shape {instance_map.shape[:2]} does not match RGB frame shape {(height, width)} "
            f"for {frame_path}"
        )

    per_instance_metrics = []
    union_mask = np.zeros((height, width), dtype=bool)
    overlay_np = frame_np.copy()

    for idx, item in enumerate(drawn_instances):
        object_name = item["object_name"]
        instance_id = int(item["instance_id"])
        bbox = item["bbox_2d"]
        color = PALETTE[idx % len(PALETTE)]

        instance_mask = mask_from_instance_id(instance_map, instance_id)
        has_mask = bool(instance_mask.any())
        bbox_mask = bbox_to_mask(bbox, (height, width))

        target_mask = instance_mask if has_mask else bbox_mask
        region_metrics = compute_region_metrics(attn_unit, attn_prob, attn_binary, target_mask)
        bbox_metrics = compute_region_metrics(attn_unit, attn_prob, attn_binary, bbox_mask)

        union_mask |= target_mask
        overlay_np = contour_overlay(overlay_np, target_mask, color_rgb=color, thickness=2)

        per_instance_metrics.append(
            {
                "object_name": object_name,
                "instance_id": instance_id,
                "bbox_2d": bbox,
                "used_mask_type": "instance_mask" if has_mask else "bbox_fallback",
                "instance_mask_available": has_mask,
                "region_metrics": region_metrics,
                "bbox_metrics": bbox_metrics,
            }
        )

    grouped = group_instances_by_object(drawn_instances)
    per_object_metrics = []
    for object_name, items in grouped.items():
        object_union_mask = np.zeros((height, width), dtype=bool)
        object_union_bbox = np.zeros((height, width), dtype=bool)
        for item in items:
            inst_mask = mask_from_instance_id(instance_map, int(item["instance_id"]))
            has_mask = bool(inst_mask.any())
            object_union_mask |= (inst_mask if has_mask else bbox_to_mask(item["bbox_2d"], (height, width)))
            object_union_bbox |= bbox_to_mask(item["bbox_2d"], (height, width))

        per_object_metrics.append(
            {
                "object_name": object_name,
                "num_instances": len(items),
                "region_metrics": compute_region_metrics(attn_unit, attn_prob, attn_binary, object_union_mask),
                "bbox_metrics": compute_region_metrics(attn_unit, attn_prob, attn_binary, object_union_bbox),
            }
        )

    union_metrics = compute_region_metrics(attn_unit, attn_prob, attn_binary, union_mask)
    background_metrics = compute_region_metrics(attn_unit, attn_prob, attn_binary, ~union_mask)

    heatmap_uint8 = np.clip(attn_unit * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(overlay_np, 0.55, heatmap_rgb, 0.45, 0.0)
    overlay_pil = Image.fromarray(overlay_rgb)

    draw = ImageDraw.Draw(overlay_pil)
    font = exp0.find_font(18)
    title = f"[BEST FRAME] {frame_path.name} | union_mass={union_metrics['attention_mass']:.3f} | bg_mass={background_metrics['attention_mass']:.3f}"
    title_bbox = draw.textbbox((0, 0), title, font=font)
    draw.rectangle([0, 0, title_bbox[2] - title_bbox[0] + 16, title_bbox[3] - title_bbox[1] + 10], fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255), font=font)

    return (
        {
            "frame_path": str(frame_path),
            "frame_instance_path": str(instance_map_path),
            "threshold_quantile": float(threshold_quantile),
            "union_relevant_metrics": union_metrics,
            "background_metrics": background_metrics,
            "per_instance_metrics": per_instance_metrics,
            "per_object_metrics": per_object_metrics,
        },
        attn_unit,
        attn_binary.astype(np.uint8),
        overlay_pil,
    )


def save_binary_attention_visualization(
    frame_path: Path,
    attn_binary: np.ndarray,
    output_path: Path,
    title: str,
):
    frame = Image.open(frame_path).convert("RGB")
    frame_np = np.array(frame)
    binary_color = np.zeros_like(frame_np)
    binary_color[..., 0] = attn_binary * 255
    overlay = cv2.addWeighted(frame_np, 0.72, binary_color, 0.28, 0.0)
    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)
    font = exp0.find_font(18)
    text_bbox = draw.textbbox((0, 0), title, font=font)
    draw.rectangle([0, 0, text_bbox[2] - text_bbox[0] + 16, text_bbox[3] - text_bbox[1] + 10], fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255), font=font)
    overlay_pil.save(output_path)


def main():
    args = parse_args()
    manifest = exp0.load_manifest(args.manifest_path)
    case = exp0.select_case(manifest, args)

    frame_paths_all = exp0.load_scene_frame_paths(case.frame_dir)
    try:
        best_frame_idx = frame_paths_all.index(case.best_frame_path)
    except ValueError as exc:
        raise ValueError(f"Best frame {case.best_frame_path} not found in {case.frame_dir}") from exc

    sampled_indices = exp0.sample_frame_indices(len(frame_paths_all), args.num_frames, forced_idx=best_frame_idx)
    sampled_frame_paths = [frame_paths_all[i] for i in sampled_indices]

    tokenizer, model, image_processor, _ = exp0.load_model_and_processor(args)
    output_text, attn_weights, capture_source = exp0.run_inference_with_attention(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        frame_paths=sampled_frame_paths,
        prompt=case.prompt,
        conv_mode=args.conv_mode,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    patch_side = exp0.get_spatial_patch_side(model)
    frame_maps, attn_meta = exp0.aggregate_attention_maps(attn_weights, patch_side_hint=patch_side)
    if frame_maps.shape[0] != len(sampled_frame_paths):
        raise RuntimeError(
            f"Attention frame count {frame_maps.shape[0]} != sampled frame count {len(sampled_frame_paths)}"
        )

    best_local_idx = sampled_frame_paths.index(case.best_frame_path)
    best_frame_map = frame_maps[best_local_idx]

    grounding_summary, attn_unit, attn_binary, overlay_pil = evaluate_best_frame_grounding(
        case=case,
        best_frame_map=best_frame_map,
        threshold_quantile=args.threshold_quantile,
    )

    run_dir = args.output_root / case.scene_name / f"{case.qa_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = run_dir / "best_frame_grounding_overlay.jpg"
    overlay_pil.save(overlay_path)

    binary_overlay_path = run_dir / "best_frame_attention_binary.jpg"
    save_binary_attention_visualization(
        frame_path=case.best_frame_path,
        attn_binary=attn_binary,
        output_path=binary_overlay_path,
        title=f"Attention binary map @ q={args.threshold_quantile:.2f}",
    )

    attn_unit_path = run_dir / "best_frame_attention_unit.npy"
    np.save(attn_unit_path, attn_unit)

    summary = {
        "qa_id": case.qa_id,
        "scene_name": case.scene_name,
        "prompt": case.prompt,
        "prediction": output_text,
        "capture_source": capture_source,
        "attention_meta": attn_meta,
        "best_frame_path": str(case.best_frame_path),
        "sampled_frame_paths": [str(p) for p in sampled_frame_paths],
        "grounding_summary": grounding_summary,
        "overlay_path": str(overlay_path),
        "binary_overlay_path": str(binary_overlay_path),
        "attention_unit_path": str(attn_unit_path),
        "manifest_question_entry": case.question_entry,
    }
    summary_path = run_dir / "grounding_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"[Experiment 1] scene={case.scene_name} qa_id={case.qa_id}")
    print(f"Prompt    : {case.prompt}")
    print(f"Prediction: {output_text}")
    print(f"Best frame: {case.best_frame_path.name}")
    print(f"Union mass: {grounding_summary['union_relevant_metrics']['attention_mass']:.4f}")
    print(f"Saved overlay to: {overlay_path}")
    print(f"Saved summary to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
