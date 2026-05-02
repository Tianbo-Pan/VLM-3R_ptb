#!/usr/bin/env python3
"""Experiment 2: patch-score distribution analysis for selective pooling.

This script is intended as a lightweight prior experiment for the current
question-conditioned patch-selection prototype. It reuses the existing ptb_test
manifest / attention extraction pipeline and adds analyses for:

1. Raw patch-score distributions per frame.
2. Best-frame GT region coverage under score-based top-k patch ranking.
3. Agreement between question-cosine patch scores and fusion-attention maps.
4. Saved overlays / raw numpy dumps for qualitative inspection.

Compared with experiment 0 / 1, this script focuses on the *score signal*
produced by ptb_patch_selection.selective_pooling rather than only attention.
"""

from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
sys.path.append("/local_home/pantianbo/projects/vision_reasoning/VLM-3R")

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import experiment0_attention_rollout as exp0
import experiment1_attention_grounding as exp1
from ptb_patch_selection.methods import build_selective_patch_video_features


DEFAULT_OUTPUT_ROOT = Path(
    "/local_home/pantianbo/projects/vision_reasoning/VLM-3R/ptb_test/experiment2_patch_score_analysis"
)

DEFAULT_DEMO_ARGV = [
    "--model-path", "Journey9ni/vlm-3r-llava-qwen2-lora",
    "--model-base", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2",
    "--flat-index", "0",
    "--overwrite",
    "--mm-spatial-pool-stride", "2",
    "--mm-spatial-pool-mode", "average",
    "--mm-newline-position", "grid",
    "--num-frames", "32",
    "--fine-topk", "16",
    "--scoring-mode", "question_cosine",
    "--compare-topk", "4,8,16,32",
    "--coverage-thresholds", "0.0,0.25,0.5",
    "--save-frame-overlays",
]


def inject_default_demo_argv():
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_DEMO_ARGV)
        print("No CLI arguments detected. Using built-in demo arguments.")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2: patch-score distribution analysis.")
    parser.add_argument("--manifest-path", type=Path, default=exp0.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--qa-id", type=str, default=None)
    parser.add_argument("--scene-name", type=str, default=None)
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--flat-index", type=int, default=None)
    parser.add_argument("--run-all", action="store_true", help="Run all questions in the manifest.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap when --run-all is set.")

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

    parser.add_argument("--fine-topk", type=int, default=16)
    parser.add_argument("--scoring-mode", type=str, default="question_cosine")
    parser.add_argument("--fine-scale", type=float, default=1.0)
    parser.add_argument("--include-coarse", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--append-newline", type=lambda x: str(x).lower() == "true", default=True)

    parser.add_argument(
        "--compare-topk",
        type=str,
        default="4,8,16,32",
        help="Comma-separated top-k values for score/attention overlap and GT patch coverage.",
    )
    parser.add_argument(
        "--coverage-thresholds",
        type=str,
        default="0.0,0.25,0.5",
        help="Comma-separated thresholds for converting patch coverage ratios into GT-positive patches.",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--save-frame-overlays", action="store_true")
    parser.add_argument(
        "--visualize-topk",
        type=int,
        default=None,
        help="Top-k patches to visualize for score-vs-attention comparison on the best frame. Defaults to fine_topk.",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError(f"Failed to parse any integers from: {text}")
    return sorted(set(out))


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise ValueError(f"Failed to parse any floats from: {text}")
    return sorted(set(out))


def build_case_list(manifest: Dict[str, Any], args) -> List[exp0.SelectedCase]:
    if not args.run_all:
        return [exp0.select_case(manifest, args)]

    cases: List[exp0.SelectedCase] = []
    flat = exp0.flatten_manifest_questions(manifest)
    for idx, (scene_entry, question_entry) in enumerate(flat):
        frame_path = Path(question_entry["render_info"]["frame_color_path"])
        case = exp0.SelectedCase(
            qa_id=question_entry["qa_id"],
            scene_name=scene_entry["scene_name"],
            scene_entry=scene_entry,
            question_entry=question_entry,
            frame_dir=frame_path.parent,
            best_frame_path=frame_path,
            prompt=question_entry["parsed_question"]["question_text"],
        )
        cases.append(case)
        if args.max_cases is not None and len(cases) >= args.max_cases:
            break
    return cases


def prepare_video_inputs(
    model,
    tokenizer,
    image_processor,
    frame_paths: Sequence[Path],
    prompt: str,
    conv_mode: str,
    device: str,
):
    pil_images = exp0.load_pil_images(frame_paths)
    pixel_values = image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"].half().to(device)
    video = [pixel_values]

    full_prompt = exp0.build_video_prompt(model, prompt, conv_mode)
    input_ids = exp0.tokenizer_image_token(
        full_prompt,
        tokenizer,
        exp0.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 151643
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)
    return video, input_ids, attention_mask


@torch.no_grad()
def run_generation_with_attention(
    model,
    tokenizer,
    video,
    input_ids,
    attention_mask,
    conv_mode: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
):
    conv = exp0.conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != exp0.SeparatorStyle.TWO else conv.sep2
    stopping_criteria = exp0.KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with exp0.FusionAttentionRecorder(model) as recorder:
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=video,
                attention_mask=attention_mask,
                modalities="video",
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)]
    output_text = output_text.strip()

    if recorder.captured is None:
        raise RuntimeError("No fusion attention was captured. Check the fusion block type and hooks.")
    return output_text, recorder.captured, recorder.capture_source


@torch.no_grad()
def compute_patch_selection_metadata(
    model,
    input_ids,
    video,
    attention_mask,
    args,
):
    _, metadata = build_selective_patch_video_features(
        model=model,
        input_ids=input_ids,
        images=video,
        attention_mask=attention_mask,
        modalities="video",
        fine_topk=args.fine_topk,
        scoring_mode=args.scoring_mode,
        fine_scale=args.fine_scale,
        include_coarse=args.include_coarse,
        append_newline=args.append_newline,
    )
    return metadata


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - float(np.max(x))
    ex = np.exp(x)
    denom = float(ex.sum())
    if denom <= 0:
        return np.zeros_like(ex)
    return ex / denom


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    if a.size != b.size or a.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_score_stats(frame_scores: np.ndarray) -> Dict[str, float]:
    flat = frame_scores.reshape(-1).astype(np.float64)
    sorted_desc = np.sort(flat)[::-1]
    prob = softmax_np(flat)
    entropy = float(-(prob * np.log(prob + 1e-12)).sum())
    numel = flat.size

    def top_mean(k: int) -> float:
        if numel == 0:
            return 0.0
        return float(sorted_desc[: min(k, numel)].mean())

    top1 = float(sorted_desc[0]) if numel > 0 else 0.0
    top2 = float(sorted_desc[1]) if numel > 1 else top1
    return {
        "mean": float(flat.mean()) if numel > 0 else 0.0,
        "std": float(flat.std()) if numel > 0 else 0.0,
        "min": float(flat.min()) if numel > 0 else 0.0,
        "max": float(flat.max()) if numel > 0 else 0.0,
        "entropy": entropy,
        "softmax_entropy": entropy,
        "top1": top1,
        "top5_mean": top_mean(5),
        "top10_mean": top_mean(10),
        "top16_mean": top_mean(16),
        "margin_top1_top2": float(top1 - top2),
        "positive_fraction": float((flat > 0).mean()) if numel > 0 else 0.0,
    }


def normalize_maps_global(score_maps: np.ndarray) -> np.ndarray:
    maps = score_maps.astype(np.float32).copy()
    vmin = float(maps.min())
    vmax = float(maps.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(maps, dtype=np.float32)
    return (maps - vmin) / (vmax - vmin)


def normalize_single_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)


def patch_index_to_xyxy(patch_idx: int, grid_size: int, width: int, height: int) -> Tuple[int, int, int, int]:
    row = int(patch_idx) // grid_size
    col = int(patch_idx) % grid_size
    x1 = int(round(col * width / grid_size))
    y1 = int(round(row * height / grid_size))
    x2 = int(round((col + 1) * width / grid_size))
    y2 = int(round((row + 1) * height / grid_size))
    return x1, y1, x2, y2


def build_heatmap_overlay_pil(
    frame_path: Path,
    heatmap_2d: np.ndarray,
    alpha: float,
) -> Image.Image:
    image = Image.open(frame_path).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    heatmap_unit = normalize_single_map(heatmap_2d)
    heatmap_resized = cv2.resize(heatmap_unit, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8)
    color_map_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 1.0 - alpha, color_map_rgb, alpha, 0.0)
    return Image.fromarray(overlay)


def draw_ranked_patch_boxes(
    image: Image.Image,
    patch_indices: Sequence[int],
    patch_values: Sequence[float],
    grid_size: int,
    color_rgb: Tuple[int, int, int],
    font,
    max_labels: int = 8,
):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for rank, (patch_idx, value) in enumerate(zip(patch_indices, patch_values), start=1):
        x1, y1, x2, y2 = patch_index_to_xyxy(int(patch_idx), grid_size, width, height)
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
        if rank <= max_labels:
            label = f"{rank}:{value:.3f}"
            tb = draw.textbbox((0, 0), label, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
            bg = [x1, max(0, y1 - text_h - 4), min(width, x1 + text_w + 6), y1]
            draw.rectangle(bg, fill=color_rgb)
            draw.text((x1 + 3, bg[1] + 1), label, fill=(255, 255, 255), font=font)


def add_banner(image: Image.Image, text: str, font, fill=(0, 0, 0)):
    draw = ImageDraw.Draw(image)
    tb = draw.textbbox((0, 0), text, font=font)
    text_w = tb[2] - tb[0]
    text_h = tb[3] - tb[1]
    draw.rectangle([0, 0, min(image.width, text_w + 16), text_h + 10], fill=fill)
    draw.text((8, 5), text, fill=(255, 255, 255), font=font)


def concatenate_images_horizontally(images: Sequence[Image.Image]) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(255, 255, 255))
    cur_x = 0
    for img in images:
        canvas.paste(img, (cur_x, 0))
        cur_x += img.width
    return canvas


def save_best_frame_patch_comparison(
    case,
    frame_path: Path,
    score_map: np.ndarray,
    attention_map: np.ndarray,
    selected_indices: np.ndarray,
    selected_scores: np.ndarray,
    grid_size: int,
    topk: int,
    alpha: float,
    font,
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    drawn_instances = case.question_entry.get("render_info", {}).get("drawn_instances", [])

    raw_image = Image.open(frame_path).convert("RGB")
    score_overlay = build_heatmap_overlay_pil(frame_path, score_map, alpha=alpha)
    attention_overlay = build_heatmap_overlay_pil(frame_path, attention_map, alpha=alpha)
    comparison_overlay = raw_image.copy()

    flat_attention = attention_map.reshape(-1)
    attn_topk = min(topk, flat_attention.size)
    attention_top_indices = np.argsort(flat_attention)[-attn_topk:][::-1]
    attention_top_values = flat_attention[attention_top_indices]

    score_top_indices = np.array(selected_indices[:topk], dtype=int)
    score_top_values = np.array(selected_scores[:topk], dtype=float)

    if drawn_instances:
        exp0.draw_bboxes_on_pil(score_overlay, drawn_instances, font, best=True)
        exp0.draw_bboxes_on_pil(attention_overlay, drawn_instances, font, best=True)
        exp0.draw_bboxes_on_pil(comparison_overlay, drawn_instances, font, best=True)

    draw_ranked_patch_boxes(score_overlay, score_top_indices, score_top_values, grid_size, (46, 204, 113), font)
    draw_ranked_patch_boxes(attention_overlay, attention_top_indices, attention_top_values, grid_size, (241, 196, 15), font)

    score_set = set(int(x) for x in score_top_indices.tolist())
    attention_set = set(int(x) for x in attention_top_indices.tolist())
    overlap = score_set & attention_set
    score_only = [idx for idx in score_top_indices.tolist() if int(idx) not in overlap]
    attn_only = [idx for idx in attention_top_indices.tolist() if int(idx) not in overlap]
    overlap_sorted = [idx for idx in score_top_indices.tolist() if int(idx) in overlap]

    draw_ranked_patch_boxes(comparison_overlay, score_only, [flat_attention[idx] for idx in score_only], grid_size, (46, 204, 113), font, max_labels=0)
    draw_ranked_patch_boxes(comparison_overlay, attn_only, [flat_attention[idx] for idx in attn_only], grid_size, (241, 196, 15), font, max_labels=0)
    draw_ranked_patch_boxes(comparison_overlay, overlap_sorted, [flat_attention[idx] for idx in overlap_sorted], grid_size, (231, 76, 60), font, max_labels=0)

    add_banner(
        score_overlay,
        f"Score top-{len(score_top_indices)} patches ({case.qa_id[:8]})",
        font,
    )
    add_banner(
        attention_overlay,
        f"Fusion-attn top-{len(attention_top_indices)} patches ({case.qa_id[:8]})",
        font,
    )
    add_banner(
        comparison_overlay,
        f"Green=score only | Yellow=attn only | Red=overlap | overlap={len(overlap)}",
        font,
    )

    panel = concatenate_images_horizontally([score_overlay, attention_overlay, comparison_overlay])

    score_path = output_dir / "best_frame_score_topk_overlay.jpg"
    attention_path = output_dir / "best_frame_attention_topk_overlay.jpg"
    compare_path = output_dir / "best_frame_score_attention_compare.jpg"
    score_overlay.save(score_path)
    attention_overlay.save(attention_path)
    panel.save(compare_path)

    return {
        "score_overlay_path": str(score_path),
        "attention_overlay_path": str(attention_path),
        "comparison_panel_path": str(compare_path),
    }


def build_union_target_mask(case) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    render_info = case.question_entry["render_info"]
    frame_path = Path(render_info["frame_color_path"])
    instance_map_path = Path(render_info["frame_instance_path"])
    drawn_instances = render_info.get("drawn_instances", [])

    frame_img = Image.open(frame_path).convert("RGB")
    frame_np = np.array(frame_img)
    height, width = frame_np.shape[:2]

    instance_map = exp1.load_instance_map(instance_map_path)
    if instance_map.shape[:2] != (height, width):
        raise ValueError(
            f"Instance map shape {instance_map.shape[:2]} does not match RGB frame shape {(height, width)} for {frame_path}"
        )

    union_mask = np.zeros((height, width), dtype=bool)
    per_instance = []
    for item in drawn_instances:
        instance_id = int(item["instance_id"])
        bbox = item["bbox_2d"]
        instance_mask = exp1.mask_from_instance_id(instance_map, instance_id)
        has_mask = bool(instance_mask.any())
        bbox_mask = exp1.bbox_to_mask(bbox, (height, width))
        target_mask = instance_mask if has_mask else bbox_mask
        union_mask |= target_mask
        per_instance.append(
            {
                "object_name": item["object_name"],
                "instance_id": instance_id,
                "bbox_2d": bbox,
                "used_mask_type": "instance_mask" if has_mask else "bbox_fallback",
                "pixel_area": int(target_mask.sum()),
            }
        )

    meta = {
        "frame_path": str(frame_path),
        "frame_instance_path": str(instance_map_path),
        "height": height,
        "width": width,
    }
    return union_mask, per_instance, meta


def patch_coverages_from_mask(mask: np.ndarray, grid_size: int) -> np.ndarray:
    height, width = mask.shape
    coverages = np.zeros((grid_size, grid_size), dtype=np.float32)
    for row in range(grid_size):
        y1 = int(round(row * height / grid_size))
        y2 = int(round((row + 1) * height / grid_size))
        for col in range(grid_size):
            x1 = int(round(col * width / grid_size))
            x2 = int(round((col + 1) * width / grid_size))
            patch = mask[y1:y2, x1:x2]
            if patch.size == 0:
                coverages[row, col] = 0.0
            else:
                coverages[row, col] = float(patch.mean())
    return coverages.reshape(-1)


def topk_overlap(a_scores: np.ndarray, b_scores: np.ndarray, k: int) -> Dict[str, Any]:
    flat_a = a_scores.reshape(-1)
    flat_b = b_scores.reshape(-1)
    k = min(k, flat_a.size, flat_b.size)
    if k <= 0:
        return {"k": 0, "overlap_count": 0, "jaccard": 0.0, "overlap_ratio": 0.0}
    a_idx = np.argsort(flat_a)[-k:][::-1]
    b_idx = np.argsort(flat_b)[-k:][::-1]
    a_set = set(int(x) for x in a_idx.tolist())
    b_set = set(int(x) for x in b_idx.tolist())
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return {
        "k": int(k),
        "overlap_count": int(inter),
        "overlap_ratio": float(inter / max(k, 1)),
        "jaccard": float(inter / max(union, 1)),
    }


def compute_gt_topk_metrics(
    frame_scores: np.ndarray,
    patch_coverages: np.ndarray,
    topk_values: Sequence[int],
    coverage_thresholds: Sequence[float],
) -> Dict[str, Any]:
    flat_scores = frame_scores.reshape(-1)
    order = np.argsort(flat_scores)[::-1]
    out: Dict[str, Any] = {"overall": {}}

    out["overall"] = {
        "mean_patch_coverage": float(patch_coverages.mean()),
        "max_patch_coverage": float(patch_coverages.max()),
        "positive_patch_fraction_any": float((patch_coverages > 0).mean()),
        "positive_patch_fraction_25": float((patch_coverages >= 0.25).mean()),
        "positive_patch_fraction_50": float((patch_coverages >= 0.50).mean()),
    }

    per_threshold = {}
    for thr in coverage_thresholds:
        gt_positive = patch_coverages > thr if thr <= 0 else patch_coverages >= thr
        num_gt = int(gt_positive.sum())
        per_k = []
        for k in topk_values:
            cur_k = min(k, flat_scores.size)
            top_idx = order[:cur_k]
            selected_cov = patch_coverages[top_idx]
            hit_count = int(gt_positive[top_idx].sum()) if num_gt > 0 else 0
            per_k.append(
                {
                    "k": int(cur_k),
                    "gt_positive_patch_count": num_gt,
                    "selected_gt_hit_count": hit_count,
                    "precision": float(hit_count / max(cur_k, 1)),
                    "recall": float(hit_count / max(num_gt, 1)) if num_gt > 0 else 0.0,
                    "mean_selected_patch_coverage": float(selected_cov.mean()) if selected_cov.size > 0 else 0.0,
                    "max_selected_patch_coverage": float(selected_cov.max()) if selected_cov.size > 0 else 0.0,
                    "coverage_sum": float(selected_cov.sum()) if selected_cov.size > 0 else 0.0,
                }
            )
        per_threshold[str(thr)] = per_k
    out["per_threshold"] = per_threshold
    return out


def summarize_overlap_list(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    return {
        "num_frames": len(items),
        "mean_overlap_ratio": float(np.mean([x["overlap_ratio"] for x in items])),
        "std_overlap_ratio": float(np.std([x["overlap_ratio"] for x in items])),
        "mean_jaccard": float(np.mean([x["jaccard"] for x in items])),
        "std_jaccard": float(np.std([x["jaccard"] for x in items])),
    }


def analyze_case(
    case,
    args,
    tokenizer,
    model,
    image_processor,
    topk_values: Sequence[int],
    coverage_thresholds: Sequence[float],
) -> Dict[str, Any]:
    frame_paths_all = exp0.load_scene_frame_paths(case.frame_dir)
    try:
        best_frame_idx = frame_paths_all.index(case.best_frame_path)
    except ValueError as exc:
        raise ValueError(f"Best frame {case.best_frame_path} not found in {case.frame_dir}") from exc

    sampled_indices = exp0.sample_frame_indices(len(frame_paths_all), args.num_frames, forced_idx=best_frame_idx)
    sampled_frame_paths = [frame_paths_all[i] for i in sampled_indices]
    best_local_idx = sampled_frame_paths.index(case.best_frame_path)

    video, input_ids, attention_mask = prepare_video_inputs(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        frame_paths=sampled_frame_paths,
        prompt=case.prompt,
        conv_mode=args.conv_mode,
        device=args.device,
    )

    output_text, attn_weights, capture_source = run_generation_with_attention(
        model=model,
        tokenizer=tokenizer,
        video=video,
        input_ids=input_ids,
        attention_mask=attention_mask,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    patch_meta = compute_patch_selection_metadata(
        model=model,
        input_ids=input_ids,
        video=video,
        attention_mask=attention_mask,
        args=args,
    )

    patch_scores = patch_meta["patch_scores"].numpy()
    selected_indices = patch_meta["selected_indices"].numpy()
    selected_scores = patch_meta["selected_scores"].numpy()
    grid_size = int(patch_meta["grid_size"])

    if patch_scores.ndim != 2:
        raise ValueError(f"Expected patch_scores with shape [F, T], got {patch_scores.shape}")
    if patch_scores.shape[0] != len(sampled_frame_paths):
        raise RuntimeError(
            f"Patch-score frame count {patch_scores.shape[0]} != sampled frame count {len(sampled_frame_paths)}"
        )

    score_maps = patch_scores.reshape(patch_scores.shape[0], grid_size, grid_size)

    patch_side = exp0.get_spatial_patch_side(model)
    attention_maps, attn_meta = exp0.aggregate_attention_maps(attn_weights, patch_side_hint=patch_side)
    if attention_maps.shape[0] != score_maps.shape[0]:
        raise RuntimeError(
            f"Attention frame count {attention_maps.shape[0]} != score frame count {score_maps.shape[0]}"
        )

    score_stats = [compute_score_stats(frame_scores) for frame_scores in score_maps]
    per_frame_alignment = []
    for i in range(score_maps.shape[0]):
        flat_score = score_maps[i].reshape(-1)
        flat_attn = attention_maps[i].reshape(-1)
        alignment = {
            "frame_index": int(i),
            "frame_path": str(sampled_frame_paths[i]),
            "pearson": pearson_corr(flat_score, flat_attn),
            "cosine": cosine_sim(flat_score, flat_attn),
            "topk_overlap": [topk_overlap(flat_score, flat_attn, k) for k in topk_values],
        }
        per_frame_alignment.append(alignment)

    union_mask, per_instance_meta, gt_meta = build_union_target_mask(case)
    patch_coverages = patch_coverages_from_mask(union_mask, grid_size)
    best_frame_gt_metrics = compute_gt_topk_metrics(
        score_maps[best_local_idx],
        patch_coverages,
        topk_values=topk_values,
        coverage_thresholds=coverage_thresholds,
    )

    best_selected_patch_coverages = patch_coverages[selected_indices[best_local_idx]] if selected_indices.shape[1] > 0 else np.array([])
    best_frame_selected_summary = {
        "selected_patch_count": int(selected_indices.shape[1]),
        "mean_selected_patch_coverage": float(best_selected_patch_coverages.mean()) if best_selected_patch_coverages.size > 0 else 0.0,
        "max_selected_patch_coverage": float(best_selected_patch_coverages.max()) if best_selected_patch_coverages.size > 0 else 0.0,
        "selected_patch_hit_fraction_any": float((best_selected_patch_coverages > 0).mean()) if best_selected_patch_coverages.size > 0 else 0.0,
        "selected_patch_hit_fraction_25": float((best_selected_patch_coverages >= 0.25).mean()) if best_selected_patch_coverages.size > 0 else 0.0,
        "selected_patch_hit_fraction_50": float((best_selected_patch_coverages >= 0.50).mean()) if best_selected_patch_coverages.size > 0 else 0.0,
    }

    run_dir = args.output_root / case.scene_name / case.qa_id
    run_dir.mkdir(parents=True, exist_ok=True)

    score_maps_path = run_dir / "patch_score_maps.npy"
    np.save(score_maps_path, score_maps)
    attention_maps_path = run_dir / "fusion_attention_maps.npy"
    np.save(attention_maps_path, attention_maps)
    patch_score_raw_path = run_dir / "patch_scores.npy"
    np.save(patch_score_raw_path, patch_scores)
    comparison_artifacts = {}

    if args.save_frame_overlays:
        font = exp0.find_font(args.font_size)
        overlay_dir = run_dir / "score_overlays"
        overlay_dir.mkdir(exist_ok=True)
        score_maps_unit = normalize_maps_global(score_maps)
        drawn_instances = case.question_entry.get("render_info", {}).get("drawn_instances", [])
        for i, frame_path in enumerate(sampled_frame_paths):
            overlay_path = overlay_dir / f"{frame_path.stem}_score.jpg"
            exp0.render_overlay_image(
                frame_path=frame_path,
                normalized_heatmap=score_maps_unit[i],
                stats=score_stats[i],
                output_path=overlay_path,
                alpha=args.alpha,
                font=font,
                is_best_frame=(i == best_local_idx),
                drawn_instances=drawn_instances if i == best_local_idx else None,
            )

        visualize_topk = args.visualize_topk if args.visualize_topk is not None else args.fine_topk
        comparison_artifacts = save_best_frame_patch_comparison(
            case=case,
            frame_path=case.best_frame_path,
            score_map=score_maps[best_local_idx],
            attention_map=attention_maps[best_local_idx],
            selected_indices=selected_indices[best_local_idx],
            selected_scores=selected_scores[best_local_idx],
            grid_size=grid_size,
            topk=visualize_topk,
            alpha=args.alpha,
            font=font,
            output_dir=run_dir / "patch_comparison_vis",
        )

    overlap_summary_by_k = {}
    for k in topk_values:
        overlap_items = [next(item for item in frame_item["topk_overlap"] if item["k"] == min(k, grid_size * grid_size)) for frame_item in per_frame_alignment]
        overlap_summary_by_k[str(k)] = summarize_overlap_list(overlap_items)

    result = {
        "qa_id": case.qa_id,
        "scene_name": case.scene_name,
        "prompt": case.prompt,
        "prediction": output_text,
        "capture_source": capture_source,
        "sampled_frame_paths": [str(p) for p in sampled_frame_paths],
        "best_frame_local_index": int(best_local_idx),
        "best_frame_path": str(case.best_frame_path),
        "attention_meta": attn_meta,
        "patch_selection_meta": {
            "grid_size": grid_size,
            "fine_topk": int(args.fine_topk),
            "scoring_mode": args.scoring_mode,
            "fine_scale": float(args.fine_scale),
            "coarse_token_count": int(patch_meta["coarse_token_count"]),
            "fine_token_count": int(patch_meta["fine_token_count"]),
            "combined_token_count": int(patch_meta["combined_token_count"]),
        },
        "score_stats_per_frame": [
            {
                "frame_index": int(i),
                "frame_path": str(sampled_frame_paths[i]),
                **score_stats[i],
            }
            for i in range(len(score_stats))
        ],
        "score_attention_alignment": {
            "per_frame": per_frame_alignment,
            "summary_by_k": overlap_summary_by_k,
            "mean_pearson": float(np.mean([x["pearson"] for x in per_frame_alignment])),
            "std_pearson": float(np.std([x["pearson"] for x in per_frame_alignment])),
            "mean_cosine": float(np.mean([x["cosine"] for x in per_frame_alignment])),
            "std_cosine": float(np.std([x["cosine"] for x in per_frame_alignment])),
        },
        "best_frame_gt_patch_coverage": {
            "gt_meta": gt_meta,
            "per_instance_meta": per_instance_meta,
            "selected_patch_summary": best_frame_selected_summary,
            "metrics": best_frame_gt_metrics,
            "selected_indices": selected_indices[best_local_idx].tolist(),
            "selected_scores": selected_scores[best_local_idx].tolist(),
        },
        "artifacts": {
            "patch_score_maps_path": str(score_maps_path),
            "attention_maps_path": str(attention_maps_path),
            "patch_scores_path": str(patch_score_raw_path),
            **comparison_artifacts,
        },
        "manifest_question_entry": case.question_entry,
    }

    summary_path = run_dir / "patch_score_analysis_summary.json"
    result["summary_path"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def aggregate_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    selected_hit_any = [
        x["best_frame_gt_patch_coverage"]["selected_patch_summary"]["selected_patch_hit_fraction_any"] for x in results
    ]
    mean_pearson = [x["score_attention_alignment"]["mean_pearson"] for x in results]
    mean_cosine = [x["score_attention_alignment"]["mean_cosine"] for x in results]
    top1_margins = [
        np.mean([frame_item["margin_top1_top2"] for frame_item in x["score_stats_per_frame"]]) for x in results
    ]
    softmax_entropies = [
        np.mean([frame_item["softmax_entropy"] for frame_item in x["score_stats_per_frame"]]) for x in results
    ]
    return {
        "num_cases": len(results),
        "mean_selected_patch_hit_fraction_any": float(np.mean(selected_hit_any)),
        "std_selected_patch_hit_fraction_any": float(np.std(selected_hit_any)),
        "mean_alignment_pearson": float(np.mean(mean_pearson)),
        "std_alignment_pearson": float(np.std(mean_pearson)),
        "mean_alignment_cosine": float(np.mean(mean_cosine)),
        "std_alignment_cosine": float(np.std(mean_cosine)),
        "mean_framewise_top1_margin": float(np.mean(top1_margins)),
        "std_framewise_top1_margin": float(np.std(top1_margins)),
        "mean_framewise_softmax_entropy": float(np.mean(softmax_entropies)),
        "std_framewise_softmax_entropy": float(np.std(softmax_entropies)),
    }


def main():
    inject_default_demo_argv()
    args = parse_args()
    topk_values = parse_int_list(args.compare_topk)
    coverage_thresholds = parse_float_list(args.coverage_thresholds)

    manifest = exp0.load_manifest(args.manifest_path)
    cases = build_case_list(manifest, args)

    tokenizer, model, image_processor, _ = exp0.load_model_and_processor(args)

    all_results = []
    for idx, case in enumerate(cases):
        run_dir = args.output_root / case.scene_name / case.qa_id
        summary_path = run_dir / "patch_score_analysis_summary.json"
        if summary_path.exists() and not args.overwrite_output:
            print(f"[Skip {idx+1}/{len(cases)}] {case.scene_name} / {case.qa_id} already exists: {summary_path}")
            with summary_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            cached["summary_path"] = str(summary_path)
            all_results.append(cached)
            continue

        print(f"[Run  {idx+1}/{len(cases)}] scene={case.scene_name} qa_id={case.qa_id}")
        result = analyze_case(
            case=case,
            args=args,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            topk_values=topk_values,
            coverage_thresholds=coverage_thresholds,
        )
        all_results.append(result)
        print(
            f"  -> saved {result['summary_path']} | mean alignment pearson="
            f"{result['score_attention_alignment']['mean_pearson']:.4f} | "
            f"selected-hit-any={result['best_frame_gt_patch_coverage']['selected_patch_summary']['selected_patch_hit_fraction_any']:.4f}"
        )

    aggregate = aggregate_results(all_results)
    aggregate_path = args.output_root / "aggregate_patch_score_analysis.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "scoring_mode": args.scoring_mode,
                "fine_topk": args.fine_topk,
                "num_frames": args.num_frames,
                "compare_topk": topk_values,
                "coverage_thresholds": coverage_thresholds,
                "aggregate": aggregate,
                "case_summaries": [
                    {
                        "scene_name": x["scene_name"],
                        "qa_id": x["qa_id"],
                        "summary_path": x.get("summary_path", ""),
                        "mean_alignment_pearson": x["score_attention_alignment"]["mean_pearson"],
                        "selected_patch_hit_fraction_any": x["best_frame_gt_patch_coverage"]["selected_patch_summary"]["selected_patch_hit_fraction_any"],
                    }
                    for x in all_results
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 80)
    print(f"Processed cases: {len(all_results)}")
    print(f"Aggregate summary saved to: {aggregate_path}")
    if aggregate:
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
