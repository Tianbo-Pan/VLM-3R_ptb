#!/usr/bin/env python3
"""Experiment 0: attention extraction + per-frame visualization for ptb_test cases.

Goals
-----
1. Load one QA case from ``ptb_test/scannetpp_qa_visualizations/selection_manifest.json``.
2. Run VLM-3R inference on a sampled frame sequence from the corresponding scene.
3. Hook fusion attention during multimodal encoding.
4. Produce per-frame attention heatmaps for *all sampled frames* (not only the best frame).
5. Mark which sampled frame is the manifest-provided best frame.

This is intentionally a lightweight "sanity check" script. It does not yet compute
formal grounding metrics; it only verifies the end-to-end extraction + visualization
pipeline for later experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Limit to one GPU for easier debugging and visualization.
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


DEFAULT_MANIFEST_PATH = Path(
    "/local_home/pantianbo/projects/vision_reasoning/VLM-3R/ptb_test/scannetpp_qa_visualizations/selection_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/local_home/pantianbo/projects/vision_reasoning/VLM-3R/ptb_test/experiment0_outputs"
)


@dataclass
class SelectedCase:
    qa_id: str
    scene_name: str
    scene_entry: Dict[str, Any]
    question_entry: Dict[str, Any]
    frame_dir: Path
    best_frame_path: Path
    prompt: str


class FusionAttentionRecorder:
    """Capture fusion attention without modifying the core model code."""

    def __init__(self, model):
        self.model = model
        self.captured: Optional[torch.Tensor] = None
        self.capture_source: Optional[str] = None
        self._handles = []

    def _fusion_hook(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
            self.captured = output[1].detach().float().cpu()
            self.capture_source = "fusion_block"

    def _inner_mha_hook(self, module, inputs, output):
        if self.captured is not None:
            return
        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
            self.captured = output[1].detach().float().cpu()
            self.capture_source = "inner_cross_attention"

    def __enter__(self):
        fusion_block = self.model.get_model().get_fusion_block()
        if fusion_block is None:
            raise RuntimeError("Model has no fusion block; cannot capture multimodal attention.")
        self._handles.append(fusion_block.register_forward_hook(self._fusion_hook))
        if hasattr(fusion_block, "cross_attention"):
            self._handles.append(fusion_block.cross_attention.register_forward_hook(self._inner_mha_hook))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 0: extract and visualize per-frame fusion attention.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--qa-id", type=str, default=None, help="Exact qa_id from the manifest.")
    parser.add_argument("--scene-name", type=str, default=None, help="Scene name if selecting by scene.")
    parser.add_argument("--question-index", type=int, default=0, help="Question index within a scene entry.")
    parser.add_argument("--flat-index", type=int, default=None, help="Flattened question index across the full manifest.")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--mm-spatial-pool-mode", type=str, default="average")
    parser.add_argument("--mm-newline-position", type=str, default="grid")

    parser.add_argument("--num-frames", type=int, default=32, help="Number of scene frames to sample for inference. Use <=0 for all frames.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha.")
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--draw-best-frame-boxes", action="store_true", help="Draw manifest bboxes on the best frame overlay.")
    return parser.parse_args()


def find_font(size: int):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_manifest_questions(manifest: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    flat = []
    for scene_entry in manifest["scenes"]:
        for question_entry in scene_entry["questions"]:
            flat.append((scene_entry, question_entry))
    return flat


def select_case(manifest: Dict[str, Any], args) -> SelectedCase:
    flat = flatten_manifest_questions(manifest)

    scene_entry = None
    question_entry = None

    if args.qa_id is not None:
        for s, q in flat:
            if q["qa_id"] == args.qa_id:
                scene_entry, question_entry = s, q
                break
        if question_entry is None:
            raise ValueError(f"qa_id={args.qa_id} not found in manifest.")
    elif args.flat_index is not None:
        if args.flat_index < 0 or args.flat_index >= len(flat):
            raise IndexError(f"flat_index={args.flat_index} is out of range [0, {len(flat)-1}].")
        scene_entry, question_entry = flat[args.flat_index]
    elif args.scene_name is not None:
        scene_candidates = [s for s in manifest["scenes"] if s["scene_name"] == args.scene_name]
        if not scene_candidates:
            raise ValueError(f"scene_name={args.scene_name} not found in manifest.")
        scene_entry = scene_candidates[0]
        if args.question_index < 0 or args.question_index >= len(scene_entry["questions"]):
            raise IndexError(
                f"question_index={args.question_index} is out of range [0, {len(scene_entry['questions'])-1}] "
                f"for scene {args.scene_name}."
            )
        question_entry = scene_entry["questions"][args.question_index]
    else:
        scene_entry, question_entry = flat[0]

    prompt = question_entry["parsed_question"]["question_text"]
    best_frame_path = Path(question_entry["render_info"]["frame_color_path"])
    frame_dir = best_frame_path.parent

    return SelectedCase(
        qa_id=question_entry["qa_id"],
        scene_name=scene_entry["scene_name"],
        scene_entry=scene_entry,
        question_entry=question_entry,
        frame_dir=frame_dir,
        best_frame_path=best_frame_path,
        prompt=prompt,
    )


def sample_frame_indices(length: int, target_num: int, forced_idx: Optional[int] = None) -> List[int]:
    if length <= 0:
        return []
    if target_num <= 0 or target_num >= length:
        indices = list(range(length))
    else:
        indices = np.linspace(0, length - 1, target_num, dtype=int).tolist()

    if forced_idx is None or forced_idx in indices:
        return sorted(set(indices))

    # Always include the forced frame, then refill if duplicates reduced count.
    indices.append(forced_idx)
    indices = sorted(set(indices))

    if target_num > 0 and len(indices) > target_num:
        removable = [idx for idx in indices if idx != forced_idx]
        removable.sort(key=lambda x: abs(x - forced_idx))
        while len(indices) > target_num and removable:
            victim = removable.pop(0)
            if victim in indices:
                indices.remove(victim)

    if target_num > 0 and len(indices) < target_num:
        canonical = np.linspace(0, length - 1, target_num, dtype=int).tolist()
        for idx in canonical:
            if idx not in indices:
                indices.append(idx)
            if len(indices) == target_num:
                break
        if len(indices) < target_num:
            for idx in range(length):
                if idx not in indices:
                    indices.append(idx)
                if len(indices) == target_num:
                    break

    return sorted(indices)


def load_scene_frame_paths(frame_dir: Path) -> List[Path]:
    frame_paths = sorted(
        [p for p in frame_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")
    return frame_paths


def load_pil_images(frame_paths: Sequence[Path]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in frame_paths]


def build_overwrite_config(args, model_path: str) -> Dict[str, Any]:
    overwrite_config: Dict[str, Any] = {}
    if not args.overwrite:
        return overwrite_config

    overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
    overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
    overwrite_config["mm_newline_position"] = args.mm_newline_position

    cfg_pretrained = AutoConfig.from_pretrained(model_path)
    if "qwen" not in model_path.lower():
        least_token_number = 32 * (24 // args.mm_spatial_pool_stride) ** 2 + 1000
        scaling_factor = math.ceil(least_token_number / 4096)
        if scaling_factor >= 2:
            if "vicuna" in cfg_pretrained._name_or_path.lower():
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
    return overwrite_config


def load_model_and_processor(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    overwrite_config = build_overwrite_config(args, args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        overwrite_config=overwrite_config if overwrite_config else None,
    )
    model.to(args.device)
    model.eval()
    return tokenizer, model, image_processor, context_len


def build_video_prompt(model, prompt: str, conv_mode: str) -> str:
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_inference_with_attention(
    model,
    tokenizer,
    image_processor,
    frame_paths: Sequence[Path],
    prompt: str,
    conv_mode: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    pil_images = load_pil_images(frame_paths)
    pixel_values = image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"].half().to(device)
    video = [pixel_values]

    full_prompt = build_video_prompt(model, prompt, conv_mode)
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 151643

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with FusionAttentionRecorder(model) as recorder:
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=video,
                attention_mask=attention_mask,
                modalities="video",
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
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


def get_spatial_patch_side(model) -> Optional[int]:
    spatial_tower = model.get_model().get_spatial_tower()
    if spatial_tower is None:
        return None
    return getattr(spatial_tower, "num_patches_per_side", None)


def aggregate_attention_maps(attn_weights: torch.Tensor, patch_side_hint: Optional[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert captured attention into frame-wise 2D patch maps.

    Returns
    -------
    heatmaps : np.ndarray
        Shape [num_frames, h, w]
    meta : dict
        Includes source token counts and whether a camera token seems present.
    """
    if attn_weights.ndim == 4:
        attn_weights = attn_weights.mean(dim=1)
    if attn_weights.ndim != 3:
        raise ValueError(f"Expected attention with 3 or 4 dims, got shape {tuple(attn_weights.shape)}")

    num_frames, num_query, src_len = attn_weights.shape

    if patch_side_hint is not None:
        patch_tokens = patch_side_hint * patch_side_hint
    else:
        patch_side_hint = int(round(math.sqrt(src_len)))
        patch_tokens = patch_side_hint * patch_side_hint

    has_camera_token = False
    if src_len == patch_tokens + 1:
        attn_patch = attn_weights[:, :, 1:]
        has_camera_token = True
    elif src_len == patch_tokens:
        attn_patch = attn_weights
    elif src_len > patch_tokens:
        # Fallback: assume patch tokens are at the tail if extra conditioning tokens exist.
        attn_patch = attn_weights[:, :, -patch_tokens:]
    else:
        side = int(round(math.sqrt(src_len)))
        if side * side != src_len:
            raise ValueError(
                f"Cannot reshape source attention of length {src_len} into a 2D grid. "
                f"patch_side_hint={patch_side_hint}"
            )
        patch_side_hint = side
        patch_tokens = src_len
        attn_patch = attn_weights

    frame_maps = attn_patch.mean(dim=1).numpy().reshape(num_frames, patch_side_hint, patch_side_hint)
    meta = {
        "num_frames": num_frames,
        "num_query_tokens": num_query,
        "num_source_tokens": src_len,
        "num_patch_tokens": patch_tokens,
        "patch_side": patch_side_hint,
        "has_camera_token": has_camera_token,
    }
    return frame_maps, meta


def normalize_heatmaps_global(frame_maps: np.ndarray) -> np.ndarray:
    maps = frame_maps.copy()
    vmin = float(maps.min())
    vmax = float(maps.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(maps, dtype=np.float32)
    return ((maps - vmin) / (vmax - vmin)).astype(np.float32)


def compute_heatmap_stats(frame_map: np.ndarray) -> Dict[str, float]:
    flat = frame_map.reshape(-1).astype(np.float64)
    flat_sum = flat.sum()
    if flat_sum > 0:
        prob = flat / flat_sum
        entropy = float(-(prob * np.log(prob + 1e-12)).sum())
    else:
        entropy = 0.0
    return {
        "mean": float(flat.mean()),
        "max": float(flat.max()),
        "min": float(flat.min()),
        "entropy": entropy,
    }


def draw_bboxes_on_pil(image: Image.Image, drawn_instances: Sequence[Dict[str, Any]], font, best: bool):
    draw = ImageDraw.Draw(image)
    palette = [
        (231, 76, 60),
        (52, 152, 219),
        (155, 89, 182),
        (26, 188, 156),
        (243, 156, 18),
        (46, 204, 113),
    ]
    for idx, item in enumerate(drawn_instances):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = item["bbox_2d"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = item["object_name"]
        tw = draw.textbbox((0, 0), label, font=font)
        text_w = tw[2] - tw[0]
        text_h = tw[3] - tw[1]
        text_bg = [x1, max(0, y1 - text_h - 6), x1 + text_w + 8, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 4, text_bg[1] + 2), label, fill=(255, 255, 255), font=font)

    if best:
        draw.rectangle([1, 1, image.width - 2, image.height - 2], outline=(255, 0, 0), width=6)


def render_overlay_image(
    frame_path: Path,
    normalized_heatmap: np.ndarray,
    stats: Dict[str, float],
    output_path: Path,
    alpha: float,
    font,
    is_best_frame: bool,
    drawn_instances: Optional[Sequence[Dict[str, Any]]] = None,
):
    image = Image.open(frame_path).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    heatmap_resized = cv2.resize(normalized_heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8)
    color_map_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 1.0 - alpha, color_map_rgb, alpha, 0.0)
    overlay_pil = Image.fromarray(overlay)

    if drawn_instances:
        draw_bboxes_on_pil(overlay_pil, drawn_instances, font, best=is_best_frame)

    draw = ImageDraw.Draw(overlay_pil)
    banner_text = (
        f"{frame_path.name} | peak={stats['max']:.4f} | mean={stats['mean']:.4f} | H={stats['entropy']:.3f}"
    )
    if is_best_frame:
        banner_text = "[BEST FRAME] " + banner_text
    text_bbox = draw.textbbox((0, 0), banner_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    bg_h = text_h + 10
    draw.rectangle([0, 0, min(width, text_w + 16), bg_h], fill=(0, 0, 0))
    draw.text((8, 5), banner_text, fill=(255, 255, 255), font=font)

    if is_best_frame and not drawn_instances:
        draw.rectangle([1, 1, overlay_pil.width - 2, overlay_pil.height - 2], outline=(255, 0, 0), width=6)

    overlay_pil.save(output_path)


def make_contact_sheet(
    frame_paths: Sequence[Path],
    overlay_paths: Sequence[Path],
    best_frame_name: str,
    output_path: Path,
    font,
):
    thumbs = []
    target_size = (320, 240)
    for frame_path, overlay_path in zip(frame_paths, overlay_paths):
        tile = Image.open(overlay_path).convert("RGB")
        tile.thumbnail(target_size)
        canvas = Image.new("RGB", target_size, (255, 255, 255))
        offset = ((target_size[0] - tile.width) // 2, (target_size[1] - tile.height) // 2)
        canvas.paste(tile, offset)
        draw = ImageDraw.Draw(canvas)
        label = frame_path.name
        text_bbox = draw.textbbox((0, 0), label, font=font)
        draw.rectangle([0, target_size[1] - 24, max(80, text_bbox[2] - text_bbox[0] + 10), target_size[1]], fill=(0, 0, 0))
        draw.text((5, target_size[1] - 21), label, fill=(255, 255, 255), font=font)
        if frame_path.name == best_frame_name:
            draw.rectangle([1, 1, target_size[0] - 2, target_size[1] - 2], outline=(255, 0, 0), width=6)
            draw.rectangle([0, 0, 100, 24], fill=(180, 0, 0))
            draw.text((5, 3), "BEST FRAME", fill=(255, 255, 255), font=font)
        thumbs.append(canvas)

    cols = min(4, max(1, int(math.ceil(math.sqrt(len(thumbs))))))
    rows = int(math.ceil(len(thumbs) / cols))
    margin = 12
    sheet_w = cols * target_size[0] + (cols + 1) * margin
    sheet_h = rows * target_size[1] + (rows + 1) * margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (240, 240, 240))

    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = margin + col * (target_size[0] + margin)
        y = margin + row * (target_size[1] + margin)
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest_path)
    case = select_case(manifest, args)

    frame_paths_all = load_scene_frame_paths(case.frame_dir)
    try:
        best_frame_idx = frame_paths_all.index(case.best_frame_path)
    except ValueError as exc:
        raise ValueError(
            f"Best frame {case.best_frame_path} is not present in sampled directory {case.frame_dir}"
        ) from exc

    sampled_indices = sample_frame_indices(len(frame_paths_all), args.num_frames, forced_idx=best_frame_idx)
    sampled_frame_paths = [frame_paths_all[i] for i in sampled_indices]

    tokenizer, model, image_processor, _ = load_model_and_processor(args)
    output_text, attn_weights, capture_source = run_inference_with_attention(
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

    patch_side = get_spatial_patch_side(model)
    frame_maps, attn_meta = aggregate_attention_maps(attn_weights, patch_side_hint=patch_side)
    if frame_maps.shape[0] != len(sampled_frame_paths):
        raise RuntimeError(
            f"Captured attention frame count {frame_maps.shape[0]} does not match sampled frame count {len(sampled_frame_paths)}."
        )

    frame_maps_norm = normalize_heatmaps_global(frame_maps)

    run_dir = args.output_root / case.scene_name / f"{case.qa_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_output_dir = run_dir / "per_frame"
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    font = find_font(args.font_size)
    overlay_paths = []
    per_frame_stats = []
    drawn_instances = case.question_entry["render_info"].get("drawn_instances", [])

    for frame_path, raw_map, norm_map in zip(sampled_frame_paths, frame_maps, frame_maps_norm):
        is_best_frame = frame_path == case.best_frame_path
        stats = compute_heatmap_stats(raw_map)
        overlay_path = frame_output_dir / f"{frame_path.stem}_attn.jpg"
        render_overlay_image(
            frame_path=frame_path,
            normalized_heatmap=norm_map,
            stats=stats,
            output_path=overlay_path,
            alpha=args.alpha,
            font=font,
            is_best_frame=is_best_frame,
            drawn_instances=drawn_instances if (is_best_frame and args.draw_best_frame_boxes) else None,
        )
        overlay_paths.append(overlay_path)
        per_frame_stats.append(
            {
                "frame_path": str(frame_path),
                "overlay_path": str(overlay_path),
                "is_best_frame": is_best_frame,
                **stats,
            }
        )

    contact_sheet_path = run_dir / "all_sampled_frames_attention.jpg"
    make_contact_sheet(
        frame_paths=sampled_frame_paths,
        overlay_paths=overlay_paths,
        best_frame_name=case.best_frame_path.name,
        output_path=contact_sheet_path,
        font=font,
    )

    summary = {
        "qa_id": case.qa_id,
        "scene_name": case.scene_name,
        "prompt": case.prompt,
        "prediction": output_text,
        "best_frame_path": str(case.best_frame_path),
        "best_frame_name": case.best_frame_path.name,
        "sampled_frame_paths": [str(p) for p in sampled_frame_paths],
        "capture_source": capture_source,
        "attention_meta": attn_meta,
        "contact_sheet_path": str(contact_sheet_path),
        "per_frame_stats": per_frame_stats,
        "manifest_question_entry": case.question_entry,
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"[Experiment 0] scene={case.scene_name} qa_id={case.qa_id}")
    print(f"Prompt    : {case.prompt}")
    print(f"Prediction: {output_text}")
    print(f"Saved per-frame overlays to: {frame_output_dir}")
    print(f"Saved contact sheet to     : {contact_sheet_path}")
    print(f"Saved summary json to      : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
