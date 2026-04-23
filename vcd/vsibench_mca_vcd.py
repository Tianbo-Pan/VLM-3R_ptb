#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoConfig

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from vcd.methods import build_2d_feature_branch, build_spatial_noise_branch, build_weak_fusion_branch
from vcd.option_demo_utils import ensure_pad_token, load_video, score_candidate_with_details, score_candidate_with_video_features, set_seed


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

SETTING_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "two_d_feature_vcd": "2D-feature VCD",
    "weak_fusion_vcd": "Weak-fusion VCD",
    "spatial_noise_vcd": "Spatial-noise VCD",
}

STATE_COLORS = {
    "original": "#4C78A8",
    "degraded": "#F58518",
    "combined": "#54A24B",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 2D-feature and spatial-feature VCD on VSIBench MCA question types.")
    parser.add_argument("--source_log_json", type=str, default=None, help="Optional lmms_eval VSIBench log JSON used as the doc source.")
    parser.add_argument("--dataset_cache_dir", type=str, default=os.path.expanduser(os.getenv("VSIBENCH_CACHE_DIR", "~/.cache/huggingface/vsibench")))
    parser.add_argument("--output_root", type=str, default="vcd/results/vsibench_mca_vcd")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--per_type_limit", type=int, default=5)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--question_types", type=str, default=",".join(MCA_QUESTION_TYPES))

    parser.add_argument("--model-path", type=str, default="Journey9ni/vlm-3r-llava-qwen2-lora")
    parser.add_argument("--model-base", type=str, default="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--mm_newline_position", type=str, default="grid")
    parser.add_argument("--overwrite", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--load_8bit", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--force_sample", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--add_time_instruction", type=lambda x: str(x).lower() == "true", default=False)

    parser.add_argument("--pre_prompt", type=str, default="These are frames of a video.")
    parser.add_argument("--mca_post_prompt", type=str, default="Answer with exactly one complete option from the list above.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--visual_drop_rate", type=float, default=0.10)
    parser.add_argument("--visual_noise_std", type=float, default=0.01)
    parser.add_argument("--fusion_weak_ratio", type=float, default=0.35)
    parser.add_argument("--fusion_drop_rate", type=float, default=0.05)
    parser.add_argument("--fusion_noise_std", type=float, default=0.005)
    parser.add_argument("--spatial_drop_rate", type=float, default=0.15)
    parser.add_argument("--spatial_noise_std", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def question_types_from_arg(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def extract_docs_from_log(log_path: Path) -> List[dict]:
    data = json.loads(log_path.read_text())
    return [item["doc"] for item in data.get("logs", []) if item.get("doc")]


def load_docs_from_hf(args) -> List[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is not installed in the current environment. Please pass --source_log_json or run inside the lmms_eval environment."
        ) from exc

    dataset = load_dataset("nyu-visionx/VSI-Bench", split="test", cache_dir=args.dataset_cache_dir)
    return [dataset[i] for i in range(len(dataset))]


def sample_docs_per_type(docs: Sequence[dict], question_types: Sequence[str], per_type_limit: int, seed: int) -> List[dict]:
    filtered = [doc for doc in docs if doc.get("question_type") in question_types and doc.get("options")]
    rng = np.random.default_rng(seed)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for doc in filtered:
        grouped[doc["question_type"]].append(doc)

    selected = []
    for question_type in question_types:
        candidates = grouped.get(question_type, [])
        if not candidates:
            continue
        order = rng.permutation(len(candidates)).tolist() if len(candidates) > 1 else [0]
        for idx in order[: min(per_type_limit, len(candidates))]:
            selected.append(candidates[idx])
    return selected


def resolve_video_path(doc: dict, dataset_cache_dir: Path) -> Path:
    path = dataset_cache_dir / doc["dataset"] / f"{doc['scene_name']}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find video for doc {doc.get('id')} at: {path}")
    return path


def parse_option_entries(options: Sequence[str]) -> List[dict]:
    entries = []
    for idx, option in enumerate(options):
        match = re.match(r"^\s*([A-Z])[\.)]\s*(.*)$", option)
        if match:
            label = match.group(1)
            text = match.group(2).strip()
        else:
            label = chr(ord("A") + idx)
            text = option.strip()
        display_text = f"{label}. {text}"
        entries.append(
            {
                "label": label,
                "text": text,
                "display_text": display_text,
            }
        )
    return entries


def build_vsibench_prompt(doc: dict, args, model, frame_time: str, video_time: float, option_entries: Sequence[dict]) -> str:
    question = doc["question"]
    if args.add_time_instruction:
        question = (
            f"The video lasts for {video_time:.2f} seconds, and {len(frame_time.split(','))} frames are uniformly "
            f"sampled from it. These frames are located at {frame_time}. Please answer the following question related to this video.\n{question}"
        )

    qs = "\n".join(
        [
            args.pre_prompt,
            question,
            "Options:",
            *[entry["display_text"] for entry in option_entries],
            args.mca_post_prompt,
        ]
    )

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_model(args):
    model_name = get_model_name_from_path(args.model_path)
    if not args.overwrite:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.model_path, args.model_base, model_name, load_8bit=args.load_8bit
        )
    else:
        overwrite_config = {
            "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
            "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
            "mm_newline_position": args.mm_newline_position,
        }
        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)
        if "qwen" not in args.model_path.lower():
            if "224" in cfg_pretrained.mm_vision_tower:
                least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
            else:
                least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000
            scaling_factor = math.ceil(least_token_number / 4096)
            if scaling_factor >= 2:
                if "vicuna" in cfg_pretrained._name_or_path.lower():
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            load_8bit=args.load_8bit,
            overwrite_config=overwrite_config,
        )

    model.to("cuda")
    ensure_pad_token(tokenizer)
    return tokenizer, model, image_processor, max_length


def load_video_tensor(video_path: Path, args, image_processor):
    video_np, frame_time, video_time = load_video(str(video_path), args)
    video_tensor = image_processor.preprocess(video_np, return_tensors="pt")["pixel_values"].half().cuda()
    return [video_tensor], frame_time, video_time


def _combine_lists(original: Sequence[float], degraded: Sequence[float], alpha: float) -> List[float]:
    return [(1.0 + alpha) * orig - alpha * deg for orig, deg in zip(original, degraded)]


def evaluate_setting(
    tokenizer,
    model,
    prompt_prefix: str,
    option_entries: Sequence[dict],
    ground_truth_label: str,
    video,
    branch_bundle: Optional[dict],
    alpha: float,
) -> dict:
    results = []
    for entry in option_entries:
        candidate = entry["display_text"]
        orig_images = video if branch_bundle is None or branch_bundle.get("orig_images") is None else branch_bundle["orig_images"]
        orig_spatial = None if branch_bundle is None else branch_bundle.get("orig_spatial_features")
        orig_video_features = None if branch_bundle is None else branch_bundle.get("orig_video_features")
        if orig_video_features is not None:
            orig_metrics = score_candidate_with_video_features(
                tokenizer=tokenizer,
                model=model,
                prompt_prefix=prompt_prefix,
                candidate=candidate,
                video_features=orig_video_features,
            )
        else:
            orig_metrics = score_candidate_with_details(
                tokenizer=tokenizer,
                model=model,
                prompt_prefix=prompt_prefix,
                candidate=candidate,
                images=orig_images,
                spatial_features=orig_spatial,
            )

        item = {
            "candidate": candidate,
            "candidate_label": entry["label"],
            "candidate_text": entry["text"],
            "is_correct": entry["label"] == ground_truth_label,
            "answer_token_count": orig_metrics["valid_token_count"],
            "token_ids": orig_metrics["token_ids"],
            "token_texts": orig_metrics["token_texts"],
            "original_sequence_logprob": orig_metrics["sequence_logprob"],
            "original_avg_logprob": orig_metrics["avg_logprob"],
            "original_token_logits": orig_metrics["token_logits"],
            "original_token_logprobs": orig_metrics["token_logprobs"],
        }

        if branch_bundle is None:
            item["combined_score"] = orig_metrics["sequence_logprob"]
            item["combined_token_logits"] = list(orig_metrics["token_logits"])
            item["combined_token_logprobs"] = list(orig_metrics["token_logprobs"])
        else:
            degraded_video_features = branch_bundle.get("degraded_video_features")
            if degraded_video_features is not None:
                degraded_metrics = score_candidate_with_video_features(
                    tokenizer=tokenizer,
                    model=model,
                    prompt_prefix=prompt_prefix,
                    candidate=candidate,
                    video_features=degraded_video_features,
                )
            else:
                degraded_metrics = score_candidate_with_details(
                    tokenizer=tokenizer,
                    model=model,
                    prompt_prefix=prompt_prefix,
                    candidate=candidate,
                    images=branch_bundle["degraded_images"],
                    spatial_features=branch_bundle.get("degraded_spatial_features"),
                )
            item["degraded_sequence_logprob"] = degraded_metrics["sequence_logprob"]
            item["degraded_avg_logprob"] = degraded_metrics["avg_logprob"]
            item["degraded_token_logits"] = degraded_metrics["token_logits"]
            item["degraded_token_logprobs"] = degraded_metrics["token_logprobs"]
            item["combined_score"] = (1.0 + alpha) * item["original_sequence_logprob"] - alpha * item["degraded_sequence_logprob"]
            item["combined_token_logits"] = _combine_lists(item["original_token_logits"], item["degraded_token_logits"], alpha)
            item["combined_token_logprobs"] = _combine_lists(item["original_token_logprobs"], item["degraded_token_logprobs"], alpha)
        results.append(item)

    combined_scores = np.array([item["combined_score"] for item in results], dtype=np.float32)
    posterior = np.exp(combined_scores - combined_scores.max())
    posterior = posterior / posterior.sum()
    for item, prob in zip(results, posterior.tolist()):
        item["posterior_over_choices"] = prob
        item["mean_original_token_logit"] = float(np.mean(item["original_token_logits"]))
        item["mean_combined_token_logit"] = float(np.mean(item["combined_token_logits"]))
        if "degraded_token_logits" in item:
            item["mean_degraded_token_logit"] = float(np.mean(item["degraded_token_logits"]))
            item["mean_token_logit_delta"] = float(np.mean(np.array(item["combined_token_logits"]) - np.array(item["original_token_logits"])))
        else:
            item["mean_token_logit_delta"] = 0.0

    best_idx = int(combined_scores.argmax())
    predicted_item = results[best_idx]
    return {
        "prediction": predicted_item["candidate"],
        "prediction_label": predicted_item["candidate_label"],
        "results": results,
        "metadata": None if branch_bundle is None else branch_bundle.get("metadata"),
    }


def build_summary(doc_results: Sequence[dict], setting_names: Sequence[str], question_types: Sequence[str]) -> dict:
    summary = {"overall": {}, "per_question_type": {}}
    for setting in setting_names:
        correct = sum(1 for item in doc_results if item["settings"][setting]["prediction_label"] == item["ground_truth"])
        summary["overall"][setting] = {
            "num_docs": len(doc_results),
            "accuracy": correct / len(doc_results) if doc_results else 0.0,
        }

    for question_type in question_types:
        subset = [item for item in doc_results if item["question_type"] == question_type]
        if not subset:
            continue
        summary["per_question_type"][question_type] = {}
        for setting in setting_names:
            correct = sum(1 for item in subset if item["settings"][setting]["prediction_label"] == item["ground_truth"])
            correct_deltas = []
            wrong_deltas = []
            for item in subset:
                for result in item["settings"][setting]["results"]:
                    if result["candidate_label"] == item["ground_truth"]:
                        correct_deltas.append(result["mean_token_logit_delta"])
                    else:
                        wrong_deltas.append(result["mean_token_logit_delta"])
            summary["per_question_type"][question_type][setting] = {
                "num_docs": len(subset),
                "accuracy": correct / len(subset),
                "avg_correct_option_mean_token_logit_delta": float(np.mean(correct_deltas)) if correct_deltas else 0.0,
                "avg_wrong_option_mean_token_logit_delta": float(np.mean(wrong_deltas)) if wrong_deltas else 0.0,
            }
    return summary


def save_summary_csv(summary: dict, csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scope",
                "question_type",
                "setting",
                "num_docs",
                "accuracy",
                "avg_correct_option_mean_token_logit_delta",
                "avg_wrong_option_mean_token_logit_delta",
            ]
        )
        for setting, stats in summary.get("overall", {}).items():
            writer.writerow(["overall", "all", setting, stats["num_docs"], stats["accuracy"], "", ""])
        for question_type, per_setting in summary.get("per_question_type", {}).items():
            for setting, stats in per_setting.items():
                writer.writerow(
                    [
                        "per_question_type",
                        question_type,
                        setting,
                        stats["num_docs"],
                        stats["accuracy"],
                        stats["avg_correct_option_mean_token_logit_delta"],
                        stats["avg_wrong_option_mean_token_logit_delta"],
                    ]
                )


def sanitize_filename(text: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", text).strip("_")
    return safe[:max_len] if safe else "item"


def pretty_token(token: str) -> str:
    token = token.replace("Ġ", "␠")
    token = token.replace("▁", "␠")
    token = token.replace("Ċ", "\\n")
    return token


def option_group_bounds(results: Sequence[dict]):
    bounds = []
    tick_positions = []
    tick_labels = []
    token_centers = []
    cursor = 0.0
    option_gap = 1.4
    token_gap = 1.0
    for result in results:
        start = cursor
        for token_idx, token in enumerate(result["token_texts"]):
            tick_positions.append(cursor)
            tick_labels.append(f"t{token_idx}\n{pretty_token(token)}")
            token_centers.append(cursor)
            cursor += token_gap
        end = cursor - token_gap if result["token_texts"] else start
        bounds.append(
            {
                "candidate_label": result["candidate_label"],
                "candidate": result["candidate"],
                "candidate_text": result["candidate_text"],
                "start": start,
                "end": end,
                "center": (start + end) / 2 if end >= start else start,
            }
        )
        cursor += option_gap
    return bounds, tick_positions, tick_labels, token_centers


def draw_setting_subplot(ax, setting_name: str, setting_result: dict, ground_truth_label: str):
    results = setting_result["results"]
    bounds, tick_positions, tick_labels, token_centers = option_group_bounds(results)
    states = ["original"] if setting_name == "baseline" else ["original", "degraded", "combined"]
    width = 0.22 if len(states) == 3 else 0.5
    offsets = np.linspace(-width, width, num=len(states)) if len(states) > 1 else np.array([0.0])

    for bound in bounds:
        if bound["candidate_label"] == ground_truth_label:
            ax.axvspan(bound["start"] - 0.6, bound["end"] + 0.6, color="#9FD89F", alpha=0.20, zorder=0)

    for state_idx, state in enumerate(states):
        xs = []
        ys = []
        for center, result in zip(bounds, results):
            values = result[f"{state}_token_logits"]
            for token_offset, value in enumerate(values):
                xs.append(center["start"] + token_offset + offsets[state_idx])
                ys.append(value)
        ax.bar(xs, ys, width=width, color=STATE_COLORS[state], alpha=0.88, label=state, zorder=2)

    max_y = max(max(result[f"{states[0]}_token_logits"]) for result in results if result[f"{states[0]}_token_logits"])
    min_y = min(min(result[f"{states[0]}_token_logits"]) for result in results if result[f"{states[0]}_token_logits"])
    for state in states[1:]:
        state_max = max(max(result[f"{state}_token_logits"]) for result in results if result[f"{state}_token_logits"])
        state_min = min(min(result[f"{state}_token_logits"]) for result in results if result[f"{state}_token_logits"])
        max_y = max(max_y, state_max)
        min_y = min(min_y, state_min)

    text_y = max_y + max(0.8, 0.08 * (max_y - min_y + 1e-6))
    for bound, result in zip(bounds, results):
        option_prefix = f"{result['candidate_label']}"
        if result["candidate_label"] == ground_truth_label:
            option_prefix += " [GT]"
        if result["candidate_label"] == setting_result["prediction_label"]:
            option_prefix += " [Pred]"
        option_text = result["candidate_text"]
        if len(option_text) > 36:
            option_text = option_text[:33] + "..."
        ax.text(bound["center"], text_y, f"{option_prefix}\n{option_text}", ha="center", va="bottom", fontsize=9)
        ax.axvline(bound["end"] + 0.7, color="#CCCCCC", linewidth=0.8, zorder=1)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_ylabel("token logit")
    ax.set_title(f"{SETTING_DISPLAY_NAMES[setting_name]} | pred={setting_result['prediction_label']} | gt={ground_truth_label}")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(min_y - 0.8, text_y + 0.8)


def plot_per_question(doc_results: Sequence[dict], setting_names: Sequence[str], plots_root: Path):
    plots_root.mkdir(parents=True, exist_ok=True)
    for item in doc_results:
        question_dir = plots_root / item["question_type"]
        question_dir.mkdir(parents=True, exist_ok=True)
        total_tokens = sum(len(result["token_texts"]) for result in item["settings"][setting_names[0]]["results"])
        fig_width = max(18, total_tokens * 0.55)
        fig_height = 4.8 * len(setting_names) + 2.5
        fig, axes = plt.subplots(len(setting_names), 1, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes[:, 0]

        question_title = item["question"]
        if len(question_title) > 180:
            question_title = question_title[:177] + "..."
        fig.suptitle(
            f"Question type: {item['question_type']}\n"
            f"doc_id={item['doc_id']} | scene={item['scene_name']} | ground truth={item['ground_truth']}\n"
            f"{question_title}",
            fontsize=14,
            y=0.995,
        )

        for ax, setting_name in zip(axes, setting_names):
            draw_setting_subplot(ax, setting_name, item["settings"][setting_name], item["ground_truth"])

        axes[-1].set_xlabel("option-token position")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        stem = sanitize_filename(f"doc_{item['doc_index']:03d}_id_{item['doc_id']}_{item['scene_name']}")
        fig.savefig(question_dir / f"{stem}.png", dpi=220)
        plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    question_types = question_types_from_arg(args.question_types)

    if args.source_log_json:
        source_docs = extract_docs_from_log(Path(args.source_log_json))
    else:
        source_docs = load_docs_from_hf(args)

    docs = sample_docs_per_type(source_docs, question_types, args.per_type_limit, args.sample_seed)
    if not docs:
        raise RuntimeError("No VSIBench MCA docs were selected. Please check --source_log_json / dataset cache / --question_types.")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / run_name
    plots_root = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer, model, image_processor, _ = load_model(args)
    branch_args = SimpleNamespace(
        visual_drop_rate=args.visual_drop_rate,
        visual_noise_std=args.visual_noise_std,
        fusion_weak_ratio=args.fusion_weak_ratio,
        fusion_drop_rate=args.fusion_drop_rate,
        fusion_noise_std=args.fusion_noise_std,
        spatial_drop_rate=args.spatial_drop_rate,
        spatial_noise_std=args.spatial_noise_std,
    )

    doc_results = []
    for index, doc in enumerate(docs, start=1):
        video_path = resolve_video_path(doc, Path(args.dataset_cache_dir))
        video, frame_time, video_time = load_video_tensor(video_path, args, image_processor)
        option_entries = parse_option_entries(doc["options"])
        prompt_prefix = build_vsibench_prompt(doc, args, model, frame_time, video_time, option_entries)

        baseline = evaluate_setting(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            option_entries=option_entries,
            ground_truth_label=doc["ground_truth"],
            video=video,
            branch_bundle=None,
            alpha=args.alpha,
        )
        two_d_feature_vcd = evaluate_setting(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            option_entries=option_entries,
            ground_truth_label=doc["ground_truth"],
            video=video,
            branch_bundle=build_2d_feature_branch(branch_args, model, video),
            alpha=args.alpha,
        )
        weak_fusion_vcd = evaluate_setting(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            option_entries=option_entries,
            ground_truth_label=doc["ground_truth"],
            video=video,
            branch_bundle=build_weak_fusion_branch(branch_args, model, video),
            alpha=args.alpha,
        )
        spatial_vcd = evaluate_setting(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            option_entries=option_entries,
            ground_truth_label=doc["ground_truth"],
            video=video,
            branch_bundle=build_spatial_noise_branch(branch_args, model, video),
            alpha=args.alpha,
        )

        doc_results.append(
            {
                "doc_index": index,
                "doc_id": doc.get("id"),
                "dataset": doc["dataset"],
                "scene_name": doc["scene_name"],
                "question_type": doc["question_type"],
                "question": doc["question"],
                "ground_truth": doc["ground_truth"],
                "options": doc["options"],
                "video_path": str(video_path),
                "frame_time": frame_time,
                "prompt_prefix": prompt_prefix,
                "option_entries": option_entries,
                "settings": {
                    "baseline": baseline,
                    "two_d_feature_vcd": two_d_feature_vcd,
                    "weak_fusion_vcd": weak_fusion_vcd,
                    "spatial_noise_vcd": spatial_vcd,
                },
            }
        )
        print(
            f"[{index}/{len(docs)}] {doc['question_type']} | doc_id={doc.get('id')} | "
            f"GT={doc['ground_truth']} | baseline={baseline['prediction_label']} | "
            f"2d={two_d_feature_vcd['prediction_label']} | "
            f"weak_fusion={weak_fusion_vcd['prediction_label']} | "
            f"spatial={spatial_vcd['prediction_label']}"
        )

    setting_names = ["baseline", "two_d_feature_vcd", "weak_fusion_vcd", "spatial_noise_vcd"]
    summary = build_summary(doc_results, setting_names, question_types)
    plot_per_question(doc_results, setting_names, plots_root)

    payload = {
        "config": vars(args),
        "question_types": question_types,
        "selected_doc_count": len(doc_results),
        "summary": summary,
        "docs": doc_results,
    }

    results_json = output_root / "results.json"
    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    results_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    save_summary_csv(summary, summary_csv)

    print(f"Saved detailed results to: {results_json}")
    print(f"Saved summary to: {summary_json}")
    print(f"Saved summary csv to: {summary_csv}")
    print(f"Saved plots to: {plots_root}")


if __name__ == "__main__":
    main()
