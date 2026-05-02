#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from vcd.vcd_feature_degradation.option_demo_utils import ensure_pad_token, load_video, score_candidate_with_video_features, set_seed
from vcd.vcd_vision_token.methods import build_stage_setting_bundles


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

SETTING_DISPLAY_NAMES = {
    "stage0_coarse_only": "Stage 0: Coarse only",
    "stage0_semantic_negative_coarse": "Stage 0-neg: Coarse after corrupting semantic top-ratio patches",
    "stage0_semantic_vcd_combined": "Stage 0-VCD: (1+α)·orig - α·neg",
    "stage1_semantic_fine": "Stage 1: Coarse + semantic fine",
    "stage2_fusion_guided_fine": "Stage 2: Coarse + fusion-guided fine",
    "stage3_joint_semantic_fusion": "Stage 3: Coarse + joint semantic/fusion fine",
}

STATE_COLORS = {
    "stage0_coarse_only": "#4C78A8",
    "stage0_semantic_negative_coarse": "#E45756",
    "stage0_semantic_vcd_combined": "#72B7B2",
    "stage1_semantic_fine": "#F58518",
    "stage2_fusion_guided_fine": "#54A24B",
    "stage3_joint_semantic_fusion": "#B279A2",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate coarse vs semantic-negative-coarse vision-token branches on VSIBench MCA question types.")
    parser.add_argument("--source_log_json", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=os.path.expanduser(os.getenv("VSIBENCH_CACHE_DIR", "~/.cache/huggingface/vsibench")))
    parser.add_argument("--output_root", type=str, default="vcd/vcd_vision_token/results")
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

    parser.add_argument("--append_newline", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--stage1_topk", type=int, default=16)
    parser.add_argument("--stage1_scoring_mode", type=str, default="question_cosine")
    parser.add_argument("--stage1_fine_scale", type=float, default=1.0)
    parser.add_argument("--semantic_neg_ratio", type=float, default=0.5)
    parser.add_argument("--semantic_neg_corrupt_mode", type=str, default="zero", choices=["frame_mean", "zero"])
    parser.add_argument("--alpha", type=float, default=1.0, help="VCD contrastive weight for the negative branch.")
    parser.add_argument("--stage2_topk", type=int, default=16)
    parser.add_argument("--stage2_scoring_mode", type=str, default="fusion_2d3d")
    parser.add_argument("--stage2_fine_scale", type=float, default=1.0)
    parser.add_argument("--stage2_fusion_2d_weight", type=float, default=1.0)
    parser.add_argument("--stage2_fusion_3d_weight", type=float, default=1.0)
    parser.add_argument("--stage3_topk", type=int, default=16)
    parser.add_argument("--stage3_fine_scale", type=float, default=1.0)
    parser.add_argument("--stage3_semantic_weight", type=float, default=1.0)
    parser.add_argument("--stage3_fusion_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def question_types_from_arg(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def to_jsonable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return obj


def extract_docs_from_log(log_path: Path) -> List[dict]:
    data = json.loads(log_path.read_text())
    return [item["doc"] for item in data.get("logs", []) if item.get("doc")]


def load_docs_from_hf(args) -> List[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is not installed. Please pass --source_log_json or use the lmms_eval environment.") from exc
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
        entries.append({"label": label, "text": text, "display_text": f"{label}. {text}"})
    return entries


def build_vsibench_prompt(doc: dict, args, model, frame_time: str, video_time: float, option_entries: Sequence[dict]) -> str:
    question = doc["question"]
    if args.add_time_instruction:
        question = (
            f"The video lasts for {video_time:.2f} seconds, and {len(frame_time.split(','))} frames are uniformly "
            f"sampled from it. These frames are located at {frame_time}. Please answer the following question related to this video.\n{question}"
        )
    qs = "\n".join([args.pre_prompt, question, "Options:", *[x["display_text"] for x in option_entries], args.mca_post_prompt])
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_prompt_input_ids(tokenizer, prompt_prefix: str):
    input_ids = tokenizer_image_token(prompt_prefix, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    return input_ids, attention_mask


def load_model(args):
    model_name = get_model_name_from_path(args.model_path)
    if not args.overwrite:
        tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit)
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


def evaluate_setting(tokenizer, model, prompt_prefix: str, option_entries: Sequence[dict], ground_truth_label: str, video_features: torch.Tensor, metadata=None) -> dict:
    results = []
    for entry in option_entries:
        metrics = score_candidate_with_video_features(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            candidate=entry["display_text"],
            video_features=video_features,
        )
        results.append(
            {
                "candidate": entry["display_text"],
                "candidate_label": entry["label"],
                "candidate_text": entry["text"],
                "is_correct": entry["label"] == ground_truth_label,
                "answer_token_count": metrics["valid_token_count"],
                "token_ids": metrics["token_ids"],
                "token_texts": metrics["token_texts"],
                "sequence_logprob": metrics["sequence_logprob"],
                "avg_logprob": metrics["avg_logprob"],
                "token_logits": metrics["token_logits"],
                "token_logprobs": metrics["token_logprobs"],
                "first_token_logit": metrics["first_token_logit"],
                "mean_token_logit": float(np.mean(metrics["token_logits"])),
            }
        )
    scores = np.array([x["sequence_logprob"] for x in results], dtype=np.float32)
    posterior = np.exp(scores - scores.max())
    posterior = posterior / posterior.sum()
    for item, prob in zip(results, posterior.tolist()):
        item["posterior_over_choices"] = prob
    best_idx = int(scores.argmax())
    pred = results[best_idx]
    return {
        "prediction": pred["candidate"],
        "prediction_label": pred["candidate_label"],
        "results": results,
        "metadata": metadata,
    }


def evaluate_vcd_combined_setting(
    original_setting: dict,
    negative_setting: dict,
    alpha: float,
) -> dict:
    orig_results = original_setting["results"]
    neg_results = negative_setting["results"]
    if len(orig_results) != len(neg_results):
        raise ValueError("Original and negative settings must have the same number of options.")

    results = []
    for orig_item, neg_item in zip(orig_results, neg_results):
        if orig_item["candidate_label"] != neg_item["candidate_label"]:
            raise ValueError("Original and negative settings must align on candidate labels.")
        combined_score = (1.0 + alpha) * orig_item["sequence_logprob"] - alpha * neg_item["sequence_logprob"]
        results.append(
            {
                "candidate": orig_item["candidate"],
                "candidate_label": orig_item["candidate_label"],
                "candidate_text": orig_item["candidate_text"],
                "is_correct": orig_item["is_correct"],
                "sequence_logprob": float(combined_score),
                "original_sequence_logprob": orig_item["sequence_logprob"],
                "negative_sequence_logprob": neg_item["sequence_logprob"],
                "score_delta_vs_original": float(combined_score - orig_item["sequence_logprob"]),
            }
        )

    scores = np.array([x["sequence_logprob"] for x in results], dtype=np.float32)
    posterior = np.exp(scores - scores.max())
    posterior = posterior / posterior.sum()
    for item, prob in zip(results, posterior.tolist()):
        item["posterior_over_choices"] = prob
    best_idx = int(scores.argmax())
    pred = results[best_idx]
    return {
        "prediction": pred["candidate"],
        "prediction_label": pred["candidate_label"],
        "results": results,
        "metadata": {
            "stage_name": "stage0_semantic_vcd_combined",
            "alpha": float(alpha),
            "original_stage": "stage0_coarse_only",
            "negative_stage": "stage0_semantic_negative_coarse",
        },
    }


def build_summary(doc_results: Sequence[dict], setting_names: Sequence[str], question_types: Sequence[str]) -> dict:
    summary = {"overall": {}, "per_question_type": {}}
    for setting in setting_names:
        correct = sum(1 for item in doc_results if item["settings"][setting]["prediction_label"] == item["ground_truth"])
        summary["overall"][setting] = {"num_docs": len(doc_results), "accuracy": correct / len(doc_results) if doc_results else 0.0}

    baseline_name = setting_names[0]
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
                baseline_results = {x["candidate_label"]: x for x in item["settings"][baseline_name]["results"]}
                cur_results = {x["candidate_label"]: x for x in item["settings"][setting]["results"]}
                for label, base_item in baseline_results.items():
                    delta = cur_results[label]["sequence_logprob"] - base_item["sequence_logprob"]
                    if label == item["ground_truth"]:
                        correct_deltas.append(delta)
                    else:
                        wrong_deltas.append(delta)
            summary["per_question_type"][question_type][setting] = {
                "num_docs": len(subset),
                "accuracy": correct / len(subset),
                "avg_correct_option_score_delta_vs_stage0": float(np.mean(correct_deltas)) if correct_deltas else 0.0,
                "avg_wrong_option_score_delta_vs_stage0": float(np.mean(wrong_deltas)) if wrong_deltas else 0.0,
            }
    return summary


def save_summary_csv(summary: dict, csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scope", "question_type", "setting", "num_docs", "accuracy",
            "avg_correct_option_score_delta_vs_stage0", "avg_wrong_option_score_delta_vs_stage0",
        ])
        for setting, stats in summary.get("overall", {}).items():
            writer.writerow(["overall", "all", setting, stats["num_docs"], stats["accuracy"], "", ""])
        for question_type, per_setting in summary.get("per_question_type", {}).items():
            for setting, stats in per_setting.items():
                writer.writerow([
                    "per_question_type", question_type, setting, stats["num_docs"], stats["accuracy"],
                    stats["avg_correct_option_score_delta_vs_stage0"],
                    stats["avg_wrong_option_score_delta_vs_stage0"],
                ])


def sanitize_filename(text: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", text).strip("_")
    return safe[:max_len] if safe else "item"


def pretty_token(token: str) -> str:
    token = token.replace("Ġ", "␠").replace("▁", "␠").replace("Ċ", "\\n")
    return token


def option_group_bounds(results: Sequence[dict]):
    bounds = []
    tick_positions = []
    tick_labels = []
    cursor = 0.0
    option_gap = 1.4
    token_gap = 1.0
    for result in results:
        start = cursor
        for token_idx, token in enumerate(result["token_texts"]):
            tick_positions.append(cursor)
            tick_labels.append(f"t{token_idx}\\n{pretty_token(token)}")
            cursor += token_gap
        end = cursor - token_gap if result["token_texts"] else start
        bounds.append({
            "candidate_label": result["candidate_label"],
            "candidate_text": result["candidate_text"],
            "start": start,
            "end": end,
            "center": (start + end) / 2 if end >= start else start,
        })
        cursor += option_gap
    return bounds, tick_positions, tick_labels


def draw_option_score_summary(ax, item: dict, setting_names: Sequence[str]):
    baseline_results = item["settings"][setting_names[0]]["results"]
    option_labels = [result["candidate_label"] for result in baseline_results]
    option_texts = [result["candidate_text"] for result in baseline_results]
    x = np.arange(len(option_labels), dtype=np.float32)
    bar_width = 0.8 / max(len(setting_names), 1)

    for option_idx, option_label in enumerate(option_labels):
        if option_label == item["ground_truth"]:
            ax.axvspan(option_idx - 0.5, option_idx + 0.5, color="#9FD89F", alpha=0.20, zorder=0)

    for setting_idx, setting_name in enumerate(setting_names):
        setting_results = item["settings"][setting_name]["results"]
        scores = [result["sequence_logprob"] for result in setting_results]
        offset = (setting_idx - (len(setting_names) - 1) / 2.0) * bar_width
        bars = ax.bar(
            x + offset,
            scores,
            width=bar_width * 0.92,
            color=STATE_COLORS[setting_name],
            alpha=0.88,
            label=SETTING_DISPLAY_NAMES[setting_name],
            zorder=2,
        )
        baseline_by_label = {
            result["candidate_label"]: result["sequence_logprob"]
            for result in baseline_results
        }
        for bar, result in zip(bars, setting_results):
            score = result["sequence_logprob"]
            delta_text = ""
            if setting_name != setting_names[0]:
                delta = score - baseline_by_label[result["candidate_label"]]
                delta_text = f"\\nΔ{delta:+.2f}"
            pred_tag = " [Pred]" if result["candidate_label"] == item["settings"][setting_name]["prediction_label"] else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + (0.06 if score >= 0 else -0.06),
                f"{score:.2f}{delta_text}{pred_tag}",
                ha="center",
                va="bottom" if score >= 0 else "top",
                fontsize=8,
                rotation=90,
            )

    tick_labels = []
    for option_label, option_text in zip(option_labels, option_texts):
        short_text = option_text if len(option_text) <= 22 else option_text[:19] + "..."
        gt_tag = " [GT]" if option_label == item["ground_truth"] else ""
        tick_labels.append(f"{option_label}{gt_tag}\\n{short_text}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("sequence logprob")
    ax.set_title("Final option scores (higher is better; bars show score changes vs coarse baseline)")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=1)
    ax.legend(loc="best", fontsize=8)


def plot_per_question(doc_results: Sequence[dict], setting_names: Sequence[str], plots_root: Path):
    plots_root.mkdir(parents=True, exist_ok=True)
    for item in doc_results:
        question_dir = plots_root / item["question_type"]
        question_dir.mkdir(parents=True, exist_ok=True)
        fig_width = max(14, len(item["settings"][setting_names[0]]["results"]) * 3.2)
        fig_height = 6.2
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        question_title = item["question"]
        if len(question_title) > 180:
            question_title = question_title[:177] + "..."
        fig.suptitle(
            f"Question type: {item['question_type']}\\n"
            f"doc_id={item['doc_id']} | scene={item['scene_name']} | ground truth={item['ground_truth']}\\n"
            f"{question_title}",
            fontsize=14,
            y=0.995,
        )
        draw_option_score_summary(ax, item, setting_names)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        stem = sanitize_filename(f"doc_{item['doc_index']:03d}_id_{item['doc_id']}_{item['scene_name']}")
        fig.savefig(question_dir / f"{stem}.png", dpi=220)
        plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    question_types = question_types_from_arg(args.question_types)
    source_docs = extract_docs_from_log(Path(args.source_log_json)) if args.source_log_json else load_docs_from_hf(args)
    docs = sample_docs_per_type(source_docs, question_types, args.per_type_limit, args.sample_seed)
    if not docs:
        raise RuntimeError("No VSIBench MCA docs were selected.")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_vsibench_mca_vtoken_{args.per_type_limit}_per_type"
    output_root = Path(args.output_root) / run_name
    plots_root = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer, model, image_processor, _ = load_model(args)
    setting_names = [
        "stage0_coarse_only",
        "stage0_semantic_negative_coarse",
        "stage0_semantic_vcd_combined",
    ]

    doc_results = []
    for index, doc in enumerate(docs, start=1):
        video_path = resolve_video_path(doc, Path(args.dataset_cache_dir))
        video, frame_time, video_time = load_video_tensor(video_path, args, image_processor)
        option_entries = parse_option_entries(doc["options"])
        prompt_prefix = build_vsibench_prompt(doc, args, model, frame_time, video_time, option_entries)
        input_ids, attention_mask = build_prompt_input_ids(tokenizer, prompt_prefix)

        stage_bundles = build_stage_setting_bundles(args, model, video, input_ids, attention_mask)
        coarse_setting = evaluate_setting(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            option_entries=option_entries,
            ground_truth_label=doc["ground_truth"],
            video_features=stage_bundles["stage0_coarse_only"]["video_features"],
            metadata=stage_bundles["stage0_coarse_only"].get("metadata"),
        )
        negative_setting = evaluate_setting(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            option_entries=option_entries,
            ground_truth_label=doc["ground_truth"],
            video_features=stage_bundles["stage0_semantic_negative_coarse"]["video_features"],
            metadata=stage_bundles["stage0_semantic_negative_coarse"].get("metadata"),
        )
        combined_setting = evaluate_vcd_combined_setting(
            original_setting=coarse_setting,
            negative_setting=negative_setting,
            alpha=args.alpha,
        )
        settings = {
            "stage0_coarse_only": coarse_setting,
            "stage0_semantic_negative_coarse": negative_setting,
            "stage0_semantic_vcd_combined": combined_setting,
        }

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
                "settings": settings,
            }
        )
        print(
            f"[{index}/{len(docs)}] {doc['question_type']} | doc_id={doc.get('id')} | GT={doc['ground_truth']} | "
            + " | ".join(f"{name}={settings[name]['prediction_label']}" for name in setting_names)
        )

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
    results_json.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))
    summary_json.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
    save_summary_csv(summary, summary_csv)
    print(f"Saved detailed results to: {results_json}")
    print(f"Saved summary to: {summary_json}")
    print(f"Saved summary csv to: {summary_csv}")
    print(f"Saved plots to: {plots_root}")


if __name__ == "__main__":
    main()
