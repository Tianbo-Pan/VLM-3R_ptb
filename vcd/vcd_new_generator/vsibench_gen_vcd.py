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
import re
from collections import Counter, defaultdict
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoConfig

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from vcd.vcd_feature_degradation.option_demo_utils import ensure_pad_token, load_video, set_seed
from vcd.vcd_vision_token.generation_vcd import generate_with_vcd
from vcd.vcd_vision_token.methods import build_generation_branch_bundle


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

DEFAULT_QUESTION_TYPES = MCA_QUESTION_TYPES + NA_QUESTION_TYPES

SETTING_DISPLAY_NAMES = {
    "baseline_generate": "Baseline generate",
    "pairwise_gen_vcd": "Pairwise generation-time VCD",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate general generation-time VCD on selected VSIBench question types.")
    parser.add_argument("--source_log_json", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=os.path.expanduser(os.getenv("VSIBENCH_CACHE_DIR", "~/.cache/huggingface/vsibench")))
    parser.add_argument("--output_root", type=str, default="vcd/vcd_new_generator/results")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--per_type_limit", type=int, default=5)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--question_types", type=str, default="all")

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
    parser.add_argument("--mca_post_prompt", type=str, default="Answer with the option's letter from the given choices directly.")
    parser.add_argument("--na_post_prompt", type=str, default="Please answer the question using a single word or phrase.")

    parser.add_argument("--append_newline", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--stage1_topk", type=int, default=16)
    parser.add_argument("--stage1_scoring_mode", type=str, default="question_cosine")
    parser.add_argument("--stage1_fine_scale", type=float, default=1.0)
    parser.add_argument("--semantic_neg_ratio", type=float, default=0.5)
    parser.add_argument("--semantic_neg_corrupt_mode", type=str, default="zero", choices=["frame_mean", "zero"])
    parser.add_argument("--stage2_topk", type=int, default=16)
    parser.add_argument("--stage2_scoring_mode", type=str, default="fusion_2d3d")
    parser.add_argument("--stage2_fine_scale", type=float, default=1.0)
    parser.add_argument("--stage2_fusion_2d_weight", type=float, default=1.0)
    parser.add_argument("--stage2_fusion_3d_weight", type=float, default=1.0)
    parser.add_argument("--stage3_topk", type=int, default=16)
    parser.add_argument("--stage3_fine_scale", type=float, default=1.0)
    parser.add_argument("--stage3_semantic_weight", type=float, default=1.0)
    parser.add_argument("--stage3_fusion_weight", type=float, default=1.0)

    parser.add_argument("--pairwise_alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def question_types_from_arg(raw: str) -> List[str]:
    if str(raw).strip().lower() == "all":
        return []
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
    filtered = [doc for doc in docs if doc.get("question_type") in question_types]
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


def fuzzy_matching(pred: str) -> str:
    return str(pred).split(" ")[0].rstrip(".").strip()


def parse_first_number(text: str) -> Optional[float]:
    if text is None:
        return None
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", str(text).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def abs_dist_norm(pred: float, target: float) -> float:
    return abs(pred - target) / max(abs(target), 1e-12)


def mean_relative_accuracy(pred: Optional[float], target: Optional[float], start: float = 0.5, end: float = 0.95, interval: float = 0.05) -> float:
    if pred is None or target is None:
        return 0.0
    num_pts = int((end - start) / interval + 2)
    conf_intervals = np.linspace(start, end, num_pts)
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervals
    return float(np.mean(accuracy))


def score_prediction(doc: dict, raw_text: str) -> dict:
    question_type = doc["question_type"]
    if question_type in MCA_QUESTION_TYPES:
        prediction_label = parse_first_option_letter(raw_text)
        score = 1.0 if prediction_label == doc["ground_truth"] else 0.0
        return {
            "prediction_label": prediction_label,
            "score": score,
            "metric_name": "accuracy",
            "is_correct": bool(score),
        }

    if question_type in NA_QUESTION_TYPES:
        prediction_number = parse_first_number(fuzzy_matching(raw_text))
        target_number = parse_first_number(str(doc["ground_truth"]))
        score = mean_relative_accuracy(prediction_number, target_number)
        return {
            "prediction_number": prediction_number,
            "target_number": target_number,
            "score": score,
            "metric_name": "MRA:.5:.95:.05",
            "is_correct": bool(prediction_number == target_number) if prediction_number is not None and target_number is not None else False,
        }

    raise ValueError(f"Unsupported question_type: {question_type}")


def format_prediction_for_log(setting_payload: dict) -> str:
    if setting_payload.get("prediction_label") is not None:
        return str(setting_payload["prediction_label"])
    if setting_payload.get("prediction_number") is not None:
        return str(setting_payload["prediction_number"])
    return str(setting_payload.get("raw_text", "")).strip()


def build_vsibench_prompt(doc: dict, args, model, frame_time: str, video_time: float, option_entries: Optional[Sequence[dict]] = None) -> str:
    question = doc["question"]
    if args.add_time_instruction:
        question = (
            f"The video lasts for {video_time:.2f} seconds, and {len(frame_time.split(','))} frames are uniformly "
            f"sampled from it. These frames are located at {frame_time}. Please answer the following question related to this video.\n{question}"
        )

    if doc["question_type"] in MCA_QUESTION_TYPES:
        if not option_entries:
            raise ValueError(f"MCA question type {doc['question_type']} requires non-empty option_entries.")
        qs = "\n".join([args.pre_prompt, question, "Options:", *[x["display_text"] for x in option_entries], args.mca_post_prompt])
    elif doc["question_type"] in NA_QUESTION_TYPES:
        qs = "\n".join([args.pre_prompt, question, args.na_post_prompt])
    else:
        raise ValueError(f"Unsupported question_type: {doc['question_type']}")

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


def parse_first_option_letter(text: str) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)
    text = text[:1].upper()
    return text if text and text.isalpha() else None


def decode_generate_output(tokenizer, output_ids: torch.Tensor, prompt_input_ids: torch.Tensor) -> str:
    if output_ids.ndim != 2:
        raise ValueError(f"Expected output_ids with shape [B, T], got {tuple(output_ids.shape)}.")
    prompt_len = int(prompt_input_ids.shape[1])
    if output_ids.shape[1] > prompt_len:
        candidate = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
        if candidate:
            return candidate
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def generate_baseline(model, tokenizer, prompt_input_ids, attention_mask, video, stop_str: str, args) -> str:
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, prompt_input_ids)
    output_ids = model.generate(
        inputs=prompt_input_ids,
        images=video,
        attention_mask=attention_mask,
        modalities=["video" for _ in video],
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=None,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
    )
    text = decode_generate_output(tokenizer, output_ids, prompt_input_ids)
    if stop_str and stop_str in text:
        text = text.split(stop_str, 1)[0].strip()
    return text


def evaluate_doc(doc: dict, args, tokenizer, model, image_processor):
    video_path = resolve_video_path(doc, Path(args.dataset_cache_dir))
    video, frame_time, video_time = load_video_tensor(video_path, args, image_processor)
    option_entries = parse_option_entries(doc["options"]) if doc.get("options") else None
    prompt_prefix = build_vsibench_prompt(doc, args, model, frame_time, video_time, option_entries)
    prompt_input_ids, attention_mask = build_prompt_input_ids(tokenizer, prompt_prefix)
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    baseline_text = generate_baseline(model, tokenizer, prompt_input_ids, attention_mask, video, stop_str, args)

    branch_args = SimpleNamespace(
        append_newline=args.append_newline,
        stage1_topk=args.stage1_topk,
        stage1_scoring_mode=args.stage1_scoring_mode,
        stage1_fine_scale=args.stage1_fine_scale,
        semantic_neg_ratio=args.semantic_neg_ratio,
        semantic_neg_corrupt_mode=args.semantic_neg_corrupt_mode,
        stage2_topk=args.stage2_topk,
        stage2_scoring_mode=args.stage2_scoring_mode,
        stage2_fine_scale=args.stage2_fine_scale,
        stage2_fusion_2d_weight=args.stage2_fusion_2d_weight,
        stage2_fusion_3d_weight=args.stage2_fusion_3d_weight,
        stage3_topk=args.stage3_topk,
        stage3_fine_scale=args.stage3_fine_scale,
        stage3_semantic_weight=args.stage3_semantic_weight,
        stage3_fusion_weight=args.stage3_fusion_weight,
    )

    pairwise_bundle = build_generation_branch_bundle(
        branch_args,
        model,
        video,
        prompt_input_ids,
        attention_mask,
        branch_mode="pairwise",
    )
    pairwise_output = generate_with_vcd(
        tokenizer=tokenizer,
        model=model,
        prompt_input_ids=prompt_input_ids,
        branch_bundle=pairwise_bundle,
        max_new_tokens=args.max_new_tokens,
        contrast_mode="pairwise",
        contrast_alphas=[args.pairwise_alpha],
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=[stop_str],
    )

    settings = {
        "baseline_generate": {
            "raw_text": baseline_text,
        },
        "pairwise_gen_vcd": {
            "raw_text": pairwise_output["text"],
            "metadata": pairwise_output,
        },
    }
    for payload in settings.values():
        payload.update(score_prediction(doc, payload["raw_text"]))

    return {
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
        "settings": settings,
    }


def build_summary(doc_results: Sequence[dict], setting_names: Sequence[str], question_types: Sequence[str]) -> dict:
    summary = {"overall": {}, "per_question_type": {}, "sampled_question_type_counts": dict(Counter(item["question_type"] for item in doc_results))}
    for setting_name in setting_names:
        scores = [float(item["settings"][setting_name]["score"]) for item in doc_results]
        summary["overall"][setting_name] = {
            "num_docs": len(doc_results),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
        }

    for question_type in question_types:
        subset = [item for item in doc_results if item["question_type"] == question_type]
        if not subset:
            continue
        summary["per_question_type"][question_type] = {}
        for setting_name in setting_names:
            metric_names = sorted({item["settings"][setting_name]["metric_name"] for item in subset})
            scores = [float(item["settings"][setting_name]["score"]) for item in subset]
            summary["per_question_type"][question_type][setting_name] = {
                "num_docs": len(subset),
                "metric_name": metric_names[0] if len(metric_names) == 1 else ",".join(metric_names),
                "mean_score": float(np.mean(scores)) if scores else 0.0,
            }
    return summary


def save_summary_csv(summary: dict, csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "question_type", "setting", "metric_name", "num_docs", "mean_score"])
        for setting, stats in summary.get("overall", {}).items():
            writer.writerow(["overall", "all", setting, "mixed", stats["num_docs"], stats["mean_score"]])
        for question_type, per_setting in summary.get("per_question_type", {}).items():
            for setting, stats in per_setting.items():
                writer.writerow(["per_question_type", question_type, setting, stats["metric_name"], stats["num_docs"], stats["mean_score"]])


def main():
    args = parse_args()
    set_seed(args.seed)
    source_docs = extract_docs_from_log(Path(args.source_log_json)) if args.source_log_json else load_docs_from_hf(args)
    question_types = question_types_from_arg(args.question_types)
    if not question_types:
        question_types = sorted({doc["question_type"] for doc in source_docs if doc.get("question_type")})
    docs = sample_docs_per_type(source_docs, question_types, args.per_type_limit, args.sample_seed)
    if not docs:
        raise RuntimeError("No VSIBench docs were selected.")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_vsibench_gen_vcd_{args.per_type_limit}_per_type"
    output_root = Path(args.output_root) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer, model, image_processor, _ = load_model(args)
    setting_names = ["baseline_generate", "pairwise_gen_vcd"]

    doc_results = []
    for index, doc in enumerate(docs, start=1):
        result = evaluate_doc(doc, args, tokenizer, model, image_processor)
        result["doc_index"] = index
        doc_results.append(result)
        print(
            f"[{index}/{len(docs)}] {doc['question_type']} | doc_id={doc.get('id')} | GT={doc['ground_truth']} | "
            + " | ".join(f"{name}={format_prediction_for_log(result['settings'][name])}" for name in setting_names)
        )

    summary = build_summary(doc_results, setting_names, question_types)
    payload = {
        "config": vars(args),
        "question_types": question_types,
        "selected_doc_count": len(doc_results),
        "setting_display_names": SETTING_DISPLAY_NAMES,
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


if __name__ == "__main__":
    main()
