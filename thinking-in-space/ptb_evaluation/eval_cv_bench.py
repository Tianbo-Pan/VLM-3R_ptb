#!/usr/bin/env python3
"""Standalone CV-Bench evaluation for VLM-3R.

This script evaluates a VLM-3R checkpoint on CV-Bench without modifying the
existing lmms-eval task registry. It reuses the native VLM-3R / LLaVA image
inference path, so it is a good first step before integrating CV-Bench into
`thinking-in-space/lmms_eval`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from loguru import logger
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


ANSWER_PREFIXES = [
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLM-3R on CV-Bench.")
    parser.add_argument("--model-path", required=True, help="Checkpoint path or HF repo for the LoRA / merged model.")
    parser.add_argument("--model-base", default=None, help="Base model path when loading a LoRA checkpoint.")
    parser.add_argument("--conv-mode", default="qwen_1_5", help="Conversation template.")
    parser.add_argument("--dataset-path", default="nyu-visionx/CV-Bench", help="HF dataset path or local dataset script/path.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache dir.")
    parser.add_argument("--output-dir", required=True, help="Directory to save predictions and metrics.")
    parser.add_argument("--output-name", default="vlm_3r_cv_bench", help="Prediction/metric file stem.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda / cuda:0 / cpu.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model in 8-bit mode.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite selected config values while loading the model.")
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=4)
    parser.add_argument("--mm-spatial-pool-mode", default="average")
    parser.add_argument("--mm-newline-position", default="no_token")
    parser.add_argument("--pre-prompt", default="", help="Optional prefix before the CV-Bench question.")
    parser.add_argument(
        "--mca-post-prompt",
        default="Answer with the option's letter from the given choices directly.",
        help="Suffix appended after the question/options.",
    )
    return parser.parse_args()


def build_overwrite_config(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.overwrite:
        return {}
    return {
        "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
        "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
        "mm_newline_position": args.mm_newline_position,
    }


def build_question(sample: Dict[str, Any], pre_prompt: str, post_prompt: str) -> str:
    chars = string.ascii_uppercase
    options = "Options:\n" + "\n".join(f"{chars[i]}. {choice}" for i, choice in enumerate(sample["choices"]))
    parts = []
    if pre_prompt:
        parts.append(pre_prompt)
    parts.extend([sample["question"], options, post_prompt])
    return "\n".join(parts)


def extract_choice_letter(text: str) -> str:
    text = text.strip()
    for prefix in ANSWER_PREFIXES:
        text = text.replace(prefix, "")
    if len(text.split()) > 10 and not re.search(r"[ABCDEF]", text):
        return ""
    match = re.search(r"[ABCDEF]", text)
    return match.group(0) if match else ""


def normalize_gold_letter(answer: Any) -> str:
    if answer is None:
        return ""
    match = re.search(r"[ABCDEF]", str(answer))
    return match.group(0) if match else ""


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    def mean(items: List[float]) -> float:
        return sum(items) / len(items) if items else 0.0

    metrics: Dict[str, float] = {}

    ade_results = [r["correct"] for r in results if r["source"] == "ADE20K"]
    coco_results = [r["correct"] for r in results if r["source"] == "COCO"]
    omni_results = [r["correct"] for r in results if r["source"] == "Omni3D"]

    accuracy_2d_ade = mean(ade_results)
    accuracy_2d_coco = mean(coco_results)
    accuracy_2d = mean([x for x in [accuracy_2d_ade, accuracy_2d_coco] if x != 0.0 or ade_results or coco_results])
    accuracy_3d = mean(omni_results)
    combined_accuracy = mean([accuracy_2d, accuracy_3d])

    metrics["accuracy_2d_ade20k"] = accuracy_2d_ade
    metrics["accuracy_2d_coco"] = accuracy_2d_coco
    metrics["accuracy_2d"] = accuracy_2d
    metrics["accuracy_3d"] = accuracy_3d
    metrics["combined_accuracy"] = combined_accuracy

    for task_name in ["Count", "Relation", "Distance", "Depth"]:
        task_values = [r["correct"] for r in results if r["task"] == task_name]
        if task_values:
            metrics[task_name] = mean(task_values)

    return metrics


def load_model(args: argparse.Namespace):
    disable_torch_init()
    overwrite_config = build_overwrite_config(args)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        overwrite_config=overwrite_config or None,
    )

    if not args.load_8bit:
        model.to(args.device)

    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        logger.info("Tokenizer pad_token_id is None; setting it to Qwen BOS token id 151643.")
        tokenizer.pad_token_id = 151643

    return tokenizer, model, image_processor, model_name


def prepare_image_tensors(image_processor, images: List[Image.Image], device: str) -> List[torch.Tensor]:
    pixel_values = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return [image.half().to(device) for image in pixel_values]


def generate_answer(
    *,
    tokenizer,
    model,
    image_processor,
    sample: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    image = sample["image"].convert("RGB")
    images = [image]
    image_tensors = prepare_image_tensors(image_processor, images, args.device)

    question = build_question(sample, args.pre_prompt, args.mca_post_prompt)
    if model.config.mm_use_im_start_end:
        image_placeholders = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n") * len(images)
    else:
        image_placeholders = (DEFAULT_IMAGE_TOKEN + "\n") * len(images)
    qs = image_placeholders + question

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensors,
            attention_mask=attention_masks,
            modalities=["image"] * len(images),
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    return outputs.strip()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / f"{args.output_name}.jsonl"
    metric_path = output_dir / f"{args.output_name}_metrics.json"
    config_path = output_dir / f"{args.output_name}_config.json"

    logger.info("Loading model...")
    tokenizer, model, image_processor, model_name = load_model(args)

    logger.info(f"Loading dataset: {args.dataset_path} [{args.split}]")
    dataset = load_dataset(args.dataset_path, split=args.split, cache_dir=args.cache_dir)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        logger.info(f"Evaluation limited to {len(dataset)} samples.")

    results: List[Dict[str, Any]] = []

    with pred_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating CV-Bench")):
            try:
                raw_output = generate_answer(
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    sample=sample,
                    args=args,
                )
                pred_answer = extract_choice_letter(raw_output)
                gold_answer = normalize_gold_letter(sample.get("answer"))
                correct = int(pred_answer == gold_answer)
                record = {
                    "index": idx,
                    "question": sample["question"],
                    "choices": sample["choices"],
                    "raw_output": raw_output,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "correct": correct,
                    "source": sample.get("source", ""),
                    "task": sample.get("task", ""),
                    "model_id": model_name,
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Failed on sample {idx}: {exc}")
                record = {
                    "index": idx,
                    "question": sample.get("question", ""),
                    "choices": sample.get("choices", []),
                    "raw_output": "",
                    "pred_answer": "",
                    "gold_answer": normalize_gold_letter(sample.get("answer")),
                    "correct": 0,
                    "source": sample.get("source", ""),
                    "task": sample.get("task", ""),
                    "model_id": model_name,
                    "error": str(exc),
                }

            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    metrics = aggregate_metrics(results)
    logger.info(f"CV-Bench metrics: {metrics}")

    metric_payload = {
        "model_id": model_name,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_samples": len(results),
        "metrics": metrics,
    }

    with metric_path.open("w", encoding="utf-8") as f:
        json.dump(metric_payload, f, ensure_ascii=False, indent=2)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved predictions to {pred_path}")
    logger.info(f"Saved metrics to {metric_path}")


if __name__ == "__main__":
    main()
