import argparse
import json
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoConfig

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.feature_cd.common import build_inputs_embeds_from_video_features


DEFAULT_PHASE1_DEMO_ARGV = [
    "--model-path",
    "Journey9ni/vlm-3r-llava-qwen2-lora",
    "--model-base",
    "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2",
    "--video_path",
    "playground/demo/47334096.mp4",
    "--output_dir",
    "./work_dirs/phase1_option_demo",
    "--output_name",
    "pred",
    "--mm_spatial_pool_stride",
    "2",
    "--for_get_frames_num",
    "32",
    "--conv-mode",
    "qwen_1_5",
    "--mm_spatial_pool_mode",
    "average",
    "--mm_newline_position",
    "grid",
    "--prompt",
    "If I am standing by the stool and facing the stove, is the sofa to my left, right, or back?",
    "--choices",
    "left|||right|||back",
]


def inject_default_demo_argv(extra_argv: Optional[List[str]] = None):
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_PHASE1_DEMO_ARGV)
        if extra_argv:
            sys.argv.extend(extra_argv)
        print("No CLI arguments detected. Using built-in Phase-1 option demo arguments.")


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--video_path", help="Path to the video file.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--choices",
        type=str,
        required=True,
        help="Candidate answers. Use JSON list or separate items with '|||', e.g. left|||right|||back",
    )
    parser.add_argument(
        "--append_choices_to_prompt",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Whether to append the option list to the user prompt.",
    )
    parser.add_argument(
        "--answer_instruction",
        type=str,
        default="Answer with exactly one option from the list above.",
        help="Extra instruction appended after the choice list.",
    )
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument(
        "--image_grid_pinpoints",
        type=str,
        default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]",
    )
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--load_8bit", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--mm_newline_position", type=str, default="grid")
    parser.add_argument("--force_sample", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--add_time_instruction", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0, help="Contrastive weight for the degraded branch.")


def parse_choices(raw_choices: str) -> List[str]:
    raw_choices = raw_choices.strip()
    if raw_choices.startswith("["):
        choices = json.loads(raw_choices)
        if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
            raise ValueError("--choices JSON must be a list of strings.")
        return [item.strip() for item in choices if item.strip()]
    return [item.strip() for item in raw_choices.split("|||") if item.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_video(video_path, args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "0.00s", 0.0

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def load_model_and_video(args):
    model_name = get_model_name_from_path(args.model_path)
    if not args.overwrite:
        tokenizer, model, image_processor, _ = load_pretrained_model(
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
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            load_8bit=args.load_8bit,
            overwrite_config=overwrite_config,
        )

    model.to("cuda")

    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction

    video_np, frame_time, video_time = load_video(args.video_path, args)
    video_tensor = image_processor.preprocess(video_np, return_tensors="pt")["pixel_values"].half().cuda()
    return tokenizer, model, [video_tensor], frame_time, video_time


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 151643


def build_prompt(question: str, choices: List[str], args, model, frame_time: str, video_time: float) -> str:
    qs = question
    if args.add_time_instruction:
        qs = (
            f"The video lasts for {video_time:.2f} seconds, and {len(frame_time.split(','))} frames are uniformly "
            f"sampled from it. These frames are located at {frame_time}. Please answer the following question "
            f"related to this video.\n{qs}"
        )
    if args.append_choices_to_prompt:
        qs = f"{qs}\nOptions: {', '.join(choices)}\n{args.answer_instruction}"

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.no_grad()
def extract_spatial_features(model, video):
    spatial_tower = model.get_model().get_spatial_tower()
    if spatial_tower is None:
        return None
    camera_tokens, patch_tokens = spatial_tower(video[0])
    return [{"camera_tokens": camera_tokens, "patch_tokens": patch_tokens}]


def clone_spatial_features(spatial_features):
    if spatial_features is None:
        return None
    cloned = []
    for item in spatial_features:
        cloned.append(
            {
                "camera_tokens": None if item["camera_tokens"] is None else item["camera_tokens"].clone(),
                "patch_tokens": None if item["patch_tokens"] is None else item["patch_tokens"].clone(),
            }
        )
    return cloned


def apply_gaussian_noise_to_video(video, noise_std: float):
    noisy = video[0].clone()
    noisy = noisy + torch.randn_like(noisy) * noise_std
    noisy = noisy.clamp(min=video[0].min().item(), max=video[0].max().item())
    return [noisy]


def apply_patch_token_noise(spatial_features, drop_rate: float, noise_std: float):
    if spatial_features is None:
        raise ValueError("This model does not expose spatial features, so spatial-token-noise VCD cannot run.")
    degraded = clone_spatial_features(spatial_features)
    patch_tokens = degraded[0]["patch_tokens"]
    if patch_tokens is None:
        raise ValueError("Patch tokens are missing; cannot build spatial-token-noise VCD.")
    if drop_rate > 0:
        keep_mask = (torch.rand(patch_tokens.shape[:-1], device=patch_tokens.device) > drop_rate).to(patch_tokens.dtype)
        patch_tokens = patch_tokens * keep_mask.unsqueeze(-1)
    if noise_std > 0:
        patch_tokens = patch_tokens + torch.randn_like(patch_tokens) * noise_std
    degraded[0]["patch_tokens"] = patch_tokens
    return degraded


def apply_view_shuffle(spatial_features):
    if spatial_features is None:
        raise ValueError("This model does not expose spatial features, so camera-order-shuffle VCD cannot run.")
    degraded = clone_spatial_features(spatial_features)
    camera_tokens = degraded[0]["camera_tokens"]
    patch_tokens = degraded[0]["patch_tokens"]
    if camera_tokens is None:
        raise ValueError("Camera tokens are missing; cannot build camera-order-shuffle VCD.")
    perm = torch.randperm(camera_tokens.shape[0], device=camera_tokens.device)
    degraded[0]["camera_tokens"] = camera_tokens[perm]
    if patch_tokens is not None:
        degraded[0]["patch_tokens"] = patch_tokens[perm]
    return degraded, perm.detach().cpu().tolist()


def build_candidate_inputs(
    tokenizer,
    prompt_prefix: str,
    candidate: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    prefix_ids = tokenizer_image_token(prompt_prefix, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    full_ids = tokenizer_image_token(prompt_prefix + candidate, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    labels = full_ids.clone()
    prefix_len = prefix_ids.shape[1]
    labels[:, :prefix_len] = IGNORE_INDEX
    attention_mask = full_ids.ne(tokenizer.pad_token_id).long().cuda()
    return prefix_ids, full_ids, labels, prefix_len, attention_mask


def build_labels_from_video_features(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: torch.Tensor,
    video_features: torch.Tensor,
) -> Optional[torch.Tensor]:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)

    original_attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()

    filtered_input_ids = [cur_ids[cur_mask] for cur_ids, cur_mask in zip(input_ids, attention_mask)]
    filtered_labels = [cur_labels[cur_mask] for cur_labels, cur_mask in zip(labels, attention_mask)]
    new_labels = []

    for cur_input_ids, cur_labels in zip(filtered_input_ids, filtered_labels):
        image_token_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        if not image_token_positions:
            new_labels.append(cur_labels)
            continue

        image_token_indices = [-1] + image_token_positions + [cur_input_ids.shape[0]]
        cur_new_labels = []
        for idx in range(len(image_token_indices) - 1):
            cur_new_labels.append(cur_labels[image_token_indices[idx] + 1 : image_token_indices[idx + 1]])
            if idx < len(image_token_indices) - 2:
                cur_new_labels.append(
                    torch.full(
                        (video_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )
        new_labels.append(torch.cat(cur_new_labels))

    tokenizer_model_max_length = getattr(model.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length is not None:
        new_labels = [cur_labels[:tokenizer_model_max_length] for cur_labels in new_labels]

    max_len = max(cur_labels.shape[0] for cur_labels in new_labels)
    padded_labels = torch.full((len(new_labels), max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    padding_side = getattr(model.config, "tokenizer_padding_side", "right")
    for idx, cur_labels in enumerate(new_labels):
        cur_len = cur_labels.shape[0]
        if padding_side == "left":
            padded_labels[idx, -cur_len:] = cur_labels
        else:
            padded_labels[idx, :cur_len] = cur_labels

    if original_attention_mask is None:
        return None
    return padded_labels


@torch.no_grad()
def score_candidate_with_video_features(
    tokenizer,
    model,
    prompt_prefix: str,
    candidate: str,
    video_features: torch.Tensor,
):
    _, full_ids, labels, prefix_len, attention_mask = build_candidate_inputs(tokenizer, prompt_prefix, candidate)
    position_ids, final_attention_mask, inputs_embeds = build_inputs_embeds_from_video_features(
        model,
        full_ids,
        attention_mask,
        video_features,
    )
    expanded_labels = build_labels_from_video_features(
        model,
        full_ids,
        attention_mask,
        labels,
        video_features,
    )

    outputs = model(
        position_ids=position_ids,
        attention_mask=final_attention_mask,
        inputs_embeds=inputs_embeds,
        labels=expanded_labels,
        return_dict=True,
    )

    valid_token_count = int((labels != IGNORE_INDEX).sum().item())
    if valid_token_count <= 0:
        raise ValueError(f"Candidate `{candidate}` produced no scoreable tokens. Consider a non-empty candidate string.")

    candidate_logits = outputs.logits[:, prefix_len - 1 : full_ids.shape[1] - 1, :]
    candidate_token_ids = full_ids[:, prefix_len:]
    selected_logits = candidate_logits.gather(dim=-1, index=candidate_token_ids.unsqueeze(-1)).squeeze(-1)
    log_probs = candidate_logits.log_softmax(dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=candidate_token_ids.unsqueeze(-1)).squeeze(-1)

    sequence_logprob = float(selected_log_probs.sum().item())
    avg_logprob = sequence_logprob / valid_token_count
    token_id_values = candidate_token_ids[0].detach().cpu().tolist()
    token_text_values = tokenizer.convert_ids_to_tokens(token_id_values)
    token_logit_values = selected_logits[0].detach().float().cpu().tolist()
    token_logprob_values = selected_log_probs[0].detach().float().cpu().tolist()

    return {
        "sequence_logprob": sequence_logprob,
        "valid_token_count": valid_token_count,
        "avg_logprob": avg_logprob,
        "token_ids": token_id_values,
        "token_texts": token_text_values,
        "token_logits": token_logit_values,
        "token_logprobs": token_logprob_values,
        "first_token_logit": float(token_logit_values[0]),
        "first_token_logprob": float(token_logprob_values[0]),
    }


@torch.no_grad()
def score_candidate_with_details(
    tokenizer,
    model,
    prompt_prefix: str,
    candidate: str,
    images,
    spatial_features=None,
):
    _, full_ids, labels, prefix_len, attention_mask = build_candidate_inputs(tokenizer, prompt_prefix, candidate)

    outputs = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        labels=labels,
        images=images,
        spatial_features=spatial_features,
        modalities=["video"],
        return_dict=True,
    )

    valid_token_count = int((labels != IGNORE_INDEX).sum().item())
    if valid_token_count <= 0:
        raise ValueError(f"Candidate `{candidate}` produced no scoreable tokens. Consider a non-empty candidate string.")

    candidate_logits = outputs.logits[:, prefix_len - 1 : full_ids.shape[1] - 1, :]
    candidate_token_ids = full_ids[:, prefix_len:]
    selected_logits = candidate_logits.gather(dim=-1, index=candidate_token_ids.unsqueeze(-1)).squeeze(-1)
    log_probs = candidate_logits.log_softmax(dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=candidate_token_ids.unsqueeze(-1)).squeeze(-1)

    sequence_logprob = float(selected_log_probs.sum().item())
    avg_logprob = sequence_logprob / valid_token_count
    token_id_values = candidate_token_ids[0].detach().cpu().tolist()
    token_text_values = tokenizer.convert_ids_to_tokens(token_id_values)
    token_logit_values = selected_logits[0].detach().float().cpu().tolist()
    token_logprob_values = selected_log_probs[0].detach().float().cpu().tolist()

    return {
        "sequence_logprob": sequence_logprob,
        "valid_token_count": valid_token_count,
        "avg_logprob": avg_logprob,
        "token_ids": token_id_values,
        "token_texts": token_text_values,
        "token_logits": token_logit_values,
        "token_logprobs": token_logprob_values,
        "first_token_logit": float(token_logit_values[0]),
        "first_token_logprob": float(token_logprob_values[0]),
    }


@torch.no_grad()
def score_candidate(
    tokenizer,
    model,
    prompt_prefix: str,
    candidate: str,
    images,
    spatial_features=None,
):
    return score_candidate_with_details(
        tokenizer=tokenizer,
        model=model,
        prompt_prefix=prompt_prefix,
        candidate=candidate,
        images=images,
        spatial_features=spatial_features,
    )


def _softmax(values: List[float]) -> List[float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    probs = torch.softmax(tensor, dim=0)
    return [float(x) for x in probs.tolist()]


def run_option_demo(
    args,
    setting_name: str,
    degraded_branch_builder: Optional[Callable] = None,
):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model, video, frame_time, video_time = load_model_and_video(args)
    ensure_pad_token(tokenizer)
    choices = parse_choices(args.choices)
    prompt_prefix = build_prompt(args.prompt, choices, args, model, frame_time, video_time)

    branch_bundle = None
    if degraded_branch_builder is not None:
        branch_bundle = degraded_branch_builder(args, model, video)

    results = []
    for candidate in choices:
        orig_images = video if branch_bundle is None or branch_bundle.get("orig_images") is None else branch_bundle["orig_images"]
        orig_spatial = None if branch_bundle is None else branch_bundle.get("orig_spatial_features")
        original_metrics = score_candidate_with_details(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            candidate=candidate,
            images=orig_images,
            spatial_features=orig_spatial,
        )

        item = {
            "candidate": candidate,
            "original_sequence_logprob": original_metrics["sequence_logprob"],
            "original_avg_logprob": original_metrics["avg_logprob"],
            "original_first_token_logit": original_metrics["first_token_logit"],
            "answer_token_count": original_metrics["valid_token_count"],
            "token_ids": original_metrics["token_ids"],
            "token_texts": original_metrics["token_texts"],
            "original_token_logits": original_metrics["token_logits"],
            "original_token_logprobs": original_metrics["token_logprobs"],
        }

        if branch_bundle is None:
            item["combined_score"] = original_metrics["sequence_logprob"]
            item["combined_first_token_logit"] = original_metrics["first_token_logit"]
            item["combined_token_logits"] = original_metrics["token_logits"]
            item["combined_token_logprobs"] = original_metrics["token_logprobs"]
        else:
            degraded_metrics = score_candidate_with_details(
                tokenizer=tokenizer,
                model=model,
                prompt_prefix=prompt_prefix,
                candidate=candidate,
                images=branch_bundle["degraded_images"],
                spatial_features=branch_bundle.get("degraded_spatial_features"),
            )
            combined_score = (1.0 + args.alpha) * original_metrics["sequence_logprob"] - args.alpha * degraded_metrics["sequence_logprob"]
            combined_first_token_logit = (1.0 + args.alpha) * original_metrics["first_token_logit"] - args.alpha * degraded_metrics["first_token_logit"]
            item["degraded_sequence_logprob"] = degraded_metrics["sequence_logprob"]
            item["degraded_avg_logprob"] = degraded_metrics["avg_logprob"]
            item["degraded_first_token_logit"] = degraded_metrics["first_token_logit"]
            item["degraded_token_logits"] = degraded_metrics["token_logits"]
            item["degraded_token_logprobs"] = degraded_metrics["token_logprobs"]
            item["combined_score"] = combined_score
            item["combined_first_token_logit"] = combined_first_token_logit
            item["combined_token_logits"] = [
                (1.0 + args.alpha) * orig - args.alpha * deg
                for orig, deg in zip(original_metrics["token_logits"], degraded_metrics["token_logits"])
            ]
            item["combined_token_logprobs"] = [
                (1.0 + args.alpha) * orig - args.alpha * deg
                for orig, deg in zip(original_metrics["token_logprobs"], degraded_metrics["token_logprobs"])
            ]

        results.append(item)

    combined_scores = [item["combined_score"] for item in results]
    posterior = _softmax(combined_scores)
    for item, prob in zip(results, posterior):
        item["posterior_over_choices"] = prob

    best_idx = int(np.argmax(combined_scores))
    best_item = results[best_idx]

    output_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    payload = {
        "setting": setting_name,
        "video_path": args.video_path,
        "prompt": args.prompt,
        "scoring_prompt": prompt_prefix,
        "choices": choices,
        "best_choice": best_item["candidate"],
        "alpha": args.alpha,
        "frame_time": frame_time,
        "results": results,
    }
    if branch_bundle is not None and branch_bundle.get("metadata") is not None:
        payload["degraded_branch_metadata"] = branch_bundle["metadata"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"[{setting_name}] Best choice: {best_item['candidate']}")
    for item in results:
        print(
            f"  - {item['candidate']}: combined={item['combined_score']:.4f}, "
            f"orig={item['original_sequence_logprob']:.4f}, "
            f"posterior={item['posterior_over_choices']:.4f}"
            + (
                ""
                if "degraded_sequence_logprob" not in item
                else f", degraded={item['degraded_sequence_logprob']:.4f}, combined_logit={item['combined_first_token_logit']:.4f}"
            )
        )
    print(f"Saved detailed result to: {output_path}")
