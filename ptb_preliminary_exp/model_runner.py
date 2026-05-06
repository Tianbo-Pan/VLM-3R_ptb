from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

import ptb_test.experiment0_attention_rollout as exp0
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model.feature_cd.common import build_inputs_embeds_from_video_features
from vcd.option_demo_utils import score_candidate_with_video_features

from .perturbation_utils import build_encoded_video_and_patch_scores, pack_encoded_video_features


def load_model_bundle(args):
    tokenizer, model, image_processor, _ = exp0.load_model_and_processor(args)
    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 151643
    return tokenizer, model, image_processor


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
    prompt_prefix = exp0.build_video_prompt(model, prompt, conv_mode)
    input_ids = tokenizer_image_token(prompt_prefix, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)
    return video, prompt_prefix, input_ids, attention_mask


@torch.no_grad()
def build_original_video_features(
    tokenizer,
    model,
    video,
    input_ids,
    attention_mask,
    scoring_mode: str,
    fusion_2d_weight: float,
    fusion_3d_weight: float,
):
    encoded_video_features, patch_scores, metadata = build_encoded_video_and_patch_scores(
        model,
        tokenizer,
        video,
        input_ids,
        attention_mask,
        scoring_mode=scoring_mode,
        fusion_2d_weight=fusion_2d_weight,
        fusion_3d_weight=fusion_3d_weight,
    )
    branch_features = metadata.pop("branch_features", {})
    video_features = pack_encoded_video_features(model, encoded_video_features)
    metadata = copy.deepcopy(metadata)
    metadata.update(
        {
            "encoded_video_shape": list(encoded_video_features.shape),
            "patch_scores_shape": list(patch_scores.shape),
        }
    )
    aux = {
        "branch_features": branch_features,
    }
    return encoded_video_features, patch_scores, video_features, metadata, aux


@torch.no_grad()
def score_options_with_video_features(
    tokenizer,
    model,
    prompt_prefix: str,
    option_labels: Sequence[str],
    video_features: torch.Tensor,
) -> Dict[str, Dict]:
    option_metrics: Dict[str, Dict] = {}
    for label in option_labels:
        option_metrics[label] = score_candidate_with_video_features(
            tokenizer=tokenizer,
            model=model,
            prompt_prefix=prompt_prefix,
            candidate=label,
            video_features=video_features,
        )
    return option_metrics


def option_metrics_to_scores(option_metrics: Dict[str, Dict], score_key: str = "avg_logprob") -> Dict[str, float]:
    return {label: float(metrics[score_key]) for label, metrics in option_metrics.items()}


@torch.no_grad()
def generate_answer_with_video_features(
    tokenizer,
    model,
    prompt_prefix: str,
    video_features: torch.Tensor,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float = 0.1,
) -> str:
    input_ids = tokenizer_image_token(prompt_prefix, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(video_features.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(video_features.device)
    position_ids, final_attention_mask, inputs_embeds = build_inputs_embeds_from_video_features(
        model,
        input_ids,
        attention_mask,
        video_features,
    )
    conv = conv_templates["qwen_1_5"].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    output_ids = model.generate(
        position_ids=position_ids,
        attention_mask=final_attention_mask,
        inputs_embeds=inputs_embeds,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if stop_str and text.endswith(stop_str):
        text = text[: -len(stop_str)]
    return text.strip()
