import copy
import re
import sys
from pathlib import Path
from typing import List, Optional

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.registry import register_model

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

from vcd.vcd_feature_degradation.option_demo_utils import ensure_pad_token, score_candidate_with_video_features
from vcd.vcd_vision_token.methods import (
    build_stage0_coarse_only_branch,
    build_stage0_semantic_negative_coarse_branch,
)

from .vlm_3r import Vlm3r


MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}


def _parse_option_entries(options: List[str]) -> List[dict]:
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


@register_model("vlm_3r_semantic_vcd")
class Vlm3rSemanticVCD(Vlm3r):
    def __init__(
        self,
        stage1_scoring_mode: str = "question_cosine",
        semantic_neg_ratio: float = 0.5,
        semantic_neg_corrupt_mode: str = "zero",
        alpha: float = 1.0,
        candidate_mode: str = "label",
        **kwargs,
    ) -> None:
        self.stage1_scoring_mode = str(stage1_scoring_mode)
        self.semantic_neg_ratio = float(semantic_neg_ratio)
        self.semantic_neg_corrupt_mode = str(semantic_neg_corrupt_mode)
        self.alpha = float(alpha)
        self.candidate_mode = str(candidate_mode)
        self.latest_vcd_metadata: Optional[dict] = None
        super().__init__(**kwargs)
        ensure_pad_token(self.tokenizer)
        eval_logger.info(
            "Enabled semantic VCD MCA scoring: "
            f"stage1_scoring_mode={self.stage1_scoring_mode}, "
            f"semantic_neg_ratio={self.semantic_neg_ratio}, "
            f"semantic_neg_corrupt_mode={self.semantic_neg_corrupt_mode}, "
            f"alpha={self.alpha}, candidate_mode={self.candidate_mode}"
        )

    def _score_mca_with_semantic_vcd(self, prompt: str, doc: dict, videos: List[torch.Tensor]) -> str:
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_token_ids).long().cuda()

        class _Args:
            pass

        branch_args = _Args()
        branch_args.stage1_scoring_mode = self.stage1_scoring_mode
        branch_args.semantic_neg_ratio = self.semantic_neg_ratio
        branch_args.semantic_neg_corrupt_mode = self.semantic_neg_corrupt_mode

        coarse_branch = build_stage0_coarse_only_branch(branch_args, self.model, videos, input_ids, attention_mask)
        negative_branch = build_stage0_semantic_negative_coarse_branch(branch_args, self.model, videos, input_ids, attention_mask)

        option_entries = _parse_option_entries(doc["options"])
        scored_options = []
        for entry in option_entries:
            candidate = entry["label"] if self.candidate_mode == "label" else entry["display_text"]
            orig_metrics = score_candidate_with_video_features(
                tokenizer=self.tokenizer,
                model=self.model,
                prompt_prefix=prompt,
                candidate=candidate,
                video_features=coarse_branch["video_features"],
            )
            neg_metrics = score_candidate_with_video_features(
                tokenizer=self.tokenizer,
                model=self.model,
                prompt_prefix=prompt,
                candidate=candidate,
                video_features=negative_branch["video_features"],
            )
            combined_score = (1.0 + self.alpha) * orig_metrics["sequence_logprob"] - self.alpha * neg_metrics["sequence_logprob"]
            scored_options.append(
                {
                    "label": entry["label"],
                    "display_text": entry["display_text"],
                    "candidate": candidate,
                    "original_score": orig_metrics["sequence_logprob"],
                    "negative_score": neg_metrics["sequence_logprob"],
                    "combined_score": combined_score,
                }
            )

        best_item = max(scored_options, key=lambda x: x["combined_score"])
        self.latest_vcd_metadata = {
            "options": scored_options,
            "coarse_metadata": coarse_branch.get("metadata"),
            "negative_metadata": negative_branch.get("metadata"),
            "alpha": self.alpha,
        }
        return best_item["label"] if self.candidate_mode == "label" else best_item["display_text"]

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc = self.task_dict[task][split][doc_id]
            visuals = [doc_to_visual(doc)]
            if visuals != [None]:
                visuals = self.flatten(visuals)
                videos = []
                try:
                    for visual in visuals:
                        if self.video_decode_backend == "decord":
                            video = self.load_video(visual, self.max_frames_num)
                        else:
                            from lmms_eval.models.model_utils.load_video import read_video_pyav

                            video = read_video_pyav(visual, num_frm=self.max_frames_num)
                        video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                        videos.append(video)
                except Exception as e:
                    eval_logger.info(f"{e}")
                    eval_logger.info(f"Video {visuals} can not load, check the source")
                    video_path = "\n".join(visuals)
                    res.append(f"Video {video_path} can not load, check the source")
                    pbar.update(1)
                    continue

                qs = contexts
                if self.model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs
            else:
                videos = None
                qs = contexts

            conv = copy.deepcopy(conv_templates[self.conv_template]) if "llama_3" in self.conv_template else conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            if doc.get("question_type") in MCA_QUESTION_TYPES and doc.get("options") and videos is not None:
                outputs = self._score_mca_with_semantic_vcd(prompt, doc, videos)
                res.append(outputs)
                pbar.update(1)
                continue

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if "llama_3" in self.conv_template:
                pad_token_ids = 0
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                output_ids = self._generate_with_optional_feature_cd(
                    input_ids=input_ids,
                    videos=videos,
                    attention_masks=attention_masks,
                    stopping_criteria=stopping_criteria,
                    gen_kwargs=gen_kwargs,
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            res.append(outputs)
            pbar.update(1)
        return res
