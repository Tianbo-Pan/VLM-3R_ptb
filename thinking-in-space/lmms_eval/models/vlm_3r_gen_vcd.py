import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

from vcd.vcd_feature_degradation.option_demo_utils import ensure_pad_token
from vcd.vcd_vision_token.generation_vcd import generate_with_vcd
from vcd.vcd_vision_token.methods import build_generation_branch_bundle

from .vlm_3r import Vlm3r


def _parse_float_list(raw_value, expected_len: Optional[int] = None) -> Optional[List[float]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)):
        values = [float(x) for x in raw_value]
    else:
        text = str(raw_value).strip()
        if text.lower() in {"", "none"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)} from {raw_value}.")
    return values


def _parse_bool(raw_value) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).lower() == "true"


@register_model("vlm_3r_gen_vcd")
class Vlm3rGenVCD(Vlm3r):
    def __init__(
        self,
        branch_mode: str = "pairwise",
        contrast_mode: str = "pairwise",
        contrast_alphas: Optional[Sequence[float]] = None,
        beta: float = 0.05,
        append_newline: bool = True,
        stage1_topk: int = 16,
        stage1_scoring_mode: str = "question_cosine",
        stage1_fine_scale: float = 1.0,
        semantic_neg_ratio: float = 0.5,
        semantic_neg_corrupt_mode: str = "zero",
        stage2_topk: int = 16,
        stage2_scoring_mode: str = "fusion_2d3d",
        stage2_fine_scale: float = 1.0,
        stage2_fusion_2d_weight: float = 1.0,
        stage2_fusion_3d_weight: float = 1.0,
        stage3_topk: int = 16,
        stage3_fine_scale: float = 1.0,
        stage3_semantic_weight: float = 1.0,
        stage3_fusion_weight: float = 1.0,
        **kwargs,
    ) -> None:
        self.branch_mode = str(branch_mode).lower()
        self.contrast_mode = str(contrast_mode).lower()
        if self.branch_mode != "pairwise" or self.contrast_mode != "pairwise":
            raise ValueError("Only pairwise generation-time VCD is kept in the current implementation.")
        expected_alpha_len = 1
        self.contrast_alphas = _parse_float_list(contrast_alphas, expected_len=expected_alpha_len)
        self.beta = float(beta)
        self.append_newline = _parse_bool(append_newline)
        self.stage1_topk = int(stage1_topk)
        self.stage1_scoring_mode = str(stage1_scoring_mode)
        self.stage1_fine_scale = float(stage1_fine_scale)
        self.semantic_neg_ratio = float(semantic_neg_ratio)
        self.semantic_neg_corrupt_mode = str(semantic_neg_corrupt_mode)
        self.stage2_topk = int(stage2_topk)
        self.stage2_scoring_mode = str(stage2_scoring_mode)
        self.stage2_fine_scale = float(stage2_fine_scale)
        self.stage2_fusion_2d_weight = float(stage2_fusion_2d_weight)
        self.stage2_fusion_3d_weight = float(stage2_fusion_3d_weight)
        self.stage3_topk = int(stage3_topk)
        self.stage3_fine_scale = float(stage3_fine_scale)
        self.stage3_semantic_weight = float(stage3_semantic_weight)
        self.stage3_fusion_weight = float(stage3_fusion_weight)
        self.latest_vcd_metadata: Optional[dict] = None

        super().__init__(**kwargs)
        ensure_pad_token(self.tokenizer)
        eval_logger.info(
            "Enabled general generation-time VCD: "
            f"branch_mode={self.branch_mode}, "
            f"contrast_mode={self.contrast_mode}, "
            f"contrast_alphas={self.contrast_alphas}, "
            f"beta={self.beta}"
        )

    def _build_branch_args(self):
        return SimpleNamespace(
            append_newline=self.append_newline,
            stage1_topk=self.stage1_topk,
            stage1_scoring_mode=self.stage1_scoring_mode,
            stage1_fine_scale=self.stage1_fine_scale,
            semantic_neg_ratio=self.semantic_neg_ratio,
            semantic_neg_corrupt_mode=self.semantic_neg_corrupt_mode,
            stage2_topk=self.stage2_topk,
            stage2_scoring_mode=self.stage2_scoring_mode,
            stage2_fine_scale=self.stage2_fine_scale,
            stage2_fusion_2d_weight=self.stage2_fusion_2d_weight,
            stage2_fusion_3d_weight=self.stage2_fusion_3d_weight,
            stage3_topk=self.stage3_topk,
            stage3_fine_scale=self.stage3_fine_scale,
            stage3_semantic_weight=self.stage3_semantic_weight,
            stage3_fusion_weight=self.stage3_fusion_weight,
        )

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

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            prompt_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, prompt_input_ids)

            if videos is None:
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                attention_masks = prompt_input_ids.ne(pad_token_ids).long().cuda()
                with torch.inference_mode():
                    output_ids = self._generate_with_optional_feature_cd(
                        input_ids=prompt_input_ids,
                        videos=videos,
                        attention_masks=attention_masks,
                        stopping_criteria=stopping_criteria,
                        gen_kwargs=gen_kwargs,
                    )
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                res.append(outputs)
                pbar.update(1)
                continue

            if gen_kwargs.get("num_beams", 1) != 1:
                eval_logger.warning("generation-time VCD currently ignores num_beams and uses greedy/sample decoding only.")

            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            attention_mask = prompt_input_ids.ne(pad_token_ids).long().cuda()

            branch_args = self._build_branch_args()
            branch_bundle = build_generation_branch_bundle(
                branch_args,
                self.model,
                videos,
                prompt_input_ids,
                attention_mask,
                branch_mode=self.branch_mode,
            )

            with torch.inference_mode():
                generation_output = generate_with_vcd(
                    tokenizer=self.tokenizer,
                    model=self.model,
                    prompt_input_ids=prompt_input_ids,
                    branch_bundle=branch_bundle,
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    contrast_mode=self.contrast_mode,
                    contrast_alphas=self.contrast_alphas,
                    beta=self.beta,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_strings=gen_kwargs.get("until"),
                )

            self.latest_vcd_metadata = {
                "doc_id": doc_id,
                "question_type": doc.get("question_type"),
                "branch_mode": branch_bundle.get("branch_mode"),
                "branch_names": generation_output.get("branch_names"),
                "steps": generation_output.get("steps"),
                "branch_metadata": {
                    branch["name"]: branch.get("metadata")
                    for branch in branch_bundle.get("branches", [])
                },
            }
            res.append(generation_output["text"])
            pbar.update(1)
        return res
