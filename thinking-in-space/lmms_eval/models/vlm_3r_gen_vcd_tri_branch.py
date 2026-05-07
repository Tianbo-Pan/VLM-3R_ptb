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
from vcd.vcd_new_generator_patch_warp.methods import build_generation_patch_warp_branch_bundle
from vcd.vcd_vision_token.generation_vcd import generate_with_vcd

from .vlm_3r import Vlm3r


def _parse_float_list(raw_value) -> Optional[List[float]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)):
        return [float(x) for x in raw_value]
    text = str(raw_value).strip()
    if text.lower() in {"", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if "," in text:
        parts = text.split(",")
    elif ":" in text:
        parts = text.split(":")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = [text]
    return [float(part.strip()) for part in parts if part.strip()]


def _parse_bool(raw_value) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).lower() == "true"


@register_model("vlm_3r_gen_vcd_tri_branch")
class Vlm3rGenVCDTriBranch(Vlm3r):
    def __init__(
        self,
        branch_mode: str = "tri",
        contrast_mode: str = "tri_rectified",
        contrast_alphas: Optional[Sequence[float]] = None,
        beta: float = 0.05,
        reference_mode: str = "max_original_augmented",
        append_newline: bool = True,
        patch_warp_ratio: float = 0.2,
        patch_warp_selection_mode: str = "question_cosine",
        patch_warp_selection_scope: str = "per_frame",
        patch_warp_shift_size: int = 1,
        patch_warp_mix_ratio: float = 0.65,
        patch_warp_fusion_2d_weight: float = 1.0,
        patch_warp_fusion_3d_weight: float = 1.0,
        aug_patch_topk: int = 16,
        aug_patch_ratio: Optional[float] = 0.01,
        aug_selection_scope: str = "global",
        aug_scoring_mode: str = "question_cosine",
        aug_injection_mode: str = "inplace_boost_coarse",
        aug_boost_factor: float = 1.5,
        aug_background_decay: float = 0.98,
        aug_fine_scale: float = 1.0,
        aug_include_coarse: bool = True,
        aug_append_newline: bool = True,
        aug_fusion_2d_weight: float = 1.0,
        aug_fusion_3d_weight: float = 1.0,
        **kwargs,
    ) -> None:
        self.branch_mode = str(branch_mode).lower()
        self.contrast_mode = str(contrast_mode).lower()
        if self.branch_mode not in {"tri", "pairwise"}:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}. Expected tri or pairwise.")
        if self.branch_mode == "tri" and self.contrast_mode not in {"tri_simple", "tri_rectified"}:
            raise ValueError(
                f"tri branch_mode expects contrast_mode in {{tri_simple, tri_rectified}}, got {contrast_mode}."
            )
        if self.branch_mode == "pairwise" and self.contrast_mode != "pairwise":
            raise ValueError("pairwise branch_mode expects contrast_mode=pairwise.")

        self.contrast_alphas = _parse_float_list(contrast_alphas)
        if self.branch_mode == "tri" and self.contrast_alphas is not None and len(self.contrast_alphas) not in {1, 2}:
            raise ValueError("tri-branch contrast_alphas must contain 1 or 2 floats.")
        if self.branch_mode == "pairwise" and self.contrast_alphas is not None and len(self.contrast_alphas) != 1:
            raise ValueError("pairwise contrast_alphas must contain exactly 1 float.")

        self.beta = float(beta)
        self.reference_mode = str(reference_mode).lower()
        self.append_newline = _parse_bool(append_newline)
        self.patch_warp_ratio = float(patch_warp_ratio)
        self.patch_warp_selection_mode = str(patch_warp_selection_mode)
        self.patch_warp_selection_scope = str(patch_warp_selection_scope)
        self.patch_warp_shift_size = int(patch_warp_shift_size)
        self.patch_warp_mix_ratio = float(patch_warp_mix_ratio)
        self.patch_warp_fusion_2d_weight = float(patch_warp_fusion_2d_weight)
        self.patch_warp_fusion_3d_weight = float(patch_warp_fusion_3d_weight)
        self.aug_patch_topk = int(aug_patch_topk)
        self.aug_patch_ratio = None if aug_patch_ratio in [None, "", "none"] else float(aug_patch_ratio)
        self.aug_selection_scope = str(aug_selection_scope)
        self.aug_scoring_mode = str(aug_scoring_mode)
        self.aug_injection_mode = str(aug_injection_mode)
        self.aug_boost_factor = float(aug_boost_factor)
        self.aug_background_decay = float(aug_background_decay)
        self.aug_fine_scale = float(aug_fine_scale)
        self.aug_include_coarse = _parse_bool(aug_include_coarse)
        self.aug_append_newline = _parse_bool(aug_append_newline)
        self.aug_fusion_2d_weight = float(aug_fusion_2d_weight)
        self.aug_fusion_3d_weight = float(aug_fusion_3d_weight)
        self.latest_vcd_metadata: Optional[dict] = None

        super().__init__(**kwargs)
        ensure_pad_token(self.tokenizer)
        eval_logger.info(
            "Enabled tri-branch generation-time VCD: "
            f"branch_mode={self.branch_mode}, "
            f"contrast_mode={self.contrast_mode}, "
            f"contrast_alphas={self.contrast_alphas}, "
            f"reference_mode={self.reference_mode}, "
            f"beta={self.beta}, "
            f"patch_warp_selection_mode={self.patch_warp_selection_mode}, "
            f"patch_warp_selection_scope={self.patch_warp_selection_scope}, "
            f"aug_scoring_mode={self.aug_scoring_mode}, "
            f"aug_injection_mode={self.aug_injection_mode}"
        )

    def _build_branch_args(self):
        return SimpleNamespace(
            append_newline=self.append_newline,
            patch_warp_ratio=self.patch_warp_ratio,
            patch_warp_selection_mode=self.patch_warp_selection_mode,
            patch_warp_selection_scope=self.patch_warp_selection_scope,
            patch_warp_shift_size=self.patch_warp_shift_size,
            patch_warp_mix_ratio=self.patch_warp_mix_ratio,
            patch_warp_fusion_2d_weight=self.patch_warp_fusion_2d_weight,
            patch_warp_fusion_3d_weight=self.patch_warp_fusion_3d_weight,
            aug_patch_topk=self.aug_patch_topk,
            aug_patch_ratio=self.aug_patch_ratio,
            aug_selection_scope=self.aug_selection_scope,
            aug_scoring_mode=self.aug_scoring_mode,
            aug_injection_mode=self.aug_injection_mode,
            aug_boost_factor=self.aug_boost_factor,
            aug_background_decay=self.aug_background_decay,
            aug_fine_scale=self.aug_fine_scale,
            aug_include_coarse=self.aug_include_coarse,
            aug_append_newline=self.aug_append_newline,
            aug_fusion_2d_weight=self.aug_fusion_2d_weight,
            aug_fusion_3d_weight=self.aug_fusion_3d_weight,
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
            branch_bundle = build_generation_patch_warp_branch_bundle(
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
                    reference_mode=self.reference_mode,
                )

            self.latest_vcd_metadata = {
                "doc_id": doc_id,
                "question_type": doc.get("question_type"),
                "branch_mode": branch_bundle.get("branch_mode"),
                "branch_names": generation_output.get("branch_names"),
                "reference_mode": self.reference_mode,
                "steps": generation_output.get("steps"),
                "branch_metadata": {
                    branch["name"]: branch.get("metadata")
                    for branch in branch_bundle.get("branches", [])
                },
            }
            res.append(generation_output["text"])
            pbar.update(1)
        return res
