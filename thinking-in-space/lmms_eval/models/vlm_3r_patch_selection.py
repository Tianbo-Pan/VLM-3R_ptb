from typing import Optional

import torch
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptb_patch_selection import generate_with_selective_patch_pooling
from .vlm_3r import Vlm3r


@register_model("vlm_3r_patch_selection")
class Vlm3rPatchSelection(Vlm3r):
    def __init__(
        self,
        fine_topk: int = 16,
        fine_ratio: Optional[float] = None,
        selection_scope: str = "per_frame",
        scoring_mode: str = "question_cosine",
        fine_scale: float = 1.0,
        fusion_2d_weight: float = 1.0,
        fusion_3d_weight: float = 1.0,
        include_coarse: bool = True,
        append_newline: bool = True,
        coarse_mode: str = "full",
        coarse_context_radius: int = 0,
        coarse_context_topk: Optional[int] = None,
        coarse_context_scale: float = 1.0,
        contextual_coarse_first: bool = True,
        save_patch_selection_metadata: bool = False,
        **kwargs,
    ) -> None:
        self.patch_selection_enabled = True
        self.fine_topk = int(fine_topk)
        self.fine_ratio = None if fine_ratio in [None, "", "none"] else float(fine_ratio)
        self.selection_scope = str(selection_scope)
        self.scoring_mode = str(scoring_mode)
        self.fine_scale = float(fine_scale)
        self.fusion_2d_weight = float(fusion_2d_weight)
        self.fusion_3d_weight = float(fusion_3d_weight)
        self.include_coarse = include_coarse if isinstance(include_coarse, bool) else str(include_coarse).lower() == "true"
        self.append_newline = append_newline if isinstance(append_newline, bool) else str(append_newline).lower() == "true"
        self.coarse_mode = str(coarse_mode)
        self.coarse_context_radius = int(coarse_context_radius)
        self.coarse_context_topk = None if coarse_context_topk in [None, "", "none"] else int(coarse_context_topk)
        self.coarse_context_scale = float(coarse_context_scale)
        self.contextual_coarse_first = (
            contextual_coarse_first
            if isinstance(contextual_coarse_first, bool)
            else str(contextual_coarse_first).lower() == "true"
        )
        self.save_patch_selection_metadata = (
            save_patch_selection_metadata
            if isinstance(save_patch_selection_metadata, bool)
            else str(save_patch_selection_metadata).lower() == "true"
        )
        self.latest_patch_selection_metadata: Optional[dict] = None
        super().__init__(**kwargs)
        eval_logger.info(
            f"Enabled patch-selection inference: fine_topk={self.fine_topk}, "
            f"fine_ratio={self.fine_ratio}, selection_scope={self.selection_scope}, "
            f"scoring_mode={self.scoring_mode}, include_coarse={self.include_coarse}, "
            f"coarse_mode={self.coarse_mode}, coarse_context_radius={self.coarse_context_radius}, "
            f"coarse_context_topk={self.coarse_context_topk}, coarse_context_scale={self.coarse_context_scale}, "
            f"fusion_2d_weight={self.fusion_2d_weight}, fusion_3d_weight={self.fusion_3d_weight}"
        )

    def _generate_with_optional_feature_cd(
        self,
        input_ids,
        videos,
        attention_masks,
        stopping_criteria,
        gen_kwargs,
    ):
        if videos is not None:
            if len(videos) != 1:
                raise ValueError(
                    f"Patch-selection eval currently expects exactly one video per request, got {len(videos)}."
                )
            output = generate_with_selective_patch_pooling(
                self.model,
                input_ids=input_ids,
                images=videos,
                attention_mask=attention_masks,
                tokenizer=self.tokenizer,
                modalities="video",
                fine_topk=self.fine_topk,
                fine_ratio=self.fine_ratio,
                selection_scope=self.selection_scope,
                scoring_mode=self.scoring_mode,
                fine_scale=self.fine_scale,
                fusion_2d_weight=self.fusion_2d_weight,
                fusion_3d_weight=self.fusion_3d_weight,
                include_coarse=self.include_coarse,
                append_newline=self.append_newline,
                coarse_mode=self.coarse_mode,
                coarse_context_radius=self.coarse_context_radius,
                coarse_context_topk=self.coarse_context_topk,
                coarse_context_scale=self.coarse_context_scale,
                contextual_coarse_first=self.contextual_coarse_first,
                return_metadata=self.save_patch_selection_metadata,
                use_cache=self.use_cache,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )
            if self.save_patch_selection_metadata:
                output_ids, metadata = output
                self.latest_patch_selection_metadata = metadata
                return output_ids
            self.latest_patch_selection_metadata = None
            return output

        return super()._generate_with_optional_feature_cd(
            input_ids=input_ids,
            videos=videos,
            attention_masks=attention_masks,
            stopping_criteria=stopping_criteria,
            gen_kwargs=gen_kwargs,
        )
