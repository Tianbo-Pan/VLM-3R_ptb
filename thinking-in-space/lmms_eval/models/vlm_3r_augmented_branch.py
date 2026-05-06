from typing import Optional

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptb_patch_selection import generate_with_augmented_patch_pooling
from .vlm_3r import Vlm3r


@register_model("vlm_3r_augmented_branch")
class Vlm3rAugmentedBranch(Vlm3r):
    def __init__(
        self,
        patch_topk: int = 16,
        patch_ratio: Optional[float] = 0.01,
        selection_scope: str = "global",
        scoring_mode: str = "question_cosine",
        injection_mode: str = "inplace_boost_coarse",
        boost_factor: float = 2.0,
        background_decay: float = 0.98,
        fine_scale: float = 1.0,
        include_coarse: bool = True,
        append_newline: bool = True,
        fusion_2d_weight: float = 1.0,
        fusion_3d_weight: float = 1.0,
        save_patch_selection_metadata: bool = False,
        **kwargs,
    ) -> None:
        self.patch_topk = int(patch_topk)
        self.patch_ratio = None if patch_ratio in [None, "", "none"] else float(patch_ratio)
        self.selection_scope = str(selection_scope)
        self.scoring_mode = str(scoring_mode)
        self.injection_mode = str(injection_mode)
        self.boost_factor = float(boost_factor)
        self.background_decay = float(background_decay)
        self.fine_scale = float(fine_scale)
        self.include_coarse = include_coarse if isinstance(include_coarse, bool) else str(include_coarse).lower() == "true"
        self.append_newline = append_newline if isinstance(append_newline, bool) else str(append_newline).lower() == "true"
        self.fusion_2d_weight = float(fusion_2d_weight)
        self.fusion_3d_weight = float(fusion_3d_weight)
        self.save_patch_selection_metadata = (
            save_patch_selection_metadata
            if isinstance(save_patch_selection_metadata, bool)
            else str(save_patch_selection_metadata).lower() == "true"
        )
        self.latest_patch_selection_metadata: Optional[dict] = None
        super().__init__(**kwargs)
        eval_logger.info(
            "Enabled augmentation-branch inference: "
            f"patch_topk={self.patch_topk}, "
            f"patch_ratio={self.patch_ratio}, "
            f"selection_scope={self.selection_scope}, "
            f"scoring_mode={self.scoring_mode}, "
            f"injection_mode={self.injection_mode}, "
            f"boost_factor={self.boost_factor}, "
            f"background_decay={self.background_decay}, "
            f"include_coarse={self.include_coarse}, "
            f"fine_scale={self.fine_scale}"
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
                    f"Augmented-branch eval currently expects exactly one video per request, got {len(videos)}."
                )
            output = generate_with_augmented_patch_pooling(
                self.model,
                input_ids=input_ids,
                images=videos,
                attention_mask=attention_masks,
                tokenizer=self.tokenizer,
                modalities="video",
                patch_topk=self.patch_topk,
                patch_ratio=self.patch_ratio,
                selection_scope=self.selection_scope,
                scoring_mode=self.scoring_mode,
                injection_mode=self.injection_mode,
                boost_factor=self.boost_factor,
                background_decay=self.background_decay,
                fine_scale=self.fine_scale,
                include_coarse=self.include_coarse,
                append_newline=self.append_newline,
                fusion_2d_weight=self.fusion_2d_weight,
                fusion_3d_weight=self.fusion_3d_weight,
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
