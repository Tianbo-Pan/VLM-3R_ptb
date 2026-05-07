from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

from llava.model.feature_cd.common import build_inputs_embeds_from_video_features


AllowedTokensFn = Callable[[int, torch.Tensor], Optional[torch.Tensor]]


def _as_batch(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim == 1:
        return input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids to have shape [T] or [B, T], got {tuple(input_ids.shape)}.")
    return input_ids


@torch.no_grad()
def next_token_logits_with_video_features(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    video_features: torch.Tensor,
) -> torch.Tensor:
    input_ids = _as_batch(input_ids)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    elif attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)

    position_ids, final_attention_mask, inputs_embeds = build_inputs_embeds_from_video_features(
        model,
        input_ids,
        attention_mask,
        video_features,
    )
    outputs = model(
        position_ids=position_ids,
        attention_mask=final_attention_mask,
        inputs_embeds=inputs_embeds,
        return_dict=True,
        use_cache=False,
    )
    return outputs.logits[:, -1, :]


def combine_branch_logits(
    branch_logits: Sequence[torch.Tensor],
    contrast_mode: str = "pairwise",
    contrast_alphas: Optional[Sequence[float]] = None,
    branch_names: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    if not branch_logits:
        raise ValueError("branch_logits must contain at least one tensor.")

    contrast_mode = str(contrast_mode).lower()
    if contrast_mode == "none" or len(branch_logits) == 1:
        return branch_logits[-1].clone()

    if contrast_mode == "pairwise":
        if len(branch_logits) != 2:
            raise ValueError(f"Pairwise contrast expects exactly 2 branches, got {len(branch_logits)}.")
        alpha = 1.0 if not contrast_alphas else float(contrast_alphas[0])
        weak_logits, strong_logits = branch_logits
        return (1.0 + alpha) * strong_logits - alpha * weak_logits

    if contrast_mode in {"tri_simple", "tri_rectified"}:
        if len(branch_logits) != 3:
            raise ValueError(f"{contrast_mode} expects exactly 3 branches, got {len(branch_logits)}.")
        if not branch_names or len(branch_names) != len(branch_logits):
            raise ValueError(f"{contrast_mode} requires 3 branch_names aligned with branch_logits.")
        branch_logit_map = {str(name): logits for name, logits in zip(branch_names, branch_logits)}
        required_names = {"degraded", "original", "augmented"}
        missing = sorted(required_names - set(branch_logit_map.keys()))
        if missing:
            raise ValueError(f"{contrast_mode} requires branches {sorted(required_names)}, missing {missing}.")

        if not contrast_alphas:
            alpha_aug, alpha_deg = 1.0, 1.0
        elif len(contrast_alphas) == 1:
            alpha_aug = alpha_deg = float(contrast_alphas[0])
        else:
            alpha_aug = float(contrast_alphas[0])
            alpha_deg = float(contrast_alphas[1])

        degraded_logits = branch_logit_map["degraded"]
        original_logits = branch_logit_map["original"]
        augmented_logits = branch_logit_map["augmented"]
        augmented_delta = augmented_logits - original_logits
        degraded_delta = degraded_logits - original_logits

        if contrast_mode == "tri_rectified":
            augmented_delta = torch.relu(augmented_delta)
            degraded_delta = torch.relu(degraded_delta)

        return original_logits + alpha_aug * augmented_delta - alpha_deg * degraded_delta

    raise ValueError(
        f"Unsupported contrast_mode: {contrast_mode}. Supported: none, pairwise, tri_simple, tri_rectified."
    )


def build_reference_logits(
    branch_logits: Sequence[torch.Tensor],
    branch_names: Optional[Sequence[str]] = None,
    reference_mode: str = "last",
) -> torch.Tensor:
    if not branch_logits:
        raise ValueError("branch_logits must contain at least one tensor.")

    reference_mode = str(reference_mode).lower()
    if reference_mode == "last":
        return branch_logits[-1]

    if not branch_names or len(branch_names) != len(branch_logits):
        raise ValueError(f"reference_mode={reference_mode} requires branch_names aligned with branch_logits.")

    branch_logit_map = {str(name): logits for name, logits in zip(branch_names, branch_logits)}
    if reference_mode in branch_logit_map:
        return branch_logit_map[reference_mode]

    if reference_mode == "max_original_augmented":
        return torch.maximum(branch_logit_map["original"], branch_logit_map["augmented"])
    if reference_mode == "mean_original_augmented":
        return 0.5 * (branch_logit_map["original"] + branch_logit_map["augmented"])

    raise ValueError(
        "Unsupported reference_mode: "
        f"{reference_mode}. Supported: last, degraded, original, augmented, "
        "max_original_augmented, mean_original_augmented."
    )


def apply_plausibility_constraint(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    beta: float = 0.05,
    allowed_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.shape != reference_logits.shape:
        raise ValueError(
            f"logits and reference_logits must match, got {tuple(logits.shape)} vs {tuple(reference_logits.shape)}."
        )
    constrained = logits.clone()

    if beta is not None and beta > 0:
        reference_probs = reference_logits.softmax(dim=-1)
        cutoff = float(beta) * reference_probs.max(dim=-1, keepdim=True).values
        plausibility_mask = reference_probs >= cutoff
        constrained = constrained.masked_fill(~plausibility_mask, -float("inf"))

    if allowed_mask is not None:
        if allowed_mask.ndim == 1:
            allowed_mask = allowed_mask.unsqueeze(0)
        if allowed_mask.shape != constrained.shape:
            raise ValueError(
                f"allowed_mask must match logits shape, got {tuple(allowed_mask.shape)} vs {tuple(constrained.shape)}."
            )
        constrained = constrained.masked_fill(~allowed_mask.bool(), -float("inf"))

    if not torch.isfinite(constrained).any():
        return reference_logits
    return constrained


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits with shape [1, vocab], got {tuple(logits.shape)}.")

    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)

    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > top_p
        remove_mask[..., 0] = False
        filtered_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sampled_sorted_idx).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def trim_stop_strings(text: str, stop_strings: Optional[Sequence[str]]) -> str:
    if not stop_strings:
        return text
    best_idx = None
    for stop_string in stop_strings:
        if not stop_string:
            continue
        idx = text.find(stop_string)
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx = idx
    if best_idx is None:
        return text
    return text[:best_idx]


@torch.no_grad()
def generate_with_vcd(
    tokenizer,
    model,
    prompt_input_ids: torch.Tensor,
    branch_bundle: Dict[str, object],
    max_new_tokens: int = 16,
    contrast_mode: str = "pairwise",
    contrast_alphas: Optional[Sequence[float]] = None,
    beta: float = 0.05,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    stop_strings: Optional[Sequence[str]] = None,
    allowed_tokens_fn: Optional[AllowedTokensFn] = None,
    reference_mode: str = "last",
) -> Dict[str, object]:
    branches = branch_bundle.get("branches")
    if not branches:
        raise ValueError("branch_bundle must contain a non-empty `branches` list.")
    first_branch_device = branches[0]["video_features"].device
    prompt_input_ids = _as_batch(prompt_input_ids).to(first_branch_device)
    attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long, device=prompt_input_ids.device)

    branch_names = [str(branch["name"]) for branch in branches]
    generated_token_ids: List[int] = []
    step_records: List[Dict[str, object]] = []
    current_input_ids = prompt_input_ids
    current_attention_mask = attention_mask
    eos_token_id = tokenizer.eos_token_id if eos_token_id is None else eos_token_id

    for step_idx in range(int(max_new_tokens)):
        step_branch_logits: List[torch.Tensor] = []
        for branch in branches:
            branch_logits = next_token_logits_with_video_features(
                model=model,
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                video_features=branch["video_features"],
            )
            step_branch_logits.append(branch_logits)

        combined_logits = combine_branch_logits(
            step_branch_logits,
            contrast_mode=contrast_mode,
            contrast_alphas=contrast_alphas,
            branch_names=branch_names,
        )
        reference_logits = build_reference_logits(
            step_branch_logits,
            branch_names=branch_names,
            reference_mode=reference_mode,
        )

        allowed_mask = None
        if allowed_tokens_fn is not None:
            allowed_mask = allowed_tokens_fn(step_idx, current_input_ids)
            if allowed_mask is not None and allowed_mask.ndim == 1:
                allowed_mask = allowed_mask.unsqueeze(0).to(device=combined_logits.device)

        constrained_logits = apply_plausibility_constraint(
            combined_logits,
            reference_logits=reference_logits,
            beta=beta,
            allowed_mask=allowed_mask,
        )
        next_token = sample_from_logits(
            constrained_logits,
            temperature=temperature,
            top_p=top_p,
        )
        token_id = int(next_token.item())
        generated_token_ids.append(token_id)

        step_records.append(
            {
                "step": step_idx,
                "selected_token_id": token_id,
                "selected_token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                "branch_names": branch_names,
                "branch_top1_token_ids": [int(x.argmax(dim=-1).item()) for x in step_branch_logits],
                "combined_top1_token_id": int(combined_logits.argmax(dim=-1).item()),
                "constrained_top1_token_id": int(constrained_logits.argmax(dim=-1).item()),
            }
        )

        next_token_batch = next_token.view(1, 1).to(device=current_input_ids.device, dtype=current_input_ids.dtype)
        current_input_ids = torch.cat([current_input_ids, next_token_batch], dim=-1)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=-1,
        )

        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        if eos_token_id is not None and token_id == eos_token_id:
            break
        if stop_strings and any(stop_string and stop_string in generated_text for stop_string in stop_strings):
            break

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    generated_text = trim_stop_strings(generated_text, stop_strings).strip()
    return {
        "text": generated_text,
        "generated_token_ids": generated_token_ids,
        "steps": step_records,
        "branch_mode": branch_bundle.get("branch_mode"),
        "branch_names": branch_names,
    }
