from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ptb_test.experiment0_attention_rollout as exp0


DEFAULT_MANIFEST_PATH = exp0.DEFAULT_MANIFEST_PATH


@dataclass
class CaseRecord:
    case: exp0.SelectedCase
    flat_index: int


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Dict:
    return exp0.load_manifest(path)


def build_case_records(
    manifest: Dict,
    require_options: bool = True,
) -> List[CaseRecord]:
    records: List[CaseRecord] = []
    flat = exp0.flatten_manifest_questions(manifest)
    for idx, (scene_entry, question_entry) in enumerate(flat):
        parsed = question_entry.get("parsed_question", {})
        if require_options and not parsed.get("options_map"):
            continue
        frame_path = Path(question_entry["render_info"]["frame_color_path"])
        selected = exp0.SelectedCase(
            qa_id=question_entry["qa_id"],
            scene_name=scene_entry["scene_name"],
            scene_entry=scene_entry,
            question_entry=question_entry,
            frame_dir=frame_path.parent,
            best_frame_path=frame_path,
            prompt=parsed.get("question_text", ""),
        )
        records.append(CaseRecord(case=selected, flat_index=idx))
    return records


def filter_case_records(
    records: Sequence[CaseRecord],
    qa_id: Optional[str] = None,
    scene_name: Optional[str] = None,
    flat_index: Optional[int] = None,
    max_cases: Optional[int] = None,
) -> List[CaseRecord]:
    out = list(records)
    if qa_id is not None:
        out = [x for x in out if x.case.qa_id == qa_id]
    if scene_name is not None:
        out = [x for x in out if x.case.scene_name == scene_name]
    if flat_index is not None:
        out = [x for x in out if x.flat_index == flat_index]
    if max_cases is not None:
        out = out[: max_cases]
    return out


def sample_case_frame_paths(case: exp0.SelectedCase, num_frames: int) -> List[Path]:
    frame_paths = exp0.load_scene_frame_paths(case.frame_dir)
    best_idx = None
    for idx, frame_path in enumerate(frame_paths):
        if frame_path.name == case.best_frame_path.name:
            best_idx = idx
            break
    sampled_indices = exp0.sample_frame_indices(len(frame_paths), num_frames, forced_idx=best_idx)
    return [frame_paths[idx] for idx in sampled_indices]


def find_mismatched_case(
    reference: CaseRecord,
    candidates: Sequence[CaseRecord],
) -> Optional[CaseRecord]:
    ref_qtype = reference.case.question_entry.get("parsed_question", {}).get("question_type")
    ref_option_count = len(reference.case.question_entry.get("parsed_question", {}).get("options_map") or {})
    for candidate in candidates:
        if candidate.case.qa_id == reference.case.qa_id:
            continue
        cand_parsed = candidate.case.question_entry.get("parsed_question", {})
        if cand_parsed.get("question_type") != ref_qtype:
            continue
        if len(cand_parsed.get("options_map") or {}) != ref_option_count:
            continue
        return candidate
    for candidate in candidates:
        if candidate.case.qa_id != reference.case.qa_id:
            return candidate
    return None

