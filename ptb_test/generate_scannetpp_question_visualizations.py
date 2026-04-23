#!/usr/bin/env python3
"""Generate bbox/mask visualizations for 20 processed ScanNet++ train scenes.

The script:
1. Reads the 20 local scene folders under Scannetpp_processed/color/train.
2. Loads QA labels from merged_qa_scannetpp_train.json.
3. For each scene, automatically selects 2 questions that can be fully visualized
   on a single input frame.
4. Overlays instance masks + 2D boxes for question-related objects.
5. Writes all outputs into ptb_test/scannetpp_qa_visualizations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_ROOT = Path(
    "/local_home/pantianbo/dataset_all/vision_reasoning/Scannetpp_processed"
)
DEFAULT_QA_PATH = (
    DEFAULT_DATASET_ROOT
    / "vsibench_train"
    / "merged_qa_scannetpp_train.json"
)
DEFAULT_FRAME_META_PATH = (
    DEFAULT_DATASET_ROOT
    / "metadata"
    / "train"
    / "scannetpp_frame_metadata_train.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/local_home/pantianbo/projects/vision_reasoning/VLM-3R/ptb_test/scannetpp_qa_visualizations"
)


QUESTION_TYPE_PRIORITY = {
    "mc_distance_compare": 0,
    "pair_distance": 1,
    "left_right_obj": 2,
    "egocentric3": 3,
    "quadrant3": 4,
    "count": 5,
    "single_object_size": 6,
}


PALETTE = [
    (231, 76, 60),   # red
    (52, 152, 219),  # blue
    (155, 89, 182),  # purple
    (26, 188, 156),  # teal
    (243, 156, 18),  # orange
    (46, 204, 113),  # green
    (230, 126, 34),  # dark orange
    (241, 196, 15),  # yellow
]


@dataclass
class ParsedQuestion:
    question_type: str
    relevant_objects: List[str]
    display_objects: List[str]
    question_text: str
    answer_value: str
    answer_object: Optional[str] = None
    options_map: Optional[Dict[str, str]] = None


@dataclass
class CandidateQuestion:
    scene_name: str
    qa_id: str
    parsed: ParsedQuestion
    raw_item: Dict
    best_frame: Dict
    best_frame_score: float
    drawn_objects: Dict[str, List[Dict]]


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def safe_filename(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.-]+", "_", text).strip("_")


def compact_question_text(question_text: str) -> str:
    text = normalize_space(question_text)
    text = re.sub(r"^These are frames of a video\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Options:.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Answer with .*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Please answer .*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def find_font(size: int) -> ImageFont.ImageFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_options_map(question_text: str) -> Dict[str, str]:
    option_pairs = re.findall(r"\n([A-Z])\.\s+([^\n]+)", question_text)
    return {k.strip(): v.strip() for k, v in option_pairs}


def parse_question(raw_question_text: str, raw_answer: str) -> Optional[ParsedQuestion]:
    question_text = normalize_space(raw_question_text)
    options_map = parse_options_map(raw_question_text)

    m = re.search(
        r"Measuring from the closest point of each object, which of these objects \((.*?)\) "
        r"is the (closest|farthest|nearest) to the ([^?]+)\?",
        question_text,
        flags=re.IGNORECASE,
    )
    if m:
        choices = [x.strip() for x in m.group(1).split(",")]
        reference = m.group(3).strip()
        answer_object = options_map.get(raw_answer.strip(), raw_answer.strip())
        return ParsedQuestion(
            question_type="mc_distance_compare",
            relevant_objects=choices + [reference],
            display_objects=choices + [reference],
            question_text=question_text,
            answer_value=raw_answer.strip(),
            answer_object=answer_object,
            options_map=options_map,
        )

    m = re.search(
        r"Measuring from the closest point of each object, what is the direct distance between "
        r"the (.*?) and the ([^?]+) \(in meters\)\?",
        question_text,
        flags=re.IGNORECASE,
    )
    if m:
        obj_a = m.group(1).strip()
        obj_b = m.group(2).strip()
        return ParsedQuestion(
            question_type="pair_distance",
            relevant_objects=[obj_a, obj_b],
            display_objects=[obj_a, obj_b],
            question_text=question_text,
            answer_value=raw_answer.strip(),
        )

    m = re.search(
        r"If I am standing by the (.*?) and facing the (.*?), is the (.*?) to the left or the right of the (.*?)\?",
        question_text,
        flags=re.IGNORECASE,
    )
    if m:
        objs = [m.group(i).strip() for i in range(1, 5)]
        return ParsedQuestion(
            question_type="left_right_obj",
            relevant_objects=objs,
            display_objects=objs,
            question_text=question_text,
            answer_value=raw_answer.strip(),
            options_map=options_map or None,
        )

    m = re.search(
        r"If I am standing by the (.*?) and facing the (.*?), is the (.*?) to my left, right, or back\?",
        question_text,
        flags=re.IGNORECASE,
    )
    if m:
        objs = [m.group(i).strip() for i in range(1, 4)]
        return ParsedQuestion(
            question_type="egocentric3",
            relevant_objects=objs,
            display_objects=objs,
            question_text=question_text,
            answer_value=raw_answer.strip(),
            options_map=options_map or None,
        )

    m = re.search(
        r"If I am standing by the (.*?) and facing the (.*?), is the (.*?) to my front-left, "
        r"front-right, back-left, or back-right\?",
        question_text,
        flags=re.IGNORECASE,
    )
    if m:
        objs = [m.group(i).strip() for i in range(1, 4)]
        return ParsedQuestion(
            question_type="quadrant3",
            relevant_objects=objs,
            display_objects=objs,
            question_text=question_text,
            answer_value=raw_answer.strip(),
            options_map=options_map or None,
        )

    m = re.search(
        r"How many (.*)\(s\) are in this room\?",
        question_text,
        flags=re.IGNORECASE,
    )
    if m:
        obj = m.group(1).strip()
        return ParsedQuestion(
            question_type="count",
            relevant_objects=[obj],
            display_objects=[obj],
            question_text=question_text,
            answer_value=raw_answer.strip(),
        )

    m = re.search(
        r"What is the length of the longest dimension .* of the (.*?), measured in centimeters\?",
        question_text,
        flags=re.IGNORECASE,
    )
    if m:
        obj = m.group(1).strip()
        return ParsedQuestion(
            question_type="single_object_size",
            relevant_objects=[obj],
            display_objects=[obj],
            question_text=question_text,
            answer_value=raw_answer.strip(),
        )

    return None


def bbox_area(bbox: Sequence[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def frame_objects_for_question(frame: Dict, relevant_objects: Iterable[str]) -> Dict[str, List[Dict]]:
    relevant_set = set(relevant_objects)
    grouped = defaultdict(list)
    for box in frame.get("bboxes_2d", []):
        name = box["category_name"]
        if name in relevant_set:
            grouped[name].append(box)
    return dict(grouped)


def score_frame(grouped_objects: Dict[str, List[Dict]]) -> float:
    score = 0.0
    for boxes in grouped_objects.values():
        if not boxes:
            continue
        largest = max(boxes, key=lambda x: bbox_area(x["bbox_2d"]))
        score += math.sqrt(max(1, bbox_area(largest["bbox_2d"])))
        score += 6.0 * max(0, len(boxes) - 1)
    return score


def build_candidate_question(scene_name: str, qa_item: Dict, frame_meta_scene: Dict) -> Optional[CandidateQuestion]:
    question_text = qa_item["conversations"][0]["value"]
    answer_text = qa_item["conversations"][1]["value"]
    parsed = parse_question(question_text, answer_text)
    if parsed is None:
        return None

    best_frame = None
    best_frame_score = -1.0
    best_grouped = None
    required = set(parsed.relevant_objects)

    for frame in frame_meta_scene["frames"]:
        grouped = frame_objects_for_question(frame, parsed.relevant_objects)
        if not required.issubset(grouped.keys()):
            continue
        score = score_frame(grouped)
        if score > best_frame_score:
            best_frame = frame
            best_frame_score = score
            best_grouped = grouped

    if best_frame is None or best_grouped is None:
        return None

    return CandidateQuestion(
        scene_name=scene_name,
        qa_id=qa_item["id"],
        parsed=parsed,
        raw_item=qa_item,
        best_frame=best_frame,
        best_frame_score=best_frame_score,
        drawn_objects=best_grouped,
    )


def question_sort_key(candidate: CandidateQuestion) -> Tuple[int, float, int]:
    priority = QUESTION_TYPE_PRIORITY.get(candidate.parsed.question_type, 999)
    return (priority, -candidate.best_frame_score, len(candidate.parsed.relevant_objects))


def object_overlap_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pick_two_questions(candidates: List[CandidateQuestion]) -> List[CandidateQuestion]:
    if len(candidates) <= 2:
        return sorted(candidates, key=question_sort_key)

    by_type = defaultdict(list)
    for cand in candidates:
        by_type[cand.parsed.question_type].append(cand)
    for cand_list in by_type.values():
        cand_list.sort(key=question_sort_key)

    selected: List[CandidateQuestion] = []

    preferred_first_types = [
        "mc_distance_compare",
        "pair_distance",
        "left_right_obj",
        "egocentric3",
        "quadrant3",
        "count",
        "single_object_size",
    ]
    preferred_second_types = [
        "pair_distance",
        "mc_distance_compare",
        "left_right_obj",
        "egocentric3",
        "quadrant3",
        "count",
        "single_object_size",
    ]

    for qtype in preferred_first_types:
        if by_type[qtype]:
            selected.append(by_type[qtype][0])
            break

    if not selected:
        selected.append(sorted(candidates, key=question_sort_key)[0])

    seen_ids = {selected[0].qa_id}
    for qtype in preferred_second_types:
        pool = [x for x in by_type[qtype] if x.qa_id not in seen_ids]
        if not pool:
            continue
        pool.sort(
            key=lambda x: (
                object_overlap_ratio(x.parsed.relevant_objects, selected[0].parsed.relevant_objects),
                -x.best_frame_score,
                len(x.parsed.relevant_objects),
            )
        )
        selected.append(pool[0])
        break

    if len(selected) < 2:
        remaining = [x for x in sorted(candidates, key=question_sort_key) if x.qa_id not in seen_ids]
        if remaining:
            selected.append(remaining[0])

    return selected[:2]


def get_color_map(parsed: ParsedQuestion) -> Dict[str, Tuple[int, int, int]]:
    color_map: Dict[str, Tuple[int, int, int]] = {}
    objects = parsed.display_objects

    if parsed.question_type == "mc_distance_compare":
        reference = objects[-1]
        color_map[reference] = (241, 196, 15)  # gold
        palette_idx = 0
        for obj in objects[:-1]:
            if parsed.answer_object and obj == parsed.answer_object:
                color_map[obj] = (46, 204, 113)  # green
            else:
                while PALETTE[palette_idx % len(PALETTE)] in color_map.values():
                    palette_idx += 1
                color_map[obj] = PALETTE[palette_idx % len(PALETTE)]
                palette_idx += 1
        return color_map

    for idx, obj in enumerate(objects):
        color_map[obj] = PALETTE[idx % len(PALETTE)]
    return color_map


def add_mask_overlay(
    rgba: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.28,
) -> None:
    if not mask.any():
        return
    mask3 = mask[:, :, None].astype(np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    rgba[:, :, :3] = (rgb * (1.0 - alpha * mask3) + np.array(color, dtype=np.float32) * (alpha * mask3)).astype(np.uint8)


def draw_rounded_box(draw: ImageDraw.ImageDraw, bbox: Sequence[int], color: Tuple[int, int, int], width: int = 3):
    x1, y1, x2, y2 = bbox
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
):
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rounded_rectangle(
        [left - pad, top - pad, right + pad, bottom + pad],
        radius=6,
        fill=(0, 0, 0, 180),
        outline=color,
        width=2,
    )
    draw.text((x, y), text, fill=color, font=font)


def render_visualization(
    candidate: CandidateQuestion,
    dataset_root: Path,
    output_path: Path,
) -> Dict:
    frame = candidate.best_frame
    color_path = dataset_root / frame["file_path_color"]
    instance_path = dataset_root / frame["file_path_instance"]

    image = Image.open(color_path).convert("RGBA")
    instance_map = np.array(Image.open(instance_path), dtype=np.uint16)
    rgba = np.array(image)

    color_map = get_color_map(candidate.parsed)
    label_font = find_font(18)
    info_font = find_font(26)
    small_font = find_font(20)

    draw_entries = []
    per_object_counter = defaultdict(int)

    for obj_name in candidate.parsed.display_objects:
        boxes = sorted(
            candidate.drawn_objects.get(obj_name, []),
            key=lambda x: bbox_area(x["bbox_2d"]),
            reverse=True,
        )
        for box in boxes:
            inst_id = int(box["instance_id"])
            mask = instance_map == inst_id
            add_mask_overlay(rgba, mask, color_map[obj_name], alpha=0.30)
            per_object_counter[obj_name] += 1
            draw_entries.append(
                {
                    "object_name": obj_name,
                    "instance_id": inst_id,
                    "bbox_2d": [int(v) for v in box["bbox_2d"]],
                    "instance_rank": per_object_counter[obj_name],
                    "area": bbox_area(box["bbox_2d"]),
                }
            )

    overlay = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    for entry in sorted(draw_entries, key=lambda x: x["area"], reverse=True):
        obj_name = entry["object_name"]
        bbox = entry["bbox_2d"]
        color = color_map[obj_name]
        draw_rounded_box(draw, bbox, color, width=3 if candidate.parsed.answer_object != obj_name else 4)
        label = f"{obj_name}"
        if len(candidate.drawn_objects.get(obj_name, [])) > 1:
            label += f" #{entry['instance_rank']}"
        label += f" [{entry['instance_id']}]"
        draw_label(draw, (bbox[0] + 4, max(4, bbox[1] - 24)), label, color, label_font)

    overlay_rgb = overlay.convert("RGB").resize((960, 720), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1440, 720), color=(20, 22, 28))
    canvas.paste(overlay_rgb, (0, 0))
    panel = ImageDraw.Draw(canvas)

    panel.text((990, 28), f"Scene: {candidate.scene_name}", fill=(255, 255, 255), font=info_font)
    panel.text(
        (990, 70),
        f"Frame: {Path(frame['file_path_color']).name}",
        fill=(190, 200, 215),
        font=small_font,
    )
    panel.text(
        (990, 104),
        f"Type: {candidate.parsed.question_type}",
        fill=(190, 200, 215),
        font=small_font,
    )

    q_lines = textwrap.fill(compact_question_text(candidate.parsed.question_text), width=34)
    panel.multiline_text(
        (990, 150),
        f"Question\n{q_lines}",
        fill=(240, 240, 240),
        font=small_font,
        spacing=6,
    )

    if candidate.parsed.question_type == "mc_distance_compare":
        answer_desc = candidate.parsed.answer_object or candidate.parsed.answer_value
        answer_line = f"GT answer: {candidate.parsed.answer_value} = {answer_desc}"
    elif candidate.parsed.options_map and candidate.parsed.answer_value in candidate.parsed.options_map:
        answer_line = f"GT answer: {candidate.parsed.answer_value} = {candidate.parsed.options_map[candidate.parsed.answer_value]}"
    else:
        answer_line = f"GT answer: {candidate.parsed.answer_value}"
    panel.multiline_text(
        (990, 410),
        textwrap.fill(answer_line, width=34),
        fill=(255, 255, 255),
        font=small_font,
        spacing=6,
    )

    y = 480
    panel.text((990, y), "Legend", fill=(255, 255, 255), font=info_font)
    y += 42
    for obj_name in candidate.parsed.display_objects:
        color = color_map[obj_name]
        panel.rounded_rectangle([990, y + 4, 1016, y + 30], radius=6, fill=color)
        role = ""
        if candidate.parsed.question_type == "mc_distance_compare":
            if obj_name == candidate.parsed.display_objects[-1]:
                role = "reference object"
            elif candidate.parsed.answer_object == obj_name:
                role = "correct option"
            else:
                role = "other option"
        panel.text(
            (1028, y),
            f"{obj_name}{' - ' + role if role else ''}",
            fill=(230, 230, 230),
            font=small_font,
        )
        y += 36

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)

    return {
        "output_path": str(output_path),
        "frame_color_path": str(color_path),
        "frame_instance_path": str(instance_path),
        "drawn_instances": draw_entries,
    }


def make_scene_contact_sheet(image_paths: List[Path], output_path: Path) -> None:
    if not image_paths:
        return
    images = [Image.open(p).convert("RGB") for p in image_paths]
    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    canvas = Image.new("RGB", (width, height), color=(15, 15, 15))
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def generate_readme(output_root: Path, manifest: Dict) -> None:
    lines = [
        "# ScanNet++ QA bbox/mask visualizations",
        "",
        f"- Generated scenes: {len(manifest['scenes'])}",
        f"- Total selected questions: {manifest['total_questions']}",
        "",
        "## Files",
        "",
        "- `selection_manifest.json`: full machine-readable summary",
        "- `scene_overviews/<scene>.jpg`: stitched 2-question overview per scene",
        "- `per_scene/<scene>/*.jpg`: individual question visualizations",
        "",
        "## Scene overview list",
        "",
    ]
    for scene in manifest["scenes"]:
        lines.append(
            f"- `{scene['scene_name']}`: "
            + ", ".join(
                f"{item['parsed_question']['question_type']} -> {Path(item['output_path']).name}"
                for item in scene["questions"]
            )
        )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--frame-meta-path", type=Path, default=DEFAULT_FRAME_META_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    qa_items = load_json(args.qa_path)
    frame_meta = load_json(args.frame_meta_path)
    scene_names = sorted(
        p.name for p in (args.dataset_root / "color" / "train").iterdir() if p.is_dir()
    )

    qa_by_scene = defaultdict(list)
    for item in qa_items:
        if item.get("scene_name") in scene_names:
            qa_by_scene[item["scene_name"]].append(item)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_root": str(args.dataset_root), "total_questions": 0, "scenes": []}

    for scene_name in scene_names:
        frame_meta_scene = frame_meta[scene_name]
        candidates = []
        for item in qa_by_scene[scene_name]:
            cand = build_candidate_question(scene_name, item, frame_meta_scene)
            if cand is not None:
                candidates.append(cand)

        selected = pick_two_questions(candidates)
        if len(selected) < 2:
            raise RuntimeError(f"Scene {scene_name} only yielded {len(selected)} visualizable questions.")

        scene_output_dir = args.output_root / "per_scene" / scene_name
        scene_entries = []
        scene_image_paths = []

        for idx, cand in enumerate(selected, start=1):
            filename = (
                f"{idx:02d}_{safe_filename(cand.parsed.question_type)}_"
                f"{safe_filename(cand.qa_id[:8])}.jpg"
            )
            output_path = scene_output_dir / filename
            render_info = render_visualization(cand, args.dataset_root, output_path)
            scene_image_paths.append(output_path)

            scene_entries.append(
                {
                    "qa_id": cand.qa_id,
                    "scene_name": cand.scene_name,
                    "output_path": str(output_path),
                    "best_frame_file": cand.best_frame["file_path_color"],
                    "best_frame_score": cand.best_frame_score,
                    "parsed_question": {
                        "question_type": cand.parsed.question_type,
                        "question_text": cand.parsed.question_text,
                        "relevant_objects": cand.parsed.relevant_objects,
                        "answer_value": cand.parsed.answer_value,
                        "answer_object": cand.parsed.answer_object,
                        "options_map": cand.parsed.options_map,
                    },
                    "render_info": render_info,
                }
            )

        overview_path = args.output_root / "scene_overviews" / f"{scene_name}.jpg"
        make_scene_contact_sheet(scene_image_paths, overview_path)
        manifest["scenes"].append(
            {
                "scene_name": scene_name,
                "overview_path": str(overview_path),
                "questions": scene_entries,
            }
        )
        manifest["total_questions"] += len(scene_entries)

    manifest_path = args.output_root / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_readme(args.output_root, manifest)
    print(f"Done. Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
