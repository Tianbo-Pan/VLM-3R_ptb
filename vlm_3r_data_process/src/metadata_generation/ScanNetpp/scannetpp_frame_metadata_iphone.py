import argparse
import json
import logging
import os
import shutil
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lz4.block
import numpy as np
from PIL import Image

try:
    from src.base_processor import BaseProcessorConfig, AbstractSceneProcessor
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Failed to import base processor classes. Falling back to lightweight local definitions.")

    @dataclass
    class BaseProcessorConfig:
        save_dir: str = "output"
        output_filename: str = "metadata.json"
        num_workers: int = 1
        overwrite: bool = False
        random_seed: int = 42

    class AbstractSceneProcessor:
        def __init__(self, config):
            self.config = config

        def _load_scene_list(self):
            raise NotImplementedError

        def _process_single_scene(self, scene_id):
            raise NotImplementedError

        def process_all_scenes(self):
            scene_ids = self._load_scene_list()
            results = {}
            for scene_id in scene_ids:
                result = self._process_single_scene(scene_id)
                if result is not None:
                    results[scene_id] = result
            output_dir = Path(self.config.save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / self.config.output_filename
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)


logger = logging.getLogger(__name__)


PLY_TYPE_TO_DTYPE = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


def infer_split_from_scene_list(scene_list_file: str) -> str:
    name = Path(scene_list_file).stem.lower()
    for split in ("train", "val", "test"):
        if split in name:
            return split
    return "unknown"


def load_scene_list(scene_list_file: str) -> List[str]:
    with open(scene_list_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def save_matrix_txt(matrix: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, matrix)


def save_intrinsic_txt(intrinsic: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, intrinsic)


def parse_ply_vertex_properties(ply_path: Path) -> Tuple[np.ndarray, int]:
    vertex_count = None
    vertex_props: List[Tuple[str, str]] = []
    in_vertex = False
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Unsupported PLY property line: {line}")
                _, ply_type, prop_name = parts
                if ply_type not in PLY_TYPE_TO_DTYPE:
                    raise ValueError(f"Unsupported PLY type: {ply_type}")
                vertex_props.append((prop_name, ply_type))
            elif line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"Could not find vertex count in {ply_path}")

        dtype = np.dtype([(name, PLY_TYPE_TO_DTYPE[ply_type]) for name, ply_type in vertex_props])
        vertex_data = np.fromfile(f, dtype=dtype, count=vertex_count)
    return vertex_data, vertex_count


def load_scene_points_and_instances(scene_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    ply_path = scene_dir / "scans" / "mesh_aligned_0.05_semantic.ply"
    segments_path = scene_dir / "scans" / "segments.json"
    segments_anno_path = scene_dir / "scans" / "segments_anno.json"

    vertex_data, vertex_count = parse_ply_vertex_properties(ply_path)
    xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)

    with open(segments_path, "r") as f:
        segments = json.load(f)["segIndices"]
    seg_indices = np.asarray(segments, dtype=np.int32)
    if seg_indices.shape[0] != vertex_count:
        raise ValueError(
            f"segments.json length ({seg_indices.shape[0]}) does not match vertex count ({vertex_count})"
        )

    with open(segments_anno_path, "r") as f:
        seg_ann = json.load(f)

    max_seg = int(seg_indices.max())
    seg_to_instance = np.full(max_seg + 1, -1, dtype=np.int32)
    instance_to_label: Dict[int, str] = {}
    for obj in seg_ann.get("segGroups", []):
        instance_id = int(obj.get("objectId", obj.get("id", -1)))
        if instance_id < 0:
            continue
        label = obj.get("label", "unknown")
        instance_to_label[instance_id] = label
        for seg_id in obj.get("segments", []):
            if 0 <= seg_id <= max_seg:
                seg_to_instance[seg_id] = instance_id

    vertex_instance_ids = seg_to_instance[seg_indices]
    valid_mask = vertex_instance_ids > 0
    return xyz[valid_mask], vertex_instance_ids[valid_mask], instance_to_label


def infer_original_image_size(raw_iphone_dir: Path) -> Tuple[int, int]:
    exif_path = raw_iphone_dir / "exif.json"
    if exif_path.exists():
        with open(exif_path, "r") as f:
            exif = json.load(f)
        if isinstance(exif, dict) and exif:
            first_val = next(iter(exif.values()))
            width = int(first_val.get("PixelXDimension", 1920))
            height = int(first_val.get("PixelYDimension", 1440))
            return width, height
    return 1920, 1440


def scale_intrinsic_matrix(intrinsic: np.ndarray, src_width: int, src_height: int, dst_width: int, dst_height: int) -> np.ndarray:
    scaled = intrinsic.astype(np.float32).copy()
    scaled[0, :] *= float(dst_width) / float(src_width)
    scaled[1, :] *= float(dst_height) / float(src_height)
    return scaled


def depth_bin_extract_selected(
    depth_bin_path: Path,
    target_frame_ids: List[int],
    output_dir: Path,
    target_size: Tuple[int, int],
    original_depth_size: Tuple[int, int] = (192, 256),
) -> Dict[int, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = sorted(set(target_frame_ids))
    target_id_set = set(target_ids)
    target_height, target_width = target_size
    depth_frames: Dict[int, np.ndarray] = {}

    if not depth_bin_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_bin_path}")

    original_height, original_width = original_depth_size

    def finalize_depth(frame_id: int, depth_uint16: np.ndarray) -> None:
        if (depth_uint16.shape[0], depth_uint16.shape[1]) != (target_height, target_width):
            depth_img = Image.fromarray(depth_uint16)
            depth_img = depth_img.resize((target_width, target_height), Image.NEAREST)
            depth_uint16 = np.array(depth_img, dtype=np.uint16)
        out_path = output_dir / f"frame_{frame_id:06d}.png"
        Image.fromarray(depth_uint16).save(out_path)
        depth_frames[frame_id] = depth_uint16

    try:
        with open(depth_bin_path, "rb") as infile:
            compressed = infile.read()
            decompressed = zlib.decompress(compressed, wbits=-zlib.MAX_WBITS)
            all_depth = np.frombuffer(decompressed, dtype=np.float32).reshape(-1, original_height, original_width)
        for frame_id in target_ids:
            if frame_id >= all_depth.shape[0]:
                logger.warning(f"Requested depth frame {frame_id} out of bounds ({all_depth.shape[0]}).")
                continue
            depth_uint16 = (all_depth[frame_id] * 1000.0).astype(np.uint16)
            finalize_depth(frame_id, depth_uint16)
        return depth_frames
    except Exception:
        logger.info("Global zlib depth decompression failed; falling back to per-frame decoding.")

    current_frame_id = 0
    with open(depth_bin_path, "rb") as infile:
        while True:
            size_bytes = infile.read(4)
            if not size_bytes:
                break
            frame_size = int.from_bytes(size_bytes, byteorder="little")
            frame_payload = infile.read(frame_size)
            if current_frame_id not in target_id_set:
                current_frame_id += 1
                continue

            depth_uint16 = None
            try:
                decoded = lz4.block.decompress(frame_payload, uncompressed_size=original_height * original_width * 2)
                depth_uint16 = np.frombuffer(decoded, dtype=np.uint16).reshape(original_height, original_width)
            except Exception:
                try:
                    decoded = zlib.decompress(frame_payload, wbits=-zlib.MAX_WBITS)
                    depth_float = np.frombuffer(decoded, dtype=np.float32).reshape(original_height, original_width)
                    depth_uint16 = (depth_float * 1000.0).astype(np.uint16)
                except Exception as exc:
                    logger.warning(f"Failed to decode depth frame {current_frame_id}: {exc}")

            if depth_uint16 is not None:
                finalize_depth(current_frame_id, depth_uint16)
            current_frame_id += 1
            if len(depth_frames) == len(target_ids):
                break

    return depth_frames


def extract_selected_rgb_frames(
    video_path: Path,
    frame_ids: List[int],
    output_dir: Path,
    target_size: Tuple[int, int],
    backend: str = "auto",
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = sorted(set(frame_ids))
    target_height, target_width = target_size

    tried_backends: List[str] = []

    def _save_rgb(frame_rgb: np.ndarray, frame_id: int) -> None:
        img = Image.fromarray(frame_rgb.astype(np.uint8))
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), Image.BILINEAR)
        img.save(output_dir / f"frame_{frame_id:06d}.jpg", quality=95)

    backend_candidates = ["cv2", "imageio", "ffmpeg"] if backend == "auto" else [backend]

    for candidate in backend_candidates:
        tried_backends.append(candidate)
        if candidate == "cv2":
            try:
                import cv2  # type: ignore
            except Exception:
                continue
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                continue
            try:
                for frame_id in frame_ids:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, frame_bgr = cap.read()
                    if not ok:
                        logger.warning(f"Failed to read RGB frame {frame_id} from {video_path}")
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    _save_rgb(frame_rgb, frame_id)
                return True
            finally:
                cap.release()

        if candidate == "imageio":
            try:
                import imageio.v3 as iio
            except Exception:
                continue
            try:
                for frame_id in frame_ids:
                    frame_rgb = iio.imread(video_path, index=frame_id)
                    _save_rgb(frame_rgb, frame_id)
                return True
            except Exception:
                continue

        if candidate == "ffmpeg":
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin is None:
                continue
            success = True
            for frame_id in frame_ids:
                out_path = output_dir / f"frame_{frame_id:06d}.jpg"
                cmd = [
                    ffmpeg_bin,
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"select=eq(n\\,{frame_id}),scale={target_width}:{target_height}",
                    "-vframes",
                    "1",
                    str(out_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"ffmpeg failed on frame {frame_id}: {result.stderr.strip()}")
                    success = False
            if success:
                return True

    logger.warning(
        "RGB extraction skipped because no usable backend was available. Tried backends: %s",
        ", ".join(tried_backends),
    )
    return False


def project_instances_to_frame(
    points_h: np.ndarray,
    point_instance_ids: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsic: np.ndarray,
    depth_uint16: np.ndarray,
    instance_to_label: Dict[int, str],
    depth_tolerance_ratio: float = 0.2,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    depth_m = depth_uint16.astype(np.float32) / 1000.0
    height, width = depth_m.shape

    world_to_camera = np.linalg.inv(pose_c2w).T
    points_cam = points_h @ world_to_camera
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    positive_z = z > 1e-6
    u = np.rint((x * fx / np.maximum(z, 1e-6)) + cx).astype(np.int32)
    v = np.rint((y * fy / np.maximum(z, 1e-6)) + cy).astype(np.int32)

    valid = positive_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.uint16), []

    valid_indices = np.where(valid)[0]
    u_valid = u[valid_indices]
    v_valid = v[valid_indices]
    z_valid = z[valid_indices]
    inst_valid = point_instance_ids[valid_indices]

    flat_idx = v_valid * width + u_valid
    image_depth = depth_m[v_valid, u_valid]
    depth_ok = (image_depth > 0) & (np.abs(image_depth - z_valid) <= depth_tolerance_ratio * np.maximum(image_depth, 1e-6))
    if not np.any(depth_ok):
        return np.zeros((height, width), dtype=np.uint16), []

    flat_idx = flat_idx[depth_ok]
    z_valid = z_valid[depth_ok]
    inst_valid = inst_valid[depth_ok]

    order = np.lexsort((z_valid, flat_idx))
    flat_idx_sorted = flat_idx[order]
    z_sorted = z_valid[order]
    inst_sorted = inst_valid[order]

    keep = np.ones(flat_idx_sorted.shape[0], dtype=bool)
    keep[1:] = flat_idx_sorted[1:] != flat_idx_sorted[:-1]

    pixel_instance = np.zeros(height * width, dtype=np.uint16)
    pixel_instance[flat_idx_sorted[keep]] = inst_sorted[keep].astype(np.uint16)
    instance_mask = pixel_instance.reshape(height, width)

    bboxes_2d: List[Dict[str, Any]] = []
    for inst_id in np.unique(instance_mask):
        if inst_id <= 0:
            continue
        ys, xs = np.where(instance_mask == inst_id)
        if ys.size == 0:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        bbox_entry: Dict[str, Any] = {
            "instance_id": int(inst_id),
            "bbox_2d": bbox,
        }
        if int(inst_id) in instance_to_label:
            bbox_entry["category_name"] = instance_to_label[int(inst_id)]
        bboxes_2d.append(bbox_entry)

    return instance_mask, bboxes_2d


@dataclass
class ScanNetppIPhoneFrameProcessorConfig(BaseProcessorConfig):
    scene_list_file: str = "data/raw_data/scannetpp/splits/nvs_sem_val.txt"
    raw_data_dir: str = "data/raw_data/scannetpp/data"
    processed_data_dir: str = "data/processed/scannetpp_iphone"
    max_frames: Optional[int] = 32
    img_height: int = 480
    img_width: int = 640
    export_color: bool = True
    video_backend: str = "auto"
    split: str = ""

    def __post_init__(self) -> None:
        if not self.split:
            self.split = infer_split_from_scene_list(self.scene_list_file)
        if self.save_dir == BaseProcessorConfig.save_dir:
            self.save_dir = os.path.join(self.processed_data_dir, "metadata", self.split)
        if self.output_filename == BaseProcessorConfig.output_filename:
            self.output_filename = f"scannetpp_frame_metadata_{self.split}.json"


class ScanNetppIPhoneFrameProcessor(AbstractSceneProcessor):
    def _load_scene_list(self) -> List[str]:
        try:
            return load_scene_list(self.config.scene_list_file)
        except FileNotFoundError:
            logger.error(f"Scene list file not found: {self.config.scene_list_file}")
            return []

    def _select_frame_ids(self, pose_data: Dict[str, Any]) -> List[int]:
        frame_ids = sorted(int(key.split("_")[-1]) for key in pose_data.keys())
        if self.config.max_frames is None or self.config.max_frames <= 0 or len(frame_ids) <= self.config.max_frames:
            return frame_ids
        sampled_idx = np.linspace(0, len(frame_ids) - 1, self.config.max_frames, dtype=int)
        sampled_idx = np.unique(sampled_idx)
        return [frame_ids[i] for i in sampled_idx]

    def _process_single_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        raw_scene_dir = Path(self.config.raw_data_dir) / scene_id
        iphone_dir = raw_scene_dir / "iphone"
        if not iphone_dir.exists():
            logger.error(f"iPhone directory missing for scene {scene_id}: {iphone_dir}")
            return None

        pose_path = iphone_dir / "pose_intrinsic_imu.json"
        depth_bin_path = iphone_dir / "depth.bin"
        rgb_video_path = iphone_dir / "rgb.mkv"

        with open(pose_path, "r") as f:
            pose_data = json.load(f)

        selected_frame_ids = self._select_frame_ids(pose_data)
        target_size = (self.config.img_height, self.config.img_width)
        original_width, original_height = infer_original_image_size(iphone_dir)

        base_out = Path(self.config.processed_data_dir)
        split = self.config.split
        color_out_dir = base_out / "color" / split / scene_id
        depth_out_dir = base_out / "depth" / split / scene_id
        instance_out_dir = base_out / "instance" / split / scene_id
        pose_out_dir = base_out / "pose" / split / scene_id
        intrinsic_out_path = base_out / "intrinsic" / split / f"intrinsics_{scene_id}.txt"

        color_exported = False
        if self.config.export_color:
            color_exported = extract_selected_rgb_frames(
                rgb_video_path,
                selected_frame_ids,
                color_out_dir,
                target_size,
                backend=self.config.video_backend,
            )

        depth_frames = depth_bin_extract_selected(depth_bin_path, selected_frame_ids, depth_out_dir, target_size)
        if not depth_frames:
            logger.error(f"No depth frames extracted for scene {scene_id}.")
            return None

        points_world, point_instance_ids, instance_to_label = load_scene_points_and_instances(raw_scene_dir)
        points_h = np.concatenate(
            [points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)],
            axis=1,
        )

        frame_entries: List[Dict[str, Any]] = []
        representative_intrinsic = None
        for frame_id in selected_frame_ids:
            frame_key = f"frame_{frame_id:06d}"
            if frame_key not in pose_data:
                logger.warning(f"Pose entry missing for scene {scene_id}, frame {frame_key}")
                continue
            if frame_id not in depth_frames:
                logger.warning(f"Depth frame missing after extraction for scene {scene_id}, frame {frame_id}")
                continue

            pose_entry = pose_data[frame_key]
            intrinsic = np.asarray(pose_entry["intrinsic"], dtype=np.float32)
            scaled_intrinsic = scale_intrinsic_matrix(
                intrinsic,
                original_width,
                original_height,
                self.config.img_width,
                self.config.img_height,
            )
            pose_c2w = np.asarray(pose_entry["aligned_pose"], dtype=np.float32)

            if representative_intrinsic is None:
                representative_intrinsic = scaled_intrinsic.copy()
                save_intrinsic_txt(representative_intrinsic, intrinsic_out_path)

            save_matrix_txt(pose_c2w, pose_out_dir / f"frame_{frame_id:06d}.txt")

            instance_mask, bboxes_2d = project_instances_to_frame(
                points_h,
                point_instance_ids,
                pose_c2w,
                scaled_intrinsic,
                depth_frames[frame_id],
                instance_to_label,
            )
            instance_out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(instance_mask.astype(np.uint16)).save(
                instance_out_dir / f"frame_{frame_id:06d}.png"
            )

            frame_entries.append(
                {
                    "frame_id": frame_id,
                    "file_path_color": (
                        os.path.join("color", split, scene_id, f"frame_{frame_id:06d}.jpg")
                        if color_exported
                        else None
                    ),
                    "file_path_depth": os.path.join("depth", split, scene_id, f"frame_{frame_id:06d}.png"),
                    "file_path_instance": os.path.join("instance", split, scene_id, f"frame_{frame_id:06d}.png"),
                    "camera_pose_camera_to_world": pose_c2w.tolist(),
                    "camera_intrinsics": {
                        "fx": float(scaled_intrinsic[0, 0]),
                        "fy": float(scaled_intrinsic[1, 1]),
                        "cx": float(scaled_intrinsic[0, 2]),
                        "cy": float(scaled_intrinsic[1, 2]),
                    },
                    "bboxes_2d": bboxes_2d,
                }
            )

        if representative_intrinsic is None:
            logger.error(f"No valid frames processed for scene {scene_id}.")
            return None

        return {
            "camera_intrinsics": {
                "fx": float(representative_intrinsic[0, 0]),
                "fy": float(representative_intrinsic[1, 1]),
                "cx": float(representative_intrinsic[0, 2]),
                "cy": float(representative_intrinsic[1, 2]),
            },
            "img_width": self.config.img_width,
            "img_height": self.config.img_height,
            "frames": frame_entries,
            "source_video_path": str(rgb_video_path),
            "color_exported": color_exported,
            "split": split,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ScanNet++ iPhone frame-level metadata from raw data.")
    parser.add_argument("--scene_list_file", type=str, required=True, help="Scene list txt file.")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="ScanNet++ raw data root (contains scene folders).")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Output processed data root.")
    parser.add_argument("--save_dir", type=str, default=BaseProcessorConfig.save_dir, help="Metadata output directory.")
    parser.add_argument("--output_filename", type=str, default=BaseProcessorConfig.output_filename, help="Metadata json filename.")
    parser.add_argument("--split", type=str, default="", help="Optional split name override.")
    parser.add_argument("--max_frames", type=int, default=32, help="Uniformly sampled frame count per scene; <=0 means all.")
    parser.add_argument("--img_height", type=int, default=480, help="Output image height.")
    parser.add_argument("--img_width", type=int, default=640, help="Output image width.")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite metadata json if it exists.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--export_color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export sampled RGB frames from rgb.mkv.",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="auto",
        choices=["auto", "cv2", "imageio", "ffmpeg"],
        help="Backend used for RGB frame extraction.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    config = ScanNetppIPhoneFrameProcessorConfig(
        scene_list_file=args.scene_list_file,
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        img_height=args.img_height,
        img_width=args.img_width,
        export_color=args.export_color,
        video_backend=args.video_backend,
        split=args.split,
        save_dir=args.save_dir,
        output_filename=args.output_filename,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
    )
    processor = ScanNetppIPhoneFrameProcessor(config)
    processor.process_all_scenes()


if __name__ == "__main__":
    main()
