import os
import re
from typing import List, Tuple, Dict, Any, Optional

def get_frame_number_from_filename(filename: str) -> Optional[int]:
    """
    从文件名中提取帧编号。

    启发式方法：
    1. 获取文件名（不含扩展名）。
    2. 在基本文件名中查找所有数字序列。
    3. 假设找到的最后一个数字序列是帧标识符。
    4. 将此数字序列转换为整数。

    例如:
    - "DSC01752.JPG" -> 1752
    - "000000.jpg" -> 0
    - "scene0000_00_123.jpg" -> 123
    - "frame.001.jpg" -> 1

    如果找不到数字或无法转换，则返回None。
    """
    basename = os.path.splitext(filename)[0]
    numbers = re.findall(r'\d+', basename)
    
    if not numbers:
        return None
    
    try:
        # 假设最后一个数字是帧号
        return int(numbers[-1])
    except ValueError:
        return None

def calculate_frame_timestamps(filenames: List[str], fps: float) -> Tuple[List[Dict[str, Any]], float]:
    """
    根据文件名列表和原始视频的FPS，计算每帧的采样时间及总采样时长。

    假设：
    1. FPS 是原始视频的帧率。
    2. 文件名中包含可提取的、代表原始视频中顺序的帧编号。
    3. 提供的文件名列表是“均匀采样”的结果，意味着采样的时间点在总时长内均匀分布。

    参数:
    - filenames (List[str]): 图片文件名的列表。
    - fps (float): 原始视频的每秒帧数 (FPS)。

    返回:
    - Tuple[List[Dict[str, Any]], float]:
        - 一个字典列表，每个字典包含：
            - 'original_filename' (str): 原始文件名。
            - 'frame_number' (int): 从文件名中提取的帧编号。
            - 'timestamp' (float): 该帧的采样时间（秒），相对于序列中的第一帧。
          此列表按帧编号（时间顺序）排序。
        - 总采样时长（秒），即从第一个采样点到最后一个采样点的时间跨度。
    
    异常:
    - ValueError: 如果 fps <= 0。
    """
    if fps <= 0:
        raise ValueError("FPS 必须是正数。")

    if not filenames:
        return [], 0.0

    extracted_frames: List[Dict[str, Any]] = []
    for fname in filenames:
        frame_num = get_frame_number_from_filename(fname)
        if frame_num is not None:
            extracted_frames.append({'original_filename': fname, 'frame_number': frame_num})
        else:
            print(f"警告：无法从文件名 '{fname}' 中提取帧编号。已跳过此文件。")

    if not extracted_frames:
        return [], 0.0

    # 按提取的帧编号排序
    sorted_frames = sorted(extracted_frames, key=lambda x: x['frame_number'])

    num_sampled_frames = len(sorted_frames)

    if num_sampled_frames == 1:
        # 如果只有一个采样帧，其时间戳为0，总时长也为0
        sorted_frames[0]['timestamp'] = 0.0
        return sorted_frames, 0.0

    # 获取第一个和最后一个采样帧的原始编号
    n_first = sorted_frames[0]['frame_number']
    n_last = sorted_frames[-1]['frame_number']

    # 计算这些采样帧在原始视频中覆盖的总时长
    # 如果 n_last == n_first (所有采样帧都对应同一个原始帧号)，则 duration_orig_span 为 0
    duration_orig_span = (n_last - n_first) / fps

    # 计算均匀采样的时间间隔
    # 如果 duration_orig_span 为 0 (例如所有采样帧来自同一个原始帧), 则 delta_t_sample 为 0
    if duration_orig_span == 0:
        delta_t_sample = 0.0
    else:
        # num_sampled_frames 保证大于1, 所以 num_sampled_frames - 1 >= 1
        delta_t_sample = duration_orig_span / (num_sampled_frames - 1)

    # 为每个采样帧分配时间戳
    for i, frame_info in enumerate(sorted_frames):
        frame_info['timestamp'] = i * delta_t_sample

    # 总采样时长即为 duration_orig_span
    total_sequence_time = duration_orig_span
    
    # 也可以通过 (num_sampled_frames - 1) * delta_t_sample 计算，结果一致（处理浮点精度问题）
    if num_sampled_frames > 1:
        total_sequence_time = (num_sampled_frames - 1) * delta_t_sample


    return sorted_frames, total_sequence_time