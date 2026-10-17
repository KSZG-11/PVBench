"""
帧采样器 — 从视频中提取关键帧
"""
import os
from typing import Optional
import numpy as np
from PIL import Image
import cv2


class FrameSampler:
    """视频帧采样器"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.max_frames = self.config.get("max_frames", 16)
        self.uniform = self.config.get("uniform", True)
        self.event_detection = self.config.get("event_detection", False)

    def sample_from_video(self, video_path: str) -> tuple[list[Image.Image], float]:
        """
        从视频文件采样帧

        Args:
            video_path: 视频文件路径

        Returns:
            (PIL Image 列表, 采样序列的等效 fps)。均匀抽帧时等效 fps 会按
            原始帧间隔缩放，保证速度、加速度、角速度等时间导数量纲正确。
        """
        if not os.path.exists(video_path):
            print(f"[Sampler] 视频文件不存在: {video_path}")
            return [], 24.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Sampler] 无法打开视频: {video_path}")
            return [], 24.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else 24.0
        if total_frames <= 0:
            cap.release()
            return [], fps

        # 确定采样帧索引
        n_sample = min(self.max_frames, total_frames)
        if self.uniform:
            indices = np.linspace(0, total_frames - 1, n_sample, dtype=int)
        else:
            # 等间隔（保留首尾）
            if n_sample >= total_frames:
                indices = np.arange(total_frames)
            else:
                step = total_frames / n_sample
                indices = np.array([int(i * step) for i in range(n_sample)])

        # ⚠️ 性能优化：顺序读取（一次遍历），遇到目标帧索引才保留。
        #    原实现用 cap.set(CAP_PROP_POS_FRAMES) 逐帧随机 seek，对 mp4/H.264
        #    每次 seek 都要解码器跳帧定位，32 帧采样慢 10-20 倍（日志 @3.1fps）。
        target = set(int(i) for i in indices)
        frames = []
        loaded_indices = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in target:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                loaded_indices.append(idx)
            idx += 1

        cap.release()

        # 如果启用了事件检测，额外采样关键事件帧
        if self.event_detection and len(frames) >= 4:
            keyframes = self._detect_event_frames(frames)
            # 合并去重
            existing_hashes = {self._frame_hash(f) for f in frames}
            for kf in keyframes:
                if self._frame_hash(kf) not in existing_hashes:
                    frames.append(kf)

        if len(loaded_indices) >= 2:
            mean_stride = float(np.mean(np.diff(loaded_indices)))
            effective_fps = fps / max(mean_stride, 1.0)
        else:
            effective_fps = fps
        self.last_sample_indices = loaded_indices
        self.last_raw_fps = fps
        self.last_effective_fps = effective_fps
        return frames, effective_fps

    def sample_all_from_video(self, video_path: str) -> tuple[list[Image.Image], float]:
        """
        加载视频全部帧（不做降采样），R21 物体恒存等需要完整时序覆盖的场景使用。

        Args:
            video_path: 视频文件路径

        Returns:
            (PIL Image 列表（全部帧）, 原始视频 fps)；若 fps 不可读默认返回 24.0
        """
        if not os.path.exists(video_path):
            print(f"[Sampler] 视频文件不存在: {video_path}")
            return [], 24.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Sampler] 无法打开视频: {video_path}")
            return [], 24.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and raw_fps > 0 else 24.0
        if total_frames <= 0:
            cap.release()
            return [], fps

        frames = []
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames, fps

    def sample_from_directory(self, image_dir: str,
                              extensions: tuple = (".png", ".jpg", ".jpeg")
                              ) -> list[Image.Image]:
        """从图片目录采样（用于已有帧序列）"""
        images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ])
        if not images:
            return []

        total = len(images)
        n_sample = min(self.max_frames, total)
        indices = np.linspace(0, total - 1, n_sample, dtype=int)

        frames = []
        for idx in indices:
            path = os.path.join(image_dir, images[idx])
            frames.append(Image.open(path).convert("RGB"))

        return frames

    @staticmethod
    def _detect_event_frames(frames: list[Image.Image]) -> list[Image.Image]:
        """基于帧间变化检测关键事件帧"""
        if len(frames) < 3:
            return []

        keyframes = []
        prev = np.array(frames[0].convert("L"), dtype=np.float32)

        for i in range(1, len(frames)):
            curr = np.array(frames[i].convert("L"), dtype=np.float32)
            diff = np.abs(curr - prev).mean()

            # 变化超过阈值 -> 可能发生事件
            if diff > 15:
                keyframes.append(frames[i])

            prev = curr

        return keyframes

    @staticmethod
    def _frame_hash(frame: Image.Image) -> int:
        """帧的简单哈希（用于去重）"""
        arr = np.array(frame.resize((32, 32)).convert("L"))
        return hash(arr.tobytes())
