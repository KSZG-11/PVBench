"""
DA3Metric-Large — 度量深度估计模型封装（真实推理，无 stub）

用途：从单帧估计米制深度（绝对尺度）
涉及的物理量：R04绝对尺度

官方 API (depth-anything/DA3METRIC-LARGE, ByteDance-Seed, 已查证):
  from depth_anything_3.api import DepthAnything3
  model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device)
  pred = model.inference(image_paths)   # 传路径列表
  # pred.depth[N,H,W], pred.conf, pred.intrinsics[N,3,3], pred.extrinsics[N,3,4]
  # METRIC 模型需换算: focal=(K[0,0]+K[1,1])/2; metric = focal * depth / 300.0 (米)

设计: 加载或推理失败 → 抛异常，evaluator 记 status=error。
"""
import os
import tempfile
import numpy as np
import torch
from typing import Optional
from PIL import Image

from .base import BaseSpecialistModel, DepthOutput


class DA3Model(BaseSpecialistModel):
    """DA3Metric-Large 封装（真实推理）"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda", **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path or "depth-anything/DA3METRIC-LARGE"

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": "depth-anything/DA3METRIC-LARGE",
            "checkpoint": self.model_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def load(self, **kwargs) -> None:
        """加载 DA3Metric-Large。失败抛 ImportError/RuntimeError。"""
        from depth_anything_3.api import DepthAnything3
        from .base import resolve_local_model

        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        # 离线优先: 解析本地权重目录 (魔搭 flat: hf_cache/models/depth-anything--DA3METRIC-LARGE)
        src = resolve_local_model(self.model_path, cache_dir)
        is_local = src != self.model_path
        load_kwargs = {}
        if cache_dir and not is_local:
            load_kwargs["cache_dir"] = cache_dir
        if is_local:
            print(f"[DA3] 使用本地权重目录: {src}")
        self._model = DepthAnything3.from_pretrained(src, **load_kwargs)
        self._model = self._model.to(device=self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        self._is_loaded = True
        print(f"[DA3Metric-Large] ✓ 模型加载成功 ({self.model_path})")

    def unload(self) -> None:
        self._model = None
        self._is_loaded = False
        torch.cuda.empty_cache()

    def estimate_metric_depth(self, frame: np.ndarray) -> DepthOutput:
        """度量深度估计 (R04)。返回米制深度图。失败抛 RuntimeError。"""
        if not self._is_loaded:
            raise RuntimeError("DA3 未加载")

        # DA3 官方 inference 接受图片路径列表
        tmp_dir = tempfile.mkdtemp(prefix="da3_")
        path = os.path.join(tmp_dir, "frame.png")
        try:
            Image.fromarray(frame.astype(np.uint8)).save(path)
            pred = self._model.inference([path])
        finally:
            try:
                os.remove(path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

        depth = np.asarray(pred.depth)          # [N,H,W] 或 [H,W]
        conf = np.asarray(pred.conf) if getattr(pred, "conf", None) is not None else None

        d = (depth[0] if depth.ndim == 3 else depth).astype(np.float64)
        c = (conf[0] if conf is not None and conf.ndim == 3 else conf)

        # ── 米制换算 ──────────────────────────────────────────────
        # DA3 的官方度量公式 (apply_metric_scaling): metric = depth * focal / 300,
        # 即度量分支的原始深度需乘 (focal/300) 才是米。两种情况:
        #  1) is_metric 真 (如 da3nested-*): forward 内部已完成对齐+缩放, depth 已是米, 直接用。
        #  2) 否则 (如 da3metric-large 独立使用: DPT 纯深度头, 无相机解码器):
        #     pred.intrinsics 为 None → 用相机内参焦距; 若缺失则按 ~60° 水平 FOV
        #     估计焦距 (focal ≈ 0.866 * max(H,W)), 再套 focal/300。
        is_metric = bool(getattr(pred, "is_metric", 0))
        if is_metric:
            metric_depth = d
        else:
            intr = getattr(pred, "intrinsics", None)
            if intr is not None:
                intr = np.asarray(intr)
                K = intr[0] if intr.ndim == 3 else intr
                focal = (float(K[0, 0]) + float(K[1, 1])) / 2.0
            else:
                Hh, Ww = d.shape[-2], d.shape[-1]
                focal = 0.866 * float(max(Hh, Ww))   # 60° FOV 默认焦距(像素)
            metric_depth = focal * d / 300.0

        # ⚠️ DA3 内部按自身分辨率(如 504x504)输出深度, 与输入帧尺寸(如 256x256)不一致。
        #   下游(scale_pose)会用同尺寸的 SAM mask 做 depth_map[mask] 布尔索引 → 维度不匹配报错。
        #   故把深度(和置信度)缩放回输入帧的 (H,W), 保证所有消费方尺寸一致。
        Hin, Win = int(frame.shape[0]), int(frame.shape[1])
        if metric_depth.shape[-2:] != (Hin, Win):
            import cv2
            metric_depth = cv2.resize(metric_depth.astype(np.float32), (Win, Hin),
                                      interpolation=cv2.INTER_LINEAR)
            if c is not None:
                c = cv2.resize(np.asarray(c).astype(np.float32), (Win, Hin),
                               interpolation=cv2.INTER_LINEAR)

        # 显式释放 DA3 输出对象 (可能持有 GPU 中间张量)
        try:
            del pred
        except (NameError, UnboundLocalError):
            pass
        import torch as _torch
        _torch.cuda.empty_cache()

        return DepthOutput(
            depth_map=metric_depth.astype(np.float32),
            metric_depth=float(np.median(metric_depth[metric_depth > 0])) if np.any(metric_depth > 0) else None,
            confidence=c.astype(np.float32) if c is not None else None,
        )
