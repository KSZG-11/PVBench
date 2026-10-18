"""
StableNormal — 表面法线估计模型封装（真实推理，无 stub）

用途：表面法向量估计
涉及的物理量：R02表面法线, R13光影空间一致性

官方 API (Stable-X/StableNormal, SIGGRAPH Asia 2024, 已查证):
  predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal",
                             trust_repo=True, local_cache_dir="./weights")
  normal_pil = predictor(input_pil)   # 输入/输出均为 PIL.Image
  # 加速版 entrypoint: "StableNormal_turbo" (快 10x)

设计: 加载/推理失败 → 抛异常，evaluator 记 status=error。
"""
import os
import sys
import importlib.util
import numpy as np
import torch
from typing import Optional
from PIL import Image

from .base import BaseSpecialistModel, NormalOutput

# ── 模块级 DINOv2 预加载缓存 ──
_DINOV2_MODEL = None  # 预加载的 DINOv2 模型实例 (跨 StableNormalModel 实例共享)
_DINOV2_LOADED = False  # 是否尝试过加载


def _patch_dinov2_offline():
    """离线预加载 DINOv2 (dinov2_vitl14), 劫持 torch.hub.load 返回预加载实例。

    StableNormal 的 DINOv2_Encoder 硬编码:
      torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    这个调用链会触发:
    1. 下载/校验 facebookresearch/dinov2 源码 (可命中 $TORCH_HOME/hub/ 缓存)
    2. 调用 hubconf.dinov2_vitl14() 创建模型
    3. 内部调用 torch.hub.load_state_dict_from_url(...) 下载权重
       → 在 HF_HUB_OFFLINE=1 且无外网的 GPU 节点上，步骤 3 失败。

    解决方案: 从本地缓存直接构造 DINOv2 模型, 劫持 torch.hub.load,
    对所有 facebookresearch/dinov2 的调用返回预加载实例。

    预置条件:
      $TORCH_HOME/hub/checkpoints/dinov2_vitl14_pretrain.pth  ← 权重
      $TORCH_HOME/hub/facebookresearch_dinov2_main/            ← 源码
      (源码中已打 from __future__ import annotations 补丁以兼容 py3.9)
    """
    global _DINOV2_MODEL, _DINOV2_LOADED
    if _DINOV2_LOADED:
        return  # 已尝试过, 幂等
    _DINOV2_LOADED = True

    _torch_home = os.environ.get(
        "TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    _dinov2_dir = os.path.join(_torch_home, "hub",
                                "facebookresearch_dinov2_main")
    _ckpt = os.path.join(_torch_home, "hub", "checkpoints",
                         "dinov2_vitl14_pretrain.pth")

    if not os.path.isdir(_dinov2_dir):
        print("[StableNormal] DINOv2 源码缓存缺失 "
              f"({_dinov2_dir}), torch.hub.load 可能失败")
        return
    if not os.path.isfile(_ckpt):
        print("[StableNormal] DINOv2 权重缓存缺失 "
              f"({_ckpt}), torch.hub.load 可能失败")
        return

    try:
        # 将 dinov2 源码根加入 sys.path, 使 import dinov2.xxx 可解析
        if _dinov2_dir not in sys.path:
            sys.path.insert(0, _dinov2_dir)

        # 直接 import dinov2 hub backbones (绕过 torch.hub 校验链)
        import dinov2.hub.backbones as _backbones
        model = _backbones.dinov2_vitl14(pretrained=False)
        sd = torch.load(_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        model.eval()
        _DINOV2_MODEL = model

        # 劫持 torch.hub.load: 对 facebookresearch/dinov2 返回预加载模型
        _orig_load = torch.hub.load

        def _intercept_hub_load(repo_or_dir, model_name, *args, **kwargs):
            if str(repo_or_dir) == "facebookresearch/dinov2":
                # 多个 DINOv2_Encoder 实例可共享同一冻结模型
                dev = kwargs.get("device", "cpu")
                _DINOV2_MODEL.to(dev)
                return _DINOV2_MODEL
            return _orig_load(repo_or_dir, model_name, *args, **kwargs)

        torch.hub.load = _intercept_hub_load
        print("[StableNormal] DINOv2 离线预加载 + torch.hub.load 劫持成功")
    except Exception as e:
        print(f"[StableNormal] DINOv2 离线预加载失败 ({e}), "
              f"将回退到 torch.hub.load (需要外网)")
        _DINOV2_MODEL = None


class StableNormalModel(BaseSpecialistModel):
    """StableNormal 封装（真实推理）"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda",
                 use_turbo: bool = False,
                 local_cache_dir: Optional[str] = None,
                 **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path
        self.use_turbo = use_turbo
        # 权重缓存目录: 优先显式指定, 否则用 HF cache 下的子目录
        self.local_cache_dir = local_cache_dir or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "stable_normal",
        )

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": "Stable-X/StableNormal",
            "variant": "turbo" if self.use_turbo else "v0-1",
            "checkpoint": self.model_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def load(self, **kwargs) -> None:
        """加载 StableNormal。失败抛异常。

        策略: 绕过 torch.hub.load (其内部 sys.path/validation 在离线 GPU 节点易失败),
        改为直接 import hubconf.py 调用工厂函数。
        权重从 local_cache_dir/{yoso-normal-v0-3, stable-normal-v0-1} 本地加载,
        不联网。
        """
        entrypoint = "StableNormal_turbo" if self.use_turbo else "StableNormal"
        os.makedirs(self.local_cache_dir, exist_ok=True)

        # ── 0) DINOv2 离线预加载 ──
        # StableNormal pipeline 内部 DINOv2_Encoder 硬编码:
        #   torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        # 该调用会触发 load_state_dict_from_url 联网下载权重。
        # 解决方案: 预加载 DINOv2 模型到内存, 劫持 torch.hub.load 返回预加载实例。
        _patch_dinov2_offline()

        # StableNormal pipeline 内部 DINOv2_Encoder 硬编码 torch.hub.load,
        # 绕过其内部校验逻辑 (api.github.com 403 / 信任检查)
        import torch.hub as _th_hub
        for _fn in ("_validate_not_a_forked_repo", "_check_repo_is_trusted"):
            if hasattr(_th_hub, _fn):
                try:
                    setattr(_th_hub, _fn, lambda *a, **k: None)
                except Exception:
                    pass

        _repo = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "third_party", "StableNormal"))
        if not os.path.isdir(_repo):
            raise RuntimeError(
                f"StableNormal repo 不存在: {_repo}\n"
                f"请执行: git clone https://github.com/Stable-X/StableNormal {_repo}"
            )

        # ── 方案 A: torch.hub.load 本地加载 (优先) ──
        try:
            self._model = torch.hub.load(
                _repo, entrypoint, source="local", trust_repo=True,
                local_cache_dir=self.local_cache_dir, device=self.device,
            )
            print(f"[StableNormal] torch.hub.load 成功 ({entrypoint})")
        except Exception as e_hub:
            print(f"[StableNormal] torch.hub.load 失败 ({e_hub}), "
                  f"回退到直接 import hubconf ...")
            # ── 方案 B: 直接 import hubconf, 绕过 torch.hub 的校验/syspath 管理 ──
            _hub_path = os.path.join(_repo, "hubconf.py")
            if not os.path.isfile(_hub_path):
                raise RuntimeError(
                    f"hubconf.py 不存在: {_hub_path}"
                ) from e_hub
            spec = importlib.util.spec_from_file_location(
                "stable_normal_hubconf", _hub_path)
            hubconf = importlib.util.module_from_spec(spec)
            # 把 repo 根加入 sys.path, 让 hubconf 内 from nirne.xxx 可解析
            if _repo not in sys.path:
                sys.path.insert(0, _repo)
            spec.loader.exec_module(hubconf)
            factory = getattr(hubconf, entrypoint, None)
            if factory is None:
                raise RuntimeError(
                    f"hubconf.py 中未找到 {entrypoint} 函数"
                ) from e_hub
            self._model = factory(
                local_cache_dir=self.local_cache_dir,
                device=self.device,
            )
            print(f"[StableNormal] 直接 import hubconf 成功 ({entrypoint})")

        # torch.hub 返回的 predictor 通常已在正确设备; 尝试 to(device)
        if hasattr(self._model, "to"):
            try:
                self._model = self._model.to(self.device)
            except Exception:
                pass
        self._is_loaded = True
        print(f"[StableNormal] ✓ 模型加载成功 ({entrypoint})")

    def unload(self) -> None:
        self._model = None
        self._is_loaded = False
        torch.cuda.empty_cache()

    def estimate(self, frame: np.ndarray) -> NormalOutput:
        """单帧法向量估计 (R02)。失败抛 RuntimeError。"""
        if not self._is_loaded:
            raise RuntimeError("StableNormal 未加载")

        pil = Image.fromarray(frame.astype(np.uint8)) if isinstance(frame, np.ndarray) else frame
        normal_pil = self._model(pil)  # 官方: 直接传 PIL, 返回 PIL

        # PIL 法线图 [0,255] → [-1,1] 向量
        normal_np = np.asarray(normal_pil).astype(np.float32)
        if normal_np.ndim == 2:  # 灰度异常, 补维
            normal_np = np.stack([normal_np] * 3, axis=-1)
        normal_vec = normal_np[..., :3] / 255.0 * 2.0 - 1.0
        # 归一化
        mag = np.linalg.norm(normal_vec, axis=-1, keepdims=True) + 1e-8
        normal_vec = normal_vec / mag
        return NormalOutput(normal_map=normal_vec, confidence=None)

    def estimate_batch(self, frames: np.ndarray) -> list:
        """批量法向量估计 (R13)。逐帧调用。"""
        return [self.estimate(frames[t]) for t in range(len(frames))]
