"""
FlowSeek — 光流估计模型封装（真实推理，无 stub）

用途：稠密光流估计
涉及的物理量：R19跨帧几何一致性

官方 API (mattpoggi/flowseek, ICCV 2025, 已查证):
  FlowSeek(nn.Module + PyTorchModelHubMixin)。forward(image1, image2, test_mode=True)
  输入 [N,3,H,W] 像素 [0,255]，返回 dict {'final':[N,2,H,W] flow, ...}。
  内含冻结 DepthAnythingV2。需 git clone third_party/flowseek。

设计: 加载/推理失败 → 抛异常, evaluator 记 status=error。
"""
import os
import sys
import numpy as np
import torch
from typing import Optional
from .base import BaseSpecialistModel, OpticalFlowOutput


class FlowSeekModel(BaseSpecialistModel):
    """FlowSeek 封装（真实推理）"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda", hf_cache_dir: Optional[str] = None,
                 iters: int = 12, **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path
        self.hf_cache_dir = hf_cache_dir
        self.iters = iters

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": "mattpoggi/flowseek (ICCV 2025)",
            "checkpoint": self.model_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def load(self, **kwargs) -> None:
        """加载 FlowSeek (离线, 复刻官方 demo.py: json→args→FlowSeek(args)→load_ckpt)。失败抛异常。

        关键约束 (源码写死):
          - core/flowseek.py 用顶层 import (from depth_anything_v2..., from update...):
            需把 flowseek/core 与 flowseek 根都加进 sys.path。
          - FlowSeek.__init__ 里 torch.load('weights/depth_anything_v2_%s.pth'%da_size) 是
            相对 CWD 路径: 构造期须 chdir 到 flowseek 根。
          - 主权重用 flowseek_M_* (da_size=vitb, 与本地 DA2 vitb 权重匹配)。
        """
        import json
        import types

        _third_party = os.path.join(os.path.dirname(__file__), "..", "third_party")
        _fs_root = os.path.abspath(os.path.join(_third_party, "flowseek"))
        _fs_core = os.path.join(_fs_root, "core")
        if not os.path.isdir(_fs_root):
            raise FileNotFoundError(f"flowseek 源码目录不存在: {_fs_root}")
        for p in (_fs_core, _fs_root):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        # ── 主权重 (默认 M/TartanCT_TSKH, 泛化最好; 逐个回退) ──
        weights_dir = os.path.join(_fs_root, "weights")
        if self.model_path and os.path.isfile(self.model_path):
            main_ckpt = self.model_path
        else:
            cand = ["flowseek_M_TartanCT_TSKH.pth", "flowseek_M_TartanCT.pth",
                    "flowseek_M_Tartan.pth", "flowseek_M_TartanC.pth", "flowseek_M_CT.pth"]
            main_ckpt = next((os.path.join(weights_dir, c) for c in cand
                              if os.path.isfile(os.path.join(weights_dir, c))), None)
        if not main_ckpt or not os.path.isfile(main_ckpt):
            raise FileNotFoundError(f"未找到 FlowSeek 主权重于 {weights_dir}")

        # ── config → args (M: da_size=vitb, dim=128) ──
        cfg_path = os.path.join(_fs_root, "config", "eval", "flowseek-M.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        args = types.SimpleNamespace(**cfg)

        from core.flowseek import FlowSeek  # flowseek/core/flowseek.py

        _cwd = os.getcwd()
        try:
            os.chdir(_fs_root)  # __init__ 内 torch.load('weights/...') 依赖 CWD
            model = FlowSeek(args)
            state = torch.load(main_ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if any(str(k).startswith("module.") for k in state):
                state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        finally:
            os.chdir(_cwd)

        self._model = model.to(self.device).eval()
        self.iters = getattr(args, "iters", self.iters)
        self._is_loaded = True
        print(f"[FlowSeek] ✓ 模型加载成功 ({os.path.basename(main_ckpt)}, da_size={args.da_size})")

    def unload(self) -> None:
        self._model = None
        self._is_loaded = False
        torch.cuda.empty_cache()

    def compute_flow(self, frames: np.ndarray) -> OpticalFlowOutput:
        """相邻帧前向光流 (R19)。frames [T,H,W,3] uint8。失败抛 RuntimeError。"""
        if not self._is_loaded:
            raise RuntimeError("FlowSeek 未加载")
        T = len(frames)
        if T < 2:
            raise RuntimeError("光流至少需要 2 帧")

        flows = []
        for t in range(T - 1):
            img1 = torch.from_numpy(frames[t]).permute(2, 0, 1).float()[None].to(self.device)
            img2 = torch.from_numpy(frames[t + 1]).permute(2, 0, 1).float()[None].to(self.device)
            with torch.no_grad():
                out = self._model(img1, img2, iters=self.iters, test_mode=True)
            flow = out["final"] if isinstance(out, dict) else out
            flows.append(flow.squeeze(0).permute(1, 2, 0).float().cpu().numpy())  # [H,W,2]
            # 显式释放 GPU 中间张量 (逐帧光流累积显存)
            if isinstance(out, dict):
                for _k in list(out.keys()):
                    try:
                        del out[_k]
                    except (KeyError, TypeError):
                        pass
            del out, flow, img1, img2
        torch.cuda.empty_cache()
        flow_fwd = np.stack(flows)  # [T-1,H,W,2]
        return OpticalFlowOutput(flow_fwd=flow_fwd)
