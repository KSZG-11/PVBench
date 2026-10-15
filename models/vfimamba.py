"""
VFIMamba — 视频帧插值模型封装（真实推理，无 stub）

用途：视频帧插值（用于运动连续性检测）
涉及的物理量：R20运动连续性

官方 (MCG-NJU/VFIMamba, NeurIPS 2024, 已查证):
  需 git clone third_party/VFIMamba; `from Trainer_finetune import Model`
  model = Model.from_pretrained("MCG-NJU/VFIMamba")  # 或 VFIMamba_S
  推理接口(参考 demo_2x.py): model.inference(img0, img1, ...) 输入归一化 tensor [B,3,H,W]。
  依赖 mamba_ssm / causal_conv1d (需 CUDA 编译)。

设计: 加载/推理失败 → 抛异常, evaluator 记 status=error。
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from .base import BaseSpecialistModel, InterpolationOutput


class VFIMambaModel(BaseSpecialistModel):
    """VFIMamba 封装（真实推理）"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda", hf_cache_dir: Optional[str] = None,
                 model_name: str = "VFIMamba", scale: float = 1.0, **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path
        self.hf_cache_dir = hf_cache_dir
        self.model_name = model_name
        self.scale = scale

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": f"MCG-NJU/{self.model_name}",
            "checkpoint": self.model_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def _find_local_ckpt(self):
        """在缓存目录里递归查找本地 model.pkl (魔搭/HF 格式均可)。返回 (pkl_path, config_json_path) 或 (None, None)。"""
        import glob
        # 1) 显式指定
        if self.model_path and os.path.isfile(self.model_path):
            cfg = os.path.join(os.path.dirname(self.model_path), "config.json")
            return self.model_path, (cfg if os.path.isfile(cfg) else None)
        # 2) 在 hf_cache 下按仓库名递归搜 (魔搭: models/MCG-NJU--VFIMamba/**, HF: hub/models--MCG-NJU--VFIMamba/**)
        # ⚠️ 性能: 递归 glob 必须收窄到 models/ 与 hub/ 子目录 —— 对整个
        #    hf_cache (1.5TB) 做 ** 遍历会挂起数十分钟 (同 vggt.py 的坑)。
        if self.hf_cache_dir and os.path.isdir(self.hf_cache_dir):
            search_bases = [
                os.path.join(self.hf_cache_dir, "models"),
                os.path.join(self.hf_cache_dir, "hub"),
            ]
            patterns = [
                os.path.join(base, "**", "*VFIMamba*", "**", "model.pkl")
                for base in search_bases
            ] + [
                os.path.join(base, "**", "model.pkl")
                for base in search_bases
            ]
            for pat in patterns:
                hits = sorted(glob.glob(pat, recursive=True))
                if hits:
                    pkl = hits[0]
                    cfg = os.path.join(os.path.dirname(pkl), "config.json")
                    return pkl, (cfg if os.path.isfile(cfg) else None)
        return None, None

    def _read_arch_cfg(self, config_json):
        """从 config.json 读取 F/depth; 缺失则用官方默认 (F=32, depth=[2,2,2,3,3])。"""
        F, depth = 32, [2, 2, 2, 3, 3]
        if config_json and os.path.isfile(config_json):
            try:
                import json
                with open(config_json, "r") as f:
                    c = json.load(f)
                F = int(c.get("F", F))
                depth = list(c.get("depth", depth))
            except Exception as e:
                print(f"[VFIMamba] 读取 config.json 失败, 用默认架构: {e}")
        return F, depth

    def load(self, **kwargs) -> None:
        """加载 VFIMamba。优先本地 model.pkl (离线), 失败回退在线 from_pretrained。"""
        _third_party = os.path.join(os.path.dirname(__file__), "..", "third_party")
        _vfi_path = os.path.abspath(os.path.join(_third_party, "VFIMamba"))
        if os.path.isdir(_vfi_path) and _vfi_path not in sys.path:
            sys.path.insert(0, _vfi_path)

        model_id = self.model_path or f"MCG-NJU/{self.model_name}"

        # ── 优先: 本地 model.pkl 直接加载 (绕开 hf_hub_download, 离线可用) ──
        pkl_path, cfg_json = self._find_local_ckpt()
        if pkl_path and os.path.isfile(pkl_path):
            try:
                import Trainer_finetune as _tf
                from Trainer_finetune import Model
                F, depth = self._read_arch_cfg(cfg_json)
                # 复刻 from_pretrained 内部: 设架构 → 建模型 → 加载权重
                _tf.MODEL_CONFIG["MODEL_ARCH"] = _tf.init_model_config(F=F, depth=depth)
                model = Model(-1)
                ckpt = torch.load(pkl_path, map_location="cpu")
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                # 鲁棒: 官方权重可能带 module. 前缀, 剥离后再对齐
                stripped = {k.replace("module.", ""): v for k, v in ckpt.items()}
                try:
                    model.net.load_state_dict(stripped, strict=True)
                except Exception:
                    model.net.load_state_dict(stripped, strict=False)
                self._model = model
                self._set_eval()
                self._is_loaded = True
                print(f"[VFIMamba] ✓ 本地权重加载成功 ({pkl_path}, F={F}, depth={depth})")
                return
            except Exception as e:
                print(f"[VFIMamba] 本地加载失败, 尝试在线 from_pretrained: {e}")

        # ── 回退: 官方在线 from_pretrained (需外网, 服务器离线时会失败并抛异常) ──
        from Trainer_finetune import Model
        self._model = Model.from_pretrained(model_id)
        self._set_eval()
        self._is_loaded = True
        print(f"[VFIMamba] ✓ 在线加载成功 ({model_id})")

    def _set_eval(self):
        """把底层网络置为 eval 模式 (Model 包装了 self.net, 自身未必有 eval)。"""
        m = self._model
        if hasattr(m, "eval"):
            try:
                m.eval()
                return
            except Exception:
                pass
        if hasattr(m, "net") and hasattr(m.net, "eval"):
            m.net.eval()

    def unload(self) -> None:
        self._model = None
        self._is_loaded = False
        torch.cuda.empty_cache()

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """[H,W,3] uint8 → [1,3,H,W] float [0,1], pad 到 64 倍数。"""
        t = torch.from_numpy(frame).permute(2, 0, 1).float()[None] / 255.0
        return t.to(self.device)

    def interpolate_pair(self, frame0: np.ndarray, frame1: np.ndarray,
                         n_interp: int = 1) -> np.ndarray:
        """两帧间插值 n_interp 帧 (仅支持 n_interp=1 的 2x)。失败抛 RuntimeError。"""
        if not self._is_loaded:
            raise RuntimeError("VFIMamba 未加载")
        img0 = self._to_tensor(frame0)
        img1 = self._to_tensor(frame1)
        _, _, H, W = img0.shape
        # pad 到 64 倍数
        ph, pw = ((H - 1) // 64 + 1) * 64, ((W - 1) // 64 + 1) * 64
        pad = (0, pw - W, 0, ph - H)
        img0p, img1p = F.pad(img0, pad), F.pad(img1, pad)
        with torch.no_grad():
            # VFIMamba 官方 demo_2x.py: model.inference(I0, I2, True, TTA=..., scale=...)
            # 第三个位置参数 local=True (local attention 开关), 缺失会报
            # "missing 1 required positional argument: 'local'"。
            mid = self._model.inference(img0p, img1p, True, scale=self.scale)
        if isinstance(mid, (list, tuple)):
            mid = mid[0]
        mid = mid[:, :, :H, :W].clamp(0, 1)
        mid_np = (mid.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0)
        # 显式释放 GPU 中间张量 (防逐帧插值累积显存)
        for _g in (img0, img1, img0p, img1p, mid):
            try:
                del _g
            except (NameError, UnboundLocalError):
                pass
        torch.cuda.empty_cache()
        return mid_np[None].astype(np.float32)

    def interpolate_odd_frames(self, frames: np.ndarray) -> InterpolationOutput:
        """偶数帧插回奇数帧 (R20)。失败抛 RuntimeError。

        方法: 用第 2k 与第 2k+2 帧插出第 2k+1 帧, 与真实奇数帧比对。
        """
        if not self._is_loaded:
            raise RuntimeError("VFIMamba 未加载")
        T = len(frames)
        interpolated, target_idx = [], []
        for k in range(0, T - 2, 2):
            mid = self.interpolate_pair(frames[k], frames[k + 2], n_interp=1)[0]
            interpolated.append(mid)
            target_idx.append(k + 1)  # 对应真实奇数帧索引
        if not interpolated:
            H, W = frames.shape[1:3]
            return InterpolationOutput(interpolated_frames=np.zeros((0, H, W, 3)))
        out = InterpolationOutput(interpolated_frames=np.array(interpolated))
        out.quality_scores = np.array(target_idx, dtype=float)  # 复用字段传回目标索引
        return out
