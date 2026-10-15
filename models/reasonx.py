"""
ReasonX — MLLM 感知裁判 + 光影分析（真实推理，无 stub）

用途：光源方向/光影一致性分析
涉及的物理量：R13光影空间一致性, R18光影时间一致性

官方 (adobe-research/ReasonX, CVPR 2026, 已查证):
  权重: adirik/InternVL2_5-4B-Intrinsic-Judge (完整权重 7.44GB, 非 LoRA, base InternVL2.5-4B)
  git clone https://github.com/adobe-research/ReasonX  (提供 infer.py: load_image/draw_points)
  model = AutoModel.from_pretrained(id, torch_dtype=bfloat16, use_flash_attn=True,
                                    trust_remote_code=True).eval().cuda()
  tokenizer = AutoTokenizer.from_pretrained(id, trust_remote_code=True, use_fast=False)
  question = "<image>\nWhich point appears to be closer to the camera - red or green?"
  response = model.chat(tokenizer, pixel_values, question, dict(max_new_tokens=128, do_sample=False))
  # 对红/绿标记点做相对本征比较: 深度/法线/反照率albedo/辐照度irradiance

评测器需要"逐帧主光方向"以判断光影一致性。ReasonX 本身输出的是点对相对判断,
本封装用 ReasonX 判断多组点对的相对辐照度(irradiance), 结合法线, 反演每帧主光方向。

设计: Judge 加载失败抛异常 (不再降级经典 Lambertian)。经典 Lambertian 最小二乘
      仅作为 Judge 点对约束不足时的算法内数值兜底 (非模型降级)。
"""
import os
import sys
import numpy as np
from typing import Optional
from .base import BaseSpecialistModel, ShadingOutput


class ReasonXModel(BaseSpecialistModel):
    """ReasonX 光影分析 (InternVL2.5 Intrinsic Judge + 光度学反演)"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda",
                 model_id: str = "adirik/InternVL2_5-4B-Intrinsic-Judge",
                 use_judge: bool = True, **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path
        self.model_id = model_path or model_id
        self.use_judge = use_judge
        self._tokenizer = None
        self._judge_available = False

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": self.model_id,
            "checkpoint": self.model_path,
            "use_judge": self.use_judge,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def load(self, **kwargs) -> None:
        """加载 InternVL2.5 Intrinsic Judge。失败抛异常 (不再降级经典光度学)。"""
        import torch
        from transformers import AutoModel, AutoTokenizer

        # ── transformers 兼容补丁: InternVLChatModel 只有 _tied_weights_keys,
        #     但 transformers 4.57+ _move_missing_keys_from_meta_to_device 会
        #     访问 self.all_tied_weights_keys (不带下划线) → AttributeError。
        #     公共补丁 (与 FineVQ 共用, 幂等)。
        from .base import patch_tied_weights_keys_compat
        patch_tied_weights_keys_compat()

        # ReasonX 仓库提供 infer.load_image; 加 third_party 路径
        _rx = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "ReasonX"))
        if os.path.isdir(_rx) and _rx not in sys.path:
            sys.path.insert(0, _rx)

        from .base import resolve_local_model
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        # 离线优先: 解析本地权重目录 (ReasonX 权重多为 HF snapshot 格式)
        src = resolve_local_model(self.model_id, cache_dir)
        is_local = src != self.model_id
        load_kw = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if cache_dir and not is_local:
            load_kw["cache_dir"] = cache_dir
        if is_local:
            load_kw["local_files_only"] = True
            # ⚠️ trust_remote_code 加载自定义建模文件时, transformers 仍会尝试
            # 联网核对代码 revision → 离线报 OSError。强制离线环境变量彻底关闭联网。
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            print(f"[ReasonX] 使用本地权重目录: {src}")
        try:
            self._model = AutoModel.from_pretrained(src, use_flash_attn=True, **load_kw)
        except Exception:
            # flash-attn 不可用时退回 (仅去掉 flash-attn 加速, 模型本身仍加载)
            load_kw.pop("use_flash_attn", None)
            self._model = AutoModel.from_pretrained(src, **load_kw)
        self._model = self._model.eval().to(self.device)
        tok_kw = {"trust_remote_code": True, "use_fast": False}
        if cache_dir and not is_local:
            tok_kw["cache_dir"] = cache_dir
        if is_local:
            tok_kw["local_files_only"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(src, **tok_kw)
        self._judge_available = True
        self._is_loaded = True
        print(f"[ReasonX] ✓ InternVL2.5 Intrinsic Judge 加载成功 ({self.model_id})")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._judge_available = False
        self._is_loaded = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 经典 Lambertian 光度学反演 (算法内数值兜底: Judge 点对约束不足/无法线时) ──
    @staticmethod
    def _estimate_light_direction(frame: np.ndarray,
                                  normal: Optional[np.ndarray]) -> np.ndarray:
        gray = np.mean(frame.astype(np.float32), axis=-1) / 255.0
        if normal is None:
            gy, gx = np.gradient(gray)
            L = np.array([np.mean(gx), np.mean(gy), 1.0])
            return L / (np.linalg.norm(L) + 1e-8)
        N = normal.reshape(-1, 3)
        I = gray.reshape(-1)
        valid = np.linalg.norm(N, axis=1) > 0.5
        N, I = N[valid], I[valid]
        if len(I) < 10:
            return np.array([0.0, 0.0, 1.0])
        L, *_ = np.linalg.lstsq(N, I, rcond=None)
        return L / (np.linalg.norm(L) + 1e-8)

    # ── ReasonX Judge: 基于点对辐照度比较反演主光方向 ──
    _FIB_SPHERE_CACHE = None  # 类级缓存，避免重复计算

    @classmethod
    def _fibonacci_upper_hemi(cls, n: int = 300) -> np.ndarray:
        """Fibonacci 球面上半球采样 [M,3] (确定性，z>0)"""
        if cls._FIB_SPHERE_CACHE is not None and len(cls._FIB_SPHERE_CACHE) >= n:
            return cls._FIB_SPHERE_CACHE[:n]
        indices = np.arange(n * 2)  # 全球采样再取上半
        phi = np.arccos(1 - 2 * (indices + 0.5) / (n * 2))
        theta = np.pi * (1 + np.sqrt(5)) * indices
        sphere = np.stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ], axis=-1)
        upper = sphere[sphere[:, 2] > 0]
        cls._FIB_SPHERE_CACHE = upper
        return upper[:n]

    def _estimate_light_direction_judge(self, frame: np.ndarray,
                                         normal_map: np.ndarray,
                                         max_pairs: int = 6) -> np.ndarray:
        """用 ReasonX Judge 比较点对辐照度，反演单帧主光方向。

        策略：在图像上采样的点中选取法线差异最大的 4 点，
        对 6 对组合逐一询问 Judge 相对辐照度，构建不等式约束
        L·(N_i - N_j) > 0，再用 Fibonacci 半球搜索找最优 L。
        约束不足时降级经典 Lambertian (算法内数值兜底)。
        """
        import torch

        H, W = frame.shape[:2]
        n_flat = normal_map.reshape(-1, 3)
        valid_mask = np.linalg.norm(n_flat, axis=1) > 0.5
        if valid_mask.sum() < 10:
            return self._estimate_light_direction(frame, normal_map)

        # 4×4 网格采样
        grid_size = 4
        candidates = []
        for row in np.linspace(int(H * 0.1), int(H * 0.85), grid_size):
            for col in np.linspace(int(W * 0.1), int(W * 0.85), grid_size):
                idx = int(row) * W + int(col)
                if valid_mask[idx]:
                    candidates.append((int(col), int(row), n_flat[idx].copy()))
        if len(candidates) < 4:
            return self._estimate_light_direction(frame, normal_map)

        # 贪心选 4 个法线最分散的点
        if len(candidates) > 4:
            norms_arr = np.array([c[2] for c in candidates])
            selected = [0]
            for _ in range(3):
                best = max(
                    (i for i in range(len(candidates)) if i not in selected),
                    key=lambda i: min(
                        np.linalg.norm(norms_arr[i] - norms_arr[s])
                        for s in selected
                    )
                )
                selected.append(best)
            candidates = [candidates[i] for i in selected]

        # 6 对抗对
        constraints = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                xi, yi, ni = candidates[i]
                xj, yj, nj = candidates[j]
                n_diff = ni - nj
                if np.linalg.norm(n_diff) < 0.25:
                    continue
                try:
                    answer = self.compare_points(frame, (xi, yi), (xj, yj),
                                                 modality="irradiance")
                    ans_lower = answer.lower()
                    red_hit = "red" in ans_lower
                    green_hit = "green" in ans_lower
                    if red_hit and not green_hit:
                        constraints.append(n_diff)   # L·N_i > L·N_j
                    elif green_hit and not red_hit:
                        constraints.append(-n_diff)  # L·N_j > L·N_i
                    # 同时提到 red+green → 歧义，跳过
                except Exception:
                    continue

        if len(constraints) < 2:
            return self._estimate_light_direction(frame, normal_map)

        # Fibonacci 半球搜索最优 L
        sphere = self._fibonacci_upper_hemi(300)
        cons_arr = np.array(constraints)
        dots = np.dot(cons_arr, sphere.T)  # [n_cons, n_candidates]
        satisfied = (dots > 0).sum(axis=0)
        best_idx = int(np.argmax(satisfied))
        best_L = sphere[best_idx]
        best_cnt = int(satisfied[best_idx])
        print(f"[ReasonX Judge] 光源方向 约束={len(constraints)} "
              f"满足={best_cnt}/{len(constraints)} "
              f"L=({best_L[0]:.3f},{best_L[1]:.3f},{best_L[2]:.3f})")
        return best_L

    def analyze_spatial_lighting(self, frames: np.ndarray,
                                 normals: Optional[np.ndarray] = None,
                                 depth: Optional[np.ndarray] = None) -> ShadingOutput:
        """空间光影 (R13)。逐帧主光方向。

        Judge 可用时：基于点对辐照度比较反演 → 数据驱动。
        (无法线时: 经典 Lambertian 最小二乘作为数值兜底)。
        """
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]
        T = frames.shape[0]
        dirs = np.zeros((T, 3))
        use_judge = self._judge_available and normals is not None
        for t in range(T):
            n_t = normals[t] if normals is not None and t < len(normals) else None
            if use_judge and n_t is not None:
                dirs[t] = self._estimate_light_direction_judge(frames[t], n_t)
            else:
                dirs[t] = self._estimate_light_direction(frames[t], n_t)
        return ShadingOutput(shading_map=None, light_directions=dirs)

    def analyze_temporal_lighting(self, frames: np.ndarray,
                                  shading: Optional[np.ndarray] = None,
                                  normals: Optional[np.ndarray] = None) -> ShadingOutput:
        """时间光影 (R18)。逐帧主光方向序列。

        Judge 可用且提供 normals 时：点对辐照度比较反演。
        否则：经典灰度梯度近似。
        """
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]
        T = frames.shape[0]
        use_judge = self._judge_available and normals is not None
        dirs = []
        for t in range(T):
            n_t = normals[t] if normals is not None and t < len(normals) else None
            if use_judge and n_t is not None:
                dirs.append(self._estimate_light_direction_judge(frames[t], n_t))
            else:
                # 有法线用 Lambertian LSQ (优于纯灰度梯度 z=1.0); 无法线退回梯度近似
                dirs.append(self._estimate_light_direction(frames[t], n_t))
        return ShadingOutput(shading_map=None, light_directions=np.array(dirs))

    # ── ReasonX Judge: 红/绿点对相对本征比较 (真实模型) ──
    def compare_points(self, frame: np.ndarray, red_xy, green_xy,
                       modality: str = "irradiance") -> str:
        """用 InternVL Judge 比较红/绿两点的相对本征属性。返回自然语言判断。

        modality: "depth"(谁更近) / "irradiance"(谁更亮) / "normals" / "albedo"(材质)
        未加载 Judge 时抛 RuntimeError。
        """
        if not self._judge_available:
            raise RuntimeError("ReasonX Judge 未加载")
        import torch
        from infer import load_image, draw_points  # ReasonX 仓库

        annotated = draw_points(frame.copy(), red_xy, green_xy)
        pixel_values = load_image(annotated).to(torch.bfloat16).to(self.device)
        q_map = {
            "depth": "Which point appears to be closer to the camera - red or green?",
            "irradiance": "Which point is more illuminated - red or green?",
            "normals": "Do the red and green points face the same direction?",
            "albedo": "Are the red and green points the same material?",
        }
        question = "<image>\n" + q_map.get(modality, q_map["irradiance"])
        return self._model.chat(self._tokenizer, pixel_values, question,
                                dict(max_new_tokens=128, do_sample=False))
