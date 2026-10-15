"""
画质层专用模型封装 — 帧级质量和时序质量

模型清单:
    - DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)  主体一致性
    - CLIP ViT-L/14 (openai/clip-vit-large-patch14)      背景一致性/图文对齐
    - FineVQ (IntMeGroup/FineVQ_score)                   美学质量 → models/finevq.py 独立封装

DINO 加载策略:
    仅使用 DINOv3 (最新模型), 加载失败抛异常, 不降级。
"""
from typing import Optional
import os
import torch
import numpy as np
from PIL import Image
import cv2


# DINO 模型配置 (可通过 config 覆盖)
DEFAULT_DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"


# 复用 base.py 的完整 resolve_local_model (它处理 flat 目录 + snapshots 嵌套 + HF snapshot
# 三种格式; 本地残留的残缺 _resolve_local_model 只处理 flat 顶层, 会漏掉
# hf_cache/models/org--name/snapshots/* 嵌套格式, 导致 DINOv3/CLIP 离线找不到权重)。
def _resolve_local_model(model_id: str, cache_dir: Optional[str]) -> str:
    from .base import resolve_local_model
    return resolve_local_model(model_id, cache_dir)


class SpecialistModels:
    """画质评测专用模型集合

    模型清单:
        - DINOv3 (主体一致性, 跨帧特征相似度)
        - CLIP ViT-L/14 (背景一致性, 图文对齐)
    """

    def __init__(self, device: Optional[str] = None,
                 dino_model_id: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_model_id = dino_model_id or DEFAULT_DINO_MODEL
        self._loaded = False

        # DINO 模型缓存 (避免每次调用重新加载)
        self._dino_model = None
        self._dino_processor = None
        self._dino_actual_id: Optional[str] = None  # 实际成功加载的模型ID

        # CLIP 模型缓存 (避免每次调用重新加载)
        self._clip_model = None
        self._clip_processor = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def metadata(self) -> dict:
        """模型 provenance (与 BaseSpecialistModel 统一接口对齐)。"""
        return {
            "class": type(self).__name__,
            "dino_model_id": self.dino_model_id,
            "dino_actual_id": self._dino_actual_id,
            "clip_model": "openai/clip-vit-large-patch14",
            "device": self.device,
            "is_loaded": self._loaded,
        }

    def load(self, **kwargs) -> None:
        """加载画质模型 (统一接口; 实际 DINO/CLIP 懒加载)。"""
        self.load_all()

    def load_all(self) -> None:
        """预加载所有画质模型 (保留旧名, 供 loader.get_specialist 调用)。"""
        self._loaded = True
        print(f"[Specialist] 模型准备就绪 (device={self.device}, DINO={self.dino_model_id})")

    def to_cpu(self) -> None:
        """画质模型多为懒加载, 无长驻大模型, 默认 no-op。"""
        return None

    def to_cuda(self) -> None:
        return None

    def unload(self) -> None:
        self._loaded = False
        self._dino_model = None
        self._dino_processor = None
        self._clip_model = None
        self._clip_processor = None
        torch.cuda.empty_cache()

    # ── DINO 模型加载 (仅 DINOv3, 失败抛异常) ──
    def _get_dino_model(self):
        """
        获取 DINO 模型和 processor (懒加载 + 缓存)。

        仅使用 DINOv3 (最新模型), 加载失败抛异常, 由调用方记 status=error。
        不再有 DINOv2 降级路径。

        Returns:
            (model, processor)
        """
        if self._dino_model is not None:
            return self._dino_model, self._dino_processor

        from transformers import AutoImageProcessor, AutoModel

        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")

        model_id = self.dino_model_id
        print(f"[Specialist] 加载 DINO 模型: {model_id} ...")
        load_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        src = _resolve_local_model(model_id, cache_dir)  # 离线优先本地目录
        model = AutoModel.from_pretrained(src, **load_kwargs).to(self.device)
        processor = AutoImageProcessor.from_pretrained(src, **load_kwargs)
        self._dino_model = model
        self._dino_processor = processor
        self._dino_actual_id = model_id
        print(f"[Specialist] DINO ✓ ({model_id})")
        return model, processor

    # ── 成像质量 ──
    @staticmethod
    def imaging_quality(image: Image.Image) -> dict:
        """锐度 + 曝光合理性"""
        img = np.array(image.convert("L"), dtype=np.float32)
        lap = cv2.Laplacian(img.astype(np.uint8), cv2.CV_64F)
        sharp_norm = np.clip(np.var(lap) / 1000.0, 0, 1)

        mean_brightness = np.mean(img) / 255.0
        if mean_brightness > 0.85:
            exposure = max(0, 1 - (mean_brightness - 0.85) * 10)
        elif mean_brightness < 0.15:
            exposure = max(0, mean_brightness / 0.15)
        else:
            exposure = 1.0

        return {"sharpness": float(sharp_norm), "exposure": float(exposure)}

    # ── 主体一致性 (DINOv3 跨帧特征相似度) ──
    def image_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        两帧 DINOv3 特征余弦相似度 (用于 IF-10 参考图保真度)。
        失败抛异常 (不再像素级回退)。
        """
        model, processor = self._get_dino_model()

        import torch.nn.functional as F

        inputs1 = processor(images=img1, return_tensors="pt").to(self.device)
        inputs2 = processor(images=img2, return_tensors="pt").to(self.device)

        with torch.no_grad():
            feat1 = model(**inputs1).last_hidden_state.mean(dim=1)
            feat2 = model(**inputs2).last_hidden_state.mean(dim=1)

        cos_sim = F.cosine_similarity(feat1, feat2, dim=1).item()
        return float(cos_sim)

    def _get_clip_model(self):
        """获取 CLIP 模型和 processor (懒加载 + 缓存)。

        加载失败抛异常 (不再返回 None 降级)。
        """
        if self._clip_model is not None:
            return self._clip_model, self._clip_processor

        from transformers import CLIPProcessor, CLIPModel

        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        load_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        _clip_src = _resolve_local_model("openai/clip-vit-large-patch14", cache_dir)

        model = CLIPModel.from_pretrained(_clip_src, **load_kwargs).to(self.device).eval()
        processor = CLIPProcessor.from_pretrained(_clip_src, **load_kwargs)

        self._clip_model = model
        self._clip_processor = processor
        print(f"[Specialist] CLIP ✓ (openai/clip-vit-large-patch14)")
        return model, processor

    def image_text_similarity(self, text: str, image: Image.Image) -> float:
        """
        CLIP 图文对齐分数 (用于 IF-1 辅助参考分)。

        使用标准方法: 分别提取 image features 和 text features (L2归一化),
        计算余弦相似度。缓存 CLIP 模型避免重复加载。
        失败抛异常 (不再返回 sentinel)。

        Returns:
            float in [-1, 1]
        """
        import torch.nn.functional as F

        model, processor = self._get_clip_model()

        # ── 方法: 分别提取 image/text features 再计算余弦相似度 ──
        # 这比依赖 logits_per_image (含温度参数) 更可靠、更可解释
        img_inputs = processor(images=image, return_tensors="pt").to(self.device)
        txt_inputs = processor(text=[text], return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(self.device)

        with torch.no_grad():
            img_feat = model.get_image_features(**img_inputs)  # [1, dim], L2-normalized
            txt_feat = model.get_text_features(**txt_inputs)   # [1, dim], L2-normalized
        # transformers 5.x: get_xxx_features 可能返回 BaseModelOutputWithPooling
        # 对象而非 Tensor → cosine_similarity TypeError (TQ-4 全 3 的根因)。
        img_feat = getattr(img_feat, "pooler_output", img_feat)
        txt_feat = getattr(txt_feat, "pooler_output", txt_feat)

        # 余弦相似度 (features 已经是 L2-normalized, 直接点积即可)
        cos_sim = F.cosine_similarity(img_feat, txt_feat, dim=1).item()
        return float(cos_sim)

    def subject_consistency(self, frames: list[Image.Image]) -> float:
        """相邻帧 DINOv3 特征余弦相似度均值。失败抛异常 (不再像素级回退)。"""
        if len(frames) < 2:
            return 1.0

        model, processor = self._get_dino_model()

        import torch.nn.functional as F

        features = []
        for frame in frames:
            inputs = processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = model(**inputs).last_hidden_state.mean(dim=1)
            features.append(feat)

        similarities = []
        for i in range(len(features) - 1):
            cos_sim = F.cosine_similarity(features[i], features[i+1], dim=1)
            similarities.append(cos_sim.item())

        return float(np.mean(similarities)) if similarities else 1.0

    # ── 背景一致性 (CLIP 跨帧特征相似度) ──
    def background_consistency(self, frames: list[Image.Image]) -> float:
        """相邻帧 CLIP 图像特征余弦相似度均值 (使用缓存的CLIP模型)。
        失败抛异常 (不再回退 DINO)。"""
        if len(frames) < 2:
            return 1.0

        model, processor = self._get_clip_model()

        features = []
        for frame in frames:
            inputs = processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
            # transformers 5.x: 可能返回 BaseModelOutputWithPooling → 取 pooler_output
            feat = getattr(feat, "pooler_output", feat)
            features.append(feat)

        similarities = []
        for i in range(len(features) - 1):
            cos_sim = torch.nn.functional.cosine_similarity(
                features[i], features[i+1], dim=1
            )
            similarities.append(cos_sim.item())

        return float(np.mean(similarities)) if similarities else 1.0

    # ── DINOv3 Gram 矩阵风格距离 ──
    # 风格层: ViT-B 共 12 个 block, 浅层(纹理/色彩统计) + 中层 + 深层(抽象风格)。
    # hidden_states[0]=embedding 输出, hidden_states[i]=第 i 个 block 输出。
    _DINO_STYLE_BLOCKS = (2, 4, 6, 8, 10, 12)

    def _dino_gram_matrices(self, img: Image.Image) -> list[torch.Tensor]:
        """提取单张图 DINOv3 多层 patch token 的归一化 Gram 矩阵。

        1. DINOv3 forward (output_hidden_states=True), 取选定 block 输出
        2. 去 CLS token: tokens [N, C] → G = F^T·F / N  [C, C]
           (通道间二阶统计量, 捕获纹理/色彩风格, 与空间布局无关)
        3. G 按 F-范数归一化: 消除不同骨干/层间的特征幅值量纲, 使层间距离
           落到 [0, √2] 稳定区间, 分档阈值可跨场景复用。
        """
        model, processor = self._get_dino_model()
        inputs = processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hs = model(**inputs, output_hidden_states=True).hidden_states
        grams = []
        for b in self._DINO_STYLE_BLOCKS:
            if b >= len(hs):
                continue
            tokens = hs[b][0, 1:, :].float()            # [N, C] 去 CLS
            gram = torch.mm(tokens.t(), tokens) / tokens.shape[0]
            grams.append(gram / (gram.norm(p="fro") + 1e-8))
        return grams

    def style_gram_distance(self, img1: Image.Image,
                            img2: Image.Image) -> float:
        """DINOv3 Gram 矩阵风格距离 (Neural Style Transfer 的 Gram 思想,
        骨干从 VGG19(2014) 升级为 DINOv3(2025) 自监督 ViT 特征)。

        对两张图像:
        1. DINOv3 提取多层 block 特征 (blocks 2/4/6/8/10/12)
        2. 每层 patch tokens 计算 F-范数归一化 Gram 矩阵
        3. 逐层 Frobenius 距离 → 均值

        Returns:
            float: 风格距离, 越小越相似, 归一化后约 [0, √2]; 同图=0。
            DINOv3 加载失败抛异常 (调用方走降级链)。
        """
        # 缓存: IF-10c 对同一 ref 连续调多次(每采样帧一次), ref 的 Gram 只算一次
        g1 = self._gram_cached(img1)
        g2 = self._gram_cached(img2)
        n = min(len(g1), len(g2))
        if n == 0:
            raise RuntimeError("DINOv3 未返回可用的隐藏层特征")
        dists = [float((g1[i] - g2[i]).norm(p="fro").item()) for i in range(n)]
        return float(np.mean(dists))

    def _gram_cached(self, img: Image.Image) -> list[torch.Tensor]:
        key = id(img)
        cache = getattr(self, "_gram_cache", None)
        if cache and cache[0] == key:
            return cache[1]
        grams = self._dino_gram_matrices(img)
        self._gram_cache = (key, grams)
        return grams

    # ── 时序闪烁 ──
    @staticmethod
    def temporal_flickering(frames: list[Image.Image]) -> float:
        """相邻帧静态区域像素差异"""
        if len(frames) < 2:
            return 0.0

        flicker_values = []
        for i in range(len(frames) - 1):
            f1 = np.array(frames[i].convert("L"), dtype=np.float32)
            f2 = np.array(frames[i+1].convert("L"), dtype=np.float32)
            diff = np.abs(f1 - f2).mean() / 255.0
            flicker_values.append(diff)

        return float(np.mean(flicker_values))

    # ── 动态度 ──
    @staticmethod
    def dynamic_degree(frames: list[Image.Image]) -> float:
        """帧间光流幅度"""
        if len(frames) < 2:
            return 0.0

        flow_magnitudes = []
        for i in range(len(frames) - 1):
            f1 = np.array(frames[i].convert("L"))
            f2 = np.array(frames[i+1].convert("L"))
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
            flow_magnitudes.append(mag)

        avg_mag = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
        return min(avg_mag / 5.0, 1.0)

    # ── 运动平滑度 ──
    @staticmethod
    def motion_smoothness(frames: list[Image.Image]) -> float:
        """光流加速度 → 平滑度"""
        if len(frames) < 3:
            return 1.0

        flow_mags = []
        for i in range(len(frames) - 1):
            f1 = np.array(frames[i].convert("L"))
            f2 = np.array(frames[i+1].convert("L"))
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
            flow_mags.append(mag)

        accelerations = [abs(flow_mags[i+1] - flow_mags[i])
                        for i in range(len(flow_mags) - 1)]
        avg_acc = float(np.mean(accelerations))
        return max(0, 1 - avg_acc / 2.0)
