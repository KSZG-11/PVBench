"""
L2 画质层 — 参考图保真度评测器（全局 Benchmark，I2V/CI2V 专有）

从原 L1 IF-10 迁移，IF-10b（物体身份保真）已合并入 FQ-1（主体一致性）：
  IF-10a 场景结构保真: DINOv3 ref→首帧 特征相似度
  IF-10c 色彩风格延续: DINOv3 Gram 矩阵风格距离 (降级: 直方图 KL 散度)
  IF-10d 角色面部一致性: DINOv3 ref→视频帧 精细特征追踪 (CI2V专有)

评测方式: DINOv3 + CLIP（纯专用模型，不依赖 VLM）
I2V/CI2V 触发；带 `t2v_canonical` 角色的 T2V 使用代理参考图，其余 T2V 跳过。
"""
from typing import Optional
import numpy as np
from PIL import Image
from ..base import BaseEvaluator, EvalResult, LayerResult


REFERENCE_FIDELITY_DIMENSIONS = [
    ("IF-10a", "场景结构保真",
     "参考图→视频首帧的场景结构是否保持（构图/物体位置/背景布局）"),
    ("IF-10c", "色彩风格延续",
     "参考图的色调/光照/纹理风格是否在视频中自然延续，无突变"),
    ("IF-10d", "角色面部一致性",
     "(CI2V专有) 角色面部特征/体型/服饰是否跨帧保持稳定"),
]


class ReferenceFidelityEvaluator(BaseEvaluator):
    """
    参考条件保真度底层评测（I2V/CI2V专有，IF-10a/c/d由报告层聚合）

    IF-10b（物体身份保真）已合并入 FQ-1（主体一致性），不再独立评测。
    模型需求:
        - DINOv3: 场景结构保真 (IF-10a), 角色面部一致性 (IF-10d),
                  色彩风格延续 (IF-10c, 多层 patch token Gram 矩阵风格距离)
    输入: 参考图 + 视频帧  →  逐子指标评测  →  聚合
    """

    def __init__(self, specialist, config: Optional[dict] = None):
        super().__init__("L2-Quality/ReferenceFidelity", config)
        self.specialist = specialist

    def evaluate_global(self,
                        videos: list[list[Image.Image]],
                        ref_images: list[Optional[Image.Image]],
                        modalities: Optional[list[str]] = None,
                        reference_roles: Optional[list[str]] = None
                        ) -> LayerResult:
        """
        全局评测：对全部 (ref_image, video_frames) 对评测后聚合

        Args:
            videos: [[frame1, frame2, ...], ...] 每个视频的帧列表
            ref_images: [ref_image_or_None, ...]  None → T2V条目, 有图 → I2V/CI2V

        Returns:
            LayerResult 含聚合统计
        """
        all_results = []
        per_dim_scores = {dim_id: [] for dim_id, _, _ in REFERENCE_FIDELITY_DIMENSIONS}

        modalities = modalities or ["I2V"] * len(videos)
        reference_roles = reference_roles or ["model_condition"] * len(videos)
        for idx, (frames, ref_img, modality, reference_role) in enumerate(
                zip(videos, ref_images, modalities, reference_roles)):
            # 解包 (帧, dims) 元组 (FQ/TQ 惯例), 取帧部分并取维度标注;
            # 防止 dims 字符串被当图像传入 processor。
            dims = None
            if (isinstance(frames, tuple) and len(frames) == 2
                    and isinstance(frames[0], (list, tuple))):
                frames, dims = list(frames[0]), frames[1]
            # 按标注过滤 (与 FQ/TQ 口径一致): 清单标注了维度时, 仅评显式
            # 标注 IF-10 的条目; dims=None (无标注场景) 则全部评。
            if dims is not None and "IF-10" not in (dims or []):
                continue
            modality_name = str(modality).upper()
            is_t2v_canonical = (
                modality_name == "T2V"
                and reference_role == "t2v_canonical"
            )
            if (ref_img is None or len(frames) < 2
                    or (modality_name == "T2V" and not is_t2v_canonical)):
                continue

            # IF-10a: 场景结构保真
            score, info = self._scene_structure_fidelity(ref_img, frames[0])
            result_status = "proxy" if is_t2v_canonical else "ok"
            per_dim_scores["IF-10a"].append(score)
            all_results.append(EvalResult(
                f"IF-10a-场景结构保真 (item {idx})", score,
                [] if score >= 4 else [info[:100]],
                metrics={"item": idx, "info": info[:200],
                         "reference_role": reference_role,
                         "reference_is_gt": reference_role == "external_gt"},
                status=result_status,
            ))

            # IF-10c: 色彩风格延续
            score, info = self._color_style_continuity(ref_img, frames)
            per_dim_scores["IF-10c"].append(score)
            all_results.append(EvalResult(
                f"IF-10c-色彩风格延续 (item {idx})", score,
                [] if score >= 4 else [info[:100]],
                metrics={"item": idx, "info": info[:200],
                         "reference_role": reference_role,
                         "reference_is_gt": reference_role == "external_gt"},
                status=result_status,
            ))

            # IF-10d 只对显式 CI2V 样本执行；普通 I2V 不含角色条件。
            if modality_name == "CI2V":
                score, info = self._character_face_consistency(ref_img, frames)
                per_dim_scores["IF-10d"].append(score)
                all_results.append(EvalResult(
                    f"IF-10d-角色面部一致性 (item {idx})", score,
                    [] if score >= 4 else [info[:100]],
                    metrics={"item": idx, "info": info[:200],
                             "reference_role": reference_role,
                             "reference_is_gt": reference_role == "external_gt"},
                    status=result_status,
                ))

        # 聚合
        return self._aggregate(per_dim_scores, all_results)

    def evaluate(self, frames: list[Image.Image],
                 prompt: str = "",
                 ref_image: Optional[Image.Image] = None) -> list[EvalResult]:
        """单案例评测（兼容旧接口）"""
        results = []
        if ref_image is None or len(frames) < 2:
            return results
        score, info = self._scene_structure_fidelity(ref_image, frames[0])
        results.append(EvalResult("IF-10a-场景结构保真", score,
                      [] if score >= 4 else [info[:100]]))
        score, info = self._color_style_continuity(ref_image, frames)
        results.append(EvalResult("IF-10c-色彩风格延续", score,
                      [] if score >= 4 else [info[:100]]))
        return results

    # ── 子指标实现 ──

    def _scene_structure_fidelity(self, ref: Image.Image,
                                   first_frame: Image.Image
                                   ) -> tuple[int, str]:
        """IF-10a: 参考图→视频首帧 结构保真度"""
        try:
            sim = self.specialist.image_similarity(ref, first_frame)
            if sim > 0.95:
                score = 5
            elif sim > 0.88:
                score = 4
            elif sim > 0.75:
                score = 3
            elif sim > 0.55:
                score = 2
            else:
                score = 1
            return score, f"DINOv3 sim={sim:.3f}"
        except Exception as e:
            return 3, f"scene_structure_fidelity failed: {e}"

    def _color_style_continuity(self, ref: Image.Image,
                                 frames: list[Image.Image]
                                 ) -> tuple[int, str]:
        """IF-10c: 色彩风格延续 (DINOv3 Gram 矩阵风格距离)

        优先使用 DINOv3 Gram 矩阵 (Neural Style Transfer 的 Gram 思想,
        骨干为 DINOv3 自监督 ViT 特征, blocks 2/4/6/8/10/12):
            - patch tokens 通道间二阶统计量捕获纹理/色彩风格, 与空间布局无关
            - Gram 按 F-范数归一化 → 距离约 [0, √2], 阈值跨场景稳定

        降级方案: 64-bin RGB 直方图 KL 散度 (色彩分布比较)
        """
        # ── 优先 DINOv3 Gram 矩阵风格距离 ──
        try:
            gram_dists = []
            for frame in self._sample_frames(frames, 4):
                d = self.specialist.style_gram_distance(ref, frame)
                if d >= 0:  # sentinel < 0 表示不可用
                    gram_dists.append(d)

            if gram_dists:
                mean_gram = float(np.mean(gram_dists))
                # 归一化 Gram 距离: 越小越相似 (0=完全一致, √2≈1.41 上界)。
                # 分档已由 scripts/calibrate_if10c_gram.py 在 I2V(ref=精确开场帧)
                # 分布上校准 (2026-09-11, n=40): 同视频对 p50=0.029/p90=0.052,
                # 交叉对 p10=0.104, 分布间隔 2x。模态/数据分布变化时应重新校准。
                if mean_gram < 0.025:
                    score = 5
                elif mean_gram < 0.038:
                    score = 4
                elif mean_gram < 0.052:
                    score = 3
                elif mean_gram < 0.104:
                    score = 2
                else:
                    score = 1
                return score, f"dino_gram_dist={mean_gram:.3f}"
        except Exception as e:
            print(f"[IF-10c] DINOv3 Gram 失败: {e}, 降级为直方图 KL")

        # ── 降级: 64-bin RGB 直方图 KL 散度 ──
        try:
            ref_arr = np.array(ref.resize((256, 256)))
            kls = []
            for frame in self._sample_frames(frames, 4):
                f_arr = np.array(frame.resize((256, 256)))
                kl_sum = 0
                for c in range(3):
                    hist_ref = np.histogram(ref_arr[:, :, c], bins=64, range=(0, 255))[0].astype(float)
                    hist_f = np.histogram(f_arr[:, :, c], bins=64, range=(0, 255))[0].astype(float)
                    hist_ref = hist_ref / (hist_ref.sum() + 1)
                    hist_f = hist_f / (hist_f.sum() + 1)
                    kl = np.sum(hist_ref * np.log((hist_ref + 1e-8) / (hist_f + 1e-8)))
                    kl_sum += kl
                kls.append(kl_sum / 3)

            mean_kl = float(np.mean(kls))
            if mean_kl < 0.15:
                score = 5
            elif mean_kl < 0.35:
                score = 4
            elif mean_kl < 0.6:
                score = 3
            elif mean_kl < 1.0:
                score = 2
            else:
                score = 1
            return score, f"color_KL={mean_kl:.3f}"
        except Exception as e:
            return 3, f"color_continuity failed: {e}"

    def _character_face_consistency(self, ref: Image.Image,
                                     frames: list[Image.Image]
                                     ) -> tuple[int, str]:
        """IF-10d: 角色面部一致性 (CI2V专有) — DINO 精细特征稳定性"""
        try:
            sims = []
            for frame in self._sample_frames(frames, 6):
                sim = self.specialist.image_similarity(ref, frame)
                sims.append(sim)

            avg_sim = float(np.mean(sims))
            if avg_sim > 0.92:
                score = 5
            elif avg_sim > 0.85:
                score = 4
            elif avg_sim > 0.72:
                score = 3
            elif avg_sim > 0.55:
                score = 2
            else:
                score = 1
            return score, f"face_avg_sim={avg_sim:.3f}"
        except Exception as e:
            return 3, f"face_consistency failed: {e}"

    # ── 辅助 ──

    @staticmethod
    def _sample_frames(frames, n):
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]

    @staticmethod
    def _aggregate(per_dim, results) -> LayerResult:
        distribution = lambda s: {
            "1": s.count(1), "2": s.count(2),
            "3": s.count(3), "4": s.count(4), "5": s.count(5)
        }
        mean_scores = {}
        aggregates = {}
        for dim, scores in per_dim.items():
            if scores:
                mean_scores[f"{dim}-参考图保真"] = float(np.mean(scores))
                aggregates[f"{dim}_distribution"] = distribution(scores)

        return LayerResult(
            layer="IF-10-参考条件保真度-底层子指标",
            results=results,
            aggregates=aggregates,
            mean_scores=mean_scores,
        )
