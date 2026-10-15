"""
L2 画质层 — 帧级质量评测器（全局 Benchmark）

FQ-1 主体一致性: DINOv3 跨帧特征相似度
FQ-2 美学质量: FineVQ (CVPR 2025, InternVL2-8B + 回归头, 视频质量评分)

评测方式: 专用模型(非VLM)逐视频评测 → 跨全部视频聚合
"""
from typing import Optional
import numpy as np
from PIL import Image
from ..base import BaseEvaluator, EvalResult, LayerResult


def _finevq_to_score(aesthetic: float) -> int:
    """FineVQ raw score1 → 1-5 整数分 (分位校准映射)。

    FineVQ 的 score1 是未归一化回归输出 (官方按 SRCC/PLCC 秩相关评估,
    不保证 1-5 区间), 旧 clip(round(x),1,5) 映射在其输出分布 (~15-33)
    上失效全 5。本分档经 scripts/calibrate_fq2_finevq.py 在 MiniMax-H3
    分布上校准 (2026-09-15, n=60: p10=17.7 p25=20.6 p50=22.6 p75=27.1),
    各档约 25%; 跨数据分布使用时应重新校准 (metrics 保留 raw 值可溯)。
    """
    if aesthetic >= 27.077:
        return 5
    if aesthetic >= 22.638:
        return 4
    if aesthetic >= 20.632:
        return 3
    if aesthetic >= 17.720:
        return 2
    return 1


class FrameQualityEvaluator(BaseEvaluator):
    """
    帧级质量 全局 Benchmark

    模型需求:
        - DINOv3: 主体一致性 (FQ-1)
        - FineVQ: 美学质量 (FQ-2, 输出连续 MOS 分数 → round 映射 1-5)
    """

    def __init__(self, specialist, config: Optional[dict] = None,
                 finevq=None):
        super().__init__("L2-Quality/FrameQuality", config)
        self.specialist = specialist  # SpecialistModels (DINOv3)
        self.finevq = finevq          # FineVQModel (FQ-2 美学质量)

    def evaluate_global(self,
                        videos: list
                        ) -> LayerResult:
        """
        全局评测：对全部生成视频评测帧级质量

        Args:
            videos: [[frame1, frame2, ...], ...] 每个视频的帧列表；
                    或 [(frames, dims), ...] 带维度标注 (按 dims 过滤 FQ-1/FQ-2)

        Returns:
            LayerResult 含聚合统计
        """
        all_results = []
        fq1_scores = []
        fq2_scores = []

        for idx, item in enumerate(videos):
            # 兼容: 纯帧列表 或 (frames, dims) 对
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], list) and not isinstance(item[0], Image.Image):
                frames, dims = item[0], item[1]
            else:
                frames, dims = item, None
            if len(frames) < 2:
                continue

            # FQ-1: 主体一致性 (按维度标注过滤)
            if dims is None or "FQ-1" in dims:
                try:
                    subject_cons = self.specialist.subject_consistency(frames)
                    # DINO 相似度 → 1-5分 (用阈值表, 在 0.85~0.99 区间细分)
                    from ..thresholds import score_metric
                    fq1_score = score_metric("FQ1_subject_cons", subject_cons)
                except Exception as e:
                    subject_cons = 0.0
                    fq1_score = 1

                fq1_scores.append(fq1_score)
                all_results.append(EvalResult(
                    f"FQ-1-主体一致性 (video {idx})", fq1_score,
                    [] if fq1_score >= 4 else [f"DINO相似度={subject_cons:.3f}"],
                    metrics={"subject_consistency": round(float(subject_cons), 4)}
                ))

            # FQ-2: 美学质量 (FineVQ, 视频级 MOS 连续分数) (按维度标注过滤)
            if dims is None or "FQ-2" in dims:
                try:
                    aesthetic = self._score_aesthetic(frames)
                    fq2_score = _finevq_to_score(aesthetic)
                except Exception as e:
                    print(f"  [FQ-2] FineVQ 评分失败(video {idx}): {e}")
                    fq2_scores.append(None)
                    all_results.append(EvalResult.error(
                        f"FQ-2-美学质量 (video {idx})", f"FineVQ 评分失败: {e}"))
                    continue

                fq2_scores.append(fq2_score)
                all_results.append(EvalResult(
                    f"FQ-2-美学质量 (video {idx})", fq2_score,
                    [] if fq2_score >= 4 else [f"FineVQ MOS={aesthetic:.3f}"],
                    metrics={"finevq_mos": float(aesthetic)}
                ))

        return self._aggregate(fq1_scores, fq2_scores, all_results)

    def _score_aesthetic(self, frames: list[Image.Image]) -> float:
        """用 FineVQ 对视频做美学/质量评分, 返回连续 MOS 分数。"""
        if self.finevq is None or not getattr(self.finevq, "is_loaded", False):
            raise RuntimeError("FineVQ 未加载")
        return self.finevq.score(frames)

    def evaluate(self, frames: list[Image.Image],
                 prompt: str = "") -> list[EvalResult]:
        """单视频评测（兼容旧接口）"""
        results = []
        if len(frames) >= 2:
            subj = self.specialist.subject_consistency(frames)
            results.append(EvalResult("FQ-1-主体一致性",
                          int(np.clip(subj * 5 + 0.5, 1, 5)),
                          metrics={"subject_consistency": subj}))
        try:
            mos = self._score_aesthetic(frames)
            fq2 = _finevq_to_score(mos)
            results.append(EvalResult("FQ-2-美学质量", fq2,
                          [] if fq2 >= 4 else [f"FineVQ MOS={mos:.3f}"],
                          metrics={"finevq_mos": float(mos)}))
        except Exception as e:
            results.append(EvalResult.error("FQ-2-美学质量",
                                            f"FineVQ 评分失败: {e}"))
        return results

    def _aggregate(self, fq1, fq2, results) -> LayerResult:
        def distribution(s):
            # 过滤 None (error 时不占分布)
            s = [x for x in s if x is not None]
            return {
                "1": s.count(1), "2": s.count(2),
                "3": s.count(3), "4": s.count(4), "5": s.count(5)
            }
        fq2_valid = [x for x in fq2 if x is not None]
        return LayerResult(
            layer="L2-帧级质量",
            results=results,
            aggregates={
                "FQ-1_distribution": distribution(fq1),
                "FQ-2_distribution": distribution(fq2),
            },
            mean_scores={
                "FQ-1_主体一致性": float(np.mean(fq1)) if fq1 else 0,
                "FQ-2_美学质量": float(np.mean(fq2_valid)) if fq2_valid else 0,
            }
        )
