"""
L2 画质层 — 时序质量评测器（全局 Benchmark）

TQ-1 运动平滑度: 光流加速度 → 平滑度
TQ-2 时序闪烁: 相邻帧静态区域像素差异
TQ-3 动态程度: 光流幅度分布
TQ-4 背景一致性: CLIP 跨帧特征相似度
TQ-5 成像质量: 拉普拉斯锐度 + 曝光合理性

评测方式: 专用模型逐视频评测 → 跨全部视频聚合
"""
from typing import Optional
import numpy as np
from PIL import Image
from ..base import BaseEvaluator, EvalResult, LayerResult


class TemporalQualityEvaluator(BaseEvaluator):
    """
    时序质量 全局 Benchmark

    模型需求:
        - CLIP: 背景一致性
        - OpenCV Farneback: 光流(运动平滑度+动态度)
        - 像素级: 时序闪烁 + 成像质量
    """

    def __init__(self, specialist, config: Optional[dict] = None):
        super().__init__("L2-Quality/Temporal", config)
        self.specialist = specialist

    def evaluate_global(self,
                        videos: list
                        ) -> LayerResult:
        all_results = []
        per_dim_scores = {f"TQ-{i}": [] for i in range(1, 6)}

        for idx, item in enumerate(videos):
            # 兼容: 纯帧列表 或 (frames, dims) 对
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], list):
                frames, dims = item[0], item[1]
            else:
                frames, dims = item, None
            if len(frames) < 2:
                continue

            # TQ-1 运动平滑度 (按维度标注过滤)
            if dims is None or "TQ-1" in dims:
                try:
                    smoothness = self.specialist.motion_smoothness(frames)
                    from ..thresholds import score_metric
                    tq1 = score_metric("TQ1_smoothness", smoothness)
                except Exception:
                    smoothness = 0.0
                    tq1 = 3
                per_dim_scores["TQ-1"].append(tq1)
                all_results.append(EvalResult(
                    f"TQ-1-运动平滑度 (v{idx})", tq1,
                    metrics={"smoothness": round(float(smoothness), 4)}))

            # TQ-2 时序闪烁
            if dims is None or "TQ-2" in dims:
                try:
                    flicker = self.specialist.temporal_flickering(frames)
                    # 闪烁越低越好 → flicker越小分越高
                    tq2 = 5 if flicker < 0.02 else (4 if flicker < 0.05
                          else (3 if flicker < 0.1 else (2 if flicker < 0.2 else 1)))
                except Exception:
                    tq2 = 3
                per_dim_scores["TQ-2"].append(tq2)
                all_results.append(EvalResult(f"TQ-2-时序闪烁 (v{idx})", tq2))

            # TQ-3 动态程度
            if dims is None or "TQ-3" in dims:
                try:
                    dyn_deg = self.specialist.dynamic_degree(frames)
                    # 动态度本身无所谓好坏，但极端值(全静/全动)可能异常
                    tq3 = 5 if 0.1 < dyn_deg < 0.8 else (4 if dyn_deg > 0.05
                          else (3 if dyn_deg > 0.01 else 1))
                except Exception:
                    tq3 = 3
                per_dim_scores["TQ-3"].append(tq3)
                all_results.append(EvalResult(f"TQ-3-动态程度 (v{idx})", tq3))

            # TQ-4 背景一致性
            if dims is None or "TQ-4" in dims:
                try:
                    bg_cons = self.specialist.background_consistency(frames)
                    from ..thresholds import score_metric
                    tq4 = score_metric("TQ4_bg_cons", bg_cons)
                except Exception as e:
                    # ⚠️ 必须打印: 此前静默吞异常 → bg_cons 恒 0 → TQ-4 全 3 分
                    print(f"  [TQ-4] background_consistency 失败 (v{idx}): "
                          f"{type(e).__name__}: {e}")
                    bg_cons = 0.0
                    tq4 = 3
                per_dim_scores["TQ-4"].append(tq4)
                all_results.append(EvalResult(
                    f"TQ-4-背景一致性 (v{idx})", tq4,
                    metrics={"bg_consistency": round(float(bg_cons), 4)}))

            # TQ-5 成像质量
            if dims is None or "TQ-5" in dims:
                try:
                    key = frames[len(frames)//2]
                    img_q = self.specialist.imaging_quality(key)
                    combined = (img_q["sharpness"] + img_q["exposure"]) / 2
                    tq5 = int(np.clip(combined * 5 + 0.5, 1, 5))
                except Exception:
                    tq5 = 3
                per_dim_scores["TQ-5"].append(tq5)
                all_results.append(EvalResult(f"TQ-5-成像质量 (v{idx})", tq5))

        return self._aggregate(per_dim_scores, all_results)

    def evaluate(self, frames: list[Image.Image],
                 prompt: str = "") -> list[EvalResult]:
        """单视频评测 — 返回全部 5 个子维度 (TQ-1 ~ TQ-5)"""
        results = []
        if len(frames) < 2:
            return results

        # TQ-1 运动平滑度
        try:
            smooth = self.specialist.motion_smoothness(frames)
            tq1 = int(np.clip(smooth * 5 + 0.5, 1, 5))
        except Exception:
            tq1 = 3
        results.append(EvalResult("TQ-1-运动平滑度", tq1,
                                   metrics={"smoothness": round(smooth, 4)}))

        # TQ-2 时序闪烁
        try:
            flicker = self.specialist.temporal_flickering(frames)
            tq2 = 5 if flicker < 0.02 else (4 if flicker < 0.05
                  else (3 if flicker < 0.1 else (2 if flicker < 0.2 else 1)))
        except Exception:
            tq2 = 3
        results.append(EvalResult("TQ-2-时序闪烁", tq2,
                                   metrics={"flicker": round(flicker, 4)}))

        # TQ-3 动态程度
        try:
            dyn_deg = self.specialist.dynamic_degree(frames)
            tq3 = 5 if 0.1 < dyn_deg < 0.8 else (4 if dyn_deg > 0.05
                  else (3 if dyn_deg > 0.01 else 1))
        except Exception:
            tq3 = 3
        results.append(EvalResult("TQ-3-动态程度", tq3,
                                   metrics={"dynamic_degree": round(dyn_deg, 4)}))

        # TQ-4 背景一致性
        try:
            bg_cons = self.specialist.background_consistency(frames)
            tq4 = int(np.clip(bg_cons * 5 + 0.5, 1, 5))
        except Exception:
            tq4 = 3
        results.append(EvalResult("TQ-4-背景一致性", tq4,
                                   metrics={"bg_consistency": round(bg_cons, 4)}))

        # TQ-5 成像质量
        try:
            key = frames[len(frames)//2]
            img_q = self.specialist.imaging_quality(key)
            combined = (img_q["sharpness"] + img_q["exposure"]) / 2
            tq5 = int(np.clip(combined * 5 + 0.5, 1, 5))
        except Exception:
            tq5 = 3
        results.append(EvalResult("TQ-5-成像质量", tq5,
                                   metrics={"sharpness": img_q.get("sharpness", -1),
                                            "exposure": img_q.get("exposure", -1)}))

        return results

    def _aggregate(self, per_dim, results) -> LayerResult:
        distribution = lambda s: {  # noqa
            "1": s.count(1), "2": s.count(2),
            "3": s.count(3), "4": s.count(4), "5": s.count(5)
        }
        mean_scores = {}
        aggregates = {}
        for dim, scores in per_dim.items():
            if scores:
                mean_scores[dim] = float(np.mean(scores))
                aggregates[f"{dim}_distribution"] = distribution(scores)

        return LayerResult(
            layer="L2-时序质量",
            results=results,
            aggregates=aggregates,
            mean_scores=mean_scores,
        )
