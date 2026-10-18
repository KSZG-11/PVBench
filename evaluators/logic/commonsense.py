"""
L1 逻辑层 — 常识推理评测器（全局 Benchmark）

3个维度，跨标准化 prompt 集聚合评分：
CS-1 物理常识 | CS-2 生物常识 | CS-3 社会常识

评测方式: Qwen3.8-27B 多选判断 + 解释合理性评分
"""
from typing import Optional
import numpy as np
from PIL import Image
from ..base import BaseEvaluator, EvalResult, LayerResult


COMMONSENSE_DIMENSIONS = [
    ("CS-1", "物理常识", "基本物理直觉: 重物下落/轻物漂浮/火向上/水向下流/影子方向..."),
    ("CS-2", "生物常识", "生物行为合理性: 动物运动模式/植物生长方向/人体姿态自然度..."),
    ("CS-3", "社会常识", "场景逻辑: 桌椅摆放合理性/人物互动自然度/日常因果一致性..."),
]


class CommonsenseEvaluator(BaseEvaluator):
    """
    常识推理 全局 Benchmark
    """

    def __init__(self, vlm, config: Optional[dict] = None):
        super().__init__("L1-Logic/Commonsense", config)
        self.vlm = vlm

    def evaluate_global(self,
                        prompt_video_pairs: list[tuple[str, list[Image.Image]]]
                        ) -> LayerResult:
        """
        全局评测：对所有 prompt-video 对评测后聚合
        """
        per_dim_scores = {dim_id: [] for dim_id, _, _ in COMMONSENSE_DIMENSIONS}
        all_results = []

        for idx, item in enumerate(prompt_video_pairs):
            # 兼容新旧格式: 2元组(prompt,frames) 或 3元组(prompt,frames,dims)
            if len(item) == 2:
                prompt, frames = item
                dims = None
            else:
                prompt, frames = item[0], item[1]
                dims = item[2] if len(item) >= 3 else None

            for dim_id, dim_name, dim_desc in COMMONSENSE_DIMENSIONS:
                if dims is not None and dim_id not in dims:
                    continue  # 该样本未标注此维度 → 跳过
                score, explanation = self._evaluate_commonsense(
                    frames, prompt, dim_id, dim_name, dim_desc
                )
                per_dim_scores[dim_id].append(score)
                all_results.append(EvalResult(
                    f"CS-{dim_id}:{dim_name} (item {idx})",
                    score,
                    violations=[] if score >= 4 else [f"item {idx}: {explanation[:100]}"],
                    metrics={"prompt": prompt[:80]}
                ))

        # 聚合
        aggregates = {}
        mean_scores = {}
        for dim_id, dim_name, _ in COMMONSENSE_DIMENSIONS:
            scores = per_dim_scores[dim_id]
            if scores:
                mean_scores[f"{dim_id}-{dim_name}"] = float(np.mean(scores))
                aggregates[f"{dim_id}_distribution"] = {
                    "1": scores.count(1), "2": scores.count(2),
                    "3": scores.count(3), "4": scores.count(4), "5": scores.count(5)
                }

        return LayerResult(
            layer="L1-常识推理",
            results=all_results,
            aggregates=aggregates,
            mean_scores=mean_scores,
        )

    def evaluate(self, frames: list[Image.Image],
                 prompt: str = "") -> list[EvalResult]:
        """单案例评测（兼容旧接口）"""
        results = []
        for dim_id, dim_name, dim_desc in COMMONSENSE_DIMENSIONS:
            score, explanation = self._evaluate_commonsense(
                frames, prompt, dim_id, dim_name, dim_desc
            )
            results.append(EvalResult(
                f"CS-{dim_id}:{dim_name}", score,
                [] if score >= 4 else [explanation[:100]]
            ))
        return results

    def _evaluate_commonsense(self, frames: list[Image.Image],
                               prompt: str, dim_id: str,
                               dim_name: str, dim_desc: str
                               ) -> tuple[int, str]:
        """单维度常识评测"""
        eval_prompt = f"""你是一个视频理解专家，请评估生成视频的常识合理性。

【原始 Prompt】{prompt}

【评测维度】{dim_name}: {dim_desc}

请观察视频帧，判断以下方面是否合理:
1. 视频中的现象是否符合{dim_name}基本规律？
2. 是否有任何违背常识的异常之处？

评分标准:
- 5: 完全符合常识，无可挑剔
- 4: 基本合理，有非常微小的不自然
- 3: 存在1处明显但不严重的常识偏差
- 2: 存在多处常识错误
- 1: 严重违背常识（物理/生物/社会定律被颠覆）

回复格式:
评分: X/5
原因: <一句话>"""  # noqa

        sampled = self._sample_frames(frames, 6)
        response = self.vlm.chat(sampled, eval_prompt, max_new_tokens=256)

        score = self._extract_score(response)
        return score, response

    def _extract_score(self, text: str) -> int:
        import re
        for line in text.split("\n"):
            match = re.search(r"(\d+)\s*/\s*5", line)
            if match:
                return max(1, min(5, int(match.group(1))))
        return 3

    @staticmethod
    def _sample_frames(frames, n):
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames)-1, n, dtype=int)
        return [frames[i] for i in indices]
