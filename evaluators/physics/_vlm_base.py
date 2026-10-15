"""
VLM 评测器公共基类 — 结构化 JSON 输出 + 统一采样/错误处理

benchmark 规范: VLM 打分必须结构化(JSON schema), 不用脆弱的中文正则抠分。
失败(模型未加载/推理异常/JSON解析失败) → status=error, 不打分。
"""
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult


class VLMBaseEvaluator(BaseEvaluator):
    """所有 Qwen3-VL 评测器的公共基类"""

    def __init__(self, vlm, layer: str, config: Optional[dict] = None):
        super().__init__(layer, config)
        self.vlm = vlm

    @staticmethod
    def _sample_frames(frames: list, n: int) -> list:
        if len(frames) <= n:
            return frames
        idx = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in idx]

    def _score_with_json(self, frames: list[Image.Image], dimension: str,
                         prompt: str, n_frames: int = 6,
                         max_new_tokens: int = 512) -> EvalResult:
        """统一流程: 采样 → chat_json → 解析 {score, violations, ...}。

        VLM proxy 规则校验 (v2.4):
        - 若 JSON 含 has_* 布尔字段 (如 has_thermal_phenomena / has_mechanical_event),
          且其值为 false, 说明视频未生成 prompt 要求的物理现象 (生成模型未完成任务)。
          此时封顶 score=1 (严重不达标), 而非 VLM 可能误给的满分。
        """
        if not getattr(self.vlm, "is_loaded", False):
            return EvalResult.error(dimension, "VLM 未加载")
        sampled = self._sample_frames(frames, n_frames)
        try:
            data = self.vlm.chat_json(sampled, prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            return EvalResult.error(dimension, f"VLM 推理失败: {e}")

        if data.get("_parse_error"):
            return EvalResult.error(dimension, f"VLM 输出无法解析为JSON: {data.get('_raw','')[:100]}")

        score = data.get("score", data.get("综合评分"))
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            return EvalResult.error(dimension, "VLM 未返回有效 score 字段")
        score = max(1, min(5, score))

        violations = data.get("violations", data.get("违例", []))
        if isinstance(violations, str):
            violations = [violations] if violations.strip() else []
        elif not isinstance(violations, list):
            violations = []

        # ── VLM proxy 规则校验: has_* 布尔字段为 false → 现象缺失 → 封顶1分 ──
        # 注意: 只对明确为 False 的布尔值生效; 若字段缺失或非 bool, 不触发 (保守)。
        phenomena_fields = [k for k in data if k.startswith("has_")]
        for key in phenomena_fields:
            val = data[key]
            if isinstance(val, bool) and val is False:
                if score > 1:
                    score = 1
                msg = (f"VLM规则校验: {key}=false 表示视频未生成该物理现象"
                       f"(生成模型未完成任务), 已封顶 1 分")
                if not any(key in v for v in violations):
                    violations.insert(0, msg)
                break  # 一条即可

        # 保留完整结构化输出到 metrics
        metrics = {k: v for k, v in data.items()
                   if k not in ("score", "综合评分", "violations", "违例")}
        return EvalResult(dimension, score, [str(x)[:200] for x in violations],
                          metrics=metrics, status="proxy")
