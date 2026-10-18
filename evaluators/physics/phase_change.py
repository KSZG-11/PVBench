"""
L3 物理层 — 相变评测器
R25 相变 (Qwen3.8-27B) — 熔化/凝固/汽化/凝华/升华/燃烧
"""
from typing import Optional
from PIL import Image

from ..base import EvalResult
from ._vlm_base import VLMBaseEvaluator


class PhaseChangeEvaluator(VLMBaseEvaluator):
    """相变评测器 (R25, Qwen3.8-27B)"""

    def __init__(self, vlm, config: Optional[dict] = None):
        super().__init__(vlm, "L3-Physics/PhaseChange", config)

    def evaluate(self, frames: list[Image.Image]) -> list[EvalResult]:
        prompt = """你是物质科学专家，分析视频中的相变过程(熔化/凝固/汽化/凝华/升华/燃烧)。

【评分标准】
5: 相变方向/速率完全合理（冰受热渐融成水、水沸腾冒泡向上、蜡烛燃烧渐短、凝固由表及里）
3: 方向对但速率/形态反常（瞬间全融、融化区域错位、沸腾无气泡）
1: 方向错或凭空（水结冰同时沸腾、火焰使冰变大、固体直接消失无残留）

请评估: 相变类型、方向正确性、速率合理性、形态自然度。相变需观察全过程。

【关键判定】先判断视频中是否真的存在相变过程:
- has_phase_change_phenomena: 视频中是否出现熔化/凝固/汽化/升华/燃烧等相变过程? true/false
- 若 has_phase_change_phenomena=false (视频没有生成 prompt 要求的相变过程),
  说明生成模型未完成任务, score 必须给 1 分, 并在 violations 说明"视频未生成相变过程"。

返回 JSON: {"score": <1-5整数>, "has_phase_change_phenomena": <true/false>, "reason": "一句话总结", "violations": ["..."]}"""
        return [self._score_with_json(frames, "R25-相变", prompt, n_frames=8, max_new_tokens=1024)]
