"""
L3 物理层 — 磁电作用评测器
R14 磁电作用 (Qwen3.8-27B)
"""
from typing import Optional
from PIL import Image

from ..base import EvalResult
from ._vlm_base import VLMBaseEvaluator


class ElectromagEvaluator(VLMBaseEvaluator):
    """磁电作用评测器 (R14, Qwen3.8-27B)"""

    def __init__(self, vlm, config: Optional[dict] = None):
        super().__init__(vlm, "L3-Physics/Electromag", config)

    def evaluate(self, frames: list[Image.Image]) -> list[EvalResult]:
        prompt = """你是物理专家，分析视频中电磁/静电现象的正确性。

【评测对象】
- 磁铁: 同极相斥/异极相吸、磁场对铁磁物质的作用、作用力随距离增大而减小
- 带电体: 静电吸附轻小物体(纸屑/毛发)、电荷在导体表面分布
- 静电发生器(范德格拉夫起电机等): 高压静电放电/电弧现象，包括：
  * 电弧路径应沿电场线方向(从高压端到接地端)
  * 尖端放电效应(电荷在曲率大处集中)
  * 电弧呈树枝状分叉或明亮的等离子体通道
  * 放电伴随的介质击穿现象

【评分标准】
5: 电磁/静电现象完全正确（磁场/电场方向正确、作用力强度合理、电弧路径和形态符合物理规律）
3: 有电磁现象但存在部分物理错误（方向或强度反常、电弧路径不合理、放电形态错误）
1: 完全无电磁响应或严重物理错误（磁铁吸非磁性物质、电荷行为完全不对）

【关键判定】先判断视频中是否真的存在电磁/静电现象:
- has_electromagnetic_phenomena: 视频中是否出现磁体作用/静电吸附/电弧放电等电磁现象? true/false
- 若 has_electromagnetic_phenomena=false (视频没有生成 prompt 要求的电磁/静电现象),
  说明生成模型未完成任务, score 必须给 1 分, 并在 violations 说明"视频未生成电磁/静电现象"。

返回 JSON: {"score": <1-5整数>, "has_electromagnetic_phenomena": <true/false>, "reason": "一句话总结", "violations": ["..."]}"""
        return [self._score_with_json(frames, "R14-磁电作用", prompt, n_frames=6, max_new_tokens=1024)]
