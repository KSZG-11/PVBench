"""
L3 物理层 — 热辐射评测器
R07 热辐射 (Qwen3-VL) — 五维辐射检验(表格D列)
"""
from typing import Optional
from PIL import Image

from ..base import EvalResult
from ._vlm_base import VLMBaseEvaluator


class ThermalEvaluator(VLMBaseEvaluator):
    """热辐射评测器 (R07, Qwen3-VL)"""

    def __init__(self, vlm, config: Optional[dict] = None):
        super().__init__(vlm, "L3-Physics/Thermal", config)

    def evaluate(self, frames: list[Image.Image]) -> list[EvalResult]:
        prompt = """你是热物理专家，按五维标准检验视频中的热辐射表现。温度不可见，仅通过辐射量间接呈现。

请客观检查视频中是否存在热源（火焰/高温物体/蒸汽/热气流）及相应的热辐射现象。

【五维辐射检验】
① 发光颜色: 高温物体发光颜色是否符合黑体辐射序列(暗红→橙→黄→白)?
② 热浪畸变: 高温表面上方是否有光线扭曲闪烁(热空气折射)?
③ 对流方向: 蒸汽/火焰/热气流是否向上(浮力)?
④ 辐射梯度: 离热源越近热效应越强、远端减弱(∝1/r²)?
⑤ 跨物体一致性: 同热源多物体效应是否与距离成正比?

【评分】
5: 热辐射五维全部表现正确
3: 方向正确但细节反常（发光序列有误/梯度不够/热浪畸变缺失）
1: 方向性错误（蒸汽下沉/越远越热/颜色序列违反黑体辐射）

【关键判定】先判断视频中是否真的存在热源相关现象:
- has_thermal_phenomena: 视频中是否出现热源(火焰/高温发光物体/蒸汽/热气流等)? true/false
- 若 has_thermal_phenomena=false (视频没有生成 prompt 要求的热源内容), 说明生成模型未完成任务,
  score 必须给 1 分(严重不达标), 并在 violations 说明"视频未生成热源相关内容"。

返回 JSON: {"score": <1-5整数>, "has_thermal_phenomena": <true/false>, "reason": "一句话总结", "violations": ["..."]}"""
        return [self._score_with_json(frames, "R07-热辐射", prompt, n_frames=6, max_new_tokens=1024)]
