"""
评测器基类 — 所有评测器继承于此

v3 改动（面向标准 benchmark）：
- EvalResult 新增 status 字段: ok / proxy / degraded / error / skipped / unobservable
  * ok      : 有真实 GT 的精确定量评测
  * proxy   : 无 GT，走物理自洽性/合理性检查（T2V/I2V）
  * degraded: 首选模型缺失，用替代模型降级评测
  * error   : 模型加载/推理失败，该维度不可评（不计入均分）
  * skipped : 条件不满足（如帧数不足）而主动跳过
  * unobservable: 生成模型未满足评测前提（如 prompt 要求「遮挡时保持轮廓可见」,
            但模型把目标物体完全遮住），导致物理量不可观测。不计入均分,
            但单独统计，作为「模型空间约束遵循能力」的独立信号。
- 聚合(avg_score) 只统计 status ∈ {ok, proxy, degraded} 的维度，
  error/skipped/unobservable 不污染平均分。
- 移除所有 stub 假数据机制。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


# 计入平均分的状态
SCORING_STATUSES = {"ok", "proxy", "degraded"}


@dataclass
class EvalResult:
    """单一维度评测结果"""
    dimension: str                        # 维度名称 (如 "R01-深度", "IF-1-主体遵循")
    score: int                            # 1-5 整数分数 (status=error/skipped 时无意义)
    violations: list[str] = field(default_factory=list)  # 违例描述
    metrics: dict = field(default_factory=dict)           # 额外量化指标
    status: str = "ok"                    # ok/proxy/degraded/error/skipped
    # 内容匹配标记 (不改变 status/scoring, 供报告筛选)
    content_mismatch: bool = False        # 视频内容与 prompt 描述的物理场景不匹配
    content_warnings: list[str] = field(default_factory=list)  # 内容不匹配详情
    # 向后兼容字段
    layer: str = ""
    category: str = ""
    explanation: str = ""

    @property
    def counts_toward_score(self) -> bool:
        return self.status in SCORING_STATUSES

    def to_dict(self) -> dict:
        d = {
            "dimension": self.dimension,
            "score": self.score,
            "status": self.status,
            "violations": self.violations,
            "metrics": self.metrics,
        }
        if self.content_mismatch:
            d["content_mismatch"] = True
            d["content_warnings"] = self.content_warnings
        return d

    # ── 便捷构造器 ──
    @classmethod
    def error(cls, dimension: str, reason: str) -> "EvalResult":
        return cls(dimension, 0, [reason], status="error")

    @classmethod
    def skipped(cls, dimension: str, reason: str) -> "EvalResult":
        return cls(dimension, 0, [reason], status="skipped")

    @classmethod
    def unobservable(cls, dimension: str, reason: str) -> "EvalResult":
        """生成模型未满足评测前提（物理量不可观测），不计分但单独统计。"""
        return cls(dimension, 0, [reason], status="unobservable")


@dataclass
class LayerResult:
    """单层评测汇总结果"""
    layer: str                              # 层名 (如 "L1-指令遵循", "L3-geometry")
    results: list[EvalResult] = field(default_factory=list)
    avg_score: float = 0.0
    # 全局 Benchmark 聚合字段
    aggregates: dict = field(default_factory=dict)    # {dim_distribution: {1:n,2:n,...}}
    mean_scores: dict = field(default_factory=dict)   # {dim_name: mean_score, ...}

    def scored_results(self) -> list[EvalResult]:
        """仅返回计入平均分的结果 (排除 error/skipped)"""
        return [r for r in self.results if r.counts_toward_score]

    def compute_avg(self) -> float:
        # 全局逻辑/画质层提供mean_scores时，按“维度均值”聚合，避免
        # 某个维度因样本数或子指标数量更多而在层均值中被重复加权。
        if self.mean_scores:
            values = [float(value) for value in self.mean_scores.values()
                      if isinstance(value, (int, float))]
            self.avg_score = sum(values) / len(values) if values else 0.0
            return self.avg_score
        scored = self.scored_results()
        self.avg_score = sum(r.score for r in scored) / len(scored) if scored else 0.0
        return self.avg_score

    def to_dict(self) -> dict:
        scored = self.scored_results()
        d = {
            "layer": self.layer,
            "avg_score": round(self.compute_avg(), 2),
            "n_results": len(self.results),
            "n_scored": len(scored),
            "n_error": sum(1 for r in self.results if r.status == "error"),
            "n_skipped": sum(1 for r in self.results if r.status == "skipped"),
            "n_unobservable": sum(1 for r in self.results if r.status == "unobservable"),
            "results": [r.to_dict() for r in self.results],
        }
        if self.aggregates:
            d["aggregates"] = self.aggregates
        if self.mean_scores:
            d["mean_scores"] = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in self.mean_scores.items()
            }
        return d


@dataclass
class BenchmarkResult:
    """完整 benchmark 评测结果"""
    layers: list[LayerResult] = field(default_factory=list)
    video_path: str = ""                   # 单个视频路径(物理评测)或空(全局评测)

    def add_layer(self, layer: LayerResult):
        self.layers.append(layer)

    def overall_score(self) -> float:
        # 全局L1/L2结果均提供mean_scores时，直接按适用维度均值聚合；
        # 这样L1=14、L2=7的维度不会因层数不同或RF子指标重复而失真。
        if self.layers and all(layer.mean_scores for layer in self.layers):
            values = [float(value) for layer in self.layers
                      for value in layer.mean_scores.values()
                      if isinstance(value, (int, float))]
            if values:
                return sum(values) / len(values)
        scores = [l.compute_avg() for l in self.layers if l.scored_results()]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path or "global",
            "overall_score": round(self.overall_score(), 2),
            "layers": [l.to_dict() for l in self.layers],
        }


class BaseEvaluator(ABC):
    """评测器基类"""

    def __init__(self, layer: str = "", config: Optional[dict] = None):
        self.layer = layer
        self.config = config or {}
        self.results: list[EvalResult] = []

    @staticmethod
    def unobservable_if_declared(prompt: str, keywords: list[str],
                                 dimension: str, reason: str,
                                 skip_reason: str = "") -> EvalResult:
        """通用「可观测性失败」判定。

        用于区分两类「检测不到目标」的 skip:
        - prompt 明确声明了某个物理现象/物体(命中 keywords), 但评测器没检测到
          → 生成模型未满足评测前提 → unobservable(不计分但单独统计, 反映模型能力)。
        - prompt 未声明, 或声明但不命中 keywords → 保持 skipped(评测器/环境问题)。

        Args:
            prompt: 样本 prompt 文本
            keywords: 该维度物理现象的关键词(如碰撞 collide/impact, 运动 move/travel)
            dimension: 维度名(如 "R23-动量守恒")
            reason: 未检测到的具体原因(用于 unobservable 的 violation)
            skip_reason: 保持 skipped 时的原因(默认与 reason 相同)
        """
        if not prompt:
            return EvalResult.skipped(dimension, skip_reason or reason)
        low = prompt.lower()
        declared = any(kw in low for kw in keywords)
        if declared:
            return EvalResult.unobservable(
                dimension,
                f"prompt 声明了该物理现象({[k for k in keywords if k in low][:3]}), "
                f"但未检测到: {reason} (模型未满足评测前提)")
        return EvalResult.skipped(dimension, skip_reason or reason)

    def vlm_observable(self, frame, prompt: str,
                       max_new_tokens: int = 256) -> tuple:
        """用 VLM 看关键帧, 判断 prompt 要求的关键物体是否在画面中可观测。

        用途: 当检测器(SAM3/DA3 等)分割/追踪不到物体而准备 skip 时, 先用 VLM
        二次确认「是模型没生成物体(→ unobservable) 还是检测器能力不足(→ skipped)」。

        Args:
            frame: 关键帧 (numpy [H,W,3] 或 PIL Image)
            prompt: 样本 prompt 文本
            max_new_tokens: VLM 生成 token 上限

        Returns:
            (observable: bool, reason: str)。observable=True 表示物体在画面中可见
            (检测器能力不足); False 表示物体不可见/未生成(模型问题)。
            VLM 不可用或解析失败时返回 (None, 错误说明)。
        """
        vlm = getattr(self, "vlm", None)
        if vlm is None or not getattr(vlm, "is_loaded", False):
            return None, "VLM 不可用"
        # numpy → PIL
        import numpy as _np
        if isinstance(frame, _np.ndarray):
            frame = Image.fromarray(frame.astype("uint8"))
        vlm_prompt = (
            "你是视频生成质量评估专家。下面是一段文生视频 prompt，以及该视频的"
            "一个关键帧画面。\n\n"
            f"【prompt】\"{prompt}\"\n\n"
            "【任务】逐一遍历 prompt 中描述的每一个核心物理物体（如花瓶、雕像、"
            "球、车、机械臂、人、立方体等，不是背景环境/参照物/标记），判断"
            "每个物体是否都在画面中清晰可见、轮廓可辨认。\n\n"
            "关键要求：\n"
            "1. 必须逐个列出 prompt 里出现的每个核心物体，不能只确认其中一部分。\n"
            "2. 只要【任何一个】核心物体被完全遮挡、只露出极小一部分、或根本没有"
            "生成，observable 就必须是 false（即使其他物体都可见）。\n"
            "3. 特别注意「A 在 B 前面/后面」这类前后遮挡关系，两个物体都要"
            "可见才算 true。\n\n"
            "返回 JSON: {\"observable\": <true/false>, "
            "\"objects\": [{\"name\": \"物体名\", \"visible\": <true/false>}], "
            "\"reason\": \"<简短说明哪些物体可见、哪些缺失/被遮挡>\"}"
        )
        try:
            data = vlm.chat_json([frame], vlm_prompt,
                                 max_new_tokens=max_new_tokens)
        except Exception as e:
            return None, f"VLM 推理失败: {e}"
        if data.get("_parse_error"):
            return None, f"VLM 输出解析失败: {data.get('_raw', '')[:100]}"
        obs = data.get("observable")
        if not isinstance(obs, bool):
            return None, f"VLM 未返回有效 observable 字段: {obs}"
        reason = data.get("reason", "")
        # 兜底: 若 objects 列表里存在 visible=false, 强制 observable=false
        # (防止 VLM 的 observable 字段与逐物体判断自相矛盾)
        objects = data.get("objects", [])
        if isinstance(objects, list):
            for obj in objects:
                if isinstance(obj, dict) and obj.get("visible") is False:
                    if obs is True:
                        reason = (f"{reason} (但objects里有{obj.get('name')}不可见)"
                                  if reason else f"objects里{obj.get('name')}不可见")
                    obs = False
                    break
        return obs, reason

    @abstractmethod
    def evaluate(self, frames: list[Image.Image],
                 prompt: str = "", **kwargs) -> list[EvalResult]:
        """
        评测入口

        Args:
            frames: 视频帧列表 (PIL Images)
            prompt: 原始 T2V prompt (可选)

        Returns:
            list of EvalResult
        """
        ...

    def clear(self):
        self.results = []
