"""
L1 逻辑层 — 指令遵循+语义对齐评测器（全局 Benchmark）

统一46维口径下的L1指令子集（IF-1~9）；IF-10由报告层从参考图保真子指标聚合：
IF-1 主体遵循   (合并 SA-1 物体类别 + SA-2 多物体共现)
IF-2 属性遵循   (含 SA-4 颜色匹配)
IF-3 动作遵循   (含 SA-3 人体动作)
IF-4 空间关系遵循 (含 SA-5 空间关系)
IF-5 风格遵循   (含 SA-7 外观风格)
IF-6 时序遵循   (含 SA-8 时序语义)
IF-7 否定遵循
IF-8 场景匹配   (原 SA-6, IF 未覆盖)
IF-9 相机运动   (原 SA-9, IF 未覆盖)

评测方式: Qwen3.8-27B 逐 (prompt, video) 对评分 → 跨测试集聚合
IF-1 辅助使用 CLIP/DINO 专用模型作为客观参考分

注: IF-10 参考条件保真度由 reference_fidelity 计算底层子指标，
    再由报告层聚合为 L1 IF-10；IF-10b 已合并入 FQ-1 (主体一致性)。
"""
from typing import Optional
import numpy as np
from PIL import Image
from ..base import BaseEvaluator, EvalResult, LayerResult


# IF-10 子指标由 evaluators/quality/reference_fidelity.py 计算，报告层聚合为 L1 IF-10。
# IF-10b 已合并入 FQ-1（主体一致性），不再独立评测。

INSTRUCTION_DIMENSIONS = [
    ("IF-1", "主体遵循",
     "视频中生成的物体是否与 prompt 指定的一致（含物体类别正确性 + 多物体共现准确性 + 数量匹配）"),
    ("IF-2", "属性遵循",
     "物体的颜色/材质/大小/形状/纹理是否匹配 prompt 描述"),
    ("IF-3", "动作遵循",
     "prompt 中描述的动作/运动（含人体动作和通用物体运动）是否在视频中发生且完整呈现"),
    ("IF-4", "空间关系遵循",
     "prompt 中的空间关系（上下/左右/前后/接触/包含/支撑）是否正确呈现"),
    ("IF-5", "风格遵循",
     "指定风格（写实/卡通/水墨/油画/像素/赛博朋克/电影感...）是否呈现且帧间一致"),
    ("IF-6", "时序遵循",
     "prompt 中描述的事件顺序是否正确，是否有时序语义对齐"),
    ("IF-7", "否定遵循",
     "对否定描述（\"没有...\"\"不含...\"\"不是...\"）是否正确处理，不应出现被否定的内容"),
    ("IF-8", "场景匹配",
     "视频的场景类型（室内/室外/城市场景/自然/水下/太空...）是否匹配 prompt 描述"),
    ("IF-9", "相机运动",
     "prompt 隐含的相机运动（推拉/摇移/跟拍/俯仰/固定/第一人称...）是否正确呈现"),
]

# 注: IF-10 子指标由 evaluators/quality/reference_fidelity.py 计算，
# 报告层按样本聚合为 L1 IF-10；IF-10b 已合并入 FQ-1。


class InstructionFollowingEvaluator(BaseEvaluator):
    """
    指令遵循+语义对齐 全局 Benchmark（IF 1~9）

    输入: 标准化 prompt 测试集 → 目标模型生成视频 → 逐条评测 → 聚合
    支持 T2V / I2V / CI2V 三种模态
    参考图保真度子指标由 ReferenceFidelityEvaluator 计算并聚合为 L1 IF-10
    """

    def __init__(self, vlm, specialist=None, config: Optional[dict] = None):
        super().__init__("L1-Logic/InstructionFollowing", config)
        self.vlm = vlm
        self.specialist = specialist  # 可选，用于 CLIP IF-1 客观参考分

    def evaluate_global(self,
                        prompt_video_pairs: list,
                        ref_images: Optional[list[Optional[Image.Image]]] = None
                        ) -> LayerResult:
        """
        全局评测：对全部 (prompt, video) 对评测后聚合

        Args:
            prompt_video_pairs: [(prompt_str, [frame1, frame2, ...]), ...]
            ref_images: [ref_image_or_None, ...]  长度同 prompt_video_pairs
                       (保留参数兼容旧调用，IF-10 子指标由 ReferenceFidelityEvaluator 计算)

        Returns:
            LayerResult 含 9 维聚合统计
        """
        all_results = []
        per_dim_scores = {dim_id: [] for dim_id, _, _ in INSTRUCTION_DIMENSIONS}

        for idx, item in enumerate(prompt_video_pairs):
            # 兼容新旧格式: 2元组(old prompt,frames) 或 3元组(new prompt,frames,dims)
            if len(item) == 2:
                prompt, frames = item
                dims = None  # 旧格式无维度标注 → 跑全部维度(向后兼容)
            else:
                prompt, frames = item[0], item[1]
                dims = item[2] if len(item) >= 3 else None

            if not frames:
                continue

            # 逐维评测 IF-1~9 (按该样本标注的维度过滤)
            for dim_id, dim_name, dim_desc in INSTRUCTION_DIMENSIONS:
                if dims is not None and dim_id not in dims:
                    continue  # 该样本未标注此维度 → 跳过
                score, explanation = self._evaluate_single_dim(
                    frames, prompt, dim_id, dim_name, dim_desc
                )
                per_dim_scores[dim_id].append(score)
                all_results.append(EvalResult(
                    f"{dim_id}:{dim_name} (item {idx})",
                    score,
                    violations=[] if score >= 4 else [f"item {idx}: {explanation[:100]}"],
                    metrics={"prompt": prompt[:100], "explanation": explanation[:200]}
                ))

        # 聚合统计 IF-1~9
        aggregates = {}
        mean_scores = {}
        for dim_id, dim_name, _ in INSTRUCTION_DIMENSIONS:
            scores = per_dim_scores[dim_id]
            if scores:
                mean_scores[f"{dim_id}-{dim_name}"] = float(np.mean(scores))
                aggregates[f"{dim_id}_distribution"] = {
                    "1": scores.count(1), "2": scores.count(2),
                    "3": scores.count(3), "4": scores.count(4),
                    "5": scores.count(5)
                }

        # 可选：IF-1 辅助 CLIP 客观参考分
        clip_ref = self._compute_clip_reference(prompt_video_pairs)
        if clip_ref is not None:
            aggregates["IF-1_CLIP_reference"] = clip_ref

        return LayerResult(
            layer="L1-指令遵循+语义对齐",
            results=all_results,
            aggregates=aggregates,
            mean_scores=mean_scores,
        )

    def evaluate(self, frames: list[Image.Image],
                 prompt: str = "") -> list[EvalResult]:
        """单案例评测（兼容旧接口）"""
        results = []
        for dim_id, dim_name, dim_desc in INSTRUCTION_DIMENSIONS:
            score, explanation = self._evaluate_single_dim(
                frames, prompt, dim_id, dim_name, dim_desc
            )
            results.append(EvalResult(
                f"{dim_id}:{dim_name}", score,
                [] if score >= 4 else [explanation[:100]],
                metrics={"explanation": explanation[:200]}
            ))
        return results

    def _evaluate_single_dim(self, frames: list[Image.Image],
                              prompt: str, dim_id: str,
                              dim_name: str, dim_desc: str
                              ) -> tuple[int, str]:
        """单维度评测 — Qwen3.8-27B"""
        eval_prompt = f"""你是一个视频生成质量评测专家。请对照以下 prompt 评判生成视频的语义对齐程度。

【原始 Prompt】
{prompt}

【评测维度】{dim_name}
{dim_desc}

请仔细观察视频关键帧，判断视频内容是否遵循了 prompt 中与该维度相关的指令。

回复格式:
评分: X/5 (5=完全遵循, 4=基本遵循但有小瑕疵, 3=部分遵循但有明显偏差, 2=严重偏离, 1=完全不遵循)
原因: <一句话说明>"""

        sampled = self._sample_frames(frames, 6)
        response = self.vlm.chat(sampled, eval_prompt, max_new_tokens=256)

        score = self._extract_score(response)
        return score, response

    # ── 辅助方法 ──

    def _compute_clip_reference(self, prompt_video_pairs
                                ) -> Optional[dict]:
        """
        IF-1 CLIP 客观参考分: 关键帧图文对齐作为辅助指标
        不替代 VLM 评分，仅作为 aggregates 中的参考值
        
        image_text_similarity 失败时返回 sentinel -1.0，这里滤除无效值。
        """
        if self.specialist is None:
            return None

        clip_scores = []
        n_failed = 0
        try:
            for prompt, frames in prompt_video_pairs:
                if not frames:
                    continue
                key = frames[len(frames)//2]
                sim = self.specialist.image_text_similarity(prompt, key)
                if sim >= 0:  # 有效值 (哨兵 -1.0 表示失败)
                    clip_scores.append(float(sim))
                else:
                    n_failed += 1
        except Exception:
            return None

        if not clip_scores:
            return None
        return {
            "mean": float(np.mean(clip_scores)),
            "min": float(np.min(clip_scores)),
            "max": float(np.max(clip_scores)),
            "n": len(clip_scores),
            "n_failed": n_failed,
        }

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
