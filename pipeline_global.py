"""
全局 Benchmark 评测管线 (L1 逻辑 + L2 画质)

支持三种模态: T2V / I2V / CI2V
- T2V: 仅 prompt → video
- I2V: prompt + 场景参考图 → video (触发 IF-10a/b/c)
- CI2V: prompt + 角色参考图 → video (触发 IF-10a/b/c/d)

流程: 加载 prompt 测试集 + 参考图 → 目标模型生成视频 → 全量评测 → 聚合报告
"""
import yaml
import os
import sys
import numpy as np
from typing import Optional
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.loader import ModelLoader
from sampler import FrameSampler
from evaluators.base import BenchmarkResult, EvalResult, LayerResult
from evaluators.logic import (
    InstructionFollowingEvaluator,
    ActionCausalEvaluator,
    CommonsenseEvaluator,
)
from evaluators.quality import (
    FrameQualityEvaluator,
    TemporalQualityEvaluator,
    ReferenceFidelityEvaluator,
)
from reporter import Reporter


class GlobalBenchmarkPipeline:
    """
    全局 Benchmark 管线 — 支持 T2V / I2V / CI2V 三种模态

    评测模型整体能力：
    1. 加载标准化 prompt 测试集 + 参考图（可选）
    2. 用目标模型逐条生成视频（用户预生成或外接接口）
    3. L1 逻辑评测 + L2 画质评测
    4. 跨全部视频聚合
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        gb_cfg = self.config.get("global_benchmark", {})
        self.logic_cfg = gb_cfg.get("logic", {})
        self.quality_cfg = gb_cfg.get("quality", {})

        # 输出目录
        self.output_dir = self.config.get("global", {}).get("output_dir", "./results")
        os.makedirs(os.path.join(self.output_dir, "global"), exist_ok=True)

        self.loader: Optional[ModelLoader] = None
        self.sampler = FrameSampler(self.config.get("sampling", {}))

    def initialize(self, loader: Optional[ModelLoader] = None) -> ModelLoader:
        """初始化模型。

        loader: 可选，传入共享的 ModelLoader（UnifiedPipeline 让 L1+L2 与 L3
                共享同一 loader，避免 VLM 等模型被重复加载两份占双倍显存）。
        """
        self.loader = loader if loader is not None else ModelLoader(self.config)

        # 预加载 VLM
        try:
            self.loader.get_vlm("logic")
            print("[Pipeline] VLM loaded for global benchmark.")
        except Exception as e:
            print(f"[Pipeline] VLM load warning: {e}")

        # 预加载画质模型
        try:
            self.loader.get_specialist()
            print("[Pipeline] Specialist models loaded.")
        except Exception as e:
            print(f"[Pipeline] Specialist load warning: {e}")

        self.loader.print_status()
        return self.loader

    def load_prompts(self, set_name: str) -> list[dict]:
        """加载 prompt 测试集"""
        prompt_sets = self.config.get("global_benchmark", {}).get("prompt_sets", {})
        path = prompt_sets.get(set_name, f"prompts/{set_name}.yaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("prompts", [])

    def run(self,
            prompt_video_mapping: list,
            run_logic: bool = True,
            run_quality: bool = True,
            ref_images: Optional[list[Optional[str]]] = None
            ) -> BenchmarkResult:
        """
        执行全局评测

        Args:
            prompt_video_mapping: 支持两种格式:
                [(prompt_text, video_path), ...]  — T2V 格式
                [(prompt_text, video_path, ref_image_path), ...]  — I2V/CI2V 格式
            run_logic: 是否执行逻辑层评测
            run_quality: 是否执行画质层评测
            ref_images: [ref_image_path_or_None, ...] — 独立传入参考图列表
                        None 元素对应 T2V 条目; 有路径对应 I2V/CI2V

        Returns:
            BenchmarkResult 含全部层聚合结果
        """
        if self.loader is None:
            self.initialize()

        result = BenchmarkResult()
        reporter = Reporter(output_dir=os.path.join(self.output_dir, "global"))

        # 预处理：加载视频帧 + 参考图
        n_items = len(prompt_video_mapping)

        print(f"\n[Global Benchmark] 加载 {n_items} 个视频（支持混合模态）...")

        video_frames = []
        loaded_refs = []
        loaded_modalities = []
        loaded_reference_roles = []

        for idx, item in enumerate(prompt_video_mapping):
            if isinstance(item, (list, tuple)):
                prompt, video_path = item[0], item[1]
                ref_path = item[2] if len(item) >= 3 else None
                item_modality = (item[3] if len(item) >= 4 else
                                 ("I2V" if ref_path else "T2V"))
                reference_role = "model_condition" if ref_path else "none"
                dims = item[4] if len(item) >= 5 else []
            else:
                prompt, video_path = item.get("prompt"), item.get("video")
                ref_path = item.get("ref_image") or item.get("canonical_ref_image")
                item_modality = item.get("modality", "I2V" if ref_path else "T2V")
                reference_role = item.get("reference_role", "model_condition" if ref_path else "none")
                dims = item.get("dimensions") or []
            item_modality = str(item_modality or "T2V").upper()
            if item_modality not in {"T2V", "I2V", "CI2V"}:
                print(f"  [WARN] item {idx}: 未知模态 {item_modality}，按 T2V 处理")
                item_modality = "T2V"

            if ref_images and idx < len(ref_images) and ref_images[idx]:
                ref_path = ref_images[idx]

            frames_pil, _ = self.sampler.sample_from_video(video_path)
            if not frames_pil:
                print(f"  [WARN] item {idx}: 无法加载视频 {video_path}")
                continue

            video_frames.append((prompt, frames_pil, dims))
            loaded_modalities.append(item_modality)

            ref_img = None
            if ref_path and os.path.exists(ref_path):
                try:
                    ref_img = Image.open(ref_path).convert("RGB")
                except Exception as e:
                    print(f"  [WARN] item {idx}: 参考图加载失败 {ref_path}: {e}")
            loaded_refs.append(ref_img)
            loaded_reference_roles.append(reference_role)

        modality = (loaded_modalities[0] if loaded_modalities and
                    len(set(loaded_modalities)) == 1 else "MIXED")
        print(f"[Global Benchmark] 有效样本模态: {modality}")

        if not video_frames:
            print("[Global Benchmark] 无有效视频，跳过。")
            return result

        # ── L1 逻辑层 ──
        if run_logic and self.logic_cfg.get("enabled", True):
            print("\n=== L1 逻辑层 ===")
            logic_cfg = self.logic_cfg.get("modules", {})

            # IF 指令遵循+语义对齐（IF-1~9）
            # 注: IF-10 子指标由 ReferenceFidelityEvaluator 计算，报告层归入 L1 IF-10
            if logic_cfg.get("instruction_following", True):
                print(f"  [L1-IF] 指令遵循+语义对齐评测（9维, modality={modality}）...")
                if_eval = InstructionFollowingEvaluator(
                    self.loader.get_vlm("logic"),
                    specialist=self.loader.get_specialist()
                )
                layer_if = if_eval.evaluate_global(
                    video_frames,
                    ref_images=loaded_refs if modality != "T2V" else None
                )
                result.add_layer(layer_if)
                reporter.print_layer_summary(layer_if)

            # CS 常识推理
            if logic_cfg.get("commonsense", True):
                print("  [L1-CS] 常识推理评测...")
                cs_eval = CommonsenseEvaluator(
                    self.loader.get_vlm("logic")
                )
                layer_cs = cs_eval.evaluate_global(video_frames)
                result.add_layer(layer_cs)
                reporter.print_layer_summary(layer_cs)

            # AC 动作因果（方案A: 用 unified 46d 里标注 AC-1 的样本，单视频降级模式）
            if logic_cfg.get("action_causal", True):
                print("  [L1-AC] 动作因果效应评测（单视频降级，按 dimensions 过滤）...")
                try:
                    ac_eval = ActionCausalEvaluator(
                        self.loader.get_vlm("logic"),
                        self.loader.get_spatial_tracker()
                    )
                    # 方案A: 从 video_frames 筛选标注 AC-1 的样本(单视频, 无 control)。
                    # 把 prompt 作为 case["expected_outcome"], 让 VLM 判断"动作是否发生
                    # 且引发状态变化"(动作因果本质), 而非用位移量绝对值(对微小动作不敏感)。
                    ac_samples = [(prompt, frames) for prompt, frames, dims in video_frames
                                  if dims and "AC-1" in dims]
                    if ac_samples:
                        ac_cases = [{"expected_outcome": prompt} for prompt, _ in ac_samples]
                        ac_videos = [(frames, None) for _, frames in ac_samples]
                        layer_ac = ac_eval.evaluate_global(ac_cases, ac_videos)
                        result.add_layer(layer_ac)
                        reporter.print_layer_summary(layer_ac)
                    else:
                        print("  [L1-AC] 无标注 AC-1 的样本，跳过")
                except Exception as e:
                    print(f"  [L1-AC] 评测异常: {e}")

        # ── L2 画质层 ──
        if run_quality and self.quality_cfg.get("enabled", True):
            print("\n=== L2 画质层 ===")
            quality_cfg = self.quality_cfg.get("modules", {})

            # 提取 (帧, 维度) 列表 (带维度标注供 FQ/TQ 过滤)
            frames_only = [(f, d) for _, f, d in video_frames]

            # FQ 帧级质量 (FQ-1 DINOv3 主体一致性 + FQ-2 FineVQ 美学质量)
            if quality_cfg.get("frame_quality", True):
                print("  [L2-FQ] 帧级质量评测...")
                # FineVQ 只在 L2 层加载 (与物理层模型错峰), 用完卸载释放显存
                finevq = self.loader.get_finevq()
                try:
                    fq_eval = FrameQualityEvaluator(
                        self.loader.get_specialist(), finevq=finevq)
                    layer_fq = fq_eval.evaluate_global(frames_only)
                    result.add_layer(layer_fq)
                    reporter.print_layer_summary(layer_fq)
                finally:
                    if getattr(finevq, "is_loaded", False):
                        finevq.unload()
                        print("  [L2-FQ] FineVQ 已卸载 (释放显存)")

            # TQ 时序质量
            if quality_cfg.get("temporal_quality", True):
                print("  [L2-TQ] 时序质量评测...")
                tq_eval = TemporalQualityEvaluator(self.loader.get_specialist())
                layer_tq = tq_eval.evaluate_global(frames_only)
                result.add_layer(layer_tq)
                reporter.print_layer_summary(layer_tq)

            # RF 参考图保真度 (原IF-10a/c/d, 仅逐样本 I2V/CI2V)
            if quality_cfg.get("reference_fidelity", False) and any(r is not None for r in loaded_refs):
                print("  [L2-RF] 参考图保真度评测（逐样本区分 I2V/CI2V）...")
                rf_eval = ReferenceFidelityEvaluator(self.loader.get_specialist())
                # frames_only 是 (帧, dims) 元组列表: RF 内部解包取帧, 并用 dims
                # 做 IF-10 按标注过滤 (与 FQ/TQ 口径一致)。
                layer_rf = rf_eval.evaluate_global(
                    frames_only, ref_images=loaded_refs,
                    modalities=loaded_modalities,
                    reference_roles=loaded_reference_roles)
                result.add_layer(layer_rf)
                reporter.print_layer_summary(layer_rf)

        # IF-10a/c/d 是 IF-10 的三个实现子指标：合并成一个L1维度，
        # 从结果层移除IF-10底层子指标层，避免IF-10重复计入L2基础7维。
        self._consolidate_reference_fidelity(result, modality)

        # 生成报告
        reporter.generate(result, output_prefix="global_benchmark")

        # 打印总览
        self._print_overview(result)
        return result

    @staticmethod
    def _consolidate_reference_fidelity(result: BenchmarkResult,
                                         modality: str) -> None:
        """将RF子指标按样本聚合为L1的IF-10，并移除底层子指标层。

        I2V每个样本通常有IF-10a/c；CI2V还可能有IF-10d。先按item聚合
        可避免CI2V的第三个子指标让某些样本被重复加权。
        T2V没有参考图，保留L1 IF-10为未计分状态，不伪造分数。
        """
        rf_layer = next((layer for layer in result.layers
                         if layer.layer == "IF-10-参考条件保真度-底层子指标"), None)
        if rf_layer is None:
            if str(modality).upper() == "T2V":
                if_layer = next((layer for layer in result.layers
                                 if layer.layer == "L1-指令遵循+语义对齐"), None)
                if if_layer is not None:
                    if_layer.results.append(EvalResult.skipped(
                        "IF-10-参考条件保真度", "T2V无参考图，IF-10不适用"))
            return

        grouped: dict[int, list[EvalResult]] = {}
        for item in rf_layer.results:
            if item.status not in {"ok", "proxy", "degraded"}:
                continue
            item_index = item.metrics.get("item")
            if item_index is None:
                item_index = len(grouped)
            grouped.setdefault(int(item_index), []).append(item)

        sub_scores: dict[str, list[float]] = {
            "IF-10a": [], "IF-10c": [], "IF-10d": []}
        sample_scores = []
        sample_details = []
        for item_index, items in sorted(grouped.items()):
            scores = []
            detail = {"item": item_index}
            for item in items:
                sub_id = item.dimension.split("-", 2)[0] + "-" + item.dimension.split("-", 2)[1]
                if sub_id not in sub_scores:
                    continue
                sub_scores[sub_id].append(float(item.score))
                scores.append(float(item.score))
                detail[sub_id] = item.score
            if scores:
                sample_scores.append(float(np.mean(scores)))
                sample_details.append(detail)

        # RF层不再作为独立层计入总分；其细节保存在IF-10聚合metrics中。
        result.layers = [layer for layer in result.layers if layer is not rf_layer]
        if not sample_scores:
            return

        if_layer = next((layer for layer in result.layers
                         if layer.layer == "L1-指令遵循+语义对齐"), None)
        if if_layer is None:
            return
        mean_if10 = float(np.mean(sample_scores))
        if_layer.mean_scores["IF-10-参考条件保真度"] = mean_if10
        if_layer.aggregates["IF-10_distribution"] = {
            str(score): int(sum(round(value) == score for value in sample_scores))
            for score in range(1, 6)
        }
        is_t2v_proxy = str(modality).upper() == "T2V"
        if_layer.results.append(EvalResult(
            "IF-10-参考条件保真度",
            int(np.clip(round(mean_if10), 1, 5)),
            metrics={
                "aggregation": "per_sample_mean_of_IF-10a_c",
                "subdimension_means": {
                    key: (float(np.mean(values)) if values else None)
                    for key, values in sub_scores.items()
                },
                "n_samples": len(sample_scores),
                "sample_details": sample_details,
                "source_layer": "IF-10-参考条件保真度-底层子指标",
                "modality": modality,
                "reference_role": "t2v_canonical" if is_t2v_proxy else "model_condition",
                "reference_source": "gpt-image-2" if is_t2v_proxy else "model_condition",
                "reference_is_gt": False,
            },
            status="proxy" if is_t2v_proxy else "ok",
        ))

    @staticmethod
    def _detect_modality(mapping: list,
                         ref_images: Optional[list] = None) -> str:
        """
        根据输入自动检测模态类型

        Returns: "T2V" | "I2V" | "CI2V"
        """
        has_ref = False
        # 检查独立 ref_images
        if ref_images and any(r is not None for r in ref_images):
            has_ref = True
        # 检查 mapping 中的 triple 格式
        if not has_ref:
            for item in mapping:
                if isinstance(item, (list, tuple)) and len(item) >= 3 and item[2]:
                    has_ref = True
                    break
                elif isinstance(item, dict) and item.get("ref_image"):
                    has_ref = True
                    break

        if not has_ref:
            return "T2V"

        # 进一步区分 I2V vs CI2V: 检查 ref_images 是否含"character"/"角色"标注
        # 或 mapping 中的 "modality" 字段
        for item in mapping:
            if isinstance(item, dict):
                mod = item.get("modality", "").upper()
                if mod in ("CI2V", "CHARACTER"):
                    return "CI2V"
                if mod == "I2V":
                    return "I2V"

        # 默认: 有参考图但未标注 → I2V
        return "I2V"

    @staticmethod
    def _match_ac_videos(ac_cases: list[dict],
                         video_frames: list[tuple[str, list[Image.Image]]]
                         ) -> list:
        """
        将 video_frames 按 prompt text 匹配到各 AC test case。

        返回与 ac_cases 等长的 list，每项是 (action_frames, None) 元组
        （单视频降级模式，control 视频为 None）。
        若有 control_prompt 匹配的视频，则填为 (action_frames, control_frames)。
        """
        # 构建 prompt → frames 索引
        prompt_map: dict[str, list[Image.Image]] = {}
        for item in video_frames:
            prompt = item[0]
            frames = item[1]
            prompt_map[prompt.strip()] = frames

        matched = []
        for case in ac_cases:
            action_prompt = case.get("text", "").strip()
            control_prompt = case.get("control_prompt", "").strip()

            action_frames = prompt_map.get(action_prompt)
            if action_frames is None:
                # 模糊匹配：去掉标点后比较
                import re
                action_key = re.sub(r'[^\w\s]', '', action_prompt).lower()
                for k, v in prompt_map.items():
                    if re.sub(r'[^\w\s]', '', k).lower() == action_key:
                        action_frames = v
                        break

            control_frames = None
            if control_prompt:
                control_frames = prompt_map.get(control_prompt)
                if control_frames is None:
                    import re
                    ctrl_key = re.sub(r'[^\w\s]', '', control_prompt).lower()
                    for k, v in prompt_map.items():
                        if re.sub(r'[^\w\s]', '', k).lower() == ctrl_key:
                            control_frames = v
                            break

            if action_frames is not None:
                matched.append((action_frames, control_frames))

        # 无精确匹配时：将全部 video_frames 作为 action 视频传入
        # （AC 评测器内部会按 case idx 循环，允许 videos 数量与 cases 不一致）
        if not matched and video_frames:
            matched = [(item[1], None) for item in video_frames]

        return matched

    @staticmethod
    def _print_overview(result: BenchmarkResult):
        print(f"\n{'='*60}")
        print("Global Benchmark 总览")
        print(f"{'='*60}")
        overall = result.overall_score()
        print(f"Overall Score: {overall:.2f} / 5.00")
        for layer in result.layers:
            layer.compute_avg()
            print(f"  [{layer.layer}] avg={layer.avg_score:.2f} "
                  f"(n_results={len(layer.results)})")
        print(f"{'='*60}\n")
