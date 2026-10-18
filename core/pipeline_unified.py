#!/usr/bin/env python3
"""
L1+L2+L3 统一评测管线

整合三层评测为单一入口：
  L1 逻辑层: 指令遵循(IF) + 常识推理(CS) + 动作因果(AC)
  L2 画质层: 帧质量(FQ) + 时序质量(TQ) + 参考图保真度(RF)
  L3 物理层: 25维物理量 (几何/标度位姿/形变空间/静力学光影/材料/热/磁电/相变/运动学/持续性/动力学)

特性:
  - 逐视频 prompt 自动分析 (auto_analyzer): 根据文本关键词自动判断评测维度
  - 分步执行: L3 逐视频物理评测 → L1+L2 全局评测 → 合并报告
  - manifest 驱动: 输入 JSON 文件统一描述全部测试条目

用法:
  python pipeline_unified.py --manifest manifest.json                     # 全量评测
  python pipeline_unified.py --manifest manifest.json --auto-analyze      # 自动按维度过滤
  python pipeline_unified.py --manifest manifest.json --layers physics    # 仅物理层
  python pipeline_unified.py --manifest manifest.json --limit 3           # 仅前3条
"""

import os
import sys
import json
import argparse
import time
from typing import Optional
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# HF 缓存目录 (必须在 import HF 模型前设置)
_hf_cache = os.environ.get("HF_HOME") or os.path.join(PROJECT_ROOT, "hf_cache")
os.environ["HF_HOME"] = _hf_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = _hf_cache

from pipeline_physics import PhysicsPipeline
from pipeline_global import GlobalBenchmarkPipeline
from evaluators.auto_analyzer import PhysicsAutoAnalyzer
from evaluators.base import BenchmarkResult
from reporter import Reporter

# ── 维度编号 → 模块名映射 ──
DIMENSION_TO_MODULE = {
    "R01": "geometry", "R02": "geometry", "R03": "geometry",
    "R04": "scale_pose", "R09": "scale_pose", "R10": "scale_pose",
    "R05": "deformation_spatial", "R11": "deformation_spatial",
    "R12": "statics_lighting", "R13": "statics_lighting", "R18": "statics_lighting",
    "R06": "material",
    "R07": "thermal",
    "R14": "electromag",
    "R25": "phase_change",
    "R15": "kinematics", "R16": "kinematics", "R17": "kinematics",
    "R19": "motion_persistence", "R20": "motion_persistence", "R21": "motion_persistence",
    "R22": "dynamics", "R23": "dynamics", "R24": "dynamics",
    "R08": "multiview",
}


def dimensions_to_modules(dimensions: list[str]) -> list[str]:
    """将维度列表转换为所需模块列表。"""
    mods = set()
    for d in dimensions:
        m = DIMENSION_TO_MODULE.get(d)
        if m:
            mods.add(m)
    return sorted(mods)


class UnifiedPipeline:
    """L1+L2+L3 统一评测管线。

    使用方式典型流程:
        pipe = UnifiedPipeline("configs/config.yaml")
        pipe.run(manifest, auto_analyze=True)
    或分步:
        pipe.initialize()
        pipe.run_physics(manifest)
        pipe.run_global(manifest)
        pipe.print_report()
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.physics_pipeline: Optional[PhysicsPipeline] = None
        self.global_pipeline: Optional[GlobalBenchmarkPipeline] = None
        self.auto_analyzer = PhysicsAutoAnalyzer(default_enable=True)
        self.output_dir: str = ""

        # 累积结果
        self.physics_results: list[dict] = []
        self.global_result: Optional[dict] = None

    def initialize(self):
        """初始化所有模型 (L1+L2 VLM+专家模型 + L3 专用模型)。"""
        print(f"\n{'='*60}")
        print("初始化 Unified Pipeline — L1+L2+L3")
        print(f"{'='*60}")

        # ⚠️ 共享 loader: L1+L2 与 L3 共用同一个 ModelLoader, 避免 VLM(27GB 8bit)
        #    被 global_pipeline 和 physics_pipeline 各加载一份 → 双倍显存(54GB) OOM。
        #    两个 pipeline 通过 initialize(loader=...) 注入同一个实例。
        import yaml as _yaml
        from models.loader import ModelLoader
        with open(self.config_path, "r", encoding="utf-8") as _f:
            _config = _yaml.safe_load(_f)
        self._shared_loader = ModelLoader(_config)

        print("\n>>> L1+L2 全局模型初始化...")
        self.global_pipeline = GlobalBenchmarkPipeline(config_path=self.config_path)
        self.global_pipeline.initialize(loader=self._shared_loader)

        print("\n>>> L3 物理模型初始化...")
        self.physics_pipeline = PhysicsPipeline(config_path=self.config_path)
        self.physics_pipeline.initialize(loader=self._shared_loader)

        import yaml
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.output_dir = config.get("global", {}).get("output_dir", "./results")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"\n[Unified] 全部模型初始化完成。输出目录: {self.output_dir}")

    def pregenerate_meshes(self, manifest: list[dict]) -> None:
        """Phase 0: Any6D mesh 预生成 (在全量模型初始化之前, 显存空闲时).

        R10/R17 的 Any6D 原生路径需要 InstantMesh mesh; 主评测循环只复用缓存
        (skip_mesh_generation=True —— Zero123++ ~15-30GB 与常驻 VLM/物理模型
        共存会 OOM)。故在 initialize 之前, 用轻量 loader (仅 SAM3+DA3+Any6D)
        补齐缺失缓存, 生成完卸载, 之后主循环直接命中缓存走 Any6D 原生路径。
        与 scripts/pregenerate_any6d_mesh.py 同逻辑, 但 prompt 优先 eval_prompt。
        """
        targets = [x for x in manifest
                   if {"R10", "R17"} & set(x.get("dimensions") or [])]
        if not targets:
            return
        import yaml
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from pregenerate_any6d_mesh import mesh_cache_path

        def _cached(video):
            mp = mesh_cache_path(config, video)
            return os.path.isfile(mp) and os.path.getsize(mp) > 0

        missing = [x for x in targets if not _cached(x["video"])]
        print(f"\n[Unified/Phase0] Any6D mesh: 目标 {len(targets)}, "
              f"已缓存 {len(targets)-len(missing)}, 缺失 {len(missing)}")
        if not missing:
            print("[Unified/Phase0] mesh 缓存已齐, 跳过预生成")
            return

        import numpy as np
        from models.loader import ModelLoader
        from sampler import FrameSampler
        print("[Unified/Phase0] 加载 SAM3+DA3+Any6D (不加载 VLM, "
              "防与 Zero123++ 叠加 OOM) ...")
        loader = ModelLoader(config)
        sam3 = loader.get_sam3()
        da3 = loader.get_da3()
        any6d = loader.get_any6d()
        if not getattr(any6d, "is_loaded", False):
            print("[Unified/Phase0] Any6D 未加载, 跳过预生成 (R10/R17 走降级)")
            return
        sampler = FrameSampler(config.get("sampling", {}))

        ok, failed = 0, 0
        for i, item in enumerate(missing):
            video = item["video"]
            pid = item.get("prompt_id", "?")
            mp = mesh_cache_path(config, video)
            try:
                frames, _ = sampler.sample_from_video(video)
                if not frames:
                    raise RuntimeError("视频采样失败")
                auto = any6d.prepare_auto_inputs(
                    np.stack([np.asarray(f) for f in frames]),
                    sam3=sam3, da3=da3, dimension="R10",
                    mesh_path=(config.get("models", {}).get("physics", {})
                               .get("any6d", {}).get("mesh_path")),
                    output_dir=os.path.dirname(mp),
                    prompt=item.get("eval_prompt") or item.get("prompt"),
                )
                print(f"  [{i+1}/{len(missing)}] {pid}: ✓ {auto.get('any6d_mesh')}")
                ok += 1
            except Exception as e:
                print(f"  [{i+1}/{len(missing)}] {pid}: [FAIL] "
                      f"{type(e).__name__}: {e}")
                failed += 1
        print(f"[Unified/Phase0] mesh 预生成完成: 成功 {ok}, 失败 {failed} "
              f"(失败条目 R10/R17 走 VGGT/Kabsch 降级)")

        # 卸载 InstantMesh/SAM3/DA3, 让 initialize 干净加载全量模型
        del loader, sam3, da3, any6d
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_physics(self, manifest: list[dict],
                    auto_analyze: bool = False,
                    limit: Optional[int] = None) -> list[dict]:
        """L3 物理层逐视频评测。

        Args:
            manifest: [{prompt, video, ref_image?, modality?, dimensions?}, ...]
            auto_analyze: 是否根据 prompt 自动选择评测维度
            limit: 仅评测前 N 条

        Returns:
            [{prompt_id, prompt, dimensions, layers: {mod: score}, details: {mod: [{dim, score, status}]}}, ...]
        """
        if self.physics_pipeline is None:
            self.initialize()

        items = manifest[:limit] if limit else manifest
        results: list[dict] = []

        print(f"\n{'='*60}")
        print(f"L3 物理层评测 ({len(items)} 条)")
        if auto_analyze:
            print("  auto_analyze=ON — 根据 prompt 自动选择维度")
        print(f"{'='*60}")

        for i, item in enumerate(items):
            # eval_prompt: 物理层判据专用场景描述 (I2V 条目的 T2V 原版, 含物体
            # 颜色/前后关系声明)。缺省回退真实生成指令 prompt。L1 IF 不受影响
            # (run_global 仍传 item["prompt"])。
            prompt = item.get("eval_prompt") or item.get("prompt", "")
            video = item.get("video", "")
            ref_image = item.get("ref_image")
            modality = item.get("modality", "T2V")

            print(f"\n--- [{i+1}/{len(items)}] {item.get('prompt_id', '?')} ---")
            print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

            if not os.path.exists(video):
                print(f"  [SKIP] 视频不存在: {video}")
                continue

            # 确定评测维度
            target_dimensions = item.get("dimensions")
            modules = None

            if auto_analyze:
                # 自动分析模式
                analysis = self.auto_analyzer.analyze_from_prompt(prompt)
                if target_dimensions is None:
                    # 只评测有明确关键词匹配的维度 (非 default_enable)
                    strict_analysis = PhysicsAutoAnalyzer(default_enable=False).analyze_from_prompt(prompt)
                    target_dimensions = strict_analysis["enabled_dimensions"]
                modules = analysis["enabled_modules"]
                if target_dimensions:
                    modules = dimensions_to_modules(target_dimensions)
                print(f"  auto_analyze: matched={sorted(analysis['matched'].keys())}, "
                      f"modules={modules}, dims={target_dimensions}")
            elif target_dimensions:
                # manifest 中显式指定了维度
                modules = dimensions_to_modules(target_dimensions)
                print(f"  manifest dimensions: {target_dimensions} → modules={modules}")

            # ⚠️ 优化：dimensions 全是 L1/L2 语义维度（IF/AC/CS/FQ/TQ，无 R 维度）时，
            #    modules 为空，物理层无需评测。跳过空跑，避免浪费物理层推理时间。
            if modules is not None and len(modules) == 0:
                print(f"  [SKIP] 无物理维度(R01-R25)，跳过物理评测（该视频仅测 L1/L2 语义层）")
                continue

            t0 = time.time()
            try:
                result = self.physics_pipeline.run(
                    video_path=video,
                    modules=modules,
                    ref_image_path=ref_image,
                    modality=modality,
                    target_dimensions=target_dimensions,
                    prompt=prompt,
                    storyboard=item.get("storyboard"),
                    evaluation_method=item.get("evaluation_method", "default"),
                    panel_layout=item.get("panel_layout"),
                    evaluator_context=item.get("evaluator_context") or {},
                    multiview_image_paths=item.get("multiview_images") or [],
                )
            except Exception as e:
                print(f"  [ERROR] 物理评测失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            elapsed = time.time() - t0

            # 收集每层得分
            layer_scores = {}
            dim_details = {}
            for layer in result.layers:
                layer.compute_avg()
                mod_name = layer.layer.split("-")[-1] if "-" in layer.layer else layer.layer
                layer_scores[mod_name] = round(layer.avg_score, 2)
                per_dim = []
                for r in layer.results:
                    per_dim.append({
                        "dimension": r.dimension,
                        "score": r.score,
                        "status": r.status,
                        "violations": r.violations[:3] if r.violations else [],
                        "metrics": {
                            k: v for k, v in r.metrics.items()
                            if isinstance(v, (int, float, str, bool))
                        },
                    })
                dim_details[mod_name] = per_dim

            results.append({
                "prompt_id": item.get("prompt_id", ""),
                "prompt": prompt,
                "video": video,
                "source": item.get("source", ""),
                "dimensions": target_dimensions or [],
                "layer_scores": layer_scores,
                "dim_details": dim_details,
                "elapsed": round(elapsed, 1),
            })

            print(f"  得分: {layer_scores}  ({elapsed:.0f}s)")

        self.physics_results = results
        return results

    def run_global(self, manifest: list[dict],
                   limit: Optional[int] = None) -> dict:
        """L1+L2 全局 Benchmark 评测。

        Args:
            manifest: [{prompt, video, ref_image?, modality?}, ...]
            limit: 仅评测前 N 条

        Returns:
            {overall_score, total_videos, elapsed}
        """
        if self.global_pipeline is None:
            self.initialize()

        items = manifest[:limit] if limit else manifest

        print(f"\n{'='*60}")
        print(f"L1+L2 全局 Benchmark ({len(items)} 条)")
        print(f"{'='*60}")

        # 转换为 GlobalBenchmarkPipeline 接受的格式。
        # ⚠️ 必须带上 dimensions: 全局评测器(evaluate_global)按样本标注的维度过滤,
        #    若缺 dimensions 会得到空列表 → 所有 L1/L2 维度被跳过 → 0 results。
        mapping_tuples = []
        for item in items:
            video = item.get("video", "")
            if os.path.exists(video):
                mapping_tuples.append((
                    item["prompt"],
                    video,
                    item.get("ref_image"),
                    item.get("modality", "T2V"),
                    item.get("dimensions") or [],
                ))

        if not mapping_tuples:
            print("[Global] 无有效视频，跳过。")
            return {"overall_score": 0, "total_videos": 0, "elapsed": 0}

        t0 = time.time()
        result = self.global_pipeline.run(prompt_video_mapping=mapping_tuples)
        elapsed = time.time() - t0

        overall = result.overall_score()
        print(f"\n  全局均分: {overall:.2f} / 5.00  ({elapsed:.0f}s)")

        self.global_result = {
            "overall_score": round(overall, 2),
            "total_videos": len(mapping_tuples),
            "elapsed": round(elapsed, 1),
        }
        return self.global_result

    def run(self, manifest: list[dict],
            layers: str = "all",
            auto_analyze: bool = False,
            limit: Optional[int] = None) -> dict:
        """全流程执行。

        Args:
            manifest: 测试条目列表 (见 load_manifest)
            layers: "all" | "physics" | "global"
            auto_analyze: 物理层是否自动选择维度
            limit: 仅处理前 N 条

        Returns:
            {"physics": [...], "global": {...}}
        """
        t_start = time.time()

        if layers in ("physics", "all"):
            self.run_physics(manifest, auto_analyze=auto_analyze, limit=limit)

        # ── 阶段切换: 物理层(L3)跑完 → 全局层(L1+L2)开始前, 卸载物理专用模型 ──
        # 全局评测需要 VLM + Specialist(DINO) + FineVQ + SpatialTracker(AC动作因果),
        # 不需要 VGGT/SAM3/Any6D/DA3/StableNormal/ReasonX/FlowSeek/VFIMamba。
        # 物理层这些大模型(尤其 SAM3/Any6D/InstantMesh 残留)占满显存会导致
        # FineVQ/VLM 加载 OOM。故卸载它们腾空间。
        if layers in ("global", "all"):
            loader = self._shared_loader if hasattr(self, "_shared_loader") \
                else getattr(self.physics_pipeline, "loader", None)
            if loader is not None:
                print("\n[Unified] 物理层完成，卸载物理专用模型腾显存供全局评测...")
                try:
                    # 保留 VLM(L1逻辑层) + specialist(DINO, L2画质) + spatial_tracker(AC)
                    loader.unload_except(
                        {"vlm", "specialist", "spatial_tracker"})
                except Exception as e:
                    print(f"[Unified] 卸载物理模型失败(非阻塞): {e}")
            self.run_global(manifest, limit=limit)

        elapsed = time.time() - t_start
        print(f"\n[Unified] 全管线完成, 总耗时 {elapsed:.0f}s")
        return {
            "physics": self.physics_results,
            "global": self.global_result,
            "total_elapsed": round(elapsed, 1),
        }

    def generate_report(self, output_dir: Optional[str] = None):
        """生成合并 JSON + 控制台汇总报告。"""
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        # JSON 报告
        report = {
            "physics_results": self.physics_results,
            "global_result": self.global_result,
        }
        report_path = os.path.join(out, "unified_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[Unified] JSON 报告: {report_path}")

        # 控制台汇总
        self._print_report()

    def _print_report(self):
        """打印合并汇总报告。"""
        print(f"\n{'='*70}")
        print("  Physical World Benchmark — 统一报告 (L1+L2+L3)")
        print(f"{'='*70}")

        # ── L1+L2 全局 ──
        if self.global_result and self.global_result.get("total_videos", 0) > 0:
            g = self.global_result
            print(f"\n  [L1+L2 全局评测]")
            print(f"    总分: {g['overall_score']:.2f} / 5.00")
            print(f"    视频数: {g['total_videos']}")
            print(f"    耗时: {g['elapsed']:.0f}s")

        # ── L3 物理 ──
        if self.physics_results:
            n = len(self.physics_results)
            # 按维度聚合
            dim_scores: dict[str, list[float]] = defaultdict(list)
            dim_status: dict[str, list[str]] = defaultdict(list)
            for r in self.physics_results:
                for mod, details in r.get("dim_details", {}).items():
                    for d in details:
                        dim = d["dimension"]
                        # unobservable 的 score=0 是占位值不是物理分数, 计入均值
                        # 会把"未观测/未计分"错误显示成 0 分 (如 R01 avg=0.00)。
                        if d["status"] not in ("error", "skipped", "unobservable"):
                            dim_scores[dim].append(d["score"])
                        dim_status[dim].append(d["status"])

            print(f"\n  [L3 物理层 — {n} 个视频]")
            # 遍历 dim_status (含全 error/skipped/unobservable 的维度), 而非 dim_scores:
            # 否则全未计分的维度会整行消失, 看起来像该维度未被评测。
            for dim in sorted(dim_status):
                scores = dim_scores.get(dim, [])
                n_total = len(dim_status[dim])
                if scores:
                    avg = sum(scores) / len(scores)
                    n_ok = dim_status[dim].count("ok")
                    print(f"    {dim}: avg={avg:.2f}  (n={n_total}, ok={n_ok})")
                else:
                    sc = {k: v for k, v in dim_status[dim].items()}
                    print(f"    {dim}: [无有效计分]  (n={n_total}, {sc})")

        print(f"\n  详细报告: {self.output_dir}")
        print(f"{'='*70}\n")


# ── Manifest 工具函数 ──

def load_manifest(path: str) -> list[dict]:
    """加载 manifest JSON 文件。

    manifest 格式 (每项):
        {
            "prompt_id": "PHY-R15-001",      # 可选, 标识
            "prompt": "a billiard ball...",   # 必填
            "video": "/path/to/video.mp4",    # 必填
            "ref_image": "/path/to/ref.png",  # CI2V/I2V 可选
            "modality": "T2V",                # T2V/I2V/CI2V, 默认 T2V
            "dimensions": ["R15", "R16"],      # 手动指定目标维度, auto_analyze 离线可用
            "source": "wan2.1-14b"            # 可选, 生成源标记
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 规范化: 确保每项有 prompt + video
    validated = []
    for i, item in enumerate(manifest):
        if "prompt" not in item or "video" not in item:
            print(f"[Manifest] 条目 {i} 缺 prompt/video, 跳过")
            continue
        item.setdefault("modality", "T2V")
        validated.append(item)

    print(f"[Manifest] 加载 {len(validated)} 条 (原始 {len(manifest)})")
    return validated


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Physical World Benchmark — L1+L2+L3 统一评测入口"
    )
    parser.add_argument("--manifest", "-m", required=True,
                        help="manifest JSON 文件路径 (每条含 prompt + video + 可选 ref_image/dimensions/modality)")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                        help="配置文件路径 (默认 config.yaml)")
    parser.add_argument("--layers", choices=["physics", "global", "all"],
                        default="all", help="评测层级 (默认 all)")
    parser.add_argument("--auto-analyze", action="store_true",
                        help="根据 prompt 文本自动选择物理评测维度 (未在 manifest 声明 dimensions 时启用)")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅评测前 N 条 (快速验证)")
    parser.add_argument("--no-pregen-mesh", action="store_true",
                        help="跳过 Phase 0 Any6D mesh 预生成 (未缓存条目 R10/R17 走降级)")
    parser.add_argument("--output-dir", default=None,
                        help="报告输出目录 (默认 configs/config.yaml 中的 output_dir)")

    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Error: manifest 未找到: {args.manifest}")
        sys.exit(1)

    manifest = load_manifest(args.manifest)
    if not manifest:
        print("Error: manifest 无有效条目")
        sys.exit(1)

    if args.limit:
        manifest = manifest[:args.limit]
        print(f"限制评测数量: {args.limit}")

    pipe = UnifiedPipeline(config_path=args.config)
    # Phase 0: Any6D mesh 预生成 (在全量模型加载前, 显存空闲时补齐缓存;
    # 之后主循环直接命中缓存走 Any6D 原生路径, 无需再单独跑预处理脚本)
    if args.layers in ("physics", "all") and not args.no_pregen_mesh:
        pipe.pregenerate_meshes(manifest)
    pipe.run(
        manifest=manifest,
        layers=args.layers,
        auto_analyze=args.auto_analyze,
        limit=args.limit,
    )
    pipe.generate_report(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
