"""
物理层评测管线 (L3)

支持三种模态的差异化评估:
- T2V: 文本条件视频 → prompt 代理 / 自洽性分析
- I2V: 初始帧 + 动作 prompt → 自洽性分析，参考图仅作条件保真度而非 GT
- CI2V: 受控输入（相机/场景/角色）→ 仅使用仿真导出的外部 GT 做精确定量评测

流程: 加载视频(+可选参考图) → 逐物理组评测 → 逐模块报告
每个物理量使用表格 D 列指定的专用模型
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
from evaluators.base import BenchmarkResult, LayerResult
from evaluators.seeding import set_global_seed
from evaluators.physics import (
    GeometryEvaluator,
    ScalePoseEvaluator,
    DeformationSpatialEvaluator,
    StaticsLightingEvaluator,
    MaterialEvaluator,
    ThermalEvaluator,
    ElectromagEvaluator,
    PhaseChangeEvaluator,
    KinematicsEvaluator,
    MotionPersistenceEvaluator,
    DynamicsEvaluator,
    MultiviewEvaluator,
)
from evaluators.auto_analyzer import PhysicsAutoAnalyzer
from reporter import Reporter


class PhysicsPipeline:
    """
    物理层评测管线 — 支持 T2V / I2V / CI2V

    CI2V 模式: 从参考图自动计算 GT（深度/法线/点云/分割/位姿），
               传递给各 evaluator 做精确误差对比。
    T2V/I2V 模式: evaluator 回退到合理性/自洽性检查。
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 全局可复现: 统一随机种子
        set_global_seed(self.config.get("global", {}).get("seed", 42))

        self.physics_cfg = self.config.get("physics", {}).get("modules", {})
        self.output_dir = self.config.get("global", {}).get("output_dir", "./results")
        os.makedirs(os.path.join(self.output_dir, "physics"), exist_ok=True)

        self.loader: Optional[ModelLoader] = None
        self.sampler = FrameSampler(self.config.get("sampling", {}))
        self._last_video_path: Optional[str] = None

        # P1: 全帧模块配置 (从 config 读取, 控制哪些模块使用全帧而非采样帧)
        sampling_cfg = self.config.get("sampling", {})
        self._full_frame_modules = set(sampling_cfg.get("full_frame_modules", []) or [])
        self._full_frame_max = int(sampling_cfg.get("full_frame_max", 300) or 0)
        self._cached_full_frames: Optional[list[Image.Image]] = None

        # P2: VLM offload 开关 (H20-95GB 默认关闭, 让 VLM 常驻 GPU 避免每视频往返搬移)
        self._vlm_offload = bool(self.config.get("physics", {}).get("vlm_offload", False))

    def initialize(self, loader: Optional[ModelLoader] = None) -> ModelLoader:
        """初始化所有需要的模型。

        loader: 可选，传入共享的 ModelLoader（UnifiedPipeline 让 L1+L2 与 L3
                共享同一 loader，避免 VLM 等模型被重复加载两份占双倍显存）。
        """
        self.loader = loader if loader is not None else ModelLoader(self.config)
        print("[Physics Pipeline] 正在初始化专用模型...")

        modules = [
            # (module_key, loader_methods, description)
            ("geometry", ["get_vggt", "get_stable_normal", "get_sam3", "get_vlm"],
             "VGGT-Omega + StableNormal + SAM3 + Qwen3-VL(R03非标准形状)"),
            ("scale_pose", ["get_da3", "get_sam3", "get_vggt", "get_any6d", "get_vlm"],
             "DA3 + SAM3 + VGGT-Omega + Any6D + Qwen3-VL"),
            ("deformation_spatial", ["get_vggt", "get_spatial_tracker", "get_sam3"],
             "VGGT-Omega + SpatialTracker + SAM3"),
            ("statics_lighting", ["get_vggt", "get_sam3", "get_reasonx", "get_stable_normal"],
             "VGGT-Omega + SAM3 + ReasonX + StableNormal"),
            ("material", ["get_vlm"],
             "Qwen3-VL"),
            ("thermal", ["get_vlm"],
             "Qwen3-VL"),
            ("electromag", ["get_vlm"],
             "Qwen3-VL"),
            ("phase_change", ["get_vlm"],
             "Qwen3-VL"),
            ("kinematics", ["get_spatial_tracker", "get_any6d", "get_sam3"],
             "SpatialTracker + Any6D + SAM3"),
            ("motion_persistence", ["get_vggt", "get_flowseek", "get_vfimamba",
                                    "get_spatial_tracker", "get_sam3", "get_vlm"],
             "VGGT + FlowSeek + VFIMamba + SpatialTracker + SAM3 + Qwen3-VL"),
            ("dynamics", ["get_spatial_tracker", "get_sam3", "get_da3"],
             "SpatialTracker + SAM3(物体级轨迹) + DA3(米制锚定)"),
            ("multiview", ["get_vggt"],
             "VGGT-Omega"),
        ]

        for module_key, methods, desc in modules:
            cfg = self.physics_cfg.get(module_key, {})
            if not cfg.get("enabled", True):
                continue
            print(f"  [{module_key}] {desc} ...")
            for method in methods:
                try:
                    getattr(self.loader, method)()
                except Exception as e:
                    print(f"    Warning: {method}() failed: {e}")

        self.loader.print_status()
        return self.loader

    def run(self, video_path: str,
            modules: Optional[list[str]] = None,
            ref_image_path: Optional[str] = None,
            modality: str = "T2V",
            target_dimensions: Optional[list[str]] = None,
            prompt: Optional[str] = None,
            **kwargs) -> BenchmarkResult:
        """
        执行物理评测

        Args:
            video_path: 视频文件路径
            modules: 指定评测的模块列表。
                None (默认) = 全部启用的模块
                ["geometry", "kinematics", ...] = 指定模块
                "auto" = 根据 prompt 文本自动判断应启用的模块 (需要同时传 prompt 参数)
            ref_image_path: CI2V 角色参考图路径 (T2V/I2V 为 None)
            modality: "T2V" | "I2V" | "CI2V"
            target_dimensions: 仅对这些维度计分 (如 ["R01","R15"]), None=全部
            prompt: T2V prompt 文本。当 modules="auto" 时用于自动分析;
                     也可独立传入用于日志记录和物体验证。
            **kwargs: 额外 GT 参数 (手动传参可覆盖自动计算)

        Returns:
            BenchmarkResult
        """
        if self.loader is None:
            self.initialize()

        # ── "auto" 模式: 根据 prompt 自动判断应启用的模块和维度 ──
        auto_info = None
        if modules == "auto":
            if not prompt:
                print("[Physics] modules='auto' 但未传 prompt, 回退到全部模块")
                modules = None
            else:
                analyzer = PhysicsAutoAnalyzer(default_enable=True)
                auto_info = analyzer.analyze_from_prompt(prompt)
                modules = auto_info["enabled_modules"]
                auto_dims = auto_info["enabled_dimensions"]
                # 若未指定 target_dimensions, 用自动分析的维度
                if target_dimensions is None:
                    target_dimensions = auto_dims
                print(f"[Physics] Auto-analysis from prompt: "
                      f"matched={sorted(auto_info['matched'].keys())}, "
                      f"unmatched={auto_info['unmatched'][:5]}{'...' if len(auto_info['unmatched']) > 5 else ''}")
                print(f"[Physics] Auto-selected modules: {modules}")
                print(f"[Physics] Auto-selected dimensions: {target_dimensions[:8]}{'...' if len(target_dimensions) > 8 else ''}")

        # 采样
        self._last_video_path = video_path
        self._cached_full_frames = None  # 每视频重置全帧缓存
        self._cached_full_fps = None
        frames_pil, video_fps = self.sampler.sample_from_video(video_path)
        if not frames_pil:
            print(f"[Physics] 无法从 {video_path} 采样帧")
            return BenchmarkResult()

        self._video_fps = video_fps  # 缓存 fps 供后续模块使用
        print(f"\n[Physics] 模态={modality} | 采样 {len(frames_pil)} 帧 @ {video_fps:.1f}fps from {video_path}")

        # 参考图是生成条件，不是真值。CI2V 精确路径只接受 manifest.gt
        # 中由仿真器导出的外部数据，禁止用 VGGT/DA3/SAM 的预测自评。
        gt_data = {k: v for k, v in kwargs.items() if v is not None}
        if modality == "CI2V":
            gt_keys = {k for k in gt_data if k.endswith("_gt") or k.startswith("gt_")
                       or k in {"camera_intrinsics", "camera_poses"}}
            if gt_keys:
                print(f"[Physics] CI2V 外部 GT: {sorted(gt_keys)}")
                # ⚠️ 时间轴对齐: UE 导出的是**全帧** GT, 而 sampler 已抽帧。
                # 不对齐会让所有跨帧量(轨迹/速度/角速度/位姿)与预测错帧,
                # 量纲与对错全错。必须按实际抽帧索引同步取GT 帧。
                gt_data = self._align_temporal_gt(gt_data, len(frames_pil))
            else:
                print("[Physics] CI2V 未提供仿真 GT；本条仅运行 proxy 指标，不产生 status=ok")
        if ref_image_path and not os.path.exists(ref_image_path):
            print(f"[Physics] 参考条件图不存在: {ref_image_path}")

        result = BenchmarkResult()
        reporter = Reporter(output_dir=os.path.join(self.output_dir, "physics"))
        # 确定要运行的模块
        if modules is None:
            modules = [k for k, v in self.physics_cfg.items()
                      if v.get("enabled", True)]

        # ── T2V/I2V Any6D proxy输入自动构建 ──
        # 普通T2V/I2V没有CI2V外部mesh/K/depth/mask，按用户确认的方案从
        # 视频首帧构建：SAM3 mask → DA3全帧深度 → 近似K → InstantMesh mesh。
        # 任一环节失败时保留R10/R17的VGGT/Kabsch降级，不伪造Any6D结果。
        #
        # ⚠️ 显存策略 (方案A): InstantMesh 的 Zero123++ (~15-30GB) 与常驻的
        #    VLM(27GB 8bit) 叠加会 OOM，而 VLM 8bit 无法干净 offload(bitsandbytes<0.48)
        #    反复卸载会显存泄漏。故 mesh 一律由独立脚本 scripts/pregenerate_any6d_mesh.py
        #    预生成并缓存；主循环这里**只复用缓存 mesh**，遇到未缓存的直接跳过
        #    (走 R10/R17 的 VGGT/Kabsch 降级)，绝不触发 InstantMesh，也绝不 offload。
        if (modality in {"T2V", "I2V"}
                and set(modules) & {"scale_pose", "kinematics"}
                and not all(key in gt_data for key in (
                    "any6d_depths", "any6d_intrinsics", "any6d_masks", "any6d_mesh"))):
            # ⚠️ 缓存 key 必须区分视频来源目录(T2V vs I2V)，否则同文件名的
            #    视频会误复用对方的 mesh(首帧内容不同)。用「父目录名+文件名」。
            _mesh_key = (f"{os.path.basename(os.path.dirname(video_path))}_"
                         f"{os.path.splitext(os.path.basename(video_path))[0]}")
            _mesh_cache_dir = os.path.join(
                self.output_dir, "physics", "any6d_auto", _mesh_key)
            _mesh_cached = os.path.isfile(
                os.path.join(_mesh_cache_dir, "mesh_auto.obj"))
            if not _mesh_cached:
                print("[Any6D] mesh 未缓存 (需先跑 scripts/pregenerate_any6d_mesh.py)，"
                      "跳过 Any6D 构建，R10/R17 走降级")
            else:
                try:
                    any6d_model = self.loader.get_any6d()
                    if getattr(any6d_model, "is_loaded", False):
                        auto_inputs = any6d_model.prepare_auto_inputs(
                            np.stack([np.asarray(frame) for frame in frames_pil]),
                            sam3=self.loader.get_sam3(),
                            da3=self.loader.get_da3(),
                            dimension="R10",
                            mesh_path=(self.config.get("models", {})
                                       .get("physics", {})
                                       .get("any6d", {})
                                       .get("mesh_path")),
                            output_dir=_mesh_cache_dir,
                            prompt=prompt,
                            skip_mesh_generation=True,
                        )
                        gt_data.update(auto_inputs)
                        print("[Any6D] T2V/I2V proxy输入已构建: "
                              f"source={auto_inputs['any6d_source']}, "
                              f"mesh={auto_inputs['any6d_mesh']}")
                except Exception as e:
                    print(f"[Any6D] T2V/I2V proxy输入构建失败，保留降级: {e}")

        # ── 评测前置内容检查 ──
        pre_check = None
        # 内容预检的 VLM 兜底只对「评测本身需要 VLM 的模块」有意义。
        # 纯几何/动力学模块 (dynamics/kinematics/deformation_spatial/
        # statics_lighting/multiview) 的评测不用 VLM, 预检的 VLM 兜底会
        # 无谓加载 Qwen(几十秒~几分钟)且无实际作用 → 这些模块完全跳过预检。
        # statics_lighting 同理 (R13/R18 不需内容预检)。
        skip_content_precheck = (
            modules is not None
            and set(modules).isdisjoint(self._VLM_MODULES)
        )
        if modality in ("T2V", "I2V") and prompt and not skip_content_precheck:
            try:
                pre_check = self._pre_check_content(frames_pil, prompt)
                if pre_check.get("warnings"):
                    print(f"[PreCheck] 内容警告: {'; '.join(pre_check['warnings'][:3])}")
            except Exception as e:
                print(f"[PreCheck] 前置检查失败(非阻塞): {e}")
        gt_data["_pre_check"] = pre_check

        # 逐模块评测
        for module_name in modules:
            layer = self._run_module(module_name, frames_pil, gt_data,
                                     target_dimensions=target_dimensions,
                                     modality=modality, prompt=prompt)
            if layer:
                result.add_layer(layer)
                reporter.print_layer_summary(layer)

        # 视频级防御性显存释放: 清理本视频的采样帧/中间张量, 防止跨视频累积
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 报告
        reporter.generate(result, output_prefix=f"physics_{os.path.basename(video_path)}")
        self._print_overview(result)
        return result

    # 需要 Qwen3-VL 的评测层(其余层运行前把 VLM 挪到 CPU 腾显存)
    _VLM_MODULES = {"geometry", "material", "thermal", "electromag", "phase_change",
                    "motion_persistence", "scale_pose"}

    # GT 键→ 时间轴所在维度
    #   "frame_first": [T, ...]        逐帧图/相机 (depth/normal/mask/camera_poses)
    #   "obj_time":    [N, T, ...]     逐物体逐帧 (轨迹/速度/位姿)
    #   None:          无时间轴, 不切(内参/尺寸/点云)
    _GT_TIME_AXIS = {
        "depth_gt": "frame_first",
        "normal_gt": "frame_first",
        "gt_masks": "frame_first",
        "gt_tracking_masks": "frame_first",
        "gt_flow": "frame_first",
        "camera_poses": "frame_first",
        "gt_frame_indices": "frame_first",
        "gt_timestamps": "frame_first",
        "gt_trajectories": "obj_time",
        "gt_velocities": "obj_time",
        "gt_angular_vel": "obj_time",
        "gt_poses": "obj_time",
        "gt_deformation": None,
        "pointcloud_gt": None,
        "camera_intrinsics": None,
        "gt_sizes": None,
    }

    def _align_temporal_gt(self, gt_data: dict, n_sampled: int) -> dict:
        """把全帧 GT 按 sampler 实际抽取的帧索引同步采样。

        CI2V 的 GT 由 UE 逐帧导出（全帧），而评测前 sampler 已按 max_frames
        抽帧。若不对齐，预测第 i 帧会和GT 第 i 帧比较，而后者实际对应原视频
        的另一时刻 —— 所有跨帧物理量（位移/速度/加速度/角速度）都会错。

        对齐规则见 `_GT_TIME_AXIS`：
        - frame_first: 沿 axis=0 取抽样索引
        - obj_time:    沿 axis=1 取抽样索引（axis=0 是物体维）
        - None:        不含时间轴，原样保留

        速度/加速度类 GT 长度为 T-1 / T-2，索引会先截断到其有效范围。
        """
        indices = list(getattr(self.sampler, "last_sample_indices", []) or [])
        if not indices:
            print("  [GT对齐] sampler 未提供抽帧索引, 跳过对齐(GT 可能错帧)")
            return gt_data
        if len(indices) != n_sampled:
            print(f"  [GT对齐] 抽帧索引数({len(indices)})≠采样帧数({n_sampled}), "
                  f"按前{n_sampled} 个索引对齐")
            indices = indices[:n_sampled]

        out = dict(gt_data)
        aligned, skipped = [], []
        for key, axis_kind in self._GT_TIME_AXIS.items():
            if key not in out or axis_kind is None:
                continue
            arr = out[key]
            if not isinstance(arr, np.ndarray):
                continue
            axis = 0 if axis_kind == "frame_first" else 1
            if arr.ndim <= axis:
                skipped.append(f"{key}(维度{arr.ndim}不足)")
                continue
            T_gt = arr.shape[axis]
            # 速度/加速度长度比位置短, 索引需落在有效范围内
            idx = [i for i in indices if i < T_gt]
            if len(idx) < 2:
                skipped.append(f"{key}(GT帧数{T_gt}过少)")
                continue
            try:
                out[key] = np.take(arr, idx, axis=axis)
                aligned.append(f"{key}{arr.shape}→{out[key].shape}")
            except Exception as e:
                skipped.append(f"{key}({e})")

        if aligned:
            print(f"  [GT对齐] 已按抽帧索引同步 {len(aligned)} 项: "
                  f"{'; '.join(aligned[:4])}{'...' if len(aligned) > 4 else ''}")
        if skipped:
            print(f"  [GT对齐] 未对齐: {skipped}")
        return out

    def _get_full_frames(self) -> Optional[list[Image.Image]]:
        """加载全帧 (带缓存), 仅加载一次 per video run。
        
        full_frame_max 限制最大帧数防 OOM (0=不限)。
        """
        if self._cached_full_frames is not None:
            return self._cached_full_frames
        if not self._last_video_path:
            return None
        all_frames, full_fps = self.sampler.sample_all_from_video(self._last_video_path)
        if not all_frames:
            return None
        self._cached_full_fps = full_fps if full_fps and full_fps > 0 else 24.0
        if self._full_frame_max > 0 and len(all_frames) > self._full_frame_max:
            # 均匀降采样到上限帧数 (降采样后 fps 需等比缩小)
            step = len(all_frames) / self._full_frame_max
            indices = [int(i * step) for i in range(self._full_frame_max)]
            # 确保尾帧在内
            if indices[-1] != len(all_frames) - 1:
                indices[-1] = len(all_frames) - 1
            all_frames = [all_frames[i] for i in indices]
            self._cached_full_fps = self._cached_full_fps / step
            print(f"  [full_frames] 视频{len(all_frames)}帧超过上限, 降采样到{self._full_frame_max}帧")
        self._cached_full_frames = all_frames
        return all_frames

    def _run_module(self, module_name: str, frames: list[Image.Image],
                    gt_data: dict,
                    target_dimensions: Optional[list[str]] = None,
                    modality: str = "T2V",
                    prompt: str = "") -> Optional[LayerResult]:
        """运行单个物理评测模块，收集全部 EvalResult。若指定 target_dimensions，仅保留对应维度。"""
        # 显存管理: A100-40GB 装不下全部模型同时驻留。不需要 VLM(~16GB) 的层
        # (如 kinematics/dynamics 的 SpatialTracker 推理) 运行前先把 VLM offload 到 CPU;
        # 需要 VLM 的层由 loader.get_vlm() 自动搬回 GPU。
        # H20-95GB 显存充足时 (config.physics.vlm_offload=false) 跳过 offload, 提速。
        if self.loader is not None and self._vlm_offload \
                and module_name not in self._VLM_MODULES:
            self.loader.offload_vlm()
        try:
            if module_name == "geometry":
                evaluator = GeometryEvaluator(
                    self.loader.get_vggt(),
                    self.loader.get_stable_normal(),
                    sam3=self.loader.get_sam3(),
                    vlm=self.loader.get_vlm("physics"),
                )
                results = evaluator.evaluate(
                    frames,
                    depth_gt=gt_data.get("depth_gt"),
                    normal_gt=gt_data.get("normal_gt"),
                    pointcloud_gt=gt_data.get("pointcloud_gt"),
                    camera_poses=gt_data.get("camera_poses"),
                    prompt=prompt,
                    evaluator_context=gt_data.get("evaluator_context"),
                    target_dimensions=target_dimensions,
                )

            elif module_name == "scale_pose":
                evaluator = ScalePoseEvaluator(
                    self.loader.get_da3(),
                    self.loader.get_sam3(),
                    self.loader.get_vggt(),
                    self.loader.get_any6d(),
                    vlm=self.loader.get_vlm("physics"),
                )
                results = evaluator.evaluate(
                    frames,
                    metric_scale=gt_data.get("metric_scale"),
                    gt_sizes=gt_data.get("gt_sizes"),
                    gt_poses=gt_data.get("gt_poses"),
                    prompt=prompt,
                    any6d_depths=gt_data.get("any6d_depths"),
                    any6d_intrinsics=gt_data.get("any6d_intrinsics"),
                    any6d_masks=gt_data.get("any6d_masks"),
                    any6d_mesh=gt_data.get("any6d_mesh"),
                    target_dimensions=target_dimensions,
                )

            elif module_name == "deformation_spatial":
                evaluator = DeformationSpatialEvaluator(
                    self.loader.get_vggt(),
                    self.loader.get_spatial_tracker(),
                    self.loader.get_sam3(),
                    vlm=self.loader.get_vlm("physics"),
                )
                results = evaluator.evaluate(
                    frames,
                    gt_deformation=gt_data.get("gt_deformation"),
                    target_dimensions=target_dimensions,
                    prompt=prompt,
                )

            elif module_name == "statics_lighting":
                evaluator = StaticsLightingEvaluator(
                    self.loader.get_vggt(),
                    self.loader.get_sam3(),
                    self.loader.get_reasonx(),
                    self.loader.get_stable_normal()
                )
                full_frames = None
                if "statics_lighting" in self._full_frame_modules:
                    full_frames = self._get_full_frames()
                    if full_frames:
                        print(f"  [statics_lighting] 加载全部 {len(full_frames)} 帧用于R13/R18光影分析")
                results = evaluator.evaluate(
                    frames,
                    stability_gt=gt_data.get("stability_gt"),
                    full_frames=full_frames,
                    target_dimensions=target_dimensions,
                    prompt=prompt,
                )

            elif module_name == "material":
                evaluator = MaterialEvaluator(self.loader.get_vlm("physics"))
                results = evaluator.evaluate(
                    frames,
                    expected_materials=gt_data.get("expected_materials"),
                )

            elif module_name == "thermal":
                evaluator = ThermalEvaluator(self.loader.get_vlm("physics"))
                results = evaluator.evaluate(frames)

            elif module_name == "electromag":
                evaluator = ElectromagEvaluator(self.loader.get_vlm("physics"))
                results = evaluator.evaluate(frames)

            elif module_name == "phase_change":
                evaluator = PhaseChangeEvaluator(self.loader.get_vlm("physics"))
                results = evaluator.evaluate(frames)

            elif module_name == "kinematics":
                evaluator = KinematicsEvaluator(
                    self.loader.get_spatial_tracker(),
                    self.loader.get_any6d(),
                    sam3=self.loader.get_sam3()
                )
                full_frames = None
                if "kinematics" in self._full_frame_modules:
                    full_frames = self._get_full_frames()
                    if full_frames:
                        print(f"  [kinematics/R15] 加载全部 {len(full_frames)} 帧用于物体计数")
                results = evaluator.evaluate(
                    frames,
                    gt_trajectories=gt_data.get("gt_trajectories"),
                    gt_velocities=gt_data.get("gt_velocities"),
                    gt_angular_vel=gt_data.get("gt_angular_vel"),
                    gt_masks=gt_data.get("gt_masks"),
                    gt_tracking_masks=gt_data.get("gt_tracking_masks"),
                    full_frames=full_frames,
                    fps=getattr(self, "_video_fps", 24.0),
                    scene_type=gt_data.get("scene_type", "unspecified"),
                    any6d_depths=gt_data.get("any6d_depths"),
                    any6d_intrinsics=gt_data.get("any6d_intrinsics"),
                    any6d_masks=gt_data.get("any6d_masks"),
                    any6d_mesh=gt_data.get("any6d_mesh"),
                    target_dimensions=target_dimensions,
                    prompt=prompt,
                )

            elif module_name == "motion_persistence":
                evaluator = MotionPersistenceEvaluator(
                    self.loader.get_vggt(),
                    self.loader.get_flowseek(),
                    self.loader.get_vfimamba(),
                    self.loader.get_spatial_tracker(),
                    self.loader.get_sam3(),
                    self.loader.get_vlm("physics")
                )
                full_frames = None
                storyboard = gt_data.get("storyboard") or {}
                needs_storyboard_full_frames = bool(
                    storyboard.get("enabled")
                    and storyboard.get("full_frame_required")
                )
                if "motion_persistence" in self._full_frame_modules \
                        or needs_storyboard_full_frames:
                    full_frames = self._get_full_frames()
                    if full_frames:
                        print(f"  [motion_persistence] 加载全部 {len(full_frames)} 帧用于R20/R21")
                eval_fps = (self._cached_full_fps if full_frames and self._cached_full_fps
                            else getattr(self, "_video_fps", 24.0))
                results = evaluator.evaluate(
                    frames,
                    full_frames=full_frames,
                    fps=eval_fps,
                    prompt=prompt,
                    storyboard=gt_data.get("storyboard"),
                    evaluation_method=gt_data.get("evaluation_method", "default"),
                    target_dimensions=target_dimensions,
                )

            elif module_name == "dynamics":
                evaluator = DynamicsEvaluator(
                    self.loader.get_spatial_tracker(),
                    sam3=self.loader.get_sam3(),
                    da3=self.loader.get_da3(),
                    vlm=self.loader.get_vlm("physics"),
                )
                # R23 碰撞检测需全帧 (碰撞发生在一瞬间, 32帧采样会错过拐点)
                full_frames = None
                full_fps = 24.0
                if "dynamics" in self._full_frame_modules:
                    full_frames = self._get_full_frames()
                    full_fps = self._cached_full_fps or full_fps
                    if full_frames:
                        print(f"  [dynamics/R23] 加载全部 {len(full_frames)} 帧用于碰撞检测")
                results = evaluator.evaluate(
                    frames,
                    scene_type=gt_data.get("scene_type", "unspecified"),
                    gt_trajectories=gt_data.get("gt_trajectories"),
                    fps=getattr(self, "_video_fps", 24.0),
                    full_frames=full_frames,
                    full_fps=full_fps,
                    prompt=prompt,
                    target_dimensions=target_dimensions,
                )

            elif module_name == "multiview":
                evaluator = MultiviewEvaluator(
                    self.loader.get_vggt(), sam3=self.loader.get_sam3())
                # CI2V: 优先用UE 同帧多机位原生视图 (表格要求"同一时刻不同视角"),
                # 而非把单视频切面板 —— 后者视角来自不同时刻, 违反 R08 定义。
                mv_paths = gt_data.get("multiview_image_paths") or []
                mv_frames = None
                if mv_paths:
                    loaded = []
                    for p in mv_paths:
                        try:
                            loaded.append(Image.open(p).convert("RGB"))
                        except Exception as ex:
                            print(f"  [multiview] 视图读取失败 {p}: {ex}")
                    if len(loaded) >= 2:
                        mv_frames = loaded
                        print(f"  [multiview] 使用 {len(loaded)} 个原生同步视图 (CI2V)")
                eval_method = gt_data.get("evaluation_method", "default")
                # ── T2V/I2V 自动识别环绕镜头/时间分镜 → 触发顺序多视角 proxy ──
                # R08 的 prompt 里设计了「环绕镜头(camera orbits/visits left-center-right
                # viewpoints)」和「时间分镜(storyboard.temporal_multiview)」两种多视图来源。
                # 在无显式多视图输入(mv_frames 为空)且未显式指定 evaluation_method 时,
                # 从 prompt 自动识别并触发已有 proxy, 而非直接 skip。
                if (mv_frames is None
                        and eval_method == "default"
                        and modality in ("T2V", "I2V")):
                    prompt_l = (prompt or "").lower()
                    storyboard = gt_data.get("storyboard") or {}
                    if storyboard.get("mode") == "temporal_multiview":
                        eval_method = "temporal_multiview_proxy"
                    elif any(k in prompt_l for k in (
                            "camera visits", "camera orbits", "orbiting camera",
                            "camera pans around", "multiple viewpoints",
                            "left, center, and right", "surround",
                            "half-orbit", "full orbit", "completes an orbit",
                            "temporal shots", "consecutive shots",
                            "环绕", "多视角", "环绕镜头")):
                        eval_method = "continuous_view_sequence_proxy"

                results = evaluator.evaluate(
                    mv_frames if mv_frames else frames,
                    view_indices=(list(range(len(mv_frames))) if mv_frames
                                  else gt_data.get("view_indices")),
                    camera_poses=gt_data.get("camera_poses"),
                    modality=modality,
                    panel_layout=gt_data.get("panel_layout"),
                    storyboard=gt_data.get("storyboard"),
                    evaluation_method=eval_method,
                    prompt=prompt,
                )

            else:
                print(f"  [Physics] 未知模块: {module_name}")
                return None

            # 返回完整 LayerResult（修复: 之前只取第一个 EvalResult 的 bug）
            all_results = list(results)  # 确保是可迭代的

            # 维度过滤: 若指定 target_dimensions，仅保留匹配的维度结果
            if target_dimensions:
                all_results = [
                    r for r in all_results
                    if any(r.dimension.startswith(d + "-") for d in target_dimensions)
                ]

            # 如果全部被过滤掉，跳过该模块
            if not all_results:
                return None

            return LayerResult(
                layer=f"L3-{module_name}",
                results=all_results
            )

        except Exception as e:
            print(f"  [Physics] 模块 {module_name} 执行失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 防御性显存释放: 每个模块评测结束后清理未回收的中间张量
            # (SAM3 masks / VGGT 输出 / 点云等), 防止跨视频累积 OOM。
            # 注意: empty_cache 不释放「常驻模型」的权重, 只回收临时 tensor 碎片。
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        return None

    def _pre_check_content(self, frames_pil: list[Image.Image],
                            prompt: str) -> dict:
        """评测前置内容检查: 快速验证视频关键帧是否包含 prompt 描述的场景元素。

        分两级检查:
        Level 1 — SAM3 分割物体: 检测关键帧中是否存在 prompt 提到的主体物体
        Level 2 — (可选) VLM 快速判定: 若 SAM3 结果不确定, 用 VLM 做最终裁决

        Returns:
            dict: {
                "content_ok": bool,       # 整体通过
                "warnings": list[str],    # 警告信息
                "detected_objects": list, # SAM3 检测到的物体名
                "prompt_keywords": list,  # prompt 提取的关键物体名词
                "method": str,            # "sam3" | "sam3+vlm" | "none"
            }
        """
        import re

        result = {
            "content_ok": True,
            "warnings": [],
            "detected_objects": [],
            "prompt_keywords": [],
            "method": "none",
        }

        # 提取 prompt 关键物理名词 (复用 scale_pose 的正则)
        try:
            from evaluators.physics.scale_pose import ScalePoseEvaluator
            prompt_nouns = ScalePoseEvaluator._extract_prompt_nouns(prompt)
        except Exception:
            # 简化提取: 基本名词
            nouns_re = re.compile(
                r'\b(person|man|woman|child|ball|cup|bottle|chair|table|desk|'
                r'door|window|car|bike|bicycle|book|phone|laptop|box|bag|'
                r'plant|tree|flower|dog|cat|bird|fish|ruler|pencil|pen|'
                r'lamp|building|house|wall|floor|road|path|fence|gate|'
                r'bridge|stair|pole|stick|rod|helmet|glove|watch|ring|'
                r'vase|statue|toy|block|balloon|wheel|mirror|frame|'
                r'spoon|plate|bowl|glass|pillow|blanket|towel|key|'
                r'hat|clock|candle|umbrella|backpack|sand|bucket|water|oil|'
                r'steel|clay|wood|wooden|marble|napkin|cloth|aluminum|paper|'
                r'paperclip|foam|cardboard|brass|rubber|ice|chocolate|apple|'
                r'truck|train|carriage|drone|cannon|puck|ramp|plank|pot|'
                r'horseshoe|magnet|generator|stove|grill|sphere|cone|'
                r'cylinder|cube|coin|sofa|couch)\b',
                re.IGNORECASE
            )
            prompt_nouns = set(nouns_re.findall(prompt.lower()))

        result["prompt_keywords"] = sorted(prompt_nouns)
        if not prompt_nouns:
            # prompt 中无识别到的物理名词, 跳过
            result["method"] = "no_keywords"
            return result

        # Level 1: SAM3 分割中间帧
        try:
            key_frame = np.array(frames_pil[len(frames_pil) // 2])
            sam3 = self.loader.get_sam3()
            seg = sam3.segment_frame(key_frame, dimension="PreCheck")
        except Exception as e:
            result["warnings"].append(f"SAM3 前置检查失败: {e}")
            result["content_ok"] = True  # SAM3 不可用时不过滤, 仅记录
            result["method"] = "sam3_failed"
            return result

        masks = seg.masks
        if masks is None or len(masks) == 0:
            result["content_ok"] = False
            result["warnings"].append(
                f"SAM3未检测到任何物体, prompt提及: {prompt_nouns}")
            result["method"] = "sam3_no_detection"
            return result

        # 收集检测到的物体名
        detected_names = []
        if seg.class_names:
            detected_names = [n.lower().strip() for n in seg.class_names if n]
        result["detected_objects"] = detected_names

        # 匹配检查
        detected_set = set(detected_names)
        matched = detected_set & prompt_nouns
        missing = prompt_nouns - detected_set
        result["matched_objects"] = sorted(matched)
        result["missing_objects"] = sorted(missing)
        result["method"] = "sam3"

        # 判断逻辑:
        # - 至少匹配 1 个关键物体 → 通过
        # - 0 个匹配且 prompt_nouns ≥ 2 → Level 2 VLM 二次确认
        # - 0 个匹配且 prompt_nouns == 1 → 标记警告

        if len(matched) > 0:
            # 有匹配: 通过
            match_ratio = len(matched) / max(len(prompt_nouns), 1)
            if match_ratio < 0.3 and len(prompt_nouns) >= 3:
                result["content_ok"] = True  # 不完全拒绝, 但记警告
                result["warnings"].append(
                    f"物体匹配率低({match_ratio:.0%}): "
                    f"匹配{matched}, 缺失{missing}")
        elif len(prompt_nouns) >= 2:
            # Level 2: VLM 二次确认
            result["method"] = "sam3+vlm"
            try:
                vlm_result = self._vlm_verify_content(frames_pil, prompt)
                if vlm_result is not None:
                    result["content_ok"] = vlm_result["content_ok"]
                    if not result["content_ok"]:
                        result["warnings"].append(
                            f"VLM确认内容不匹配: {vlm_result.get('reason', '')}")
                    else:
                        result["warnings"].append(
                            "SAM3未匹配到物体但VLM认为内容合理")
            except Exception as e:
                result["content_ok"] = False
                result["warnings"].append(
                    f"SAM3未检测到prompt物体({prompt_nouns}), "
                    f"VLM确认失败: {e}")
        else:
            # 单个物体未检测到, 不阻塞
            result["warnings"].append(
                f"SAM3未检测到prompt提及的物体: {prompt_nouns}")

        return result

    def _vlm_verify_content(self, frames_pil: list[Image.Image],
                             prompt: str) -> Optional[dict]:
        """VLM 快速内容验证: 判断视频关键帧是否与 prompt 描述的物理场景匹配。

        仅在 SAM3 物体检测结果不确定时调用 (作为二次确认)。
        使用简单二分类问题, 避免长推理。
        """
        try:
            vlm = self.loader.get_vlm("physics")
        except Exception:
            return None

        # 仅取中间 3 帧
        T = len(frames_pil)
        indices = sorted({0, T // 2, T - 1})
        frames_sub = [frames_pil[i] for i in indices]

        question = (
            f"Look at these frames from a generated video. "
            f"The original prompt was: \"{prompt}\"\n\n"
            f"Does the video contain the main physical objects/scene "
            f"described in the prompt? Answer ONLY 'yes' or 'no', then "
            f"briefly explain in one sentence."
        )

        try:
            # VLMWrapper公共接口是 chat，不是旧版 ask；前置检查失败不应
            # 把有效视频误报为内容不匹配。
            answer = vlm.chat(frames_sub, question, max_new_tokens=64)
        except Exception as e:
            return {"content_ok": False, "reason": f"VLM调用失败: {e}"}

        answer_lower = answer.lower().strip()
        is_ok = answer_lower.startswith("yes") or "contain" in answer_lower[:50]
        return {
            "content_ok": is_ok,
            "reason": answer[:200],
        }

    @staticmethod
    def _print_overview(result: BenchmarkResult):
        print(f"\n{'='*60}")
        print("Physics Benchmark 总览")
        print(f"{'='*60}")
        for layer in result.layers:
            layer.compute_avg()
            for r in layer.results:
                violations = r.violations[:2] if r.violations else []
                if r.status in ("error", "skipped"):
                    reason = violations[0] if violations else ""
                    print(f"  {r.dimension}: [{r.status}] {reason}")
                else:
                    tag = "" if r.status == "ok" else f"({r.status})"
                    print(f"  {r.dimension}: score={r.score}/5 {tag} " +
                          (f"violations={violations}" if violations else ""))
        print(f"{'='*60}\n")
