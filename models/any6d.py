"""
Any6D — 6D 位姿估计模型封装。

官方 Any6D 需要 RGB-D、相机内参、物体 mask 和物体 mesh；模型加载阶段只
加载 FoundationPose 后端，真正的 Any6D estimator 在收到 mesh 后惰性初始化。
"""
import os
import sys
from typing import Optional

import numpy as np
import torch
from .base import BaseSpecialistModel, PoseOutput


class Any6DModel(BaseSpecialistModel):
    """Any6D 官方 register_any6d + track_one_any6d 适配器。"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda", hf_cache_dir: Optional[str] = None,
                 mesh_path: Optional[str] = None,
                 debug_dir: Optional[str] = None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path
        self.mesh_path = mesh_path
        self.hf_cache_dir = hf_cache_dir
        self.debug_dir = debug_dir or os.path.join(
            os.getcwd(), "results", "any6d_debug")
        self._any6d_cls = None
        self._estimators = []
        self._any6d_root = None

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": "taeyeopl/any6d (FoundationPose backend)",
            "checkpoint": self.model_path,
            "mesh_path": self.mesh_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def load(self, **kwargs) -> None:
        """加载官方后端；不在没有 mesh 时构造 estimator。"""
        third_party = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "third_party"))
        any6d_root = os.path.join(third_party, "any6d")
        if not os.path.isdir(any6d_root):
            raise FileNotFoundError(f"Any6D源码目录不存在: {any6d_root}")
        foundationpose_root = os.path.join(any6d_root, "foundationpose")
        bundlesdf_root = os.path.join(foundationpose_root, "bundlesdf")
        # root/estimater.py 是官方 Any6D 类；foundationpose/estimater.py
        # 同名但不导出 Any6D，必须把 any6d_root 放在最前，避免模块遮蔽。
        import_roots = (any6d_root, foundationpose_root, bundlesdf_root)
        for import_root in reversed(import_roots):
            if os.path.isdir(import_root) and import_root in sys.path:
                sys.path.remove(import_root)
            if os.path.isdir(import_root):
                sys.path.insert(0, import_root)

        from estimater import Any6D
        self._any6d_cls = Any6D
        self._any6d_root = any6d_root
        self._is_loaded = True
        print("[Any6D] ✓ FoundationPose后端加载成功，等待mesh/RGB-D输入")

    @staticmethod
    def _load_mesh(mesh):
        import trimesh

        if isinstance(mesh, (str, os.PathLike)):
            mesh = trimesh.load(str(mesh), force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if not geometries:
                raise ValueError("mesh Scene不包含几何体")
            mesh = trimesh.util.concatenate(geometries)
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise TypeError("mesh必须是trimesh.Trimesh或有效mesh路径")
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("mesh不包含有效顶点或面")
        return mesh

    @staticmethod
    def _normalize_masks(masks: np.ndarray, n_frames: int) -> np.ndarray:
        masks = np.asarray(masks)
        if masks.ndim == 3:
            masks = masks[None, ...]
        if masks.ndim != 4 or masks.shape[1] != n_frames:
            raise ValueError(
                f"masks应为[N,T,H,W]或[T,H,W]，实际为{masks.shape}")
        return masks.astype(bool)

    @staticmethod
    def _normalize_depths(depths: np.ndarray, n_frames: int) -> np.ndarray:
        depths = np.asarray(depths, dtype=np.float32)
        if depths.ndim != 3 or depths.shape[0] != n_frames:
            raise ValueError(
                f"depths应为[T,H,W]，实际为{depths.shape}")
        return depths

    @staticmethod
    def _normalize_intrinsics(intrinsics: np.ndarray, n_frames: int) -> np.ndarray:
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.ndim == 2:
            intrinsics = np.repeat(intrinsics[None], n_frames, axis=0)
        if intrinsics.shape != (n_frames, 3, 3):
            raise ValueError(
                f"intrinsics应为[3,3]或[T,3,3]，实际为{intrinsics.shape}")
        return intrinsics

    def unload(self) -> None:
        self._estimators.clear()
        self._model = None
        self._any6d_cls = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_mesh_from_frame(self, frame: np.ndarray,
                                   mask: np.ndarray,
                                   output_dir: str) -> str:
        """使用Any6D官方InstantMesh流程从首帧生成物体mesh。"""
        any6d_root = self._any6d_root
        if not any6d_root:
            raise RuntimeError("Any6D源码根目录未初始化")
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, "mesh_auto.obj")
        if os.path.isfile(mesh_path) and os.path.getsize(mesh_path) > 0:
            return mesh_path

        if any6d_root not in sys.path:
            sys.path.insert(0, any6d_root)
        import sam2_instantmesh as instantmesh

        old_cwd = os.getcwd()
        try:
            os.chdir(any6d_root)
            # mask已经由SAM3提供；关闭rembg，避免额外依赖和二次分割改变目标边界。
            input_image = instantmesh.preprocess_image(
                frame, mask, output_dir, name="auto", rem_bg=False,
                flip=True)
            views = instantmesh.diffusion_image_generation(
                output_dir, output_dir, name="auto", input_image=input_image,
                config_path="./instantmesh/configs/instant-mesh-large.yaml")
            # ⚠️ InstantMesh 的 FlexiCubes mesh 提取(index_add_) 要求 float32,
            #    Zero123++ 的 bf16 autocast 可能泄漏 → 显式禁用。
            with torch.autocast(device_type="cuda", enabled=False):
                instantmesh.instant_mesh_process(views, output_dir, name="auto")
        finally:
            os.chdir(old_cwd)

        if not os.path.isfile(mesh_path) or os.path.getsize(mesh_path) == 0:
            raise RuntimeError(f"InstantMesh未生成有效mesh: {mesh_path}")
        return mesh_path

    def prepare_auto_inputs(self, frames: np.ndarray, sam3, da3,
                            dimension: str = "R10", concepts=None,
                            mesh_path: Optional[str] = None,
                            output_dir: Optional[str] = None,
                            prompt: Optional[str] = None,
                            skip_mesh_generation: bool = False) -> dict:
        """为T2V/I2V从首帧自动构建Any6D所需的代理输入。

        流程：首帧SAM3 mask → 全帧DA3深度 → 近似相机内参 →
        首帧InstantMesh mesh。该路径是T2V/I2V proxy，不是真实GT。

        prompt: 可选。提供后用于从 SAM3 检测结果中筛选与 prompt 相关的目标,
                避免 SAM3 开放词汇分割误检 (如把手误判 person) 导致选错 mesh 目标。

        skip_mesh_generation: True 时，若 mesh 已缓存则直接复用缓存路径；
                若未缓存则抛异常(由调用方走 R10/R17 降级)，绝不触发 InstantMesh。
                用于主评测循环(避免 Zero123++ 与常驻模型抢显存 OOM)；
                mesh 生成统一由独立脚本 pregenerate_any6d_mesh.py 完成。
        """
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames应为[T,H,W,3]，实际为{frames.shape}")
        if sam3 is None or not getattr(sam3, "is_loaded", False):
            raise RuntimeError("SAM3未加载，无法构建Any6D mask")
        if da3 is None or not getattr(da3, "is_loaded", False):
            raise RuntimeError("DA3未加载，无法构建Any6D深度")

        # ── 首帧分割: 两级回退 (具体词 → 宽泛词) ──
        # R10 在 DIMENSION_CONCEPTS 无定义 → 默认 DEFAULT_CONCEPTS(object/thing/item)。
        # 对 toy car 等具体物体, 用 prompt 具体词召回更强; 但对 person+doorway 等
        # 场景, prompt 提取的词可能含背景词(doorway/floor)召回差。
        # 故: 先试 prompt 具体词, 失败回退宽泛词 object/thing/item, 再失败才报错。
        seg = None
        effective_concepts = concepts
        if (not effective_concepts) and prompt:
            try:
                from evaluators.physics.scale_pose import ScalePoseEvaluator
                _nouns = ScalePoseEvaluator._extract_prompt_nouns(prompt)
                if _nouns:
                    effective_concepts = sorted(_nouns)
            except Exception:
                effective_concepts = None
        if effective_concepts:
            seg = sam3.segment_frame(
                frames[0], dimension=dimension, concepts=effective_concepts)
        if seg is None or seg.masks is None or len(seg.masks) == 0:
            # 具体词召回失败 → 回退默认宽泛词
            seg = sam3.segment_frame(frames[0], dimension=dimension,
                                     concepts=None)
        if seg is None or seg.masks is None or len(seg.masks) == 0:
            raise RuntimeError("SAM3首帧未检测到Any6D目标")

        # ── 健壮的 mask 选择: 面积下限 + prompt 相关性优先 (而非纯最大面积) ──
        areas = [int(np.asarray(mask).astype(bool).sum()) for mask in seg.masks]
        H_img, W_img = frames.shape[1:3]
        min_area = int(H_img * W_img * 0.005)  # 面积下限: 过滤微小碎片 (<0.5%图像)
        valid_idx = [i for i, a in enumerate(areas) if a >= min_area]
        if not valid_idx:
            # 全部过小则回退到最大面积 (至少有个目标)
            valid_idx = [int(np.argmax(areas))]

        # prompt 相关性: 若提供 prompt, 优先选类别与 prompt 相关的目标
        chosen = valid_idx[0]
        if prompt and seg.class_names:
            from evaluators.physics.scale_pose import ScalePoseEvaluator
            prompt_nouns = ScalePoseEvaluator._extract_prompt_nouns(prompt)
            if prompt_nouns:
                for i in valid_idx:
                    name = seg.class_names[i] if i < len(seg.class_names) else ""
                    if ScalePoseEvaluator._name_matches_prompt(name, prompt_nouns):
                        chosen = i
                        break
        target_mask = np.asarray(seg.masks[chosen]).astype(bool)

        depths = []
        for frame in frames:
            depth = da3.estimate_metric_depth(frame).depth_map
            depths.append(np.asarray(depth, dtype=np.float32))
        depths = np.stack(depths, axis=0)

        height, width = frames.shape[1:3]
        focal = 0.866 * float(max(height, width))
        K = np.array([
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        masks = np.repeat(target_mask[None, None], len(frames), axis=1)

        if mesh_path is None:
            output_dir = output_dir or os.path.join(
                os.getcwd(), "results", "any6d_auto")
            if skip_mesh_generation:
                # 主评测循环: 只复用缓存 mesh, 未缓存直接抛异常走降级, 不触发 InstantMesh
                cached = os.path.join(output_dir, "mesh_auto.obj")
                if os.path.isfile(cached) and os.path.getsize(cached) > 0:
                    mesh_path = cached
                else:
                    raise RuntimeError(
                        f"mesh 未缓存: {cached}；请先运行 "
                        f"scripts/pregenerate_any6d_mesh.py 预生成")
            else:
                mesh_path = self._generate_mesh_from_frame(
                    frames[0], target_mask, output_dir)

        return {
            "any6d_depths": depths,
            "any6d_intrinsics": K,
            "any6d_masks": masks,
            "any6d_mesh": mesh_path,
            "any6d_source": "sam3_da3_instantmesh_proxy",
        }

    def estimate_pose_sequence(self, frames: np.ndarray,
                               masks: Optional[np.ndarray] = None,
                               depths: Optional[np.ndarray] = None,
                               intrinsics: Optional[np.ndarray] = None,
                               mesh=None,
                               iteration: int = 5) -> PoseOutput:
        """用官方 Any6D 对一个或多个目标执行 register + track。

        Args:
            frames: [T,H,W,3] uint8 RGB帧。
            masks: [T,H,W] 或 [N,T,H,W] 目标mask。
            depths: [T,H,W] 米制深度。
            intrinsics: [3,3] 或 [T,3,3] 相机内参。
            mesh: trimesh.Trimesh 或 mesh文件路径；所有目标共用时传单个mesh。
        """
        if not self._is_loaded or self._any6d_cls is None:
            raise RuntimeError("Any6D后端未加载")
        if mesh is None:
            mesh = self.mesh_path
        if masks is None or depths is None or intrinsics is None or mesh is None:
            raise RuntimeError(
                "Any6D需要mesh、RGB-D深度、相机内参和目标mask；"
                "请传入masks/depths/intrinsics/mesh，或配置mesh_path")

        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames应为[T,H,W,3]，实际为{frames.shape}")
        n_frames = frames.shape[0]
        masks = self._normalize_masks(masks, n_frames)
        depths = self._normalize_depths(depths, n_frames)
        intrinsics = self._normalize_intrinsics(intrinsics, n_frames)
        mesh = self._load_mesh(mesh)

        positions = np.zeros((masks.shape[0], n_frames, 3), dtype=np.float32)
        rotations = np.zeros((masks.shape[0], n_frames, 3, 3), dtype=np.float32)
        self._estimators.clear()

        os.makedirs(self.debug_dir, exist_ok=True)
        # ⚠️ 显式禁用 autocast: FoundationPose/Any6D 的 register/track 全程依赖
        #    float32 (nvdiffrast CUDA kernel + linalg.inv 都不支持 bf16)。上游
        #    (VGGT/SpatialTracker 等) 可能遗留 bf16 autocast 状态, 必须强制关闭。
        for obj_idx in range(masks.shape[0]):
            estimator = self._any6d_cls(
                mesh=mesh,
                debug=0,
                debug_dir=self.debug_dir,
            )
            with torch.autocast(device_type="cuda", enabled=False):
                first_pose = estimator.register_any6d(
                    K=intrinsics[0], rgb=frames[0], depth=depths[0],
                    ob_mask=masks[obj_idx, 0], iteration=iteration,
                    name=f"object_{obj_idx}",
                )
            poses = [np.asarray(first_pose, dtype=np.float32)]
            for frame_idx in range(1, n_frames):
                with torch.autocast(device_type="cuda", enabled=False):
                    pose = estimator.track_one_any6d(
                        rgb=frames[frame_idx], depth=depths[frame_idx],
                        K=intrinsics[frame_idx], iteration=iteration,
                    )
                poses.append(np.asarray(pose, dtype=np.float32))
            pose_array = np.stack(poses, axis=0)
            rotations[obj_idx] = pose_array[:, :3, :3]
            positions[obj_idx] = pose_array[:, :3, 3]
            self._estimators.append(estimator)

        return PoseOutput(
            positions=positions,
            rotations=rotations,
            metric=True,
        )
