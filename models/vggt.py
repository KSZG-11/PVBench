"""
VGGT — 单目/多帧 3D 重建模型封装（真实推理，无 stub）

仅使用: VGGT-Omega (CVPR 2026 Oral, facebookresearch/vggt-omega) — 精度最高

涉及的物理量：R01深度, R03形状, R05形变, R08多视图几何,
             R09相对尺度, R11空间关系, R12稳定性, R19跨帧几何, R21物体恒存

官方 API (均已查证):
  VGGT-Omega (需 git clone + pip install -e . + 本地 .pt checkpoint):
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    model = VGGTOmega().to(device).eval(); model.load_state_dict(torch.load(ckpt))
    images = load_and_preprocess_images(paths, image_resolution=512)
    preds = model(images)  # {pose_enc, images, depth, depth_conf, camera_and_register_tokens}
    extr, intr = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
    # 点云 = 深度反投影

设计: Omega 加载失败直接抛异常, 推理抛 RuntimeError。不再降级 VGGT-1B。
"""
import os
import glob
import tempfile
import torch
import numpy as np
from typing import Optional
from PIL import Image

from .base import BaseSpecialistModel, DepthOutput, PointCloudOutput, NormalOutput
from .backend_utils import get_safe_dtype


class VGGTModel(BaseSpecialistModel):
    """VGGT 3D 重建封装 (仅 VGGT-Omega)"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda",
                 model_version: str = "omega",
                 image_resolution: int = 512,
                 **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path          # Omega: .pt 路径
        self.model_version = model_version    # 保留参数兼容, 恒为 "omega"
        self.image_resolution = image_resolution
        self._loaded_version = None
        self._hf_cache_dir = (os.environ.get("HF_HOME")
                              or os.environ.get("HUGGINGFACE_HUB_CACHE"))
        # 官方工具句柄
        self._omega_load_images = None
        self._omega_enc2cam = None
        self._dtype = torch.bfloat16

    @property
    def loaded_version(self) -> Optional[str]:
        return self._loaded_version

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": "facebookresearch/vggt-omega",
            "model_version": self.model_version,
            "loaded_version": self._loaded_version,
            "checkpoint": self.model_path,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    # ── 加载 ──
    def load(self, **kwargs) -> None:
        """仅加载 VGGT-Omega。失败抛异常 (不再降级 1B)。"""
        # A100(Ampere) 支持 bfloat16
        try:
            self._dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        except Exception:
            self._dtype = get_safe_dtype()

        self._load_omega()

    def _find_omega_ckpt(self) -> Optional[str]:
        """定位 Omega 本地 .pt checkpoint。"""
        if self.model_path and os.path.isfile(self.model_path):
            return self.model_path
        # ⚠️ 性能: 绝不递归 glob 项目根 / 整个 hf_cache —— 1.5TB 共享盘上
        #    readdir 遍历上百万文件会挂起数十分钟, 表现为"加载卡死"
        #    (read_bytes/GPU 均无进展, 纯元数据 syscall)。checkpoint 只可能
        #    在以下窄目录, 秒级命中。项目根仅做非递归浅层兜底。
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        hf_cache = self._hf_cache_dir or os.path.expanduser("~/.cache/huggingface")
        search_roots = [
            os.path.join(project_root, "third_party", "vggt-omega", "checkpoints"),
            os.path.join(hf_cache, "models"),
            os.path.join(project_root, "checkpoints"),
        ]
        for root in search_roots:
            if not root or not os.path.isdir(root):
                continue
            for pat in ("**/vggt_omega*512*.pt", "**/VGGT-Omega*/**/*.pt", "**/vggt_omega*.pt"):
                hits = glob.glob(os.path.join(root, pat), recursive=True)
                if hits:
                    return hits[0]
        # 浅层兜底: 直接放项目根的 .pt (非递归, O(目录项))
        for pat in ("vggt_omega*512*.pt", "vggt_omega*.pt"):
            hits = glob.glob(os.path.join(project_root, pat))
            if hits:
                return hits[0]
        return None

    def _load_omega(self) -> None:
        import sys
        _tp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "vggt-omega"))
        if os.path.isdir(_tp) and _tp not in sys.path:
            sys.path.insert(0, _tp)

        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
        self._omega_load_images = load_and_preprocess_images
        self._omega_enc2cam = encoding_to_camera

        ckpt = self._find_omega_ckpt()
        if not ckpt:
            raise FileNotFoundError(
                "未找到 VGGT-Omega checkpoint (.pt)。请申请下载 vggt_omega_1b_512.pt "
                "并放到项目目录或 third_party/vggt-omega/checkpoints/")

        model = VGGTOmega().to(self.device).eval()
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        self._model = model
        self._is_loaded = True
        self._loaded_version = "omega"
        print(f"[VGGT-Omega] ✓ 加载成功 ({ckpt}, dtype={self._dtype})")

    def unload(self) -> None:
        self._model = None
        self._is_loaded = False
        self._loaded_version = None
        torch.cuda.empty_cache()

    # ── 预处理 ──
    def _to_paths(self, frames: np.ndarray):
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]
        tmp = tempfile.mkdtemp(prefix="vggt_")
        paths = []
        for i, f in enumerate(frames):
            p = os.path.join(tmp, f"f_{i:04d}.png")
            Image.fromarray(f.astype(np.uint8)).save(p)
            paths.append(p)
        return paths, tmp

    @staticmethod
    def _cleanup(paths, tmp):
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    # ── 统一前向: 返回 (depth[T,H,W], depth_conf, extr[T,3,4], intr[T,3,3]) ──
    def _forward(self, frames: np.ndarray):
        paths, tmp = self._to_paths(frames)
        try:
            images = self._omega_load_images(paths, image_resolution=self.image_resolution).to(self.device)
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=self._dtype):
                    preds = self._model(images)
                    extr, intr = self._omega_enc2cam(preds["pose_enc"], preds["images"].shape[-2:])
                    depth = preds["depth"]
                    conf = preds.get("depth_conf")
            depth_np = depth.squeeze().float().cpu().numpy()
            extr_np = np.asarray(extr.squeeze(0).float().cpu()) if hasattr(extr, "cpu") else np.asarray(extr)
            intr_np = np.asarray(intr.squeeze(0).float().cpu()) if hasattr(intr, "cpu") else np.asarray(intr)
            conf_np = conf.squeeze().float().cpu().numpy() if conf is not None else np.ones_like(depth_np)
            # 显式释放 GPU 中间张量 (VGGT 被大量维度调用, 泄漏累积最严重)
            for _k in list(preds.keys()):
                try:
                    del preds[_k]
                except (KeyError, TypeError):
                    pass
            del preds, images, extr, intr, depth, conf
            torch.cuda.empty_cache()
            return self._norm_shapes(depth_np, conf_np, extr_np, intr_np)
        finally:
            self._cleanup(paths, tmp)

    @staticmethod
    def _norm_shapes(depth, conf, extr, intr):
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 2:
            depth = depth[None]
        if extr.ndim == 2:
            extr = extr[None]
        if intr.ndim == 2:
            intr = intr[None]
        return depth, conf, extr, intr

    # ── 推理接口 ──
    def estimate_depth(self, frame: np.ndarray) -> DepthOutput:
        if not self._is_loaded:
            raise RuntimeError("VGGT 未加载")
        depth, conf, _, _ = self._forward(frame)
        d = depth[0] if depth.ndim == 3 else depth
        c = conf[0] if getattr(conf, "ndim", 0) == 3 else conf
        return DepthOutput(depth_map=d, confidence=c)

    def reconstruct_masked_pointcloud(self, frame: np.ndarray,
                                      mask: np.ndarray,
                                      max_points: int = 5000) -> PointCloudOutput:
        """从指定2D实例mask重建对应的相机系点云。

        R03 T2V/I2V 的表格方案要求"重建点云 vs 理想模型"。原
        reconstruct_pointcloud() 输出全画面（含背景/地面/其他物体），不能用于
        单物体canonical Chamfer。此接口用SAM3 mask筛选深度反投影点，保持
        相机系坐标和相对尺度，供后续Sim(3)对齐。
        """
        if not self._is_loaded:
            raise RuntimeError("VGGT 未加载")
        depth, _conf, extr, intr = self._forward(frame)
        d = depth[0] if depth.ndim == 3 else depth
        K = intr[0] if intr.ndim == 3 else intr
        m = np.asarray(mask) > 0.5
        if m.shape != d.shape:
            import cv2
            m = cv2.resize(m.astype(np.uint8), (d.shape[1], d.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        valid = m & np.isfinite(d) & (d > 0)
        ys, xs = np.nonzero(valid)
        if len(xs) < 30:
            return PointCloudOutput(points=np.zeros((0, 3), dtype=np.float32))
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        z = d[ys, xs].astype(np.float32)
        pts = np.stack([
            (xs - cx) * z / max(fx, 1e-6),
            (ys - cy) * z / max(fy, 1e-6),
            z,
        ], axis=-1).astype(np.float32)
        if len(pts) > max_points:
            idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts = pts[idx]
        cam = np.eye(4, dtype=np.float32)
        if extr.ndim == 3 and len(extr):
            cam[:3, :4] = extr[0]
        return PointCloudOutput(points=pts, camera_pose=cam)

    def reconstruct_pointcloud(self, frames: np.ndarray) -> PointCloudOutput:
        if not self._is_loaded:
            raise RuntimeError("VGGT 未加载")
        depth, conf, extr, intr = self._forward(frames)
        # 深度反投影为点云
        pts = self._manual_unproject(depth, intr)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) > 50000:
            idx = np.random.default_rng(0).choice(len(pts), 50000, replace=False)
            pts = pts[idx]
        cam = np.eye(4)
        if extr.ndim == 3 and extr.shape[0] > 0:
            cam[:3, :4] = extr[0]
        return PointCloudOutput(points=pts, camera_pose=cam)

    @staticmethod
    def _manual_unproject(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        pts_all = []
        T = depth.shape[0] if depth.ndim == 3 else 1
        d_stack = depth if depth.ndim == 3 else depth[None]
        K_stack = intrinsic if intrinsic.ndim == 3 else intrinsic[None]
        for t in range(min(T, len(K_stack))):
            d = d_stack[t]
            K = K_stack[t]
            H, W = d.shape
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            ys, xs = np.mgrid[0:H, 0:W]
            valid = d > 0
            X = (xs - cx) * d / max(fx, 1e-6)
            Y = (ys - cy) * d / max(fy, 1e-6)
            pts_all.append(np.stack([X[valid], Y[valid], d[valid]], axis=-1))
        return np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3))

    def estimate_camera_poses(self, frames: np.ndarray) -> np.ndarray:
        if not self._is_loaded:
            raise RuntimeError("VGGT 未加载")
        _, _, extr, _ = self._forward(frames)
        return extr

    def reconstruct_masked_pointclouds(self, frames: np.ndarray,
                                       masks: np.ndarray,
                                       max_points: int = 5000):
        """多帧联合推理 + 逐物体 mask 提取世界系点云。

        用于 R08 多视图几何一致性: 一次 _forward 对 T 帧联合推理(尺度一致),
        对每个物体用 mask 从每帧深度提取相机系点云, 再用各帧相机外参 extr
        (VGGT 约定为 world-to-camera, 需取逆变换到统一世界坐标系)。

        Args:
            frames: [T,H,W,3] uint8
            masks: [N,T,H,W] bool/float 逐物体逐帧 mask (与 frames 分辨率一致,
                   或会被 resize 到深度图分辨率)
            max_points: 每个物体每帧最多保留的点数

        Returns:
            (world_points[N,T,M_i,3], extr[T,3,4], intr[T,3,3], depth_conf[T])
            world_points 是每个物体每帧的世界系点云 (不等长, 用 list of list)。
        """
        if not self._is_loaded:
            raise RuntimeError("VGGT 未加载")
        frames = np.asarray(frames)
        masks = np.asarray(masks)
        depth, conf, extr, intr = self._forward(frames)
        T = depth.shape[0]
        N = masks.shape[0] if masks.ndim == 4 else 1
        # 归一化 masks 为 [N,T,H,W]
        if masks.ndim == 3:
            masks = masks[None]  # [1,T,H,W]
        H_d, W_d = depth.shape[1], depth.shape[2]

        import cv2
        world_pts = []  # [N][T] -> np.ndarray [M,3]
        for n in range(N):
            per_frame = []
            for t in range(T):
                m = masks[n, t] > 0.5
                if m.shape != (H_d, W_d):
                    m = cv2.resize(m.astype(np.uint8), (W_d, H_d),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                d = depth[t]
                valid = m & np.isfinite(d) & (d > 0)
                ys, xs = np.nonzero(valid)
                if len(xs) < 10:
                    per_frame.append(np.zeros((0, 3), dtype=np.float32))
                    continue
                K = intr[t]
                fx, fy = float(K[0, 0]), float(K[1, 1])
                cx, cy = float(K[0, 2]), float(K[1, 2])
                z = d[ys, xs].astype(np.float32)
                p_cam = np.stack([
                    (xs - cx) * z / max(fx, 1e-6),
                    (ys - cy) * z / max(fy, 1e-6),
                    z,
                ], axis=-1).astype(np.float32)
                # VGGT extr 是 world-to-camera ([R_w2c | t_w2c]):
                #   p_cam = R_w2c @ p_world + t_w2c
                # 故 camera-to-world: p_world = R_w2c.T @ (p_cam - t_w2c)
                # 行向量形式: p_world = (p_cam - t_w2c) @ R_w2c
                R_w2c = extr[t, :3, :3].astype(np.float32)
                t_w2c = extr[t, :3, 3].astype(np.float32)
                p_world = (p_cam - t_w2c) @ R_w2c
                if len(p_world) > max_points:
                    idx = np.random.default_rng(0).choice(
                        len(p_world), max_points, replace=False)
                    p_world = p_world[idx]
                per_frame.append(p_world)
            world_pts.append(per_frame)
        conf_out = conf[0] if conf.ndim == 3 else conf
        return world_pts, extr, intr, conf_out

    def estimate_normals(self, frame: np.ndarray) -> NormalOutput:
        if not self._is_loaded:
            raise RuntimeError("VGGT 未加载")
        d = self.estimate_depth(frame).depth_map
        gy, gx = np.gradient(d.astype(np.float32))
        n = np.stack([-gx, -gy, np.ones_like(d)], axis=-1)
        n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
        return NormalOutput(normal_map=n, confidence=np.ones(d.shape) * 0.7)
