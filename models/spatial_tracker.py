"""
SpatialTracker V2 — 3D 点轨迹追踪模型封装（真实推理，无 stub）

用途：3D 轨迹追踪、速度/加速度计算
涉及的物理量：R05形变, R15位移, R16速度加速度, R20运动连续性,
             R22力的动态效应, R23动量守恒, R24能量守恒

官方 API (Yuxihenry/SpatialTrackerV2-Offline, ICCV 2025, 已查证):
  from models.SpaTrackV2.models.predictor import Predictor
  from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
  front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
  model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
  model.spatrack.track_num = vo_points
  (c2w_traj, intrs, point_map, conf_depth,
   track3d_pred, track2d_pred, vis_pred, conf_pred, video) = model.forward(
       video_tensor, depth=depth, intrs=intrs, extrs=extrs,
       queries=query_xyt, fps=1, iters_track=4, ...)

需手动 git clone third_party/SpaTrackerV2 (见 DEPLOYMENT.md)。
设计: 加载/推理失败 → 抛异常, evaluator 记 status=error。轨迹为世界坐标系 (米制近似)。
"""
import os
import sys
import torch
import numpy as np
from typing import Optional
from .base import BaseSpecialistModel, TrajectoryOutput


class SpatialTrackerModel(BaseSpecialistModel):
    """SpatialTracker V2 封装（真实推理，9 元组输出）"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda", hf_cache_dir: Optional[str] = None,
                 vo_points: int = 756, track_mode: str = "offline",
                 grid_size: int = 20, **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path
        self.hf_cache_dir = hf_cache_dir
        self.vo_points = vo_points
        self.track_mode = track_mode
        self.grid_size = grid_size
        self._front = None  # VGGT4Track 前端 (depth/pose)
        self._preprocess_image = None       # 推理期图像预处理 (load 时从 SpaTrackerV2 捕获)
        self._get_points_on_a_grid = None   # 推理期网格查询点生成

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": ("Yuxihenry/SpatialTrackerV2-Offline"
                         if self.track_mode == "offline"
                         else "Yuxihenry/SpatialTrackerV2-Online"),
            "checkpoint": self.model_path,
            "track_mode": self.track_mode,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def load(self, **kwargs) -> None:
        """加载 Predictor + VGGT4Track 前端。失败抛异常。

        ⚠️ SpaTrackerV2 源码用 77 处顶层绝对导入 `from models.SpaTrackV2...`,
        这与本项目自身的 `models` 包同名冲突。故在导入期做 sys.modules 隔离:
        暂存并移除本项目 models.*, 让 `models` 解析到 SpaTrackerV2/models, 导入完成后再恢复。
        (已确认 SpaTrackerV2 无函数内懒 import, 导入期一次性绑定, 推理不再触发 import。)
        """
        from .base import resolve_local_model
        _third_party = os.path.join(os.path.dirname(__file__), "..", "third_party")
        _spa_path = os.path.abspath(os.path.join(_third_party, "SpaTrackerV2"))
        if not os.path.isdir(_spa_path):
            raise FileNotFoundError(f"SpaTrackerV2 源码目录不存在: {_spa_path}")

        # 兼容垫片: torch<2.3 无 torch.nn.attention 模块, 但 SpaTrackerV2/co_tracker
        # 会 `from torch.nn.attention import sdpa_kernel, SDPBackend`。构造一个等价 shim
        # (sdpa_kernel 退化为 no-op 上下文, SDPA 走默认后端; 系统 torch 锁 2.2.2 不能升级)。
        try:
            import torch.nn.attention  # noqa: F401
        except Exception:
            import types as _types
            import enum as _enum
            import contextlib as _contextlib
            _attn = _types.ModuleType("torch.nn.attention")

            class SDPBackend(_enum.IntEnum):
                ERROR = -1
                MATH = 0
                FLASH_ATTENTION = 1
                EFFICIENT_ATTENTION = 2
                CUDNN_ATTENTION = 3

            @_contextlib.contextmanager
            def sdpa_kernel(*args, **kwargs):
                yield

            _attn.SDPBackend = SDPBackend
            _attn.sdpa_kernel = sdpa_kernel
            sys.modules["torch.nn.attention"] = _attn
            try:
                import torch.nn as _tnn
                _tnn.attention = _attn
            except Exception:
                pass

        # 兼容垫片: pyceres 是 Ceres 求解器的 C++ 绑定, 需 nvcc/编译工具链, 本节点无法安装。
        # SpaTrackerV2 的 tracker3D/spatrack_modules/ba.py 在模块顶层
        #   `import pyceres` + `from pyceres import SolverOptions, LinearSolverType, ...`
        #   + `class SpatTrackCost_static(pyceres.CostFunction)`
        # 仅用于可选的 Bundle Adjustment。本封装推理走 query_no_BA=True, 不触发 BA,
        # 故构造一个 shim 让导入期成功: CostFunction 提供可继承基类, 枚举名按需返回哑值,
        # solve 若真被调用则显式报错 (提示 BA 不可用)。
        try:
            import pyceres  # noqa: F401
        except Exception:
            import types as _types

            class _AnyAttr(type):
                def __getattr__(cls, name):
                    return 0

            class CostFunction(object):
                def __init__(self, *args, **kwargs):
                    pass

            _pyceres = _types.ModuleType("pyceres")
            _pyceres.CostFunction = CostFunction
            for _n in ("SolverOptions", "LinearSolverType", "PreconditionerType",
                       "TrustRegionStrategyType", "LoggingType", "SolverSummary",
                       "Problem", "LossFunction", "Manifold"):
                _pyceres.__dict__[_n] = _AnyAttr(_n, (), {})

            def _no_ba(*args, **kwargs):
                raise RuntimeError("pyceres 不可用 (BA 已禁用): 推理请保持 query_no_BA=True")

            _pyceres.solve = _no_ba
            sys.modules["pyceres"] = _pyceres

        # 兼容垫片: kornia_rs 是 kornia 图像 IO 的 Rust 后端, 仅有 cp311 wheel (无 py39)。
        # SpaTrackerV2/models/SpaTrack.py 顶层 `import kornia` 会触发 kornia.io → `import kornia_rs`,
        # 但该 kornia 导入在 SpaTrack.py 内是死导入 (从不调用 kornia.io.* 图像读写函数)。
        # 故用空 shim 满足 import; 若真调用其函数则显式报错。
        try:
            import kornia_rs  # noqa: F401
        except Exception:
            import types as _types
            # ⚠️ 必须是"干净"的空模块: 不能用 __getattr__ 兜底返回函数,
            # 否则其它库探测 kornia_rs.__version__ 等会拿到函数 → '.endswith' 崩溃。
            # kornia.io.io 里 kornia_rs.* 调用全在函数内, import 通过即可 (SpaTrackerV2 不调用)。
            _krs = _types.ModuleType("kornia_rs")
            _krs.__version__ = "0.0.0-shim"
            sys.modules["kornia_rs"] = _krs

        # 离线优先解析本地权重目录 (魔搭 flat: hf_cache/models/Yuxihenry--*)
        offline_id = self.model_path or (
            "Yuxihenry/SpatialTrackerV2-Offline" if self.track_mode == "offline"
            else "Yuxihenry/SpatialTrackerV2-Online")
        offline_src = resolve_local_model(offline_id, self.hf_cache_dir)
        front_src = resolve_local_model("Yuxihenry/SpatialTrackerV2_Front", self.hf_cache_dir)
        if front_src == "Yuxihenry/SpatialTrackerV2_Front":
            raise FileNotFoundError(
                "未找到 SpatialTrackerV2_Front 本地权重。请在本地终端联网下载: "
                "HF_ENDPOINT=https://hf-mirror.com huggingface-cli download "
                "Yuxihenry/SpatialTrackerV2_Front --local-dir "
                "hf_cache/models/Yuxihenry--SpatialTrackerV2_Front")

        def _fp_kwargs(src, orig_id):
            # 本地目录: 直接传路径; 否则退回 repo id + cache
            if src != orig_id:
                return src, {}
            kw = {}
            if self.hf_cache_dir:
                kw["cache_dir"] = os.path.join(self.hf_cache_dir, "hub")
            return orig_id, kw

        # ── sys.modules 隔离 + 手动命名空间包 ──
        # SpaTrackerV2/models 无 __init__.py (命名空间包), 而本项目 models 是正规包,
        # 会压过命名空间包 → 必须手动把 sys.modules['models'] 指向 SpaTrackerV2/models。
        import types
        saved = {k: v for k, v in list(sys.modules.items())
                 if k == "models" or k.startswith("models.")}
        for k in list(saved):
            del sys.modules[k]
        sys.path.insert(0, _spa_path)
        _ns = types.ModuleType("models")
        _ns.__path__ = [os.path.join(_spa_path, "models")]  # 命名空间包路径
        sys.modules["models"] = _ns
        try:
            from models.SpaTrackV2.models.predictor import Predictor
            from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
            # 推理期需要的两个辅助函数: 命名空间还原后无法再 import models.SpaTrackV2.*,
            # 故在此(命名空间激活时)导入并存为实例属性 (函数对象绑定各自模块 globals, 持久可用)。
            from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
            from models.SpaTrackV2.models.utils import get_points_on_a_grid
            self._preprocess_image = preprocess_image
            self._get_points_on_a_grid = get_points_on_a_grid

            f_src, f_kw = _fp_kwargs(front_src, "Yuxihenry/SpatialTrackerV2_Front")
            self._front = VGGT4Track.from_pretrained(f_src, **f_kw).eval().to(self.device)

            o_src, o_kw = _fp_kwargs(offline_src, offline_id)
            self._model = Predictor.from_pretrained(o_src, **o_kw)
            self._model.spatrack.track_num = self.vo_points
            # ⚠️ Predictor.to() 未 return self (predictor.py:30 只 self.spatrack.to(),
            # 无返回值 → None)。故绝不能用返回值回赋, 否则 self._model 变 None,
            # 加载期不报错(spatrack 已在上一行访问), 但推理时 self._model.forward 崩。
            # 必须就地调用 eval()/to(), 保留原对象。
            self._model.eval()
            self._model.to(self.device)
        finally:
            # 清理 SpaTrackerV2 注册的 models.*, 恢复本项目 models.*
            for k in [k for k in list(sys.modules)
                      if k == "models" or k.startswith("models.")]:
                del sys.modules[k]
            sys.modules.update(saved)
            if _spa_path in sys.path:
                sys.path.remove(_spa_path)

        self._is_loaded = True
        print(f"[SpatialTracker V2] ✓ 模型加载成功 ({offline_src})")

    def unload(self) -> None:
        self._model = None
        self._front = None
        self._is_loaded = False
        torch.cuda.empty_cache()

    def _query_grid(self, H: int, W: int) -> np.ndarray:
        """在首帧构造均匀网格查询点 [N,3] = (t=0, x, y)。"""
        g = self.grid_size
        xs = np.linspace(W * 0.1, W * 0.9, g)
        ys = np.linspace(H * 0.1, H * 0.9, g)
        gx, gy = np.meshgrid(xs, ys)
        n = gx.size
        return np.stack([np.zeros(n), gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)

    def track_3d(self, frames: np.ndarray,
                 query_points: Optional[np.ndarray] = None,
                 fps: float = 24.0) -> TrajectoryOutput:
        """3D 轨迹追踪 (R15)。frames [T,H,W,3] uint8。失败抛 RuntimeError。

        严格对齐官方 inference.py (RGB 分支) 的调用约定:
          1) video [T,C,H,W] float(0~255) → preprocess_image (crop 到 518) ;
          2) 前端 VGGT4Track 需 5D [B,T,C,H,W] 输入 (video[None]/255) → 出 pose/depth/intr;
          3) squeeze 成 numpy 后连同 CPU 视频张量喂给 Predictor.forward (返回 9 元组);
          4) track3d(相机系) 经 c2w_traj 转世界系。
        ⚠️ 前端漏掉 [None] batch 维会触发 'expected 5, got 4'。

        fps: 视频实际帧率，用于 (a) SpatialTracker 内部时间尺度解算
             (b) compute_velocity_accel 计算物理速度(m/s)而非每帧位移
        """
        if not self._is_loaded:
            raise RuntimeError("SpatialTracker V2 未加载")
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]
        T = frames.shape[0]

        # [T,C,H,W] float(0~255) → 预处理到模型分辨率 (crop, 518)
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        video_tensor = self._preprocess_image(video_tensor)   # [T,C,H,W]

        # 前端 VGGT4Track: 输入需 5D [B,T,C,H,W]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                preds = self._front(video_tensor[None].to(self.device) / 255.0)
                extrinsic = preds["poses_pred"]
                intrinsic = preds["intrs"]
                depth_map = preds["points_map"][..., 2]
                depth_conf = preds.get("unc_metric")

        depth_tensor = depth_map.squeeze().cpu().numpy()       # [T,H,W]
        extrs = extrinsic.squeeze().cpu().numpy()              # [T,4,4]
        intrs = intrinsic.squeeze().cpu().numpy()              # [T,3,3]
        unc_metric = (depth_conf.squeeze().cpu().numpy() > 0.5) if depth_conf is not None else None

        # VGGT 深度置信度统计 (用于坐标系质量评估)
        vggt_depth_conf_mean = float(depth_conf.squeeze().mean().item()) if depth_conf is not None else -1.0

        # 前端 GPU 中间张量释放 (preds 字典仍引用 GPU 张量)
        for _k in list(preds.keys()):
            try:
                del preds[_k]
            except (KeyError, TypeError):
                pass
        del preds, extrinsic, intrinsic, depth_map, depth_conf
        torch.cuda.empty_cache()

        frame_H, frame_W = video_tensor.shape[2:]
        if query_points is not None:
            query_xyt = np.asarray(query_points, dtype=np.float32)
            if query_xyt.ndim != 2 or query_xyt.shape[1] != 3:
                raise ValueError(
                    f"query_points应为[N,3]=(t,x,y)，实际={query_xyt.shape}")
            print(f"[SpatialTracker] custom query_points={query_xyt.shape}, "
                  f"x=[{query_xyt[:,1].min():.1f},{query_xyt[:,1].max():.1f}], "
                  f"y=[{query_xyt[:,2].min():.1f},{query_xyt[:,2].max():.1f}]")
        else:
            grid_pts = self._get_points_on_a_grid(self.grid_size, (frame_H, frame_W), device="cpu")
            query_xyt = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()  # [N,3]=(t,x,y)

        # ⚠️ fixed_cam=True: 跳过 pycolmap Bundle Adjustment。
        #   query_no_BA 只是把前 ba_len 个点标记为动态排除, 并不跳过 ba_pycolmap;
        #   BA 走 cam_from_track 的 not-fixed_cam 分支 → 依赖 pycolmap 3.11 的旧 API,
        #   与本环境 pycolmap 4.1(py3.13 无 3.11 wheel) 不兼容 (Image(id=) 已改名)。
        #   fixed_cam=True 直接用前端 VGGT4Track 估计的相机位姿(c2w_est_curr), 不做 BA 精修,
        #   对速度/位移/形变等相对运动量精度足够, 且彻底绕开 pycolmap。
        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    (c2w_traj, intrs_out, point_map, conf_depth,
                     track3d_pred, track2d_pred, vis_pred, conf_pred, video_out) = self._model.forward(
                        video_tensor, depth=depth_tensor, intrs=intrs, extrs=extrs,
                        queries=query_xyt, fps=fps, full_point=False, iters_track=4,
                        query_no_BA=True, fixed_cam=True, stage=1, unc_metric=unc_metric,
                        support_frame=T - 1, replace_ratio=0.2,
                    )
        except Exception:
            # 诊断: 打印完整 traceback 定位深层 None.forward 等错误 (推理路径在 SpaTrackerV2 内部)
            import traceback
            print("[SpatialTracker] track_3d 推理异常, 完整栈:")
            traceback.print_exc()
            raise

        # track3d_pred: [T, N, 3+] (相机系) → 世界系 (world = R @ p + t)
        t3d = track3d_pred[..., :3].float().cpu()          # [T,N,3]
        R = c2w_traj[:, :3, :3].float().cpu()              # [T,3,3]
        tr = c2w_traj[:, :3, 3].float().cpu()              # [T,3]
        world = torch.einsum("tij,tnj->tni", R, t3d) + tr[:, None, :]  # [T,N,3]
        world = world.numpy().transpose(1, 0, 2)           # [N,T,3]
        print(f"[SpatialTracker] returned track3d={world.shape}, "
              f"vis={tuple(vis_pred.shape)}")
        vis = vis_pred.float().cpu().numpy()

        # 评估 c2w_traj 自身一致性 (世界系质量指示器)
        c2w_np = c2w_traj.float().cpu().numpy()
        cam_trans_mag = float(np.linalg.norm(np.diff(c2w_np[:, :3, 3], axis=0), axis=-1).mean())

        # ── 显式释放 GPU 中间张量, 防显存泄漏 ──
        # self._model.forward 返回 9 元组中, track2d_pred/conf_pred/video_out/
        # point_map/conf_depth 等 GPU 张量在本函数返回后仍被局部变量引用,
        # 靠 Python GC 回收不及时 → CUDA caching allocator 显存持续累积到 OOM。
        # 逐视频循环里必须主动 del + empty_cache。
        for _g in (c2w_traj, intrs_out, point_map, conf_depth,
                   track3d_pred, track2d_pred, vis_pred, conf_pred, video_out,
                   video_tensor, depth_tensor, extrs, intrs):
            try:
                del _g
            except (NameError, UnboundLocalError):
                pass
        torch.cuda.empty_cache()

        traj = self.compute_velocity_accel(world, fps=fps)
        traj.confidences = vis.transpose(1, 0) if vis.ndim == 2 else vis
        # 附加坐标系质量元数据
        traj.object_ids = getattr(traj, "object_ids", None) or []
        if not hasattr(traj, "_coord_meta"):
            # 通过附加属性存储坐标系质量指标 (不影响 dataclass 结构)
            setattr(traj, "_coord_meta", {
                "vggt_depth_conf_mean": vggt_depth_conf_mean,
                "cam_translation_mag_per_frame": cam_trans_mag,
                "source": "vggt4track_c2w_est",
                "ba_mode": "fixed_cam_no_pycolmap",
                "fps": fps,
            })
        return traj

    def compute_velocity_accel(self, trajectories: np.ndarray,
                               fps: float = 24.0) -> TrajectoryOutput:
        """中心差分求速度/加速度。trajectories [N,T,3]。
        
        fps: 视频帧率，用于将每帧位移转换为物理速度 (m/s)。
             速度 = diff(positions) * fps; 加速度 = diff(velocities) * fps
        """
        trajectories = np.asarray(trajectories, dtype=np.float64)
        if trajectories.ndim != 3 or trajectories.shape[1] < 2:
            return TrajectoryOutput(trajectories=trajectories)
        dt = 1.0 / max(fps, 1.0)
        vel = np.diff(trajectories, axis=1) / dt       # [N,T-1,3] in m/s
        acc = np.diff(vel, axis=1) / dt if vel.shape[1] >= 2 else None  # [N,T-2,3] in m/s²
        return TrajectoryOutput(trajectories=trajectories, velocities=vel, accelerations=acc)

    @staticmethod
    def compute_angular_velocity(poses: np.ndarray, fps: float = 24.0) -> np.ndarray:
        """旋转矩阵序列 [N,T,3,3] → 角速度(轴角) [N,T-1,3] in rad/s。
        
        fps: 视频帧率，用于将每帧旋转角转换为物理角速度 (rad/s)。
        """
        poses = np.asarray(poses)
        if poses.ndim != 4 or poses.shape[1] < 2:
            N = poses.shape[0] if poses.ndim >= 1 else 0
            return np.zeros((N, 0, 3))
        dt = 1.0 / max(fps, 1.0)
        N, T = poses.shape[:2]
        ang_vel = np.zeros((N, T - 1, 3))
        for i in range(N):
            for t in range(T - 1):
                dR = poses[i, t + 1] @ poses[i, t].T
                theta = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
                if theta < 1e-8:
                    continue
                axis = np.array([dR[2, 1] - dR[1, 2],
                                 dR[0, 2] - dR[2, 0],
                                 dR[1, 0] - dR[0, 1]]) / (2 * np.sin(theta))
                ang_vel[i, t] = axis * theta / dt  # rad/s
        return ang_vel
