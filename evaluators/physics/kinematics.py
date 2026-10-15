"""
L3 物理层 — 运动学组评测器
R15 位移 + R16 速度加速度 (SpatialTracker V2 + SAM3) + R17 角速度 (Any6D + VGGT)

评测思路 (依PHYSICS_TAXONOMY.md 表格):
- R15: Sim(3)对齐后 Worldline L2 / Mean&Final Drift / Fail Rate;
       无GT时按 prompt 限定的典型运动场景检验轨迹是否符合该场景物理方程
- R16: 有GT算误差; 无GT匀速CV=exp(−std/mean) / 匀加速一致性=exp(−Var(a))
- R17: 有GT角速度+转轴误差; 无GT jerk + 旋转CV

关键修正:
- **物体级语义**: R15/R16 使用 SAM3 掩码注入 query_points 聚合的物体质心轨迹,
  不再对 400 个网格点(含大量静止背景)求 CV — 背景点会把速度 CV 稀释失真。
  R17 的 Kabsch 旋转仍用稠密点轨迹(刚体旋转需要稠密点, 这是正确用法)。
"""
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult
from ..thresholds import score_metric, score_lower_better
from ._object_tracks import build_object_tracks


def _poly_r2_k(x: np.ndarray, y: np.ndarray, deg: int) -> float:
    """多项式拟合决定系数 R²，用于「轨迹是否符合场景物理方程」。

    deg=2 → 匀加速(抛体/自由落体); deg=1 → 匀速。
    数据完全静止(无方差)时返回 nan, 表示该检验无判别力。
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < deg + 2:
        return float("nan")
    x, y = x[ok], y[ok]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    try:
        coef = np.polyfit(x, y, deg)
    except Exception:
        return float("nan")
    ss_res = float(np.sum((y - np.polyval(coef, x)) ** 2))
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def _sine_r2(x: np.ndarray, y: np.ndarray) -> float:
    """单摆场景: 用FFT 找主频后做正弦最小二乘拟合, 返回 R²。

    模型 y ≈ a·sin(ωt) + b·cos(ωt) + c, ω 取FFT 主频(排除直流)。
    线性最小二乘可解, 无需迭代优化。
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return float("nan")
    x, y = x[ok], y[ok]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    yc = y - y.mean()
    dt = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    if dt <= 0:
        return float("nan")
    spec = np.abs(np.fft.rfft(yc))
    freqs = np.fft.rfftfreq(len(yc), d=dt)
    if len(spec) < 2:
        return float("nan")
    k = int(np.argmax(spec[1:]) + 1)      # 跳过直流分量
    omega = 2.0 * np.pi * float(freqs[k])
    if omega <= 0:
        return float("nan")
    A = np.stack([np.sin(omega * x), np.cos(omega * x), np.ones_like(x)], axis=-1)
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return float("nan")
    ss_res = float(np.sum((y - A @ coef) ** 2))
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


class KinematicsEvaluator(BaseEvaluator):
    """运动学组评测器（R15 / R16 / R17）

    R15: 空间轨迹 + SAM3 物体计数检测(突然出现/消失) + 碰撞反弹对称性检查。
    """

    def __init__(self, spatial_tracker, any6d, sam3=None, config: Optional[dict] = None):
        super().__init__("L3-Physics/Kinematics", config)
        self.spatial_tracker = spatial_tracker
        self.any6d = any6d
        self.sam3 = sam3  # 物体级轨迹 + 物体数量变化检测

    def evaluate(self, frames: list[Image.Image],
                 gt_trajectories: Optional[np.ndarray] = None,
                 gt_velocities: Optional[np.ndarray] = None,
                 gt_angular_vel: Optional[np.ndarray] = None,
                 gt_masks: Optional[np.ndarray] = None,
                 gt_tracking_masks: Optional[np.ndarray] = None,
                 full_frames: Optional[list[Image.Image]] = None,
                 fps: float = 24.0,
                 scene_type: str = "unspecified",
                 any6d_depths: Optional[np.ndarray] = None,
                 any6d_intrinsics: Optional[np.ndarray] = None,
                 any6d_masks: Optional[np.ndarray] = None,
                 any6d_mesh=None,
                 target_dimensions: Optional[list[str]] = None,
                 prompt: str = ""
                 ) -> list[EvalResult]:
        frame_arrays = np.stack([np.array(f) for f in frames])
        # 物体计数检测用全帧(避免采样漏掉瞬态变化)
        r22_input = (np.stack([np.array(f) for f in full_frames])
                     if full_frames is not None and len(full_frames) > 0
                     else frame_arrays)
        results = []

        # 维度级按需执行
        want = set(target_dimensions) if target_dimensions else None
        def _need(prefix: str) -> bool:
            return want is None or any(d.startswith(prefix) for d in want)

        # R15 + R16 依赖物体级轨迹 (仅在需要时跑 SpatialTracker 追踪)
        traj = None
        if _need("R15") or _need("R16"):
            try:
                # CI2V 有外部实例GT时，使用目标mask锁定物体；invalid 样本可将
                # 视频观测mask(gt_tracking_masks)与物理GT mask(gt_masks)分离。
                # 不能再次用SAM3的通用object/thing候选决定目标，否则静态桌面
                # 可能排在cue_ball之前。
                tracking_masks = (gt_tracking_masks if gt_tracking_masks is not None
                                  else gt_masks)
                traj = build_object_tracks(
                    self.spatial_tracker, self.sam3, frame_arrays, fps=fps,
                    dimension="R15", reference_masks=tracking_masks,
                    target_only=(tracking_masks is not None and gt_trajectories is not None),
                    prompt=prompt)
                if tracking_masks is not None and gt_trajectories is not None \
                        and getattr(traj, "source", "") == "grid_fallback":
                    # CI2V有外部目标mask时，网格fallback不是表格规定的R15
                    # 物体级GT评测，禁止继续标成status=ok并产生误导分数。
                    raise RuntimeError(
                        "CI2V R15 外部GT目标追踪失败: SpatialTracker未能保留"
                        "GT mask query points，拒绝grid_fallback计分")
            except Exception as e:
                if _need("R15"):
                    results.append(EvalResult.error("R15-位移", f"SpatialTracker 失败: {e}"))
                if _need("R16"):
                    results.append(EvalResult.error("R16-速度加速度", "SpatialTracker 失败"))
                traj = None
            else:
                if _need("R15"):
                    results.append(self._eval_displacement(
                        traj, gt_trajectories, r22_input,
                        fps=fps, scene_type=scene_type, prompt=prompt))
                if _need("R16"):
                    results.append(self._eval_velocity_accel(traj, gt_velocities,
                                        scene_type=scene_type))

        # R17: 首选 Any6D 物体位姿; 未部署则用稠密点轨迹 Kabsch 估计旋转降级
        if _need("R17"):
            results.append(self._eval_angular_velocity(
                frame_arrays, gt_angular_vel, traj=traj, fps=fps,
                any6d_depths=any6d_depths,
                any6d_intrinsics=any6d_intrinsics,
                any6d_masks=any6d_masks,
                any6d_mesh=any6d_mesh,
            ))
        return results

    @staticmethod
    def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Kabsch: 求把点集 P 对齐到 Q 的最优旋转矩阵 [3,3] (去质心 + SVD, 保证 det=+1)。"""
        Pc = P - P.mean(axis=0)
        Qc = Q - Q.mean(axis=0)
        H = Pc.T @ Qc
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        return Vt.T @ D @ U.T

    def _rotations_from_tracks(self, tracks: np.ndarray) -> np.ndarray:
        """从 3D 点轨迹 [N,T,3] 估计"场景/物体"帧间刚体旋转序列 [1,T,3,3]。

        把所有可靠追踪点当作一个刚体, 用 Kabsch 求首帧→第 t 帧的旋转 R_{0→t}。
        (SpatialTracker 追踪全画面网格点, 无逐物体分割时的合理角速度代理; 若场景含
        主导刚体运动即近似其旋转。) 返回可直接喂给 compute_angular_velocity。
        """
        tracks = np.asarray(tracks, dtype=np.float64)
        if tracks.ndim != 3 or tracks.shape[1] < 2:
            return np.zeros((0, 0, 3, 3))
        N, T = tracks.shape[:2]
        # 只保留全程有限的点
        finite = np.isfinite(tracks).all(axis=(1, 2))
        pts = tracks[finite]
        if pts.shape[0] < 3:   # Kabsch 至少需 3 个非共线点
            return np.zeros((0, 0, 3, 3))
        P0 = pts[:, 0, :]
        rot = np.zeros((1, T, 3, 3))
        rot[0, 0] = np.eye(3)
        for t in range(1, T):
            rot[0, t] = self._kabsch_rotation(P0, pts[:, t, :])
        return rot

    @staticmethod
    def _sim3_align(source: np.ndarray, target: np.ndarray):
        """Umeyama Sim(3)对齐 source→target，返回对齐轨迹及参数。

        R15 表格明确要求 Sim(3) 对齐后 Worldline L2 / Drift。SpatialTracker
        和Kubric的世界坐标原点、旋转和尺度不同，单纯平移对齐会把坐标系差异
        错计为运动误差。此处不改变轨迹形状，只消除全局相似变换。
        """
        x = np.asarray(source, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        if finite.sum() < 3:
            raise ValueError("Sim3至少需要3个有效时间点")
        x, y = x[finite], y[finite]
        mx, my = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - mx, y - my
        cov = xc.T @ yc / len(x)
        U, singular, Vt = np.linalg.svd(cov)
        D = np.eye(3)
        if np.linalg.det(Vt.T @ U.T) < 0:
            D[-1, -1] = -1.0
        R = Vt.T @ D @ U.T
        var_x = float(np.sum(xc * xc) / len(x))
        if var_x < 1e-12:
            raise ValueError("预测轨迹无足够空间变化，Sim3尺度不可估")
        scale = float(np.sum(singular * np.diag(D)) / var_x)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Sim3尺度非法: {scale}")
        # R采用列向量约定(与_spatial_tracker/Kabsch一致)，行向量轨迹右乘R.T。
        aligned = scale * (source - mx) @ R.T + my
        return aligned, {"scale": scale, "rotation": R,
                         "translation": my - scale * mx @ R.T}

    @staticmethod
    def _temporal_progress_error(pred: np.ndarray, gt: np.ndarray) -> float:
        """比较预测/GT沿时间的归一化累计路程进度。

        R15 的单物体直线轨迹在Sim(3)下存在退化：冻结、瞬移、倒序等错误
        可能被全局相似变换部分吸收。该量是Worldline时间轨迹的派生Mean
        Drift诊断，不改变坐标尺度，专门检验"每个时刻走到哪里"。
        """
        pred = np.asarray(pred, dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)
        if len(pred) < 2 or len(gt) < 2:
            return 1.0
        # R15当前校准为单方向平移：使用GT首尾方向的**有符号进度**，
        # 可区分正向、反向、冻结和瞬移；单纯累计绝对路程无法区分正向/倒放。
        gt_vec = gt[-1] - gt[0]
        gt_len = float(np.linalg.norm(gt_vec))
        gt_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)
        gt_path = float(gt_step.sum())
        pred_path = float(np.linalg.norm(np.diff(pred, axis=0), axis=1).sum())
        # Kubric静止刚体仍有约1e-4m的物理数值抖动；不能将其首尾
        # 微小向量归一化成任意方向，否则progress_error会爆到数百。
        if gt_len < 1e-4 and gt_path < 1e-3:
            return 0.0 if pred_path < 1e-3 else 1.0
        if gt_len >= 1e-8:
            direction = gt_vec / gt_len
            gt_signed = (gt - gt[0]) @ direction
            pred_signed = (pred - pred[0]) @ direction
            gt_den = float(gt_signed[-1] - gt_signed[0])
            pred_den = float(pred_signed[-1] - pred_signed[0])
            if abs(gt_den) < 1e-8:
                return 1.0
            gt_progress = (gt_signed - gt_signed[0]) / gt_den
            pred_progress = ((pred_signed - pred_signed[0]) / gt_den
                             if abs(pred_den) >= 1e-8
                             else np.zeros(len(pred)))
            return float(np.mean(np.abs(pred_progress - gt_progress)))

        # GT首尾重合时退回累计路程进度（例如周期运动）。
        ps = np.linalg.norm(np.diff(pred, axis=0), axis=1)
        gs = np.linalg.norm(np.diff(gt, axis=0), axis=1)
        pred_total, gt_total = float(ps.sum()), float(gs.sum())
        if gt_total < 1e-8:
            return 0.0 if pred_total < 1e-8 else 1.0
        pred_progress = (np.concatenate([[0.0], np.cumsum(ps)]) / pred_total
                         if pred_total > 1e-8 else np.zeros(len(pred)))
        gt_progress = np.concatenate([[0.0], np.cumsum(gs)]) / gt_total
        return float(np.mean(np.abs(pred_progress - gt_progress)))

    def _eval_displacement(self, traj, gt, all_frames=None, fps: float = 24.0,
                           scene_type: str = "unspecified",
                           prompt: str = "") -> EvalResult:
        """R15 位移评测: 空间轨迹 + 物体计数 + 碰撞反弹检测。

        all_frames: 全量帧 (用于 SAM3 物体计数检测), 可为None 则回退采样帧。
        fps: 视频帧率，用于合理设置跳跃阈值（高fps下per-frame位移更小）。
        scene_type: prompt 声明的物理场景, 用于「测轨迹是否符合该场景物理方程」。
        """
        T_pos = traj.trajectories
        N, Ts = T_pos.shape[:2]
        if gt is not None:
            errs, drifts, scales, progress_errors = [], [], [], []
            alignment_modes = []
            align_errors = []
            for i in range(min(N, len(gt))):
                pred_i = np.asarray(T_pos[i], dtype=np.float64)
                gt_i = np.asarray(gt[i], dtype=np.float64)
                gt_step = np.linalg.norm(np.diff(gt_i, axis=0), axis=1)
                gt_path = float(gt_step.sum())
                gt_end_disp = float(np.linalg.norm(gt_i[-1] - gt_i[0]))
                # Sim(3)在静止GT上是退化的：它可以把任意预测运动
                # 缩放成一个点，导致错误运动得到零Worldline误差。
                # 静止GT必须走“不对齐”分支，直接检查预测是否产生显著运动。
                if gt_path < 1e-3 and gt_end_disp < 1e-4:
                    pred_step = np.linalg.norm(np.diff(pred_i, axis=0), axis=1)
                    pred_path = float(pred_step.sum())
                    pred_end_disp = float(np.linalg.norm(pred_i[-1] - pred_i[0]))
                    static_motion = bool(pred_path > 0.05 or pred_end_disp > 0.02)
                    err = pred_path if static_motion else 0.0
                    drift = pred_end_disp
                    progress_error = 1.0 if static_motion else 0.0
                    scales.append(1.0)
                    alignment_modes.append("static_gt_no_alignment")
                    if static_motion:
                        align_errors.append(
                            f"GT静止但预测运动: path={pred_path:.4f}, "
                            f"end_disp={pred_end_disp:.4f}")
                    errs.append(err)
                    drifts.append(drift)
                    progress_errors.append(progress_error)
                    continue
                try:
                    aligned, params = self._sim3_align(pred_i, gt_i)
                    alignment_modes.append("sim3")
                    err = float(np.mean(np.linalg.norm(aligned - gt_i, axis=-1)))
                    pred_disp = aligned[-1] - aligned[0]
                    gt_disp = gt_i[-1] - gt_i[0]
                    drift = float(np.linalg.norm(pred_disp - gt_disp))
                    scales.append(params["scale"])
                    progress_error = self._temporal_progress_error(aligned, gt_i)
                except ValueError as exc:
                    # 轨迹退化时不伪造Sim3结果，记录不可对齐并使用原始指标
                    err = float(np.mean(np.linalg.norm(pred_i - gt_i, axis=-1)))
                    drift = float(np.linalg.norm(
                        (pred_i[-1] - pred_i[0]) - (gt_i[-1] - gt_i[0])))
                    progress_error = self._temporal_progress_error(pred_i, gt_i)
                    alignment_modes.append("raw_fallback")
                    align_errors.append(str(exc))
                errs.append(err)
                drifts.append(drift)
                progress_errors.append(progress_error)
            if errs:
                mean_err = float(np.mean(errs))
                final_drift = float(np.mean(drifts))
                progress_error = float(np.mean(progress_errors))
                worldline_score = score_metric("R15_worldline_l2", mean_err)
                # 表格的Worldline/Mean Drift需同时约束时间进度；否则单物体
                # 直线轨迹的freeze/teleport可能被Sim(3)吸收后仍拿高分。
                # 用正确Kubric valid校准集确定的进度误差基线：低速/中速/0.5m/s
                # 的正确视频约为0.11~0.15。因此5分上界取0.15，不能把
                # SpatialTracker自身基线误差当作被测视频错误。
                progress_score = score_lower_better(
                    progress_error, [0.15, 0.25, 0.35, 0.50])
                score_components = [worldline_score, progress_score]
                track_fail_rate = None
                track_fail_score = None
                confidences = getattr(traj, "confidences", None)
                if confidences is not None:
                    conf = np.asarray(confidences, dtype=np.float64)
                    if conf.ndim == 2 and conf.shape[1] == len(pred_i):
                        track_fail_rate = float(np.mean(~np.isfinite(conf) | (conf < 0.5)))
                        track_fail_score = score_lower_better(
                            track_fail_rate, [0.05, 0.15, 0.30, 0.50])
                        score_components.append(track_fail_score)
                # R15表格三项：Worldline/Mean&Final Drift/Fail Rate。
                score = min(score_components)
                violations = []
                if score < 4:
                    msg = (
                        f"WorldlineL2(Sim3)={mean_err:.3f}, "
                        f"FinalDrift(Sim3)={final_drift:.3f}, "
                        f"TemporalProgressError={progress_error:.3f}")
                    if track_fail_rate is not None:
                        msg += f", TrackFailRate={track_fail_rate:.1%}"
                    violations.append(msg)
                metrics = {
                    "worldline_l2": mean_err,
                    "final_drift": final_drift,
                    "mean_drift": mean_err,
                    "temporal_progress_error": progress_error,
                    "worldline_score": worldline_score,
                    "progress_score": progress_score,
                    "track_fail_rate": track_fail_rate,
                    "track_fail_score": track_fail_score,
                    "alignment": (alignment_modes[0] if alignment_modes
                                  and len(set(alignment_modes)) == 1 else "mixed"),
                    "alignment_modes": alignment_modes,
                    "sim3_scales": scales,
                    "n_objects_compared": len(errs),
                    "track_source": getattr(traj, "source", "unknown"),
                }
                if align_errors:
                    metrics["sim3_fallback_reasons"] = align_errors
                return EvalResult("R15-位移", score, violations,
                                  metrics=metrics, status="ok")
        if Ts < 2:
            return EvalResult.skipped("R15-位移", "轨迹点数不足")
        jumps = np.linalg.norm(np.diff(T_pos, axis=1), axis=-1)
        # 跳跃阈值按 fps 缩放: 高fps 每帧位移小, 阈值应降低; 以 24fps 为基准
        jump_threshold = 2.0 * (24.0 / max(fps, 1.0))
        fail_rate = float(np.mean(jumps > jump_threshold))
        path_len = float(np.mean(np.sum(jumps, axis=1)))
        score = score_lower_better(fail_rate, [0.05, 0.15, 0.3, 0.5])

        violations = [] if score >= 4 else [f"FailRate={fail_rate:.1%}"]
        metrics = {"avg_path_length": path_len, "fail_rate": round(fail_rate, 4)}

        # ── v3: SAM3 物体计数检测 ──
        # 检测物体数量在视频过程中的变化 (突然出现/消失/分裂)
        sam3_count_change = False
        sam3_n_objects = -1
        sam3_count_min = -1
        sam3_count_max = -1
        if self.sam3 is not None and all_frames is not None and len(all_frames) > 0:
            try:
                seg = self.sam3.segment_video(all_frames, dimension="R15",
                                              prompt=prompt)
                if seg.masks is not None and seg.masks.ndim == 4:
                    # masks: [N_objects, T, H, W]
                    # 每帧有效物体数 (mask 面积 > 10px)
                    counts = np.sum(seg.masks.sum(axis=(2, 3)) > 10, axis=0)  # [T]
                    sam3_n_objects = int(np.median(counts))
                    sam3_count_min = int(np.min(counts))
                    sam3_count_max = int(np.max(counts))

                    # 检测相邻帧物体数变化
                    count_diffs = np.abs(np.diff(counts.astype(float)))
                    # 若任意相邻帧物体数变化 >= 2 → 突然出现/消失
                    if np.any(count_diffs >= 2):
                        sam3_count_change = True
                        jump_positions = np.where(count_diffs >= 2)[0]
                        violations.append(
                            f"SAM3检测到物体数突变: "
                            f"median={sam3_n_objects}, "
                            f"帧{jump_positions.tolist()}处相邻帧差异≥2")
                    # 若前后半段物体数中位数差 >= 1 → 永久性变化
                    T_all = len(counts)
                    if T_all >= 4:
                        half = T_all // 2
                        med_first = np.median(counts[:half])
                        med_second = np.median(counts[half:])
                        if abs(med_first - med_second) >= 1:
                            sam3_count_change = True
                            violations.append(
                                f"SAM3检测到物体数永久变化: "
                                f"前半段median={int(med_first)} → 后半段median={int(med_second)}")
            except Exception:
                pass  # SAM3 不可用则静默跳过

        # ── v3: 碰撞反弹对称性检测 ──
        # 检测轨迹速度方向反转是否构成合理的碰撞反弹
        # 合理反弹: 反弹后速率 ≤ 反弹前速率, 方向在反弹平面内镜像
        bounce_violations = 0
        if traj.velocities is not None and traj.velocities.shape[1] >= 4:
            vel_all = traj.velocities  # [N, T-1, 3]
            for i in range(vel_all.shape[0]):
                vi = vel_all[i]  # [T-1, 3]
                vmag = np.linalg.norm(vi, axis=-1)
                if vmag.mean() < 1e-6:
                    continue
                # 检测速度方向突变 (余弦相似度 < 0 → 方向反转)
                for t in range(1, len(vi) - 1):
                    speed_before = float(np.linalg.norm(vi[t-1]))
                    speed_after = float(np.linalg.norm(vi[t]))
                    # 跳过极小速度帧: 静止/极慢运动下数值噪声会伪造"方向反转"
                    if speed_before < 0.01 or speed_after < 0.01:
                        continue
                    dot = np.dot(vi[t-1], vi[t]) / (speed_before * speed_after + 1e-8)
                    if dot < -0.5:  # 方向反转 > 120°
                        # 碰撞后速度不应大于碰撞前
                        if speed_after > speed_before * 1.1:
                            bounce_violations += 1
                            if bounce_violations <= 2:
                                violations.append(
                                    f"物体{i}帧{t}: 碰撞后速率增大 "
                                    f"({speed_before:.2f}→{speed_after:.2f})")
                        # 检查碰前碰后方向是否大致镜像 (dot_product ≈ -1 则是镜像反弹)
                        # 这里 dot<-0.5 已经保证了方向大反转, 基本满足碰撞后反弹
                        break  # 每个物体只检测一次反弹

        # ── 综合打分 ──
        # 基础分 = 轨迹 fail_rate 评分
        # 物体计数变化扣 1 分, 反弹异常扣 1 分
        if sam3_count_change:
            score = max(1, score - 1)
        if bounce_violations > 0:
            score = max(1, score - 1)

        metrics["sam3_n_objects"] = sam3_n_objects
        metrics["sam3_count_min"] = sam3_count_min
        metrics["sam3_count_max"] = sam3_count_max
        metrics["sam3_count_change"] = sam3_count_change
        metrics["bounce_violations"] = bounce_violations

        # ── 表格R15 代理方案核心:「prompt 限定典型运动场景，
        #     测轨迹是否符合该场景的物理方程」──
        eq_fit = self._fit_scene_equation(T_pos, scene_type, fps)
        if eq_fit is not None:
            r2, eq_name = eq_fit
            metrics["scene_equation"] = eq_name
            metrics["scene_equation_r2"] = round(float(r2), 4)
            eq_score = int(np.clip(round(r2 * 4 + 1), 1, 5))
            if r2 < 0.6:
                violations.append(
                    f"轨迹不符合{eq_name}方程 (R²={r2:.2f})")
            # 与轨迹连续性分取较严者: 场景方程是该维度的主判据
            score = min(score, eq_score)
        metrics["scene_type"] = scene_type
        metrics["n_objects"] = int(N)
        metrics["track_source"] = traj.source

        return EvalResult("R15-位移", score, violations,
                          metrics=metrics, status="proxy")

    @staticmethod
    def _fit_scene_equation(pos: np.ndarray, scene_type: str,
                            fps: float) -> Optional[tuple]:
        """检验轨迹是否符合 prompt 指定场景的物理方程，返回 (R², 方程名)。

        表格 R15「T2V/I2V 代理方案」: prompt 限定典型运动场景(如"水平抛出小球/
        单摆摆动"), 测轨迹是否符合该场景的物理方程。

        - projectile/free_fall/bounce/incline → 位移随时间二次(匀加速)
        - sliding/rolling/circular          → 匀速 → 一次
        - pendulum                          → 周期振荡: 用主频正弦拟合
        返回 None表示该场景无明确方程可判(不参与打分, 避免瞎扣分)。
        """
        QUADRATIC = {"projectile", "free_fall", "bounce", "incline"}
        LINEAR = {"sliding", "rolling", "circular"}
        if scene_type in QUADRATIC:
            deg, name = 2, "匀加速(二次)"
        elif scene_type in LINEAR:
            deg, name = 1, "匀速(一次)"
        elif scene_type == "pendulum":
            deg, name = None, "单摆(正弦)"
        else:
            return None

        N, T = pos.shape[:2]
        if T < 5:
            return None
        t = np.arange(T, dtype=float) / max(fps, 1.0)
        r2s = []
        for i in range(N):
            p = pos[i]
            if not np.isfinite(p).all():
                continue
            # 取位移最大的主运动方向 (以极差最大轴作为主轴的简化 PCA)
            span = p.max(axis=0) - p.min(axis=0)
            if float(span.max()) < 1e-8:
                continue
            axis = int(np.argmax(span))
            y = p[:, axis]
            if deg is not None:
                r2 = _poly_r2_k(t, y, deg)
            else:
                r2 = _sine_r2(t, y)
            if np.isfinite(r2):
                r2s.append(r2)
        if not r2s:
            return None
        return float(np.mean(r2s)), name

    def _eval_velocity_accel(self, traj, gt_vel, scene_type="unspecified") -> EvalResult:
        vel = traj.velocities
        if vel is None or vel.shape[1] == 0:
            return EvalResult.skipped("R16-速度加速度", "无速度数据")
        if gt_vel is not None:
            tlen = min(vel.shape[1], gt_vel.shape[1])
            err = float(np.mean(np.linalg.norm(vel[:, :tlen] - gt_vel[:, :tlen], axis=-1)))
            score = score_metric("R16_vel_err", err)
            return EvalResult("R16-速度加速度", score,
                              [] if score >= 4 else [f"速度误差={err:.2f}"],
                              metrics={"vel_error": err}, status="ok")

        # ── 无GT: 按表格分场景选择假设 ──
        # 表格: "针对匀速运动，速度的变异系数=exp(−std/mean)；
        #        针对匀加速运动，加速度一致性=exp(−Var(a))"
        # 即**场景决定用哪个指标**。此前实现取两者的较优值(max), 属于评分虚高:
        # 任何轨迹只要在两种假设之一下表现好就能拿高分, 削弱了区分度。
        vmag = np.linalg.norm(vel, axis=-1)  # [N, T-1] N=物体数

        all_vel_cvs, all_accel_cvs, modes = [], [], []
        for i in range(vmag.shape[0]):
            vi = vmag[i]
            if vi.mean() <= 1e-6 or len(vi) < 3:
                continue
            all_vel_cvs.append(float(vi.std() / (vi.mean() + 1e-8)))
            ai = np.diff(vi)
            if len(ai) > 0 and np.abs(ai).mean() > 1e-8:
                all_accel_cvs.append(float(np.std(ai) / (np.abs(ai).mean() + 1e-8)))
            else:
                all_accel_cvs.append(0.0)
            vel_first = vi[:max(1, len(vi) // 4)].mean()
            vel_last = vi[-max(1, len(vi) // 4):].mean()
            if vel_last > 1.5 * vel_first:
                modes.append("accelerating")
            elif vel_first > 1.5 * vel_last:
                modes.append("decelerating")
            else:
                modes.append("constant_velocity")

        if not all_vel_cvs:
            return EvalResult.skipped("R16-速度加速度", "无有效运动物体轨迹")

        mean_vel_cv = float(np.mean(all_vel_cvs))
        mean_accel_cv = float(np.mean(all_accel_cvs)) if all_accel_cvs else 999.0
        cv_score_v = float(np.exp(-mean_vel_cv))
        cv_score_a = float(np.exp(-mean_accel_cv * 0.5))

        # 场景 → 假设映射 (prompt 已声明 scene_type)
        CONST_VEL_SCENES = {"sliding", "circular", "rolling"}
        CONST_ACC_SCENES = {"free_fall", "projectile", "incline", "bounce"}
        if scene_type in CONST_VEL_SCENES:
            hypothesis = "constant_velocity"
        elif scene_type in CONST_ACC_SCENES:
            hypothesis = "constant_acceleration"
        else:
            # 场景未声明 → 由实测运动模式判定 (多数投票), 而非取最高分
            n_acc = sum(1 for m in modes if m in ("accelerating", "decelerating"))
            hypothesis = ("constant_acceleration"
                          if n_acc > len(modes) / 2 else "constant_velocity")

        if hypothesis == "constant_acceleration":
            score = int(np.clip(round(cv_score_a * 4 + 1), 1, 5))
            best_cv, cv_used = mean_accel_cv, cv_score_a
            violations = ([] if score >= 4
                          else [f"加速度一致性偏低 accel_cv={mean_accel_cv:.3f}, "
                                f"exp(-cv/2)={cv_score_a:.2f}"])
        else:
            score = int(np.clip(round(cv_score_v * 4 + 1), 1, 5))
            best_cv, cv_used = mean_vel_cv, cv_score_v
            violations = ([] if score >= 4
                          else [f"速度稳定性偏低 vel_cv={mean_vel_cv:.3f}, "
                                f"exp(-cv)={cv_score_v:.2f}"])

        return EvalResult("R16-速度加速度", score, violations,
                          metrics={
                              "velocity_cv": round(mean_vel_cv, 4),
                              "acceleration_cv": round(mean_accel_cv, 4),
                              "cv_score_used": round(cv_used, 4),
                              "hypothesis": hypothesis,
                              "scene_type": scene_type,
                              "observed_modes": modes[:8],
                              "n_objects": int(vmag.shape[0]),
                              "track_source": traj.source,
                          }, status="proxy")

    def _eval_angular_velocity(self, frames, gt_ang, traj=None,
                               fps: float = 24.0, any6d_depths=None,
                               any6d_intrinsics=None, any6d_masks=None,
                               any6d_mesh=None) -> EvalResult:
        source = "any6d"
        rotations = None
        try:
            poses = self.any6d.estimate_pose_sequence(
                frames,
                masks=any6d_masks,
                depths=any6d_depths,
                intrinsics=any6d_intrinsics,
                mesh=any6d_mesh,
            )
            rotations = poses.rotations  # [N,T,3,3]
        except Exception:
            # Any6D 需物体掩码/未部署 → 降级:用稠密点轨迹 Kabsch 估计刚体旋转。
            # 注意: 刚体旋转估计**必须**用稠密点(至少3个非共线点), 不能用物体质心
            # (单点无朝向信息), 故这里取 point_tracks 而非 trajectories。
            try:
                if traj is None:
                    from ._object_tracks import build_object_tracks
                    traj = build_object_tracks(self.spatial_tracker, self.sam3,
                                               frames, fps=fps, dimension="R17")
                dense = getattr(traj, "point_tracks", None)
                if dense is None or np.asarray(dense).size == 0:
                    dense = traj.trajectories
                rotations = self._rotations_from_tracks(dense)  # [1,T,3,3]
                source = "spatial_tracker_kabsch"
            except Exception as e:
                return EvalResult.error("R17-角速度", f"Any6D/SpatialTracker 均不可用: {e}")

        if rotations is None or np.asarray(rotations).size == 0:
            return EvalResult.skipped("R17-角速度", "无有效旋转数据")

        ang_vel = self.spatial_tracker.compute_angular_velocity(rotations, fps=fps)
        if ang_vel.size == 0:
            return EvalResult.skipped("R17-角速度", "无角速度数据")

        if gt_ang is not None:
            tlen = min(ang_vel.shape[1], gt_ang.shape[1])
            err = float(np.mean(np.linalg.norm(ang_vel[:, :tlen] - gt_ang[:, :tlen], axis=-1)))
            score = score_metric("R17_angvel_err", err)
            return EvalResult("R17-角速度", score,
                              [] if score >= 4 else [f"角速度误差={err:.3f}"],
                              metrics={"angular_vel_error": err}, status="ok")
        # 无GT: 旋转 CV (自由旋转 ω 应恒定)
        mag = np.linalg.norm(ang_vel, axis=-1).reshape(-1)
        rot_cv = float(mag.std() / (mag.mean() + 1e-8)) if mag.mean() > 1e-6 else 0.0
        score = score_metric("R17_rot_cv", rot_cv)
        return EvalResult("R17-角速度", score,
                          [] if score >= 4 else [f"旋转CV={rot_cv:.3f}"],
                          metrics={"rotation_cv": rot_cv, "source": source, "fps": fps}, status="proxy")
