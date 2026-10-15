"""
L3 物理层 — 动力学组评测器
R22 力的动态效应 + R23 动量守恒 + R24 能量守恒 (SpatialTracker V2 + SAM3 + DA3)

评测思路 (依PHYSICS_TAXONOMY.md 表格):
- R22: 10条尺度无关检验(牛顿一/伽利略/抛体/斜面反解μ/弹跳e²/滑动减速/静摩擦/圆周切线/支撑/滚动)
       + 【度量深度近似数值】DA3 米制 → 验a≈g
- R23: 假设动量守恒解质量比, 校验 e∈[0,1] 且 KE 不增
- R24: KE+PE 时间曲线, 守恒应恒定, 有摩擦单调下降, 不能自发上升

关键修正:
- **物体级语义**: 轨迹经 SAM3 掩码注入 query_points 聚合成物体质心 [N_obj,T,3],
  不再把 SpatialTracker 的400 个网格点误当成 400 个物体 (见 _object_tracks.py)。
- 重力方向从"自由落体物体的主加速度方向"估计(数据驱动)。
- 碰撞检测用"两物体最近距离的时间极小值"定位碰撞帧, 而非硬编码中点。
- 能量量纲: 用 DA3 米制尺度把归一化坐标锚定到米, 使 KE 与 PE=gh 可加。
"""
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult
from ..thresholds import score_metric
from ._object_tracks import build_object_tracks

# 场景类型 → 该场景应启用的 R22 检验项(表格 ①~⑩)
# 表格要求「prompt 限定典型运动场景，测轨迹是否符合该场景的物理方程」,
# 故不同场景只启用可判定的检验, 避免对不适用场景强行打分。
SCENE_CHECKS = {
    "free_fall":    ["newton_1", "projectile", "galilean", "support"],
    "projectile":   ["newton_1", "projectile", "support"],
    "incline":      ["newton_1", "incline_mu", "deceleration", "support"],
    "collision":    ["newton_1", "support"],
    "pendulum":     ["newton_1", "support"],
    "bounce":       ["newton_1", "projectile", "restitution", "support"],
    "sliding":      ["newton_1", "deceleration", "sliding_const_decel", "support"],
    "circular":     ["newton_1", "circular_tangent"],
    "rolling":      ["newton_1", "deceleration", "support"],
    "unspecified":  ["newton_1", "projectile", "deceleration", "support"],
}


def _poly_r2(x: np.ndarray, y: np.ndarray, deg: int) -> float:
    """多项式拟合的决定系数 R²，用于「轨迹是否符合该场景物理方程」。

    deg=2 → 抛物线(自由落体/抛体竖直方向); deg=1 → 匀速/匀减速直线。
    返回 [0,1]，数据退化时返回 nan (由调用方nanmean 忽略)。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < deg + 2:
        return float("nan")
    x, y = x[ok], y[ok]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")   # 完全静止, 该检验无判别力
    try:
        coef = np.polyfit(x, y, deg)
    except Exception:
        return float("nan")
    resid = y - np.polyval(coef, x)
    ss_res = float(np.sum(resid ** 2))
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def _find_peaks(series: np.ndarray, rel_prominence: float = 0.1) -> list:
    """找显著局部极大值索引 (弹跳顶点)。

    rel_prominence: 峰值需高于全序列最小值 + rel_prominence × 全幅，
    过滤掉数值噪声造成的伪峰。
    """
    s = np.asarray(series, dtype=float)
    if s.size < 3 or not np.isfinite(s).any():
        return []
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    thr = lo + rel_prominence * (hi - lo + 1e-8)
    peaks = []
    for t in range(1, len(s) - 1):
        if not np.isfinite(s[t]):
            continue
        if s[t] > s[t - 1] and s[t] >= s[t + 1] and s[t] > thr:
            peaks.append(t)
    return peaks


class DynamicsEvaluator(BaseEvaluator):
    """动力学组评测器（R22 / R23 / R24）"""

    def __init__(self, spatial_tracker, sam3=None, da3=None,
                 config: Optional[dict] = None, vlm=None):
        super().__init__("L3-Physics/Dynamics", config)
        self.spatial_tracker = spatial_tracker
        self.sam3 = sam3   # 物体级轨迹所需 (表格已为 R09/R11/R12/R21 指定 SAM3)
        self.da3 = da3     # 【度量深度近似数值】档所需, 表格 R22 明确列出
        self.vlm = vlm     # R23 可观测性二次判断 (模型没生成碰撞 vs 检测器不行)

    def evaluate(self, frames: list[Image.Image],
                 scene_type: str = "unspecified",
                 gt_trajectories: Optional[np.ndarray] = None,
                 fps: float = 24.0,
                 full_frames: Optional[list[Image.Image]] = None,
                 full_fps: float = 24.0,
                 prompt: str = "",
                 target_dimensions: Optional[list[str]] = None
                 ) -> list[EvalResult]:
        frame_arrays = np.stack([np.array(f) for f in frames])
        # 维度级按需执行
        want = set(target_dimensions) if target_dimensions else None
        def _need(prefix: str) -> bool:
            return want is None or any(d.startswith(prefix) for d in want)

        # R22/R24 需要物体级轨迹 (SpatialTracker); R23 用 SAM3 逐帧分割(不依赖 traj)
        need_traj = _need("R22") or _need("R24")
        traj = None
        if need_traj:
            try:
                traj = build_object_tracks(self.spatial_tracker, self.sam3,
                                           frame_arrays, fps=fps,
                                           dimension="R22", prompt=prompt)
            except Exception as e:
                errs = []
                if _need("R22"):
                    errs.append(EvalResult.error("R22-力的动态效应", f"SpatialTracker 推理失败: {e}"))
                if _need("R24"):
                    errs.append(EvalResult.error("R24-能量守恒", "SpatialTracker 推理失败"))
                traj = None
                # 若 R22/R24 都失败但 R23 需要, 继续算 R23
                if not _need("R23"):
                    return errs
                results = errs
                # 继续走 R23 分支
                traj = None
                g_dir = None
                metric_scale = None
                gt = None
                # 直接跳到 R23 计算
                momentum_result = self._compute_momentum(frames, full_frames, full_fps, prompt, None, gt)
                if _need("R23"):
                    results.append(momentum_result)
                return results

        gt = None
        if gt_trajectories is not None and traj is not None:
            gt = self.spatial_tracker.compute_velocity_accel(gt_trajectories, fps=fps)

        # 估计重力方向(单位向量), 供 R22/R24 使用
        g_dir = self._estimate_gravity_dir(traj) if traj is not None else None
        # 米制尺度因子 (归一化单位 → 米); None 表示无法锚定
        metric_scale = self._estimate_metric_scale(frame_arrays, traj) if traj is not None else None

        results = []
        if _need("R22"):
            results.append(self._eval_force_effects(traj, scene_type, g_dir, gt, fps, metric_scale))
        if _need("R23"):
            results.append(self._compute_momentum(frames, full_frames, full_fps, prompt, traj, gt))
        if _need("R24"):
            results.append(self._eval_energy(traj, g_dir, gt, metric_scale))
        return results

    def _compute_momentum(self, frames, full_frames, full_fps, prompt, traj, gt):
        """R23 碰撞检测: 优先 SAM3 2D 逐帧分割(更稳), 失败则回退 3D 轨迹方案。"""
        momentum_result = None
        if full_frames is not None and len(full_frames) > 4:
            full_arrays = np.stack([np.array(f) for f in full_frames])
            try:
                momentum_result = self._eval_momentum_2d(
                    self.sam3, full_arrays, concepts=None, prompt=prompt)
            except Exception as e:
                print(f"[R23] 2D bbox 碰撞检测失败, 回退 3D: {e}")
                momentum_result = None
        if momentum_result is None:
            momentum_result = self._eval_momentum(traj, gt, prompt=prompt) if traj is not None else \
                EvalResult.skipped("R23-动量守恒", "无可用轨迹")
        return momentum_result

    def _estimate_metric_scale(self, frames, traj) -> Optional[float]:
        """用 DA3 米制深度估计「归一化世界坐标 → 米」的比例因子。

        表格 R22【度量深度近似数值·T2V/I2V 可启用】: 用度量深度模型出近似米制
        → 加速度 ≈ g。做法: 比较 DA3 米制深度中位数与轨迹坐标的深度量级。
        无 DA3 时返回 None, 相关检验降级为「无量纲比值」形式(仍符合表格①~⑩)。
        """
        if self.da3 is None:
            return None
        try:
            key = frames[len(frames) // 2]
            dm = self.da3.estimate_metric_depth(key).depth_map
            dm = dm[np.isfinite(dm) & (dm > 0)]
            if dm.size < 100:
                return None
            depth_m = float(np.median(dm))          # 场景典型深度(米)
        except Exception:
            return None
        pos = np.asarray(traj.trajectories)
        if pos.size == 0:
            return None
        # 轨迹坐标中同一场景的深度量级 (取坐标模长中位数作代理)
        r = np.linalg.norm(pos.reshape(-1, 3), axis=-1)
        r = r[np.isfinite(r) & (r > 0)]
        if r.size < 3:
            return None
        depth_norm = float(np.median(r))
        if depth_norm < 1e-8:
            return None
        scale = depth_m / depth_norm
        # 合理性护栏: 比例因子应为正且不过分极端
        if not np.isfinite(scale) or scale <= 0 or scale > 1e4:
            return None
        return scale

    @staticmethod
    def _estimate_gravity_dir(traj) -> np.ndarray:
        """从**运动物体**的平均加速度方向估计重力方向(数据驱动)。

        只用加速度显著的物体(排除静止背景), 避免均值被稀释。返回单位向量。
        """
        acc = traj.accelerations
        if acc is None or acc.shape[1] == 0:
            return np.array([0.0, -1.0, 0.0])  # 缺省: -y
        amag = np.linalg.norm(acc, axis=-1)              # [N,T-2]
        per_obj = np.nanmean(amag, axis=1)               # [N]
        if np.isfinite(per_obj).any():
            thr = np.nanpercentile(per_obj, 50)
            moving = per_obj >= max(thr, 1e-9)
            acc_sel = acc[moving] if moving.any() else acc
        else:
            acc_sel = acc
        mean_acc = np.nanmean(acc_sel.reshape(-1, 3), axis=0)
        norm = np.linalg.norm(mean_acc)
        if not np.isfinite(norm) or norm < 1e-6:
            return np.array([0.0, -1.0, 0.0])
        return mean_acc / norm


    # ── R22 力的动态效应 (表格①~⑩ 尺度无关检验) ──
    def _eval_force_effects(self, traj, scene_type, g_dir, gt,
                            fps=24.0, metric_scale=None) -> EvalResult:
        pos = traj.trajectories
        vel = traj.velocities
        acc = traj.accelerations
        N, T = pos.shape[:2]
        if T < 3 or acc is None or acc.shape[1] == 0:
            return EvalResult.skipped("R22-力的动态效应", "帧数<3，无法评估")

        # CI2V GT 路径
        if gt is not None and getattr(gt, "trajectories", None) is not None:
            gp = gt.trajectories
            tlen = min(T, gp.shape[1])
            errs = []
            for i in range(min(N, gp.shape[0])):
                aligned = pos[i, :tlen] + (gp[i, 0] - pos[i, 0])
                errs.append(np.mean(np.linalg.norm(aligned - gp[i, :tlen], axis=-1)))
            mean_err = float(np.mean(errs)) if errs else 1.0
            score = score_metric("R22_traj_err", mean_err)
            v_gt = [] if score >= 4 else [f"GT轨迹误差={mean_err:.3f}"]
            # 【CI2V 绝对数值】严格验 a ≈ 9.8 m/s²
            a_check = self._check_gravity_magnitude(acc, g_dir, metric_scale=1.0)
            if a_check is not None:
                a_meas, a_ok = a_check
                if not a_ok:
                    v_gt.append(f"重力加速度={a_meas:.2f}m/s² 偏离9.8")
            return EvalResult("R22-力的动态效应", score, v_gt,
                              metrics={"gt_traj_error": mean_err,
                                       "scene_type": scene_type,
                                       "measured_g": (a_check[0] if a_check else None)},
                              status="ok")

        # ── T2V/I2V: 按场景类型启用对应检验 ──
        enabled = SCENE_CHECKS.get(scene_type, SCENE_CHECKS["unspecified"])
        checks, v = {}, []
        up = -g_dir

        # ① 牛顿第一定律: 无外力方向(垂直重力)不应出现无因加速
        if "newton_1" in enabled:
            acc_flat = acc.reshape(-1, 3)
            acc_perp = acc_flat - np.outer(acc_flat @ g_dir, g_dir)
            perp_mag = np.linalg.norm(acc_perp, axis=-1)
            # 阈值改为相对量: 横向加速度不应超过重力方向加速度典型值的 50%
            a_g_typ = float(np.median(np.abs(acc_flat @ g_dir))) + 1e-8
            spurious = float(np.mean(perp_mag > 0.5 * max(a_g_typ, 1e-6)))
            checks["newton_1"] = 1.0 - spurious
            if spurious > 0.3:
                v.append(f"①牛顿一: 横向无因加速帧比例={spurious:.1%}")

        # ② 伽利略等效性: 同重力场不同物体自由下落加速度应相同
        if "galilean" in enabled and N >= 2:
            a_g_per_obj = np.array([np.nanmean(acc[i] @ g_dir) for i in range(N)])
            a_g_per_obj = a_g_per_obj[np.isfinite(a_g_per_obj)]
            if len(a_g_per_obj) >= 2 and np.abs(a_g_per_obj).mean() > 1e-8:
                cv = float(np.std(a_g_per_obj) / (np.abs(np.mean(a_g_per_obj)) + 1e-8))
                checks["galilean"] = float(np.exp(-cv))
                if cv > 0.3:
                    v.append(f"②伽利略等效性: 各物体下落加速度CV={cv:.2f}(应相同)")

        # ③ 抛体: 轨迹拟合抛物线 + 重力方向加速度恒定 + 水平方向匀速
        if "projectile" in enabled:
            sub = []
            for i in range(N):
                p = pos[i]
                h = p @ up                          # 竖直位移
                horiz = p - np.outer(p @ up, up)    # 水平分量
                t_axis = np.arange(T, dtype=float) / max(fps, 1.0)
                # 竖直方向二次拟合 R²
                r2_v = _poly_r2(t_axis, h, deg=2)
                # 水平方向应匀速 → 一次拟合 R²
                hx = np.linalg.norm(horiz - horiz[0], axis=-1)
                r2_h = _poly_r2(t_axis, hx, deg=1)
                #竖直加速度方向应向下(沿 g_dir 为正)
                a_g = float(np.nanmean(acc[i] @ g_dir))
                sign_ok = 1.0 if a_g > 0 else 0.0
                sub.append(np.nanmean([r2_v, r2_h, sign_ok]))
            if sub:
                checks["projectile"] = float(np.nanmean(sub))
                if checks["projectile"] < 0.6:
                    v.append(f"③抛体: 抛物线/匀速/向下加速拟合度={checks['projectile']:.2f}")

        # ④ 斜面下滑: a_斜/a_落 = sinθ − μcosθ → 反解 μ应 ∈[0,1]
        if "incline_mu" in enabled:
            mus = []
            for i in range(N):
                a_vec = np.nanmean(acc[i], axis=0)
                a_mag = float(np.linalg.norm(a_vec))
                if a_mag < 1e-8:
                    continue
                # 加速度方向与重力方向夹角 → 斜面倾角 θ 的代理
                cos_a = float(np.clip(np.dot(a_vec / a_mag, g_dir), -1, 1))
                theta = np.arccos(cos_a)             # 运动方向偏离竖直的角
                s, c = np.sin(theta), np.cos(theta)
                if c < 1e-6:
                    continue
                # a_斜/a_落 比值 = ratio, 由 ratio = sinθ − μcosθ 反解 μ
                a_free = float(np.nanmax(np.abs(acc.reshape(-1, 3) @ g_dir))) + 1e-8
                ratio = a_mag / a_free
                mu = (s - ratio) / c
                if np.isfinite(mu):
                    mus.append(mu)
            if mus:
                mu_arr = np.array(mus)
                in_range = float(np.mean((mu_arr >= 0) & (mu_arr <= 1)))
                mu_cv = (float(np.std(mu_arr) / (abs(np.mean(mu_arr)) + 1e-8))
                         if len(mu_arr) >= 2 else 0.0)
                checks["incline_mu"] = float(in_range * np.exp(-mu_cv))
                if in_range < 0.8:
                    v.append(f"④斜面: 反解摩擦系数μ={np.mean(mu_arr):.2f} 越界[0,1]")

        # ⑤ 弹跳恢复: 相邻弹跳高度比 = e² ∈[0,1], 不能越弹越高
        if "restitution" in enabled:
            e_ok, e_all = [], []
            for i in range(N):
                peaks = _find_peaks(pos[i] @ up)
                if len(peaks) < 2:
                    continue
                ph = np.array([(pos[i] @ up)[t] for t in peaks])
                base = float(np.min(pos[i] @ up))
                amp = ph - base
                if np.any(amp[:-1] <= 1e-8):
                    continue
                ratios = amp[1:] / amp[:-1]
                e_vals = np.sqrt(np.clip(ratios, 0, None))
                e_all.extend(e_vals.tolist())
                e_ok.append(float(np.mean((ratios > 0) & (ratios <= 1.0))))
            if e_ok:
                checks["restitution"] = float(np.mean(e_ok))
                if checks["restitution"] < 0.9:
                    v.append(f"⑤弹跳: 存在越弹越高(e²>1), 合规比例={checks['restitution']:.2f}")

        # ⑥ 滑动减速: 应单调减速至停
        if "deceleration" in enabled and vel is not None and vel.shape[1] > 3:
            vmag = np.linalg.norm(vel, axis=-1)
            v_first = vmag[:, :max(1, vmag.shape[1] // 3)].mean(axis=1)
            v_last = vmag[:, -max(1, vmag.shape[1] // 3):].mean(axis=1)
            checks["deceleration"] = float((v_last <= v_first + 1e-6).mean())
            if checks["deceleration"] < 0.5:
                v.append(f"⑥滑动减速: 仅{checks['deceleration']:.0%}物体末速≤初速")

        # ⑥b 减速度恒定 (a = −μg 恒定)
        if "sliding_const_decel" in enabled and vel is not None and vel.shape[1] > 3:
            consts = []
            for i in range(N):
                vm = np.linalg.norm(vel[i], axis=-1)
                if len(vm) < 4 or vm.mean() < 1e-8:
                    continue
                consts.append(_poly_r2(np.arange(len(vm), dtype=float), vm, deg=1))
            if consts:
                checks["sliding_const_decel"] = float(np.nanmean(consts))
                if checks["sliding_const_decel"] < 0.6:
                    v.append(f"⑥b减速度恒定性 R²={checks['sliding_const_decel']:.2f}")

        # ⑧ 圆周运动: 松手后沿切线飞出 (非径向)
        if "circular_tangent" in enabled:
            tang = []
            for i in range(N):
                p, vv = pos[i], vel[i] if vel is not None else None
                if vv is None or len(vv) < 4:
                    continue
                center = p.mean(axis=0)
                # 末段速度方向 vs 径向方向: 切线飞出 → 两者近正交
                radial = p[-1] - center
                rn = np.linalg.norm(radial)
                vn = np.linalg.norm(vv[-1])
                if rn < 1e-8 or vn < 1e-8:
                    continue
                cos_rv = abs(float(np.dot(radial / rn, vv[-1] / vn)))
                tang.append(1.0 - cos_rv)   # 越接近正交越好
            if tang:
                checks["circular_tangent"] = float(np.nanmean(tang))
                if checks["circular_tangent"] < 0.5:
                    v.append(f"⑧圆周: 出射方向偏径向(切线度={checks['circular_tangent']:.2f})")

        # ⑨ 支撑/法向力: 落地后不应继续穿透下陷 (真实检验, 替代原硬编码 1.0)
        if "support" in enabled:
            sup = []
            for i in range(N):
                hgt = pos[i] @ up
                if len(hgt) < 4:
                    continue
                floor = float(np.percentile(hgt, 2))    # 地面高度代理
                span = float(np.ptp(hgt)) + 1e-8
                # 触地后是否持续下陷: 统计低于地面代理值的显著超出量
                sink = np.clip(floor - hgt, 0, None) / span
                sup.append(1.0 - float(np.mean(sink > 0.05)))
            if sup:
                checks["support"] = float(np.nanmean(sup))
                if checks["support"] < 0.9:
                    v.append(f"⑨支撑: 检出穿透/下陷帧, 合规度={checks['support']:.2f}")

        # 【度量深度近似数值】DA3 米制 → 验 a ≈ g
        measured_g = None
        if metric_scale is not None:
            a_check = self._check_gravity_magnitude(acc, g_dir, metric_scale)
            if a_check is not None:
                measured_g, a_ok = a_check
                checks["metric_g"] = 1.0 if a_ok else 0.0
                if not a_ok:
                    v.append(f"米制加速度={measured_g:.2f}m/s² 偏离g=9.8过多")

        if not checks:
            return EvalResult.skipped("R22-力的动态效应",
                                       f"场景{scene_type}下无可判定的力学检验项")

        avg = float(np.nanmean(list(checks.values())))
        score = int(np.clip(round(avg * 4 + 1), 1, 5))
        return EvalResult("R22-力的动态效应", score, v,
                          metrics={"law_checks": {k: round(float(x), 3)
                                                  for k, x in checks.items()},
                                   "gravity_dir": g_dir.tolist(),
                                   "scene_type": scene_type,
                                   "enabled_checks": enabled,
                                   "n_objects": int(N),
                                   "track_source": traj.source,
                                   "metric_scale": metric_scale,
                                   "measured_g_ms2": measured_g},
                          status="proxy")

    @staticmethod
    def _check_gravity_magnitude(acc, g_dir, metric_scale):
        """把归一化加速度换算成m/s² 并检验是否 ≈9.8。

        Returns (measured_g, is_ok) 或 None。容差: [0.5g, 2g] 视为量级正确。
        """
        if acc is None or acc.shape[1] == 0 or metric_scale is None:
            return None
        a_g = acc.reshape(-1, 3) @ g_dir
        a_g = a_g[np.isfinite(a_g)]
        if a_g.size < 3:
            return None
        measured = float(np.median(np.abs(a_g)) * metric_scale)
        return measured, bool(4.9 <= measured <= 19.6)


    @staticmethod
    def _extract_mass_ratio(prompt: str) -> tuple[float, float]:
        """从 prompt 提取两个碰撞物体的质量比 (m1, m2)。

        启发式: 根据 prompt 关键词判断质量关系。
        - "identical"/"same"/"equal" → 等质量 (1:1)
        - "large/heavy" vs "small/light" → 大:小 = 3:1 (粗略假设)
        - "different mass" → 默认 2:1 (未知比例, 粗略假设)
        - 其他 → 默认 1:1

        质量比用于 R23 动量守恒检验 m1*v1 + m2*v2 = m1*v1' + m2*v2'。
        2D bbox 方案无真实质量, 只能从 prompt 语义做启发式估计。
        """
        low = (prompt or "").lower()
        # 等质量关键词
        if any(w in low for w in ("identical", "same", "equal", "same diameter",
                                   "five-ball", "two identical")):
            return 1.0, 1.0
        # 大/小 或 重/轻
        has_large = any(w in low for w in ("large", "heavy", "big"))
        has_small = any(w in low for w in ("small", "light", "little"))
        if has_large and has_small:
            return 3.0, 1.0  # 大:小 = 3:1 (粗略)
        # different mass (比例未知)
        if "different" in low and "mass" in low:
            return 2.0, 1.0  # 默认 2:1
        # 默认等质量
        return 1.0, 1.0

    # ── R23 动量守恒 (逐帧单帧分割方案: 绕过 SAM3 视频追踪对小目标的失败) ──
    def _eval_momentum_2d(self, sam3, frames, concepts=None,
                          prompt: str = "") -> Optional[EvalResult]:
        """逐帧单帧分割检测 2D 碰撞。

        根因链: SAM3 视频追踪(segment_video)对「面积 0.6% 的小冰球」完全追踪不了
        (有效物体帧=0, 无论 box label/大小), 但单帧分割(segment_frame)能稳定检测到
        冰球(可视化确认 puck conf=0.39 area=0.005)。故改用「逐帧单帧分割」:
        对均匀采样的 ~40 帧分别 segment_frame, 每帧取面积最大的 2 个中等面积
        前景物体(两个冰球), 构建 2D 中心轨迹 → 检测「接近-碰撞-弹开」。

        Returns:
            EvalResult 或 None(2D 方案不可用, 由调用方回退 3D 方案)。
        """
        try:
            if concepts is None:
                from models.sam3 import DIMENSION_CONCEPTS
                r23_words = DIMENSION_CONCEPTS.get("R23", [])
                concepts_2d = list(dict.fromkeys(
                    r23_words + ["object", "thing", "disk", "circle",
                                 "target", "white disk", "round object"]))
            else:
                concepts_2d = concepts
            H, W = frames[0].shape[:2] if hasattr(frames[0], "shape") else (480, 832)
            diag = float(np.sqrt(H * H + W * W))
            # 冰球面积 ~0.5%, 过滤区间 0.2%~50% (排除碎片和背景气桌)
            min_area_frac = 0.002
            max_area_frac = 0.50

            # 用全部帧做单帧分割 (碰撞发生在一瞬间, 40 帧采样间隔约 0.25s 会漏掉
            #   碰撞拐点; 全帧 243 帧间隔约 0.04s 能精确捕获「接近-弹开」的瞬时变化)
            T_total = len(frames)
            sample_idx = np.arange(T_total, dtype=int)  # 全部帧

            # 每帧: 检测到的中等面积物体 → 取面积最大的 2 个作为候选冰球
            frame_centers = []  # 每帧 [最多2个物体] 的 2D 中心 (归一化)
            frame_valid_frames = 0
            for idx in sample_idx:
                try:
                    seg = sam3.segment_frame(frames[idx], dimension="R23",
                                             concepts=concepts_2d)
                except Exception:
                    continue
                if seg.masks is None or len(seg.masks) == 0:
                    continue
                boxes = sam3.get_bounding_boxes(seg.masks)  # [N,4] xyxy 像素
                if boxes.ndim == 1:
                    boxes = boxes.reshape(1, -1)
                if boxes.ndim != 2:
                    continue
                # 过滤中等面积物体, 优先选「冰球类概念词」的物体。
                # ⚠️ 不能只取「面积最大的 2 个」: 中等面积区间里可能混入杂物
                #    (气桌标记/手/碎片), 面积最大的未必是冰球。
                #    冰球类词: puck/ball/disk/circle/sphere/marble 等 (冰球视觉上是圆盘)。
                #    两个冰球大小相同 → 面积接近, 故配对「面积最接近的 2 个冰球类物体」。
                puck_words = ("puck", "ball", "disk", "circle", "sphere",
                              "marble", "round", "target", "white")
                cands = []  # (area, cx, cy, is_puck, name)
                for i, b in enumerate(boxes):
                    x0, y0, x1, y1 = b
                    w_norm = (x1 - x0) / W
                    h_norm = (y1 - y0) / H
                    area = w_norm * h_norm
                    if not (min_area_frac <= area <= max_area_frac):
                        continue
                    cx = ((x0 + x1) / 2) / diag
                    cy = ((y0 + y1) / 2) / diag
                    name = ""
                    if seg.class_names and i < len(seg.class_names):
                        name = str(seg.class_names[i]).lower()
                    is_puck = any(w in name for w in puck_words)
                    cands.append((area, cx, cy, is_puck, name))
                if not cands:
                    frame_centers.append([])
                    continue
                # 优先冰球类词; 若冰球类 ≥2, 取面积最接近的 2 个; 否则退回取面积最大的 2 个
                puck_cands = [c for c in cands if c[3]]
                if len(puck_cands) >= 2:
                    pick = puck_cands
                else:
                    pick = cands
                pick.sort(key=lambda x: x[0], reverse=True)
                if len(pick) >= 2:
                    # 取面积最接近的一对 (两个相同冰球)
                    best_pair = None
                    best_diff = np.inf
                    for a in range(len(pick)):
                        for b in range(a + 1, len(pick)):
                            diff = abs(pick[a][0] - pick[b][0])
                            if diff < best_diff:
                                best_diff = diff
                                best_pair = (a, b)
                    a, b = best_pair
                    frame_centers.append([(pick[a][1], pick[a][2]),
                                          (pick[b][1], pick[b][2])])
                    frame_valid_frames += 1
                elif len(pick) == 1:
                    frame_centers.append([(pick[0][1], pick[0][2])])
                else:
                    frame_centers.append([])

            if frame_valid_frames < 4:
                return None  # 能检测到 ≥2 个物体的帧太少, 回退 3D

            # 构建两个冰球的 2D 轨迹 (每个采样帧取前 2 个物体的中心)
            # 冰球 A 和 B 的区分: 用首帧两个物体中心, 后续帧按「最近邻」匹配
            traj_a = []
            traj_b = []
            # 首帧的两个物体中心作为初始
            first_pair = None
            for fc in frame_centers:
                if len(fc) >= 2:
                    first_pair = fc[:2]
                    break
            if first_pair is None:
                return None
            prev_a, prev_b = first_pair[0], first_pair[1]
            traj_a.append(prev_a)
            traj_b.append(prev_b)
            for fc in frame_centers[1:]:
                if len(fc) == 0:
                    # 该帧无检测, 延续上一帧位置 (线性保持)
                    traj_a.append(prev_a)
                    traj_b.append(prev_b)
                elif len(fc) == 1:
                    # 只有 1 个物体, 匹配最近的
                    d_a = (fc[0][0] - prev_a[0]) ** 2 + (fc[0][1] - prev_a[1]) ** 2
                    d_b = (fc[0][0] - prev_b[0]) ** 2 + (fc[0][1] - prev_b[1]) ** 2
                    if d_a <= d_b:
                        prev_a = fc[0]
                        traj_a.append(prev_a)
                        traj_b.append(prev_b)
                    else:
                        prev_b = fc[0]
                        traj_a.append(prev_a)
                        traj_b.append(prev_b)
                else:
                    # 2 个物体: 按最近邻匹配
                    c0, c1 = fc[0], fc[1]
                    d00 = (c0[0] - prev_a[0]) ** 2 + (c0[1] - prev_a[1]) ** 2
                    d01 = (c1[0] - prev_a[0]) ** 2 + (c1[1] - prev_a[1]) ** 2
                    if d00 <= d01:
                        prev_a, prev_b = c0, c1
                    else:
                        prev_a, prev_b = c1, c0
                    traj_a.append(prev_a)
                    traj_b.append(prev_b)

            traj_a = np.array(traj_a)  # [M,2]
            traj_b = np.array(traj_b)
            M = len(traj_a)

            # 两个冰球的 2D 距离随时间变化
            d = np.linalg.norm(traj_a - traj_b, axis=-1)  # [M]
            finite = np.isfinite(d)
            d_max = float(np.max(d[finite]))
            d_min = float(np.min(d[finite]))
            d_range = float(np.ptp(d[finite]))
            if d_range < 1e-6:
                return None  # 距离几乎不变, 无碰撞
            # ⚠️ 碰撞判定: 不能只看「绝对距离 < 固定阈值」, 因为 T2V 生成的碰撞
            #    有「缝隙」(两个冰球最小距离 50px, 但不会真正零距离接触)。
            #    正确判据: 距离从「远」骤降到「近」——最小距离 < 峰值的 30%,
            #    即「相向运动到很近」的碰撞特征。
            best_t = -1
            best_min_d = np.inf
            best_approach = 0.0
            best_separate = 0.0
            if d_min < d_max * 0.30:
                # 碰撞帧 = 距离最小的帧 (argmin)。
                # ⚠️ 不能用「单帧骤降(approach 大)」判定: 冰球是「缓慢相向运动」
                #    (243帧里 90帧从 422px 降到 50px, 每帧仅降 ~4px), 单帧骤降幅度小。
                #    碰撞的本质是「距离降到很近(峰值 30% 以下)」, 取 argmin 即碰撞帧。
                best_t = int(np.argmin(d))
                best_min_d = float(d[best_t])
                # 接近幅度: 从「距离开始下降」到「碰撞」的累计降幅
                # (取碰撞帧前 10 帧的平均距离作为「碰撞前距离」)
                pre_idx = max(0, best_t - 10)
                best_approach = float(np.mean(d[pre_idx:best_t + 1]) - best_min_d)
                best_approach = max(best_approach, d_range * 0.05)
                # 弹开幅度: 碰撞后 10 帧距离回升的幅度
                post_idx = min(M - 1, best_t + 10)
                best_separate = max(0.0, float(np.mean(d[best_t:post_idx + 1]) - best_min_d))

            if best_t < 0:
                print(f"[R23-2D] 逐帧分割检测到 {frame_valid_frames} 帧有≥2物体, "
                      f"但未检测到碰撞拐点 (min_d={d_min:.4f} "
                      f"max_d={d_max:.4f} ratio={d_min/max(d_max,1e-8):.3f})")
                return None  # 无碰撞拐点, 回退 3D

            # 检测到碰撞 → 检验「动量守恒」(R23 核心), 而非恢复系数 e(e 属 R24 能量守恒)。
            # 动量守恒: m₁v₁ + m₂v₂ = m₁v₁' + m₂v₂'。
            # 2D bbox 位置差分 → 速度(像素/帧)。碰撞前取速度 v₁,v₂, 碰撞后取 v₁',v₂'。
            # 质量比从 prompt 启发式提取(等质量→1:1, 大/小→3:1, different mass→2:1)。
            m1, m2 = self._extract_mass_ratio(prompt)
            # 碰撞前速度: 不能只取碰撞前 3 帧(冰球已减速接近停止, 速度≈0)。
            # 改为「距离开始明显下降」的帧到碰撞帧 (覆盖冰球真正相向运动的阶段)。
            # 从碰撞帧往前找「距离仍 >= 峰值 80%」的帧 (冰球还在两端、距离大)，
            # 该帧到碰撞帧之间就是「相向运动」阶段，速度采样取这一段。
            pre_start = 0
            for _t in range(best_t, 0, -1):
                if d[_t] >= d_max * 0.80:
                    pre_start = _t
                    break
            pre_end = best_t
            # 确保采样窗口足够长(至少 5 帧), 否则速度估计噪声大
            if best_t - pre_start < 5:
                pre_start = max(0, best_t - 5)
            # 碰撞后速度: 碰撞帧到「距离回升到稳定」的帧 (碰撞后弹开阶段)。
            # 若碰撞后几乎不弹开(碰一下就停), 则取碰撞后 10 帧。
            post_start = best_t
            post_end = min(M - 1, best_t + 10)
            # 速度 = 位置差分 (像素/帧)
            def _vel(traj, a, b):
                if b - a < 1:
                    return np.zeros(2)
                return (traj[b] - traj[a]) / (b - a)
            v1_pre = _vel(traj_a, pre_start, pre_end)
            v2_pre = _vel(traj_b, pre_start, pre_end)
            v1_post = _vel(traj_a, post_start, post_end)
            v2_post = _vel(traj_b, post_start, post_end)
            # 总动量(碰撞前/后), 归一化残差
            p_pre = m1 * v1_pre + m2 * v2_pre
            p_post = m1 * v1_post + m2 * v2_post
            p_residual = float(np.linalg.norm(p_post - p_pre))
            # ⚠️ 归一化分母不能用「总动量模长」: 等速相向碰撞时总动量≈0(两动量抵消),
            #    除以≈0 的数会残差爆炸(0/0型)。改用「各自动量绝对值之和」:
            #    denominator = |m₁v₁| + |m₂v₂| + |m₁v₁'| + |m₂v₂'|
            #    该值在相向碰撞时非零, 是稳定的量纲基准。
            p_mag_pre = (m1 * np.linalg.norm(v1_pre) + m2 * np.linalg.norm(v2_pre))
            p_mag_post = (m1 * np.linalg.norm(v1_post) + m2 * np.linalg.norm(v2_post))
            p_denom = float(p_mag_pre + p_mag_post)
            p_pre_norm = float(np.linalg.norm(p_pre))
            # 归一化: 残差 / 各自动量绝对值之和 (无量纲, 相向碰撞时稳定)
            mom_residual_norm = p_residual / max(p_denom, 1e-8)
            # 动量守恒残差小 → 高分; 残差大 → 低分 (动量不守恒)
            # 阈值: 残差 < 10% 优秀(5分), <25% 良好(4分), <50% 一般(3分),
            #       <100% 差(2分), ≥100% 严重不守恒(1分)
            if mom_residual_norm < 0.10:
                base_score = 5
            elif mom_residual_norm < 0.25:
                base_score = 4
            elif mom_residual_norm < 0.50:
                base_score = 3
            elif mom_residual_norm < 1.0:
                base_score = 2
            else:
                base_score = 1

            # ── 辅助权重 (0.5~1): 碰撞真实性 + 恢复系数质量 ──
            # 1) 碰撞真实性: 距离骤降幅度 ratio = d_min/d_max (越小碰撞越真实明显)
            ratio = d_min / max(d_max, 1e-8)
            if ratio < 0.15:
                w_real = 1.0
            elif ratio < 0.30:
                w_real = 0.9
            elif ratio < 0.50:
                w_real = 0.8
            else:
                w_real = 0.7
            # 2) 恢复系数质量: e = 弹开幅度 / 接近幅度 (弹性碰撞 e≈1 好;
            #    但非弹性碰撞 e 小在物理上也是"合理"的, 只是能量损失, 属 R24 范畴。
            #    这里只把 e 作为「碰撞是否有明确弹开」的辅助, 不重扣分)。
            e = best_separate / max(best_approach, 1e-8)
            if e >= 0.5:
                w_e = 1.0
            elif e >= 0.2:
                w_e = 0.9
            elif e >= 0.1:
                w_e = 0.8
            else:
                w_e = 0.7  # 几乎无弹开(碰一下就停), 碰撞质量差
            weight = w_real * w_e
            weight = max(0.5, min(1.0, weight))

            # 最终分 = 动量守恒基础分 × 权重 (四舍五入到整数, 下限 1)
            final_score = max(1, int(round(base_score * weight)))

            violations = []
            if base_score < 4:
                violations.append(
                    f"碰撞前后动量不守恒: 归一化残差={mom_residual_norm:.2f} "
                    f"(|p_pre|={p_pre_norm:.3f}, |Δp|={p_residual:.3f}, "
                    f"分母={p_denom:.3f})")
            if weight < 0.9:
                violations.append(
                    f"碰撞辅助权重={weight:.2f}偏低 (ratio={ratio:.2f}, e={e:.2f})")
            print(f"[R23-2D] 检测到碰撞@帧{best_t}: 动量残差={mom_residual_norm:.3f} "
                  f"(|p_pre|={p_pre_norm:.3f}, |Δp|={p_residual:.3f}, denom={p_denom:.3f}) "
                  f"→ 基础分={base_score}, 权重={weight:.2f} → 最终分={final_score}")
            # 碰撞帧对应的原始视频帧索引
            orig_collision_frame = int(sample_idx[best_t]) if best_t < len(sample_idx) else best_t
            return EvalResult(
                "R23-动量守恒", final_score, violations,
                metrics={
                    "method": "sam3_per_frame_momentum",
                    "collision_frame": orig_collision_frame,
                    "min_distance_norm": round(best_min_d, 4),
                    "momentum_residual_norm": round(mom_residual_norm, 4),
                    "p_pre_norm": round(p_pre_norm, 4),
                    "p_post_diff_norm": round(p_residual, 4),
                    "momentum_denominator": round(p_denom, 4),
                    "base_score": base_score,
                    "collision_weight": round(weight, 3),
                    "collision_ratio": round(ratio, 3),
                    "restitution_coef": round(e, 3),
                    "mass_ratio_m2_over_m1": m2 / m1,
                    "n_sampled_frames": M,
                    "n_frames_with_2obj": frame_valid_frames,
                },
                status="proxy")
        except Exception as e:
            print(f"[R23-2D] 逐帧分割失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ── R23 动量守恒 ──
    def _eval_momentum(self, traj, gt, prompt: str = "") -> EvalResult:
        vel = traj.velocities
        pos = traj.trajectories
        if vel is None or vel.shape[1] < 2:
            return EvalResult.skipped("R23-动量守恒", "速度数据不足")
        N = vel.shape[0]
        if N < 2:
            # 现在 N 是**真实物体数**(SAM3聚合), 该保护有效:
            # 单物体场景无从检验 Σmᵢvᵢ = Σmᵢvᵢ′
            # prompt 声明了碰撞/多物体但只追踪到 1 个物体 → 模型未满足前提
            return self.unobservable_if_declared(
                prompt, ["collide", "collision", "impact", "hit", "bounce",
                         "two", "both", "cradle", "puck", "marble", "碰撞", "对碰"],
                "R23-动量守恒",
                f"有效物体数={N}(<2), 无碰撞可检验 (track_source={traj.source})")

        # GT 路径
        if gt is not None and getattr(gt, "velocities", None) is not None:
            gv = gt.velocities
            tlen = min(vel.shape[1], gv.shape[1])
            errs = [np.mean(np.linalg.norm(vel[i, :tlen] - gv[i, :tlen], axis=-1))
                    for i in range(min(N, gv.shape[0]))]
            mean_err = float(np.mean(errs)) if errs else 1.0
            score = score_metric("R23_mom_err", mean_err)
            return EvalResult("R23-动量守恒", score,
                              [] if score >= 4 else [f"GT动量误差={mean_err:.3f}"],
                              metrics={"gt_momentum_error": mean_err}, status="ok")

        # 无GT: 遍历所有物体对, 用距离突降 + 局部极小值 + 阈值定位碰撞帧。
        # ⚠️ T2V 模型的"碰撞"往往是「快速接近 → 最近点(有微小缝隙) → 弹开」,
        #    真实零距离接触很难生成。故碰撞判定放宽为「距离先显著下降、再显著
        #    上升」的接近-弹开趋势, 阈值基于该物体对自身距离序列(而非全局中位数),
        #    容忍缝隙碰撞。
        Tsteps = vel.shape[1]
        best_pair = None
        best_collision_t = -1
        best_min_d = np.inf
        best_pair_info = None

        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j], axis=-1)  # [T]
                if Tsteps < 4:
                    continue
                # 该物体对自身距离序列的统计 (而非全局中位数, 避免被远距离帧拉高)
                d_median = float(np.median(d))
                d_range = float(np.ptp(d))  # 距离跨度
                if d_range < 1e-8:
                    continue
                d_min = float(np.min(d))
                # 碰撞阈值: 最近距离需 < 该对距离中位数的 50%, 且需有显著接近
                # (T2V 缝隙碰撞的最近点通常远小于其平均/中位距离)。
                collision_thresh = d_median * 0.5
                # 找局部极小值: 距离下降后再上升 (接近-弹开拐点)
                for t in range(1, Tsteps - 1):
                    if d[t] <= d[t - 1] and d[t] <= d[t + 1] and d[t] < collision_thresh:
                        # 排除端点: 碰前碰后都需至少 1 帧
                        if t < 1 or t > Tsteps - 2:
                            continue
                        # 要求「接近」和「弹开」幅度显著 (相对该对距离跨度)
                        approach = float(d[t - 1] - d[t]) if t >= 1 else 0.0
                        separate = float(d[t + 1] - d[t]) if t + 1 < Tsteps else 0.0
                        if approach < d_range * 0.02 or separate < d_range * 0.02:
                            continue  # 接近/弹开不显著, 不是碰撞
                        if d[t] < best_min_d:
                            best_min_d = d[t]
                            best_collision_t = t
                            best_pair = (i, j)
                            # 碰前/碰后各取 min(3, available) 帧做速度平均
                            pre_end = t
                            pre_start = max(0, pre_end - 3)
                            post_start = t
                            post_end = min(Tsteps, post_start + 3)
                            v_pre_i = vel[i, pre_start:pre_end, :].mean(axis=0)
                            v_pre_j = vel[j, pre_start:pre_end, :].mean(axis=0)
                            v_post_i = vel[i, post_start:post_end, :].mean(axis=0)
                            v_post_j = vel[j, post_start:post_end, :].mean(axis=0)
                            rel_pre = v_pre_i - v_pre_j
                            rel_post = v_post_i - v_post_j
                            collision_axis = pos[j, t] - pos[i, t]
                            collision_axis /= np.linalg.norm(collision_axis) + 1e-8
                            closing = float(np.dot(rel_pre, collision_axis))
                            separating = float(np.dot(rel_post, collision_axis))
                            e = abs(separating) / max(abs(closing), 1e-8)

                            # 由 m1 Δv1 + m2 Δv2 = 0 反解正质量比 m2/m1。
                            dv1 = v_post_i - v_pre_i
                            dv2 = v_post_j - v_pre_j
                            mass_ratio = -float(np.dot(dv1, dv2)) / max(float(np.dot(dv2, dv2)), 1e-8)
                            valid_mass_ratio = np.isfinite(mass_ratio) and mass_ratio > 0
                            m1, m2 = 1.0, mass_ratio if valid_mass_ratio else 1.0
                            momentum_residual = float(np.linalg.norm(m1 * dv1 + m2 * dv2))
                            momentum_scale = (float(np.linalg.norm(m1 * v_pre_i) +
                                                    np.linalg.norm(m2 * v_pre_j)) + 1e-8)
                            momentum_residual /= momentum_scale
                            ke_pre = 0.5 * (m1 * np.sum(v_pre_i ** 2) + m2 * np.sum(v_pre_j ** 2))
                            ke_post = 0.5 * (m1 * np.sum(v_post_i ** 2) + m2 * np.sum(v_post_j ** 2))
                            best_pair_info = (e, ke_pre, ke_post, int(i), int(j),
                                              mass_ratio, momentum_residual, valid_mass_ratio)

        if best_pair is None:
            # prompt 声明了碰撞(collide/impact/cradle/碰撞等)但未检测到 → 模型没生成碰撞
            return self.unobservable_if_declared(
                prompt, ["collide", "collision", "impact", "hit", "bounce",
                         "crash", "strike", "cradle", "swing",
                         "碰撞", "对碰", "相撞", "弹开"],
                "R23-动量守恒",
                "未检测到碰撞（无物体对距离<阈值且呈局部极小）")

        (e, ke_pre, ke_post, pi, pj, mass_ratio,
         momentum_residual, valid_mass_ratio) = best_pair_info
        v = []
        if not valid_mass_ratio:
            v.append(f"反解质量比无正解(m2/m1={mass_ratio:.3f})")
        if momentum_residual > 0.15:
            v.append(f"归一化动量残差={momentum_residual:.3f}")
        if e > 1.05:
            v.append(f"恢复系数e={e:.2f}>1 (碰后法向相对速率增大)")
        if ke_post > ke_pre * 1.1:
            v.append(f"碰撞后动能增加{ke_post/max(ke_pre,1e-8):.2f}x")
        score = 5 if not v else max(1, 5 - len(v))
        names = getattr(traj, "object_names", []) or []
        pair_names = [names[pi] if pi < len(names) else f"obj{pi}",
                      names[pj] if pj < len(names) else f"obj{pj}"]
        return EvalResult("R23-动量守恒", score, v,
                          metrics={"restitution_coef": float(e),
                                   "mass_ratio_m2_over_m1": float(mass_ratio),
                                   "momentum_residual_norm": momentum_residual,
                                   "kinetic_energy_ratio": float(ke_post / max(ke_pre, 1e-8)),
                                   "collision_frame": best_collision_t,
                                   "min_distance": float(best_min_d),
                                   "pair": [pi, pj],
                                   "pair_names": pair_names,
                                   "n_objects": int(N),
                                   "track_source": traj.source,
                                   "collision_threshold": float(collision_thresh)},
                          status="proxy")

    # ── R24 能量守恒 v2: 增加弹跳高度合理性分析 ──
    # v2: 不仅检查能量是否自发上升, 还检查弹跳模式的物理合理性:
    #      - 逐次弹跳高度比值应接近常数 (同一恢复系数)
    #      - 弹跳次数与能量衰减速率匹配 (不会2-3次就耗尽)
    #      - 每次弹跳后高度严格递减 (不应出现反弹更高的情况)
    def _eval_energy(self, traj, g_dir, gt, metric_scale=None) -> EvalResult:
        pos = traj.trajectories
        vel = traj.velocities
        if vel is None or vel.shape[1] < 2:
            return EvalResult.skipped("R24-能量守恒", "速度数据不足")
        N = pos.shape[0]

        # 高度 = 沿"反重力方向"的投影 (h 越大势能越高)
        up = -g_dir
        h = pos @ up  # [N,T]

        # ── 量纲一致化──
        # KE=½v² 与 PE=gh 只有在同一长度单位下才能相加。
        # 有 DA3 米制尺度 → 换算到米, PE 用 g=9.8;
        # 无尺度 → 用无量纲形式: 长度以轨迹幅度归一, g取「实测重力方向加速度中位数」,
        #   这样 KE 与 PE 处于同一(归一化)单位系,曲线形状/单调性判断仍然成立。
        if metric_scale is not None:
            pos_m = pos * metric_scale
            vel_m = vel * metric_scale
            h_m = pos_m @ up
            g_mag = 9.8
            unit = "SI(m, m/s,9.8)"
        else:
            acc = traj.accelerations
            if acc is not None and acc.shape[1] > 0:
                a_g = (acc.reshape(-1, 3) @ g_dir)
                a_g = a_g[np.isfinite(a_g)]
                g_mag = float(np.median(np.abs(a_g))) if a_g.size >= 3 else 1.0
            else:
                g_mag = 1.0
            if not np.isfinite(g_mag) or g_mag <= 1e-8:
                g_mag = 1.0
            pos_m, vel_m, h_m = pos, vel, h
            unit = "normalized(g=measured)"

        def energy(v_series, h_series):
            vmag = np.linalg.norm(v_series, axis=-1)  # [N,T-1]
            KE = 0.5 * vmag ** 2
            PE = g_mag * h_series[:, :vmag.shape[1]]
            return KE + PE

        E = energy(vel_m, h_m)

        if gt is not None and getattr(gt, "velocities", None) is not None:
            gp, gv = gt.trajectories, gt.velocities
            gh = gp @ up
            gE = energy(gv, gh)
            errs = []
            for i in range(min(N, gE.shape[0])):
                tlen = min(E.shape[1], gE.shape[1])
                errs.append(np.mean(np.abs(E[i, :tlen] - gE[i, :tlen])))
            mean_err = float(np.mean(errs)) if errs else 1.0
            score = score_metric("R24_energy_err", mean_err)
            return EvalResult("R24-能量守恒", score,
                              [] if score >= 4 else [f"GT能量误差={mean_err:.3f}"],
                              metrics={"gt_energy_error": mean_err,
                                       "energy_unit": unit}, status="ok")

        # ── 无GT: 综合能量分析 ──
        v = []
        bounce_metrics = {}

        # ① 机械能不应自发上升
        energy_violations = 0
        for i in range(N):
            es = E[i]
            if len(es) < 3:
                continue
            e0 = es[:len(es) // 3].mean()
            e1 = es[-len(es) // 3:].mean()
            change = (e1 - e0) / (abs(e0) + 1e-8)
            if change > 0.2:
                v.append(f"物体{i}机械能自发增长{change:.1%}")
                energy_violations += 1

        # ② 弹跳峰值分析
        # 关键: 弹跳高度必须相对**地面**测量(振幅), 而非世界坐标绝对值。
        # 否则坐标存在大常数偏移时 ratios≈1 恒成立, 该检验失去判别力。
        total_bounce_issues = 0
        for i in range(N):
            hi = h_m[i]  # [T] 高度序列 (与能量同一单位系)
            if len(hi) < 5:
                continue
            peaks = _find_peaks(hi, rel_prominence=0.1)
            if len(peaks) < 2:
                continue

            floor = float(np.nanmin(hi))                      # 地面高度代理
            amp = np.array([float(hi[t]) - floor for t in peaks])  # 相对地面的弹跳高度
            if np.any(amp[:-1] <= 1e-8):
                continue

            # ②a: 逐次弹跳高度比 = e²
            ratios = amp[1:] / amp[:-1]
            e_vals = np.sqrt(np.clip(ratios, 0, None))

            # ②b: 高度比= e² 必须 ∈(0,1] — 不能越弹越高
            n_increase = int(np.sum(ratios > 1.0))
            if n_increase > 0:
                v.append(f"物体{i}: {n_increase}次弹跳高度反常增加(e²>1, 违反能量守恒)")
                total_bounce_issues += n_increase

            # 恢复系数应稳定 (同一材料对同一地面)
            if len(ratios) >= 2:
                ratio_cv = float(np.std(ratios) / (np.mean(ratios) + 1e-8))
                if ratio_cv > 0.5:
                    v.append(f"物体{i}: 弹跳恢复系数不稳定 (e²的CV={ratio_cv:.2f})")
                    total_bounce_issues += 1

            # 记录弹跳指标
            bounce_metrics[f"obj{i}_n_peaks"] = len(peaks)
            bounce_metrics[f"obj{i}_e2_mean"] = round(float(np.mean(ratios)), 3)
            bounce_metrics[f"obj{i}_e_mean"] = round(float(np.mean(e_vals)), 3)

        # ── 综合打分 ──
        total_v = len(v)
        score = 5 if total_v == 0 else max(1, 5 - total_v)

        metrics = {"n_objects": int(N), "up_dir": up.tolist(),
                   "energy_violations": energy_violations,
                   "bounce_issues": total_bounce_issues,
                   "energy_unit": unit,
                   "g_used": round(float(g_mag), 4),
                   "track_source": traj.source}
        metrics.update(bounce_metrics)

        return EvalResult("R24-能量守恒", score, v,
                          metrics=metrics, status="proxy")
