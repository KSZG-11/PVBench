"""
L3 物理层 — 静力学与光影组评测器
R12 稳定性 (VGGT + SAM3) + R13 光影空间 (ReasonX + StableNormal) + R18 光影时间 (ReasonX + VGGT)

评测思路 (来自物理量分类体系.xlsx D列):
- R12: VGGT深度→3D点云→RANSAC支撑面拟合→物体底部与支撑面接触检查
- R13: 光度残差 + 阴影位置残差 + 各物体反推光源方向方差(应一致)
- R18: 主光方向L(t)时间稳定性(相机运动补偿到世界坐标系)
"""
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult
from ..thresholds import score_metric, score_lower_better


def _com_in_support_polygon(com: np.ndarray, bottom_pts: np.ndarray,
                            plane_normal: np.ndarray, plane_d: float,
                            tol: float = 0.02) -> bool:
    """检查物体重心投影是否落在底部接触点的支撑多边形内。

    Args:
        com: [3] 物体质心 (世界/相机坐标系)
        bottom_pts: [N,3] 物体底部接触点
        plane_normal: [3] 支撑平面法向量
        plane_d: float 平面方程 normal·x + d = 0 中的 d
        tol: 凸包边界容差 (相对尺度)

    Returns:
        True 若 CoM 投影在支撑多边形内（稳定），False 否则（可能倾倒）
    """
    n_pts = len(bottom_pts)
    if n_pts < 3:
        return True  # 不足 3 个接触点，无法判定，默认不报

    # 将 CoM 和底部点投影到支撑平面
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
    com_proj = com - np.dot(com, n) * n + n * (-plane_d)  # 投影到平面
    pts_proj = bottom_pts - np.outer(np.dot(bottom_pts, n), n)  # [N,3]

    # 在支撑平面上建立 2D 坐标系
    # 选两个与 n 正交的基向量
    if abs(n[0]) < 0.9:
        e1 = np.cross(n, np.array([1, 0, 0], dtype=np.float64))
    else:
        e1 = np.cross(n, np.array([0, 1, 0], dtype=np.float64))
    e1 /= (np.linalg.norm(e1) + 1e-8)
    e2 = np.cross(n, e1)

    # 投影到 2D
    def to_2d(pts3d):
        x = np.dot(pts3d, e1)
        y = np.dot(pts3d, e2)
        return np.stack([x, y], axis=-1)

    pts_2d = to_2d(pts_proj)
    com_2d = to_2d(com_proj.reshape(1, 3))[0]

    # 去重 (避免凸包退化)
    # 按网格量化去重
    scale = float(np.std(pts_2d, axis=0).mean()) + 1e-8
    grid_size = max(scale * 0.05, 1e-6)
    pts_q = np.unique(np.round(pts_2d / grid_size).astype(int), axis=0)
    if len(pts_q) < 3:
        return True
    pts_u = pts_q.astype(np.float64) * grid_size

    # 计算凸包
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts_u)
    except Exception:
        return True  # 无法计算凸包，默认不报

    # 点在凸包内判断
    # 使用 hull.equations: 每个面 n·x + d <= 0 (对于内部点)
    # 对 CoM 检查每个面的半空间约束
    eqs = hull.equations  # [N_faces, 4] (nx, ny, d)
    com_h = np.append(com_2d, 1.0)  # 齐次坐标
    inside = np.all(np.dot(eqs, com_h) <= tol * scale)

    return bool(inside)


def _fit_plane_ransac(points, n_iter=100, inlier_thresh=0.05):
    """在3D点云上用RANSAC拟合主平面(支撑面)。
    
    Args:
        points: [N, 3] 3D点云
        n_iter: RANSAC迭代次数
        inlier_thresh: 内点距离阈值(相对于点云尺度归一化)
    
    Returns:
        (normal, d, inlier_mask): 平面方程 normal·x + d = 0; 内点布尔掩码
    """
    N = len(points)
    if N < 3:
        return None, None, None
    scale = float(np.std(points, axis=0).mean()) + 1e-8
    thresh = inlier_thresh * scale
    best_inliers = np.array([], dtype=int)
    best_normal = None
    best_d = 0.0
    for _ in range(n_iter):
        idx = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -np.dot(normal, p0)
        dists = np.abs(points @ normal + d)
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal.copy()
            best_d = d
    if best_normal is None or len(best_inliers) < 10:
        return None, None, None
    # 确保法向量朝上(指向camera的反方向，即world-y或Z负方向取决于坐标系)
    # VGGT相机坐标系: Z=深度(向前), Y=向下。支撑面法向量应大致沿Y方向。
    # 不做强制翻转，保持法向量方向
    inlier_mask = np.zeros(N, dtype=bool)
    inlier_mask[best_inliers] = True
    return best_normal, best_d, inlier_mask


class StaticsLightingEvaluator(BaseEvaluator):
    """静力学与光影组评测器（R12 / R13 / R18）"""

    def __init__(self, vggt, sam3, reasonx, stable_normal, config: Optional[dict] = None):
        super().__init__("L3-Physics/StaticsLighting", config)
        self.vggt = vggt
        self.sam3 = sam3
        self.reasonx = reasonx
        self.stable_normal = stable_normal

    def evaluate(self, frames: list[Image.Image],
                 stability_gt: Optional[bool] = None,
                 full_frames: Optional[list[Image.Image]] = None,
                 target_dimensions: Optional[list[str]] = None,
                 prompt: str = ""
                 ) -> list[EvalResult]:
        frame_arrays = np.stack([np.array(f) for f in frames])
        key = frame_arrays[len(frames) // 2]
        # 维度级按需执行
        want = set(target_dimensions) if target_dimensions else None
        def _need(prefix: str) -> bool:
            return want is None or any(d.startswith(prefix) for d in want)
        # R13/R18 用全帧获得更完整的光影时序覆盖 (若提供)
        r20_input = np.stack([np.array(f) for f in full_frames]) if full_frames else frame_arrays
        results = []
        if _need("R12"):
            results.append(self._eval_stability(key, frame_arrays, stability_gt,
                                                prompt=prompt))
        if _need("R13"):
            results.append(self._eval_spatial_lighting(r20_input, prompt=prompt))
        if _need("R18"):
            results.append(self._eval_temporal_lighting(r20_input))
        return results

    # ── R12 稳定性: 3D支撑面分析 ──
    def _eval_stability(self, key_frame, frames, stability_gt,
                        prompt: str = "") -> EvalResult:
        try:
            seg = self.sam3.segment_frame(key_frame, dimension="R12",
                                          prompt=prompt)
        except Exception as e:
            return EvalResult.error("R12-稳定性", f"SAM3 推理失败: {e}")

        masks = seg.masks
        N = len(masks) if masks is not None else 0
        if N == 0:
            return EvalResult.skipped("R12-稳定性", "未检测到物体")

        # 获取 VGGT 深度图 + 内参用于3D反投影
        try:
            depth_raw, conf, extr, intr = self.vggt._forward(key_frame)
            depth_map = depth_raw[0] if depth_raw.ndim == 3 else depth_raw  # [H_d, W_d]
            K = intr[0]  # [3, 3]
        except Exception as e:
            return EvalResult.error("R12-稳定性", f"VGGT 深度推理失败: {e}")

        H_d, W_d = depth_map.shape
        H_orig, W_orig = masks[0].shape if N > 0 else (H_d, W_d)

        # 将 SAM3 掩码缩放到深度图分辨率
        import cv2
        masks_d = []
        for i in range(N):
            if H_orig != H_d or W_orig != W_d:
                m = cv2.resize(masks[i].astype(np.float32), (W_d, H_d),
                               interpolation=cv2.INTER_LINEAR)
            else:
                m = masks[i].astype(np.float32)
            masks_d.append(m > 0.5)

        # 深度反投影 → 场景3D点云 (相机坐标系)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ys, xs = np.mgrid[0:H_d, 0:W_d]
        valid_depth = (np.isfinite(depth_map)) & (depth_map > 0)
        X_full = np.where(valid_depth, (xs - cx) * depth_map / max(fx, 1e-6), 0.0)
        Y_full = np.where(valid_depth, (ys - cy) * depth_map / max(fy, 1e-6), 0.0)
        Z_full = depth_map

        scene_mask = valid_depth.ravel()
        scene_pts = np.stack([
            X_full.ravel()[scene_mask],
            Y_full.ravel()[scene_mask],
            Z_full.ravel()[scene_mask]
        ], axis=-1)  # [M, 3]

        if len(scene_pts) < 100:
            return EvalResult.skipped("R12-稳定性", "有效3D点不足以拟合支撑面")

        # RANSAC 拟合支撑平面
        normal, d, plane_inliers = _fit_plane_ransac(scene_pts)
        if normal is None:
            return EvalResult.skipped("R12-稳定性", "RANSAC无法拟合支撑平面")

        # 支撑平面点集 (内点) 的深度统计
        plane_pts = scene_pts[plane_inliers]
        plane_z_mean = float(np.mean(plane_pts[:, 2]))
        plane_z_std = float(np.std(plane_pts[:, 2])) + 1e-8

        v = []
        obj_count = 0
        n_unstable_com = 0  # 重心超出支撑多边形的物体数
        for i in range(N):
            mask_i = masks_d[i]
            obj_valid = mask_i.ravel() & scene_mask
            n_obj = int(obj_valid.sum())
            if n_obj < 30:
                continue
            obj_count += 1

            # 物体3D点
            obj_X = X_full.ravel()[obj_valid]
            obj_Y = Y_full.ravel()[obj_valid]
            obj_Z = Z_full.ravel()[obj_valid]
            obj_pts = np.stack([obj_X, obj_Y, obj_Z], axis=-1)

            # 物体底部深度 (最近于支撑面的点 → Z大 / Y大)
            # VGGT相机坐标系: Z=深度, Y≈向下。支撑面上物体的底部应靠近平面
            # 取物体点中 Z 最大的 20% (最远离相机 = 最近支撑面/地面)
            z_cutoff = np.percentile(obj_Z, 80)
            bottom_mask = obj_Z >= z_cutoff
            bottom_z = obj_Z[bottom_mask]

            if len(bottom_z) < 5:
                continue

            # 检查底部是否接触支撑面: 底部深度应接近平面深度
            bottom_z_mean = float(np.mean(bottom_z))
            # 物体与支撑面的距离 (Z方向，归一化到场景尺度)
            obj_plane_dist = abs(bottom_z_mean - plane_z_mean) / (plane_z_std + 1e-8)

            # 物体质心
            com = np.mean(obj_pts, axis=0)
            # 底部到平面的有符号距离
            bottom_pts = obj_pts[bottom_mask]
            bottom_signed_dists = bottom_pts @ normal + d

            # 判据 1: 底部应靠近支撑面
            near_plane = obj_plane_dist < 3.0
            # 判据 2: 底部点应在平面附近
            bottom_on_plane = np.mean(np.abs(bottom_signed_dists)) < 0.10 * np.mean(np.abs(plane_pts @ normal + d))

            if not near_plane:
                v.append(f"物体{i}底部距支撑面{obj_plane_dist:.1f}σ(悬空/未接触)")
                continue  # 不接触支撑面的物体不检查 CoM
            elif not bottom_on_plane:
                bottom_std = float(np.std(bottom_signed_dists))
                bottom_scale = float(np.mean(np.abs(bottom_pts[:, 2]))) + 1e-8
                if bottom_std / bottom_scale > 0.15:
                    v.append(f"物体{i}底部与支撑面不平行(std={bottom_std/bottom_scale:.2f})")

            # ── 判据 3 (表格方案): 重心竖直投影是否落在支撑多边形内 ──
            # 将底部接触点和 CoM 投影到支撑平面上
            try:
                _check_com_in_support = _com_in_support_polygon(
                    com, bottom_pts, normal, d)
                if not _check_com_in_support:
                    n_unstable_com += 1
                    if n_unstable_com <= 5:
                        v.append(f"物体{i}重心投影超出支撑多边形(可能倾倒)")
            except Exception:
                pass  # 凸包计算失败不阻塞

        if stability_gt is True and v:
            score = max(1, 5 - 2 * len(v))
            v.insert(0, "[CI2V]参考图稳定但视频检测到不稳定")
            status = "ok"
        else:
            # 每发现一个 violation 扣 1 分, 重心超出支撑多边形额外加重扣分
            score = 5 if not v else max(1, 5 - len(v) - n_unstable_com)
            status = "proxy"
        return EvalResult("R12-稳定性", score, v,
                          metrics={"n_objects": N, "n_evaluated": obj_count,
                                   "n_unstable_com": n_unstable_com,
                                   "plane_inlier_ratio": float(plane_inliers.sum()) / max(len(scene_pts), 1)},
                          status=status)

    # ── R13 光影空间一致性: 各帧主光方向方差 + 亮度梯度交叉验证 + 阴影方向验证 ──
    # v3: 新增阴影投射方向一致性检查 (物体底部暗区方向应一致).
    def _eval_spatial_lighting(self, frames, prompt: str = "") -> EvalResult:
        try:
            normals = np.stack([n.normal_map for n in self.stable_normal.estimate_batch(frames)])
        except Exception as e:
            return EvalResult.error("R13-光影空间一致性", f"StableNormal 推理失败: {e}")
        try:
            shading = self.reasonx.analyze_spatial_lighting(frames, normals=normals)
        except Exception as e:
            return EvalResult.error("R13-光影空间一致性", f"ReasonX 光源估计失败: {e}")

        L = shading.light_directions  # [T,3]
        if L is None or len(L) < 1:
            return EvalResult.skipped("R13-光影空间一致性", "无光源方向输出")
        # 归一化后各帧光源方向应一致 → 方向方差小
        norms = np.linalg.norm(L, axis=-1, keepdims=True) + 1e-8
        Ln = L / norms
        # 用与均值方向的平均夹角衡量一致性
        mean_dir = Ln.mean(0)
        mean_dir /= (np.linalg.norm(mean_dir) + 1e-8)
        dots = np.clip(Ln @ mean_dir, -1, 1)
        spread_deg = float(np.degrees(np.mean(np.arccos(dots))))
        score = score_metric("R2x_light_angle", spread_deg)

        # ── 亮度梯度交叉验证 ──
        v = [] if score >= 4 else [f"光源方向离散={spread_deg:.1f}°"]
        brightness_contradiction = False
        light_x_sign = None

        mean_lx = float(mean_dir[0])
        mean_ly = float(mean_dir[1])
        if abs(mean_lx) > 0.15:
            light_x_sign = "right" if mean_lx > 0 else "left"

        if light_x_sign is not None:
            n_consistent = 0
            n_contradict = 0
            for frame in frames:
                if frame.ndim == 3:
                    if frame.shape[-1] == 3:
                        gray = 0.299 * frame[..., 0].astype(np.float64) + \
                               0.587 * frame[..., 1].astype(np.float64) + \
                               0.114 * frame[..., 2].astype(np.float64)
                    else:
                        gray = frame.astype(np.float64)
                else:
                    gray = frame.astype(np.float64)
                H, W = gray.shape[:2]
                mid = W // 2
                left_brightness = float(gray[:, :mid].mean())
                right_brightness = float(gray[:, mid:].mean())
                diff = right_brightness - left_brightness
                brightness_side = "right" if diff > 0 else "left"
                if brightness_side == light_x_sign:
                    n_consistent += 1
                elif abs(diff) > 2.0:
                    n_contradict += 1

            total_checked = n_consistent + n_contradict
            if total_checked > 0 and n_contradict > total_checked * 0.4:
                brightness_contradiction = True
                v.append(
                    f"亮度梯度与光源方向矛盾: "
                    f"光源来自{light_x_sign}侧, "
                    f"但{n_contradict}/{total_checked}帧亮度在相反侧更高")
                score = max(1, score - 1)

        # ── 阴影投射方向验证 (v3新增) ──
        # 对中间帧做 SAM3 分割 → 检查各物体底部阴影方向是否一致
        shadow_result = None
        try:
            key_idx = len(frames) // 2
            key_frame = frames[key_idx]
            seg = self.sam3.segment_frame(key_frame, dimension="R13",
                                          prompt=prompt)
            shadow_result = self._check_shadow_consistency(
                key_frame, seg, mean_dir
            )
        except Exception:
            pass

        if shadow_result is not None:
            shadow_consistency = shadow_result.get("shadow_consistency", 1.0)
            n_shadow_objs = shadow_result.get("n_objects_checked", 0)
            light_shadow_match = shadow_result.get("light_shadow_match", "unknown")

            if n_shadow_objs >= 2:
                if shadow_consistency < 0.5:
                    v.append(
                        f"阴影方向不一致: {n_shadow_objs}个物体中"
                        f"仅{shadow_result.get('n_agree', 0)}个阴影方向一致"
                        f"(一致性={shadow_consistency:.0%})")
                    score = max(1, score - 2)
                elif shadow_consistency < 0.75:
                    v.append(
                        f"阴影方向部分不一致(一致性={shadow_consistency:.0%})")
                    score = max(1, score - 1)

                # 阴影方向 vs ReasonX 光源方向对比
                if light_shadow_match == "contradicts":
                    v.append("阴影投射方向与ReasonX估计光源方向矛盾")
                    score = max(1, score - 1)

        # R13 光影空间一致性: I2V/T2V 下光源方向一致性来自 ReasonX + StableNormal
        # 二者均为模型推断而非外部仿真 GT → status=proxy（非 CI2V）
        status = "proxy"
        metrics = {
            "light_dir_spread_deg": round(spread_deg, 1),
            "light_x_sign": light_x_sign,
            "brightness_contradiction": brightness_contradiction,
        }
        if shadow_result is not None:
            metrics["shadow_consistency"] = shadow_result

        return EvalResult("R13-光影空间一致性", score, v,
                          metrics=metrics, status=status)

    @staticmethod
    def _check_shadow_consistency(frame: np.ndarray,
                                   seg, light_dir_cam: np.ndarray) -> dict:
        """检测各物体底部阴影投射方向是否一致。

        对每个 SAM3 分割物体:
        1. 找到物体底部中心 (mask最下行像素中心)
        2. 在物体下方地面区域扫描水平亮度分布
        3. 暗区质心 → 阴影投射方向 (相对物体底部)
        4. 检查多物体间阴影方向符号是否一致
        5. 与 ReasonX 估计光源方向对比

        Returns:
            dict 或 None (物体不足/掩码无效时)
        """
        masks = seg.masks
        if masks is None or len(masks) < 2:
            return None

        H, W = masks[0].shape[:2]
        if frame.ndim == 3 and frame.shape[-1] == 3:
            gray = (0.299 * frame[..., 0].astype(np.float64) +
                    0.587 * frame[..., 1].astype(np.float64) +
                    0.114 * frame[..., 2].astype(np.float64))
        else:
            gray = frame.astype(np.float64)

        shadow_dirs = []
        for i, mask in enumerate(masks):
            mask_bin = mask > 0.5
            if not mask_bin.any():
                continue

            rows, cols = np.where(mask_bin)
            bottom_row = rows.max()

            # 物体底部中心 (取最底行的列均值)
            bottom_cols = cols[rows == bottom_row]
            obj_bottom_x = float(np.mean(bottom_cols))

            # 定义地面搜索区域: 物体底部下方 5%~15% 图像高度
            ground_top = int(bottom_row + 1)
            ground_bottom = int(min(H, bottom_row + max(5, H * 0.12)))
            if ground_bottom <= ground_top:
                continue

            ground_strip = gray[ground_top:ground_bottom, :]  # [strip_H, W]
            col_brightness = ground_strip.mean(axis=0)  # [W]

            # 亮度归一化: 暗=高权重
            b_min, b_max = col_brightness.min(), col_brightness.max()
            if b_max - b_min < 0.5:
                continue  # 地面区域无显著明暗差异
            darkness = 1.0 - (col_brightness - b_min) / (b_max - b_min)

            # 阴影质心 = 暗度加权平均列
            if darkness.sum() > 1e-8:
                shadow_x = float(np.average(np.arange(W, dtype=np.float64),
                                            weights=darkness))
                shadow_dir_x = shadow_x - obj_bottom_x  # >0 → 阴影偏右(光源偏左)
                if abs(shadow_dir_x) > W * 0.005:  # 过滤微弱信号
                    shadow_dirs.append(shadow_dir_x)

        if len(shadow_dirs) < 2:
            return None

        shadow_dirs = np.array(shadow_dirs)
        signs = np.sign(shadow_dirs)
        n_positive = int((signs > 0).sum())
        n_negative = int((signs < 0).sum())
        total = n_positive + n_negative
        if total < 2:
            return None

        # 阴影方向一致性 = 多数方向占比
        consistency = max(n_positive, n_negative) / total

        # 与 ReasonX 光源方向对比
        light_x = float(light_dir_cam[0])
        light_shadow_match = "unknown"
        if abs(light_x) > 0.1:
            # light_x > 0 → 光源偏右 → 阴影偏左 → 期望 shadow_dir_x < 0
            expected_sign = -1 if light_x > 0 else 1
            dominant_sign = 1 if n_positive > n_negative else -1
            light_shadow_match = ("consistent" if expected_sign == dominant_sign
                                  else "contradicts")

        return {
            "shadow_consistency": round(consistency, 3),
            "n_objects_checked": total,
            "n_agree": max(n_positive, n_negative),
            "light_shadow_match": light_shadow_match,
            "shadow_dirs": [round(float(d), 1) for d in shadow_dirs],
        }

    # ── R18 光影时间一致性: L(t) 相邻帧变化 (相机运动补偿 + 静态相机噪声抑制) ──
    def _eval_temporal_lighting(self, frames) -> EvalResult:
        use_judge = getattr(self.reasonx, '_judge_available', False)
        # 始终计算 StableNormal 法线: Judge 可用时用于点对辐照度比较反演,
        # Judge 不可用时传给 _estimate_light_direction 使用 Lambertian LSQ
        # (远优于纯灰度梯度近似 z=1.0)
        normals = None
        try:
            normals = np.stack([n.normal_map for n in self.stable_normal.estimate_batch(frames)])
        except Exception as e:
            print(f"[R18] StableNormal 法线提取失败，降级经典光度学: {e}")
            use_judge = False  # Judge 需要法线做点对辐照度比较
        try:
            shading = self.reasonx.analyze_temporal_lighting(frames, normals=normals)
        except Exception as e:
            return EvalResult.error("R18-光影时间一致性", f"ReasonX 推理失败: {e}")
        L = shading.light_directions  # [T, 3] 相机坐标系下的光源方向
        if L is None or len(L) < 2:
            return EvalResult.skipped("R18-光影时间一致性", "帧数不足")
        norms = np.linalg.norm(L, axis=-1, keepdims=True) + 1e-8
        Ln_cam = L / norms  # 相机坐标系下的单位光源方向

        # 相机坐标系下的逐帧变化(原始指标)
        dots_cam = np.clip(np.sum(Ln_cam[1:] * Ln_cam[:-1], axis=-1), -1, 1)
        mean_change_cam = float(np.degrees(np.mean(np.arccos(dots_cam))))

        # ── 静态相机检测 ──
        # ReasonX 光源估计有 10-25° 固有噪声，静态相机下所有变化都是噪声
        is_static_camera = False
        cam_rotation_deg = 0.0
        try:
            cam_extr = self.vggt.estimate_camera_poses(frames)  # [T, 3, 4]
            if cam_extr is not None and cam_extr.ndim == 3 and cam_extr.shape[0] >= 2:
                T_cam = cam_extr.shape[0]
                # 检测相机旋转幅度: 相邻帧旋转角度的平均值
                rot_angles = []
                for t in range(1, T_cam):
                    R1 = cam_extr[t-1, :3, :3]
                    R2 = cam_extr[t, :3, :3]
                    # 相对旋转角度 = arccos((trace(R1@R2^T) - 1) / 2)
                    R_rel = R1 @ R2.T
                    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
                    angle = np.degrees(np.arccos((trace - 1.0) / 2.0))
                    rot_angles.append(angle)
                if rot_angles:
                    cam_rotation_deg = float(np.mean(rot_angles))
                    # 若平均帧间旋转 < 1.5° → 静态/三脚架相机
                    is_static_camera = cam_rotation_deg < 1.5
        except Exception:
            pass

        # 尝试用 VGGT 相机位姿补偿: 将光源方向变换到世界坐标系
        mean_change_world = None
        compensated = False
        if not is_static_camera:
            # 仅在非静态相机时才尝试世界坐标系补偿
            try:
                cam_extr = self.vggt.estimate_camera_poses(frames)
                if cam_extr is not None and cam_extr.ndim == 3 and cam_extr.shape[0] >= 2:
                    T_cam = cam_extr.shape[0]
                    Ln_world = []
                    for t in range(min(T_cam, len(Ln_cam))):
                        R_w2c = cam_extr[t, :3, :3]
                        R_c2w = R_w2c.T
                        L_w = R_c2w @ Ln_cam[t]
                        Ln_w = L_w / (np.linalg.norm(L_w) + 1e-8)
                        Ln_world.append(Ln_w)
                    if len(Ln_world) >= 2:
                        Ln_world = np.stack(Ln_world)
                        dots_world = np.clip(np.sum(Ln_world[1:] * Ln_world[:-1], axis=-1), -1, 1)
                        mean_change_world = float(np.degrees(np.mean(np.arccos(dots_world))))
                        compensated = True
            except Exception:
                pass

        # 选择更可信的指标
        if compensated and mean_change_world is not None:
            if mean_change_world < mean_change_cam * 1.2:
                mean_change = mean_change_world
            else:
                mean_change = mean_change_cam
                compensated = False
        else:
            mean_change = mean_change_cam

        # ── 静态相机噪声抑制 ──
        # 若相机静止但光源方向变化大 → 属 ReasonX 估计噪声，降权计分
        noise_adjusted = False
        if is_static_camera and mean_change > 20.0:
            # 噪声上限: 静态场景光源不应有 > 20° 物理变化
            # 将超出部分视为噪声，扣减噪声基线后再计分
            noise_baseline = 15.0  # ReasonX 静态场景典型噪声基线
            mean_change_adj = max(0, mean_change - noise_baseline)
            noise_adjusted = True
        else:
            mean_change_adj = mean_change

        # R18 光影时间一致性: I2V/T2V 下光源时序来自 ReasonX + VGGT 相机位姿
        # ReasonX VLM judge 是评测工具而非仿真 GT → status=proxy
        score = score_metric("R2x_light_angle", mean_change_adj)
        status = "proxy"
        metrics = {
            "mean_angle_change_deg": round(mean_change, 1),
            "cam_frame_change_deg": round(mean_change_cam, 1),
            "is_static_camera": is_static_camera,
            "cam_rotation_deg": round(cam_rotation_deg, 2),
        }
        if noise_adjusted:
            metrics["noise_adjusted"] = True
            metrics["noise_baseline_deg"] = noise_baseline
            metrics["adjusted_change_deg"] = round(mean_change_adj, 1)
        if compensated:
            metrics["world_frame_change_deg"] = round(mean_change_world, 1)
            metrics["camera_compensated"] = True

        return EvalResult("R18-光影时间一致性", score,
                          [] if score >= 4 else [f"光源方向逐帧变化={mean_change:.1f}°"],
                          metrics=metrics, status=status)
