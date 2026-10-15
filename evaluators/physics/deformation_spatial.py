"""
L3 物理层 — 形变空间组评测器
R05 形变状态 (VGGT + SpatialTracker V2) + R11 空间关系 (VGGT + SAM3)

评测思路 (来自物理量分类体系.xlsx D列):
- R05: 与初始帧模板配准→形变梯度/Green应变→刚体应变≈0/det(F)≈1; 无GT查刚体距离变化
- R11: 逐物体SDF→穿透体积/悬浮间隙/接触遮挡检测; 用掩码重叠 + 深度前后序近似
  v2: 多帧分析(首/中/尾)替代单帧, 检测跨帧重叠是否持续(穿模)vs瞬态(遮挡)。
"""
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult
from ..thresholds import score_metric


class DeformationSpatialEvaluator(BaseEvaluator):
    """形变空间组评测器（R05 / R11）"""

    def __init__(self, vggt, spatial_tracker, sam3, config: Optional[dict] = None,
                 vlm=None):
        super().__init__("L3-Physics/DeformationSpatial", config)
        self.vggt = vggt
        self.spatial_tracker = spatial_tracker
        self.sam3 = sam3
        self.vlm = vlm   # R11 可观测性二次判断 (模型没生成多物体 vs 检测器不行)

    def evaluate(self, frames: list[Image.Image],
                 gt_deformation: Optional[np.ndarray] = None,
                 target_dimensions: Optional[list[str]] = None,
                 prompt: str = ""
                 ) -> list[EvalResult]:
        frame_arrays = np.stack([np.array(f) for f in frames])
        # 维度级按需执行
        want = set(target_dimensions) if target_dimensions else None
        def _need(prefix: str) -> bool:
            return want is None or any(d.startswith(prefix) for d in want)
        results = []
        # R05: 隔离异常, 单维度失败不影响 R11
        if _need("R05"):
            try:
                results.append(self._eval_deformation(frame_arrays, gt_deformation))
            except Exception as e:
                results.append(EvalResult.error("R05-形变状态",
                                f"内部异常: {type(e).__name__}: {e}"))
        # R11: 隔离异常
        if _need("R11"):
            try:
                results.append(self._eval_spatial(frame_arrays, prompt=prompt))
            except Exception as e:
                results.append(EvalResult.error("R11-空间关系",
                                f"内部异常: {type(e).__name__}: {e}"))
        return results

    def _eval_deformation(self, frames, gt_deformation) -> EvalResult:
        T = len(frames)
        if T < 2:
            return EvalResult.skipped("R05-形变状态", "帧数<2")
        try:
            pc_list = [self.vggt.reconstruct_pointcloud(frames[t]).points for t in range(T)]
        except Exception as e:
            return EvalResult.error("R05-形变状态", f"VGGT 点云重建失败: {e}")

        try:
            from scipy.spatial import cKDTree
            rng = np.random.default_rng(0)

            # GT 路径: 与参考点云对比
            if gt_deformation is not None and len(gt_deformation) >= 10:
                ref = gt_deformation
                dists = []
                for t in range(T):
                    cur = pc_list[t]
                    if len(cur) < 10:
                        continue
                    n = min(500, len(cur), len(ref))
                    c = cur[rng.choice(len(cur), n, replace=False)]
                    r = ref[rng.choice(len(ref), n, replace=False)]
                    d1 = cKDTree(r).query(c, k=1)[0].mean()
                    d2 = cKDTree(c).query(r, k=1)[0].mean()
                    scale = np.std(ref, axis=0).mean() + 1e-8
                    dists.append((d1 + d2) / 2 / scale)
                if not dists:
                    return EvalResult.skipped("R05-形变状态", "点云数据不足")
                avg = float(np.mean(dists))
                score = score_metric("R05_deform", avg)
                return EvalResult("R05-形变状态", score,
                                  [] if score >= 4 else [f"Chamfer/scale={avg:.3f}"],
                                  metrics={"avg_chamfer_norm": avg}, status="ok")

            # ── 无GT: Procrustes 刚体对齐 → 应变张量代理 ──
            if len(pc_list) < 2 or len(pc_list[0]) < 10:
                return EvalResult.skipped("R05-形变状态", "点云数据不足")

            from scipy.spatial.distance import pdist

            # 取首帧点云作为参考模板 (去质心)
            base_raw = pc_list[0][:200].astype(np.float64)
            if len(base_raw) < 20:
                return EvalResult.skipped("R05-形变状态", "参考点云点数 < 20")
            base_center = base_raw.mean(axis=0)
            base_centered = base_raw - base_center

            changes = []
            deform_metrics = []  # 逐帧形变度量

            # ── 方法1: Procrustes对齐后的Chamfer距离 ──
            aligned_chamfer = []

            # ── 方法2: 应变张量代理 (对齐后的pairwise距离变化) ──
            d0 = pdist(base_centered)  # 参考模板的pairwise距离矩阵
            d0_mean = np.mean(d0) + 1e-8

            for t in range(0, T):
                cur_raw = pc_list[t][:200].astype(np.float64)
                if len(cur_raw) < 20:
                    continue
                cur_center = cur_raw.mean(axis=0)
                cur_centered = cur_raw - cur_center

                # Procrustes: 把当前帧点云旋转对齐到首帧参考
                H = cur_centered[: len(base_centered)].T @ base_centered[: len(cur_centered)]
                U, _, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                # 对齐后的当前帧点云 (去除了刚体旋转)
                cur_aligned = cur_centered @ R.T

                # Chamfer距离 (Procrustes对齐后)
                if t > 0:
                    d1 = cKDTree(base_centered).query(cur_aligned, k=1)[0].mean()
                    d2 = cKDTree(cur_aligned).query(base_centered, k=1)[0].mean()
                    scale = np.std(base_centered, axis=0).mean() + 1e-8
                    aligned_avg = (d1 + d2) / 2 / scale
                    aligned_chamfer.append(aligned_avg)

                # 应变代理: 对齐后的pairwise距离变化率
                dt = pdist(cur_aligned)
                m = min(len(d0), len(dt))
                change = np.mean(np.abs(dt[:m] - d0[:m])) / d0_mean
                changes.append(change)
                deform_metrics.append({
                    "frame": t,
                    "strain_proxy": round(float(change), 4),
                })

            avg = float(np.mean(changes)) if changes else 0.0

            # ── 应变张量方法: 从SpatialTracker轨迹估计变形梯度 ──
            strain_gradient = None
            try:
                traj = self.spatial_tracker.track_3d(frames)
                tracks = traj.trajectories  # [N,T,3]
                if tracks.shape[1] >= 2:
                    # 对每个轨迹点, 用其邻域点估计局部变形梯度 F
                    # F = dX_t / dX_0, 格林-拉格朗日应变 E = 0.5*(F^T F - I)
                    N_tracks, T_tracks = tracks.shape[:2]
                    if N_tracks >= 4 and T_tracks >= 2:
                        P0 = tracks[:N_tracks, 0, :]  # 首帧位置
                        strain_per_t = []
                        for t in range(1, T_tracks):
                            Pt = tracks[:N_tracks, t, :]
                            # 去质心
                            c0 = P0 - P0.mean(axis=0)
                            ct = Pt - Pt.mean(axis=0)
                            # 最小二乘 F: solve c0 @ F^T ≈ ct  →  F = (c0^T c0)^{-1} c0^T ct
                            try:
                                G = np.linalg.lstsq(c0, ct, rcond=None)[0].T  # [3,3]
                                # 格林-拉格朗日应变
                                E = 0.5 * (G.T @ G - np.eye(3))
                                strain_norm = float(np.linalg.norm(E, ord="fro"))
                                strain_per_t.append(strain_norm)
                            except np.linalg.LinAlgError:
                                pass
                        if strain_per_t:
                            strain_gradient = float(np.mean(strain_per_t))
            except Exception:
                strain_gradient = None

            # ── 可变形物体检测 ──
            is_deformable = False
            if avg > 0.25 and len(changes) >= 2:
                ch = np.array(changes)
                n_up = int(np.sum(np.diff(ch) > 0.01))
                n_down = int(np.sum(np.diff(ch) < -0.01))
                n_total = n_up + n_down
                if n_total > 0:
                    dominance = abs(n_up - n_down) / n_total
                    if dominance > 0.5:
                        is_deformable = True

            # ── 综合打分 ──
            # ⚠️ R05 语义修正 (v2.5): R05 的 prompt 全部要求「物体发生形变」
            #    (钢球压黏土 / 挤压橡胶球 / 钳子弯折铝板)。因此正确判定方向是:
            #      检测到真实形变事件 → 高分; 未检测到(静态画面, 模型未完成任务) → 低分。
            #    旧实现用 R05_deform(lower_better, 形变越小越好), 导致静态画面(应变≈0)
            #     拿满分、正确形变反而低分 —— 评分方向完全颠倒。
            # 优先使用 strain_gradient (更物理), 其次 avg (Procrustes后的pdist变化)
            if strain_gradient is not None:
                # 应变张量 Frobenius 范数: 0=无应变(刚体), 0.01=轻微, 0.05=明显
                primary_metric = strain_gradient
                metric_name = "strain_frobenius"
            else:
                primary_metric = avg
                metric_name = "pdist_change_aligned"

            # ── 形变事件存在性判定 (噪声基线) ──
            # VGGT 点云重建噪声基线 ~0.15 (见 thresholds.py R05_deform 注释)。
            # 应变显著高于噪声基线 → 存在真实形变事件; 否则视频未生成 prompt 要求的形变。
            # strain_gradient(应变张量Frobenius范数)与 avg(pdist变化率)量纲不同:
            #   应变张量基线取 0.02 (0=刚体, 0.01轻微, 0.05明显);
            #   pdist变化率基线取 0.15 (与旧 R05_deform 噪声基线一致)。
            if strain_gradient is not None:
                has_deformation_event = float(strain_gradient) > 0.02
            else:
                has_deformation_event = float(avg) > 0.15

            if not has_deformation_event:
                # 视频未生成 prompt 要求的形变事件 (静态/刚体画面) → 模型未完成任务
                score = 1
                status = "proxy"
                violations = [f"未检测到形变事件(应变={primary_metric:.4f}<噪声基线), "
                              f"视频未生成 prompt 要求的形变"]
            else:
                # 存在形变事件 → 按「形变幅度合理性」评分 (中等形变最佳, 过度形变扣分)
                # 形变存在即视为完成任务, 至少 3 分起; 幅度合理区间给 5 分。
                if strain_gradient is not None:
                    # 应变张量: 0.02~0.05 轻微(5分), 0.05~0.10 中等(4分),
                    # 0.10~0.20 明显(3分), 0.20~0.40 剧烈(2分), >0.40 过度(1分)
                    if strain_gradient <= 0.05:
                        score = 5
                    elif strain_gradient <= 0.10:
                        score = 4
                    elif strain_gradient <= 0.20:
                        score = 3
                    elif strain_gradient <= 0.40:
                        score = 2
                    else:
                        score = 1
                else:
                    # pdist变化率: 0.15~0.25 轻微(5分), 0.25~0.40 中等(4分),
                    # 0.40~0.60 明显(3分), 0.60~1.0 剧烈(2分), >1.0 过度(1分)
                    if avg <= 0.25:
                        score = 5
                    elif avg <= 0.40:
                        score = 4
                    elif avg <= 0.60:
                        score = 3
                    elif avg <= 1.0:
                        score = 2
                    else:
                        score = 1
                status = "proxy"
                violations = ([] if score >= 4
                              else [f"形变幅度异常: 应变={primary_metric:.4f}"
                                    f"{'(可变形物体)' if is_deformable else ''}"])

            metrics = {
                "avg_deformation": round(avg, 4),
                "is_deformable": is_deformable,
                "has_deformation_event": has_deformation_event,
                "method": metric_name,
            }
            if aligned_chamfer:
                metrics["aligned_chamfer_mean"] = round(float(np.mean(aligned_chamfer)), 4)
            if strain_gradient is not None:
                metrics["strain_gradient_frobenius"] = round(strain_gradient, 6)
            metrics["deform_per_frame"] = deform_metrics[:8]

            return EvalResult("R05-形变状态", score, violations,
                              metrics=metrics,
                              status=status)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return EvalResult.error("R05-形变状态",
                                    f"形变计算异常: {type(e).__name__}: {e}")

    def _eval_spatial(self, frames, fps: float = 24.0,
                      prompt: str = "") -> EvalResult:
        T = len(frames)
        # 多帧分析: 首/中/尾 (至少检查2帧)
        key_indices = [0]
        if T >= 3:
            key_indices.append(T // 2)
        if T >= 2:
            key_indices.append(T - 1)
        key_indices = sorted(set(key_indices))

        # 逐帧 SAM3 分割
        frame_masks = []
        for idx in key_indices:
            try:
                seg = self.sam3.segment_frame(frames[idx], dimension="R11",
                                              prompt=prompt)
                frame_masks.append(seg.masks if seg.masks is not None else None)
            except Exception:
                frame_masks.append(None)

        # 取有效帧中物体数量最多的作为参考
        valid_masks = [m for m in frame_masks if m is not None and len(m) >= 2]
        if not valid_masks:
            # prompt 声明了多物体空间关系但检测不到 ≥2 个物体。
            # 先关键词预筛, 命中时用 VLM 看关键帧确认「物体是否真的没生成/不可见」。
            multi_kw = ["stack", "pile", "tower", "beside", "next to",
                        "two", "three", "several", "multiple", "cubes",
                        "blocks", "train", "carriage", "堆叠", "并排"]
            low = (prompt or "").lower()
            if any(kw in low for kw in multi_kw):
                obs, obs_reason = self.vlm_observable(
                    frames[len(frames) // 2], prompt)
                if obs is False:
                    return EvalResult.unobservable(
                        "R11-空间关系",
                        f"SAM3 未检测到≥2个物体, VLM 确认多物体未生成/不可见: "
                        f"{obs_reason}")
            return EvalResult.skipped(
                "R11-空间关系",
                "SAM3 在所有关键帧均未检测到≥2个物体")

        ref_masks = max(valid_masks, key=len)
        N = len(ref_masks)  # 参考帧物体数量
        max_n_objects = N

        # 逐帧获取深度图 (用于辅助判断遮挡)
        depth_maps = {}
        for idx in key_indices:
            if idx not in depth_maps:
                try:
                    depth_out = self.vggt.estimate_depth(frames[idx])
                    if depth_out.depth_map is not None:
                        depth_maps[idx] = depth_out.depth_map
                except Exception:
                    pass

        # ── 跨帧重叠分析 ──
        violations = []
        max_overlap_all = 0.0
        # 对每对物体, 检查在每帧中的重叠情况
        pair_overlap_frames = {}  # (i,j) → [frame_indices where overlap > 0.1]

        for fi, (idx, masks) in enumerate(zip(key_indices, frame_masks)):
            if masks is None or len(masks) < 2:
                continue
            depth_map = depth_maps.get(idx)
            n_cur = len(masks)
            # VGGT 深度图分辨率可能与原始帧不同(image_resolution=512),
            # SAM3 masks 为原始帧尺寸, 布尔索引前需对齐 (否则 IndexError)
            H_d, W_d = depth_map.shape if depth_map is not None else (0, 0)
            H_m, W_m = masks[0].shape if n_cur > 0 else (0, 0)
            need_depth_resize = (H_d > 0 and W_d > 0 and (H_m != H_d or W_m != W_d))
            for i in range(n_cur):
                for j in range(i + 1, n_cur):
                    mask_i = masks[i] > 0.5
                    mask_j = masks[j] > 0.5
                    overlap = mask_i & mask_j
                    overlap_px = int(overlap.sum())
                    area_i, area_j = int(mask_i.sum()), int(mask_j.sum())
                    if area_i == 0 or area_j == 0:
                        continue
                    ratio = overlap_px / max(area_i, area_j)
                    max_overlap_all = max(max_overlap_all, ratio)
                    if ratio > 0.1:
                        # 深度辅助: 若深度差大 → 遮挡(正常)
                        is_occlusion = False
                        if depth_map is not None and overlap_px > 20:
                            if need_depth_resize:
                                import cv2
                                mask_i_d = cv2.resize(mask_i.astype(np.float32),
                                                      (W_d, H_d),
                                                      interpolation=cv2.INTER_LINEAR) > 0.5
                                mask_j_d = cv2.resize(mask_j.astype(np.float32),
                                                      (W_d, H_d),
                                                      interpolation=cv2.INTER_LINEAR) > 0.5
                                overlap_d = mask_i_d & mask_j_d
                                d_i = depth_map[mask_i_d & overlap_d]
                                d_j = depth_map[mask_j_d & overlap_d]
                            else:
                                d_i = depth_map[mask_i & overlap]
                                d_j = depth_map[mask_j & overlap]
                            d_i = d_i[np.isfinite(d_i) & (d_i > 0)]
                            d_j = d_j[np.isfinite(d_j) & (d_j > 0)]
                            if len(d_i) > 0 and len(d_j) > 0:
                                med_i, med_j = np.median(d_i), np.median(d_j)
                                scene_scale = (float(np.percentile(
                                    depth_map[np.isfinite(depth_map) & (depth_map > 0)], 90
                                )) if np.any(depth_map > 0) else 1.0)
                                if abs(med_i - med_j) > 0.12 * scene_scale:
                                    is_occlusion = True
                        if not is_occlusion:
                            pair_key = (i, j)
                            pair_overlap_frames.setdefault(pair_key, []).append(idx)

        # ── 3D 穿透检测: 将mask区域反投影到3D, 检查包围盒是否相交 ──
        penetrations_3d = {}  # (i,j) → {penetration_ratio, frame_idx, ...}
        for (i, j), overlap_frames in list(pair_overlap_frames.items()):
            for fi, (idx, masks) in enumerate(zip(key_indices, frame_masks)):
                if idx not in overlap_frames:
                    continue
                if masks is None or i >= len(masks) or j >= len(masks):
                    continue
                depth_map = depth_maps.get(idx)
                if depth_map is None:
                    continue

                mask_i = masks[i] > 0.5
                mask_j = masks[j] > 0.5
                H_m, W_m = mask_i.shape
                H_d, W_d = depth_map.shape

                # 反投影到3D (简化: 使用VGGT深度图 + 针孔近似)
                try:
                    # 获取每个物体的深度值
                    import cv2
                    if H_m != H_d or W_m != W_d:
                        mask_i_d = cv2.resize(mask_i.astype(np.float32),
                                              (W_d, H_d), interpolation=cv2.INTER_NEAREST) > 0.5
                        mask_j_d = cv2.resize(mask_j.astype(np.float32),
                                              (W_d, H_d), interpolation=cv2.INTER_NEAREST) > 0.5
                    else:
                        mask_i_d = mask_i
                        mask_j_d = mask_j

                    # 物体区域的深度统计
                    depths_i = depth_map[mask_i_d]
                    depths_i = depths_i[np.isfinite(depths_i) & (depths_i > 0)]
                    depths_j = depth_map[mask_j_d]
                    depths_j = depths_j[np.isfinite(depths_j) & (depths_j > 0)]

                    if len(depths_i) < 20 or len(depths_j) < 20:
                        continue

                    # 3D 深度范围 (VGGT深度为相对尺度, 归一化)
                    scene_median = float(np.median(depth_map[
                        np.isfinite(depth_map) & (depth_map > 0)])) if np.any(depth_map > 0) else 1.0
                    if scene_median <= 0:
                        continue

                    # 物体在深度维的范围 (归一化 z 坐标)
                    z_i_min = float(np.percentile(depths_i, 10)) / scene_median
                    z_i_max = float(np.percentile(depths_i, 90)) / scene_median
                    z_j_min = float(np.percentile(depths_j, 10)) / scene_median
                    z_j_max = float(np.percentile(depths_j, 90)) / scene_median

                    # 3D 深度范围重叠 → 穿透检测
                    depth_overlap_min = max(z_i_min, z_j_min)
                    depth_overlap_max = min(z_i_max, z_j_max)
                    depth_overlap_range = depth_overlap_max - depth_overlap_min

                    # 物体各自的深度跨度
                    span_i = z_i_max - z_i_min + 1e-8
                    span_j = z_j_max - z_j_min + 1e-8

                    # 穿透比率: 重叠深度 / min(span_i, span_j)
                    penetration_ratio = depth_overlap_range / min(span_i, span_j) if depth_overlap_range > 0 else 0.0

                    if penetration_ratio > 0.15:  # 超过15%的深度范围内两者共存 → 3D穿透
                        key_pair = (i, j)
                        if key_pair not in penetrations_3d:
                            penetrations_3d[key_pair] = {
                                "penetration_ratio": penetration_ratio,
                                "frame_idx": idx,
                                "depth_i_range": f"{z_i_min:.3f}-{z_i_max:.3f}",
                                "depth_j_range": f"{z_j_min:.3f}-{z_j_max:.3f}",
                            }
                except Exception:
                    pass

        # ── 跨帧判定: 持续重叠=穿模, 非持续=遮挡或巧合 ──
        # 合并 2D 持续性 和 3D 穿透分析
        n_frames_checked = len([m for m in frame_masks if m is not None])
        pair_scores = {}  # (i,j) → "3d_penetration" | "persistent_2d" | "transient"
        for (i, j), overlap_frames in pair_overlap_frames.items():
            persistent = len(overlap_frames) >= max(2, n_frames_checked)
            has_3d = (i, j) in penetrations_3d
            if has_3d:
                pair_scores[(i, j)] = "3d_penetration"
            elif persistent:
                pair_scores[(i, j)] = "persistent_2d"
            else:
                pair_scores[(i, j)] = "transient"

        # 生成违规信息
        n_persistent = 0
        n_3d_penetration = 0
        n_transient = 0
        for (i, j), classification in pair_scores.items():
            if classification == "3d_penetration":
                n_3d_penetration += 1
                p3d = penetrations_3d.get((i, j), {})
                violations.append(
                    f"物体{i}与{j}: 3D穿透检测 (帧{p3d.get('frame_idx','?')}, "
                    f"穿透率={p3d.get('penetration_ratio',0):.2f}, "
                    f"深度i={p3d.get('depth_i_range','?')}, "
                    f"深度j={p3d.get('depth_j_range','?')})")
            elif classification == "persistent_2d":
                n_persistent += 1
                overlap_frames = pair_overlap_frames[(i, j)]
                violations.append(
                    f"物体{i}与{j}在{len(overlap_frames)}个关键帧中持续2D重叠(疑似穿模, 3D深度信息不足)")
            else:
                n_transient += 1
                overlap_frames = pair_overlap_frames[(i, j)]
                violations.append(
                    f"物体{i}与{j}在{len(overlap_frames)}/{n_frames_checked}帧重叠(瞬态遮挡/交错, 非穿模)")

        # ── 打分 ──
        # 3D穿透 权重最高; 持续2D重叠 次之
        total_penalties = n_3d_penetration * 2 + n_persistent
        if total_penalties == 0:
            score = 5
        elif total_penalties <= 1:
            score = 4
        elif total_penalties <= 3:
            score = 3
        elif total_penalties <= 5:
            score = 2
        else:
            score = 1

        return EvalResult("R11-空间关系", score, violations,
                          metrics={
                              "n_objects": N,
                              "max_n_objects": max_n_objects,
                              "n_frames_checked": n_frames_checked,
                              "n_persistent_overlaps": n_persistent,
                              "n_transient_overlaps": n_transient,
                              "n_3d_penetrations": n_3d_penetration,
                              "max_overlap_ratio": round(max_overlap_all, 4),
                              "depth_checked": any(d is not None for d in depth_maps.values()),
                              "detection_mode": "2d_overlap + 3d_depth_penetration",
                          },
                          status="proxy")
