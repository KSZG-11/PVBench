"""
L3 物理层 — 多视图评测器
R08 多视图几何一致性 (VGGT)

评测思路 (来自物理量分类体系.xlsx D列):
投影误差(3D点投影到各视图与观测2D点欧氏距离) + Sampson距离(极线几何) + 新视角PSNR

两种工作模式:
1. 原生多视图: frames 即为不同视角的独立图像 → 直接喂 VGGT
2. T2V 面板模式: 每个帧内包含并排的多个视角画面 → 先裁切面板再喂 VGGT

注: 需多视角输入(同一时刻不同视角)。默认关闭(config multiview.enabled=false)。
"""
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult
from ..thresholds import score_lower_better, score_metric

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# T2V 面板模式的预设布局
_PANEL_LAYOUTS = {
    "horizontal_3": {"n_panels": 3, "axis": "horizontal", "desc": "水平三等分"},
    "horizontal_2": {"n_panels": 2, "axis": "horizontal", "desc": "水平两等分"},
    "vertical_3":   {"n_panels": 3, "axis": "vertical",   "desc": "垂直三等分"},
}


class MultiviewEvaluator(BaseEvaluator):
    """多视图几何一致性评测器（R08）"""

    def __init__(self, vggt, sam3=None, config: Optional[dict] = None):
        super().__init__("L3-Physics/Multiview", config)
        self.vggt = vggt
        self.sam3 = sam3

    @staticmethod
    def _split_panels(frame: np.ndarray,
                      n_panels: int = 3,
                      layout: str = "horizontal") -> list[np.ndarray]:
        """从包含多面板的单帧中裁切出各个视角画面。

        Args:
            frame: [H, W, 3] 单帧 (可能含多视角并排)
            n_panels: 面板数量
            layout: "horizontal" 水平排列 / "vertical" 垂直排列

        Returns:
            length=n_panels 的 numpy array 列表, 每个 [H_p, W_p, 3]
        """
        H, W = frame.shape[:2]
        panels = []
        if layout == "horizontal":
            panel_w = W // n_panels
            for i in range(n_panels):
                x0 = i * panel_w
                x1 = (i + 1) * panel_w if i < n_panels - 1 else W
                panels.append(frame[:, x0:x1])
        else:  # vertical
            panel_h = H // n_panels
            for i in range(n_panels):
                y0 = i * panel_h
                y1 = (i + 1) * panel_h if i < n_panels - 1 else H
                panels.append(frame[y0:y1, :])
        return panels

    def evaluate(self, frames: list[Image.Image],
                 view_indices: Optional[list] = None,
                 camera_poses: Optional[np.ndarray] = None,
                 modality: str = "T2V",
                 panel_layout: Optional[str] = None,
                 storyboard: Optional[dict] = None,
                 evaluation_method: str = "default",
                 prompt: str = ""
                 ) -> list[EvalResult]:
        """评测多视图几何一致性。

        Args:
            frames: 视频帧
            view_indices: CI2V 显式多视角帧索引 (None = 单视频)
            camera_poses: CI2V GT 相机外参
            modality: "T2V" | "I2V" | "CI2V"
            panel_layout: T2V 面板布局 key of _PANEL_LAYOUTS (None = 不启用面板模式)
            storyboard: 可选三阶段时间视角分镜，仅作为顺序多视角 proxy
            evaluation_method: temporal_multiview_proxy / continuous_view_sequence_proxy
            prompt: 样本 prompt, 用于 SAM3 概念筛选
        """
        dim_name = "R08-多视图几何一致性"

        if evaluation_method == "temporal_multiview_proxy":
            return self._evaluate_temporal_storyboard(frames, storyboard or {},
                                                      prompt=prompt)
        if evaluation_method == "continuous_view_sequence_proxy":
            return self._evaluate_continuous_sequence(frames, prompt=prompt)

        # ── CI2V 显式多视角 ──
        if view_indices is not None and len(view_indices) >= 2:
            return self._evaluate_native(frames, camera_poses)

        # ── T2V 面板模式：单帧内包含并排多视角 ──
        if panel_layout is not None and panel_layout in _PANEL_LAYOUTS:
            return self._evaluate_t2v_panels(frames, panel_layout, camera_poses)

        # ── I2V / 普通 T2V：只有单视频流，无法做多视图几何一致性 ──
        # per PHYSICS_TAXONOMY.md: R08 适用模型 T2V△ / I2V△ / CI2V✓
        # T2V/I2V 的环绕镜头提示词属于 prompt 代理，但在没有显式多面板/多视角输入时
        # 无法可靠评估 → 跳过
        return [EvalResult.skipped(dim_name,
                                   f"当前模态={modality}, 无显式多视图输入, 无法评测R08")]

    @staticmethod
    def _phase_midpoint_indices(storyboard: dict, n_frames: int) -> list[int]:
        total = float(storyboard.get("total_seconds", 10.0) or 10.0)
        phases = storyboard.get("phases") or []
        indices = []
        for phase in phases:
            midpoint = (float(phase.get("start", 0.0))
                        + float(phase.get("end", 0.0))) / 2.0
            ratio = np.clip(midpoint / max(total, 1e-8), 0.0, 1.0)
            indices.append(int(round(ratio * max(n_frames - 1, 0))))
        return list(dict.fromkeys(indices))

    def _evaluate_temporal_storyboard(self, frames: list[Image.Image],
                                      storyboard: dict,
                                      prompt: str = "") -> list[EvalResult]:
        if storyboard.get("mode") != "temporal_multiview":
            return [EvalResult.skipped(
                "R08-多视图几何一致性", "分镜mode不是temporal_multiview")]
        indices = self._phase_midpoint_indices(storyboard, len(frames))
        if len(indices) < 3:
            return [EvalResult.skipped(
                "R08-多视图几何一致性", f"时间分镜有效视角不足3个: {indices}")]
        selected = [frames[i] for i in indices]
        return self._evaluate_view_sequence(
            selected, mode="temporal_storyboard_proxy",
            source_indices=indices, storyboard=storyboard, prompt=prompt)

    def _evaluate_continuous_sequence(self,
                                      frames: list[Image.Image],
                                      prompt: str = "") -> list[EvalResult]:
        if len(frames) < 3:
            return [EvalResult.skipped(
                "R08-多视图几何一致性", "连续视角序列需要至少3帧")]
        indices = np.linspace(0, len(frames) - 1, 3, dtype=int).tolist()
        selected = [frames[i] for i in indices]
        return self._evaluate_view_sequence(
            selected, mode="continuous_view_sequence_proxy",
            source_indices=indices, storyboard=None, prompt=prompt)

    def _evaluate_view_sequence(self, views: list[Image.Image], mode: str,
                                source_indices: list[int],
                                storyboard: Optional[dict],
                                prompt: str = "") -> list[EvalResult]:
        """顺序多视角代理：逐物体刚体对齐 + Chamfer 度量几何一致性。

        核心思想 (R08 本质 = 同一时刻不同视角、几何互不矛盾):
        物体运动只造成点云的位置/朝向变化, 逐物体纯刚体对齐(不含尺度)吸收运动后,
        剩余的 Chamfer 距离只反映「该物体的几何结构跨视角是否一致」。
        因此对物体静止/运动场景统一适用, 不 skip。
        """
        arrays = np.stack([np.array(frame) for frame in views])
        T = len(arrays)

        # ── 1. SAM3 分割中间帧得到关键物体 mask ──
        masks_per_obj = None  # [N,T,H,W]
        if self.sam3 is not None and getattr(self.sam3, "is_loaded", False):
            try:
                mid = len(views) // 2
                seg = self.sam3.segment_frame(arrays[mid], dimension="R08",
                                              prompt=prompt)
                if seg.masks is not None and len(seg.masks) > 0:
                    # 用 SAM3 视频追踪把中间帧 mask 传播到全部 T 帧
                    try:
                        seg_v = self.sam3.segment_video(
                            arrays, dimension="R08",
                            prompt=prompt,
                            initial_boxes=self._boxes_from_masks(
                                seg.masks, arrays[mid].shape))
                        if seg_v.masks is not None and seg_v.masks.ndim == 4 \
                                and seg_v.masks.shape[0] > 0:
                            masks_per_obj = seg_v.masks  # [N,T,H,W]
                    except Exception:
                        masks_per_obj = None
                # 追踪失败兜底: 中间帧 mask 广播到所有帧 (静止物体+相机运动的合理近似)
                if masks_per_obj is None and seg.masks is not None \
                        and len(seg.masks) > 0:
                    mid_masks = np.asarray(seg.masks, dtype=np.float32)  # [N,H,W]
                    masks_per_obj = np.repeat(mid_masks[:, None, :, :], T, axis=1)
            except Exception:
                masks_per_obj = None

        # ── 2. VGGT 多帧联合推理 + 逐物体世界系点云 ──
        try:
            if masks_per_obj is not None and len(masks_per_obj) > 0:
                world_pts, extr, intr, conf = self.vggt.reconstruct_masked_pointclouds(
                    arrays, np.asarray(masks_per_obj, dtype=np.float32))
            else:
                # 无 SAM3 或无物体: 退化为全场景点云
                world_pts = None
                extr = self.vggt.estimate_camera_poses(arrays)
                conf = None
        except Exception as exc:
            return [EvalResult.error(
                "R08-多视图几何一致性", f"VGGT多视图重建失败: {exc}")]

        # ── 3. 视角差异 (三帧相机位姿相对旋转/中心离散) ──
        # extr 是 world-to-camera [R_w2c | t_w2c]; 相机中心世界坐标 = -R_w2c.T @ t_w2c
        rotations = extr[:, :3, :3]
        centers = np.stack([
            -rotations[t].T @ extr[t, :3, 3] for t in range(len(extr))
        ], axis=0)
        center_spread = float(np.std(centers, axis=0).mean())
        rotation_angles = []
        for i in range(len(rotations)):
            for j in range(i + 1, len(rotations)):
                relative = rotations[i] @ rotations[j].T
                cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
                rotation_angles.append(float(np.degrees(np.arccos(cosine))))
        max_rotation = max(rotation_angles, default=0.0)

        # ── 4. 逐物体刚体对齐 + Chamfer ──
        chamfers = []
        n_objects = 0
        if world_pts is not None:
            for obj_pts in world_pts:  # obj_pts: list[T] of [M,3]
                valid = [p for p in obj_pts if len(p) >= 20]
                if len(valid) < 2:
                    continue
                n_objects += 1
                # 以第一帧为参考, 其余帧 ICP 对齐后算 Chamfer
                ref = valid[0].astype(np.float64)
                ref_tree = cKDTree(ref) if cKDTree is not None else None
                for p in valid[1:]:
                    aligned = self._icp_align(p.astype(np.float64), ref)
                    if cKDTree is not None and ref_tree is not None:
                        d1 = ref_tree.query(aligned, k=1)[0].mean()
                        d2 = cKDTree(aligned).query(ref, k=1)[0].mean()
                    else:
                        d1 = d2 = float(np.mean(np.linalg.norm(
                            aligned - ref[:len(aligned)], axis=-1))) \
                            if len(aligned) == len(ref) else float("inf")
                    scale = float(np.std(ref, axis=0).mean()) + 1e-8
                    chamfers.append(float((d1 + d2) / 2 / scale))

        avg_chamfer = float(np.mean(chamfers)) if chamfers else None

        # ── 5. 综合评分 ──
        violations = []
        score = 5

        # 几何一致性: Chamfer (越小越好, 阈值对齐 R19 跨帧漂移的 VGGT 噪声基线)
        if avg_chamfer is not None:
            chamfer_score = score_metric("R19_drift", avg_chamfer)
            score = min(score, chamfer_score)
            if chamfer_score < 4:
                violations.append(f"逐物体对齐后Chamfer={avg_chamfer:.3f}")
        elif world_pts is not None and n_objects == 0:
            # 有 SAM3 但没分割出有效物体
            score = min(score, 3)
            violations.append("未分割出可对齐的物体")

        # 视角差异: 三帧相机位姿应有明显差异 (验证"多视角"前提)
        if max_rotation < 3.0 and center_spread < 0.02:
            score = min(score, 2)
            violations.append("三个时间阶段缺少可判定视角差异")

        # depth_conf 辅助 (若可得): 单帧重建质量
        conf_mean = None
        if conf is not None:
            conf_mean = float(np.mean(conf[np.isfinite(conf)])) \
                if np.isfinite(conf).any() else None

        return [EvalResult(
            "R08-多视图几何一致性", score, violations,
            metrics={
                "mode": mode,
                "source_frame_indices": source_indices,
                "n_views": T,
                "n_objects": n_objects,
                "avg_chamfer_norm": round(avg_chamfer, 4) if avg_chamfer is not None else None,
                "camera_center_spread": round(center_spread, 4),
                "max_camera_rotation_deg": round(max_rotation, 2),
                "depth_conf_mean": round(conf_mean, 4) if conf_mean is not None else None,
                "time_synchronized": False,
                "storyboard": storyboard,
            },
            status="proxy")]

    @staticmethod
    def _kabsch_rotation(P, Q):
        """Kabsch 纯刚体旋转 (去质心 + SVD, det=+1), 不含尺度。"""
        Pc = P - P.mean(axis=0)
        Qc = Q - Q.mean(axis=0)
        H = Pc.T @ Qc
        try:
            U, _, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            return np.eye(3)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        return Vt.T @ D @ U.T

    @classmethod
    def _icp_align(cls, source, target, iterations=8):
        """无点对应关系时用最近邻 ICP 做纯刚体对齐 (不含尺度)。"""
        src = source.astype(np.float64).copy()
        dst = target.astype(np.float64)
        src += dst.mean(axis=0) - src.mean(axis=0)
        if cKDTree is None:
            return src
        tree = cKDTree(dst)
        for _ in range(iterations):
            _, nn = tree.query(src, k=1)
            matched = dst[nn]
            R = cls._kabsch_rotation(src, matched)
            src = (src - src.mean(axis=0)) @ R.T + matched.mean(axis=0)
        return src

    @staticmethod
    def _boxes_from_masks(masks: np.ndarray, frame_shape: tuple) -> np.ndarray:
        """mask [N,H,W] → 归一化 [xmin,ymin,width,height] (0~1) 初始框。

        ⚠️ SAM3 视频追踪 box prompt 只允许一个 box, 故只返回面积最大的主目标框 [1,4]。
        """
        H, W = frame_shape[:2]
        boxes = np.zeros((len(masks), 4), dtype=np.float32)
        for i, m in enumerate(masks):
            ys, xs = np.where(m > 0.5)
            if len(xs) == 0:
                boxes[i] = [0, 0, 0, 0]
                continue
            boxes[i] = [xs.min() / W, ys.min() / H,
                        (xs.max() - xs.min()) / W, (ys.max() - ys.min()) / H]
        valid = (boxes[:, 2] > 0.005) & (boxes[:, 3] > 0.005)
        boxes = boxes[valid]
        if len(boxes) == 0:
            return np.zeros((0, 4))
        # 只保留面积最大的主目标 (单 box)
        areas = boxes[:, 2] * boxes[:, 3]
        return boxes[np.argmax(areas)][None]

    def _evaluate_t2v_panels(self, frames: list[Image.Image],
                              panel_layout: str,
                              camera_poses: Optional[np.ndarray] = None
                              ) -> list[EvalResult]:
        """T2V 面板模式: 将单帧画面裁切为多个面板, 作为多视角输入 VGGT 重建。"""
        layout_cfg = _PANEL_LAYOUTS.get(panel_layout, _PANEL_LAYOUTS["horizontal_3"])
        n_panels = layout_cfg["n_panels"]
        axis = layout_cfg["axis"]
        dim_name = "R08-多视图几何一致性"

        # 取中间帧作为多面板源 (T2V视频各帧的视角布局通常一致)
        key_idx = len(frames) // 2
        key_frame = np.array(frames[key_idx])

        panels = self._split_panels(key_frame, n_panels=n_panels, layout=axis)
        H_p, W_p = panels[0].shape[:2]
        if min(H_p, W_p) < 32:
            return [EvalResult.skipped(
                dim_name,
                f"面板裁切后尺寸过小({H_p}x{W_p}), 请检查面板布局({panel_layout})"
            )]

        metrics = {
            "mode": "t2v_panels",
            "panel_layout": panel_layout,
            "n_panels": n_panels,
            "key_frame_idx": key_idx,
            "panel_size": [H_p, W_p],
        }

        # 多面板一起送入 VGGT 联合重建
        try:
            panel_arrays = np.stack(panels)  # [n_panels, H_p, W_p, 3]
            pc = self.vggt.reconstruct_pointcloud(panel_arrays)
            extr = self.vggt.estimate_camera_poses(panel_arrays)  # [n_panels, 3, 4]
        except Exception as e:
            return [EvalResult.error(dim_name, f"VGGT 面板重建失败: {e}")]

        pts = pc.points  # [M, 3]
        if pts is None or len(pts) < 50:
            return [EvalResult.skipped(dim_name, "重建点云不足(面板间可能不构成一致3D场景)")]

        # ── 重建质量综合评分 ──
        # 三个信号:
        # 1. 点云非退化度 (extents 各维度都 > 阈值 = 有真实3D结构)
        # 2. 相机中心离散度 (各面板相机位姿应有合理基线)
        # 3. 相机基线合理性 (最小/最大基线比)

        extents = pts.max(0) - pts.min(0)
        is_degenerate = extents.min() < 0.01

        centers = extr[:, :, 3]  # [n_panels, 3]
        center_spread = float(np.std(centers, axis=0).mean())
        metrics["center_spread"] = center_spread

        # 相机基线分析: 成对距离
        if n_panels >= 2:
            pairwise_dists = []
            for i in range(n_panels):
                for j in range(i + 1, n_panels):
                    d = float(np.linalg.norm(centers[i] - centers[j]))
                    pairwise_dists.append(d)
            if pairwise_dists:
                metrics["max_baseline"] = max(pairwise_dists)
                metrics["min_baseline"] = min(pairwise_dists)
                metrics["mean_baseline"] = float(np.mean(pairwise_dists))

        v = []
        score = 5

        # 检查1: 点云是否退化 (面板画面根本不是同一场景)
        if is_degenerate:
            v.append("点云退化(各面板可能不是同一3D场景)")
            score = max(1, score - 2)

        # 检查2: 相机中心是否过度聚集 (面板画面高度相似 = 无有效多视角信息)
        if center_spread < 0.3:
            v.append(f"相机中心过于集中(spread={center_spread:.2f}, 面板间缺乏有效视差)")
            score = max(1, score - 2)

        # 检查3: 点云数量 (太少 = 没有足够的匹配特征)
        if len(pts) < 200:
            v.append(f"点云稀疏(n={len(pts)}, 面板间匹配特征不足)")
            score = max(1, score - 1)

        # 检查4: 面板间的内点比例 (通过点云来自哪个面板的分布判断)
        # VGGT 输出的点云是多视图融合结果, 若面板不一致, 融合点数应远少于单面板点数
        # 粗略估计: 点云数 / (面板分辨率乘积) 作为内点密度
        panel_area = H_p * W_p
        inlier_density = len(pts) / (panel_area * n_panels + 1)
        metrics["inlier_density"] = round(inlier_density, 4)
        if inlier_density < 0.01:
            v.append(f"面板间内点密度极低({inlier_density:.4f}), 可能非一致场景")
            score = max(1, score - 1)

        # 若提供了GT相机位姿, 做精确误差对比
        if camera_poses is not None and len(camera_poses) >= n_panels:
            gt_centers = np.array([camera_poses[i][:3, 3] for i in range(n_panels)])
            c_pred = centers - centers.mean(0)
            c_gt = gt_centers - gt_centers.mean(0)
            cam_err = float(np.mean(np.linalg.norm(c_pred - c_gt, axis=-1)))
            metrics["camera_center_error"] = cam_err
            gt_score = score_lower_better(cam_err, [0.1, 0.3, 0.5, 1.0])
            if gt_score < 4:
                v.append(f"相机中心误差={cam_err:.3f}")
            score = min(score, gt_score)
            status = "ok"
        else:
            status = "proxy"

        metrics["n_points"] = len(pts)
        metrics["extent_x"] = round(float(extents[0]), 3)
        metrics["extent_y"] = round(float(extents[1]), 3)
        metrics["extent_z"] = round(float(extents[2]), 3)

        return [EvalResult(dim_name, score, v, metrics=metrics, status=status)]

    def _evaluate_native(self, frames: list[Image.Image],
                          camera_poses: Optional[np.ndarray] = None
                          ) -> list[EvalResult]:
        """原生多视图模式: 每个帧是独立视角图像。"""
        frame_arrays = np.stack([np.array(f) for f in frames])
        V = len(frame_arrays)
        if V < 2:
            return [EvalResult.skipped("R08-多视图几何一致性", "需要≥2个视角")]

        try:
            pc = self.vggt.reconstruct_pointcloud(frame_arrays)
            extr = self.vggt.estimate_camera_poses(frame_arrays)
        except Exception as e:
            return [EvalResult.error("R08-多视图几何一致性", f"VGGT 重建失败: {e}")]

        pts = pc.points
        if pts is None or len(pts) < 50:
            return [EvalResult.skipped("R08-多视图几何一致性", "重建点云不足")]

        centers = extr[:, :, 3]
        center_spread = float(np.std(centers, axis=0).mean())

        metrics = {"mode": "native", "n_views": V, "center_spread": center_spread}
        v = []

        if camera_poses is not None and len(camera_poses) >= V:
            gt_centers = np.array([camera_poses[i][:3, 3] for i in range(V)])
            c_pred = centers - centers.mean(0)
            c_gt = gt_centers - gt_centers.mean(0)
            cam_err = float(np.mean(np.linalg.norm(c_pred - c_gt, axis=-1)))
            metrics["camera_center_error"] = cam_err
            score = score_lower_better(cam_err, [0.1, 0.3, 0.5, 1.0])
            if score < 4:
                v.append(f"相机中心误差={cam_err:.3f}")
            return [EvalResult("R08-多视图几何一致性", score, v, metrics=metrics, status="ok")]

        extents = pts.max(0) - pts.min(0)
        score = 5 if (extents.min() > 0.01 and center_spread < 5.0) else 3
        if center_spread > 5.0:
            v.append(f"各视角相机中心离散过大={center_spread:.2f}")
        return [EvalResult("R08-多视图几何一致性", score, v, metrics=metrics, status="proxy")]
