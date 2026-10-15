"""
L3 物理层 — 运动持续性组评测器
R19 跨帧几何 (VGGT + FlowSeek) + R20 运动连续性 (VFIMamba + SpatialTracker) + R21 物体恒存 (SAM3 + VGGT + VLM)

评测思路 (来自物理量分类体系.xlsx D列):
- R19: 点云跨帧Chamfer漂移 + 光流平均端点误差AEPE
- R20: 丢奇数帧用偶数帧插回与真实奇数帧算误差 + 3D轨迹jerk
- R21: "移开→移回"ORS + 深度前后序判合法遮挡 + ID跳变统计
"""
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult
from ..thresholds import score_metric, score_lower_better, THRESHOLDS


class MotionPersistenceEvaluator(BaseEvaluator):
    """运动持续性组评测器（R19 / R20 / R21）"""

    def __init__(self, vggt, flowseek, vfimamba, spatial_tracker, sam3, vlm,
                 config: Optional[dict] = None):
        super().__init__("L3-Physics/MotionPersistence", config)
        self.vggt = vggt
        self.flowseek = flowseek
        self.vfimamba = vfimamba
        self.spatial_tracker = spatial_tracker
        self.sam3 = sam3
        self.vlm = vlm

    def evaluate(self, frames: list[Image.Image],
                 full_frames: Optional[list[Image.Image]] = None,
                 fps: float = 24.0,
                 prompt: str = "",
                 storyboard: Optional[dict] = None,
                 evaluation_method: str = "default",
                 target_dimensions: Optional[list[str]] = None) -> list[EvalResult]:
        frame_arrays = np.stack([np.array(f) for f in frames])
        # 维度级按需执行
        want = set(target_dimensions) if target_dimensions else None
        def _need(prefix: str) -> bool:
            return want is None or any(d.startswith(prefix) for d in want)
        # R21 物体恒存需要全部帧才能捕获突然出现/消失；若无全帧则回退采样帧
        r32_input = (np.stack([np.array(f) for f in full_frames])
                     if full_frames is not None and len(full_frames) > 0
                     else frame_arrays)
        # R20 运动连续性: 用采样帧即可 (VFIMamba 插帧 + SpatialTracker jerk 分析
        # 32 帧采样足够; 用全帧 241 帧会 OOM, 且 R15/R16/R17 都是 32 帧采样)。
        r30_input = frame_arrays
        # 与R09相同：只向SAM3视频Detector传当前样本实际出现的具体目标词，
        # 避免R21把 tennis ball/person/sofa/tree 等互斥概念合成一个低召回查询。
        concepts = None
        if _need("R21"):
            try:
                from .scale_pose import ScalePoseEvaluator
                concepts = ScalePoseEvaluator._prompt_concepts(prompt, "R21")
            except Exception:
                pass
        results = []
        if _need("R19"):
            results.append(self._eval_crossframe_geometry(frame_arrays))
        if _need("R20"):
            results.append(self._eval_motion_continuity(r30_input, fps=fps))
        if _need("R21"):
            results.append(self._eval_object_persistence(
                r32_input,
                concepts=concepts,
                storyboard=storyboard,
                evaluation_method=evaluation_method,
                prompt=prompt,
            ))
        return results

    # ── R19 跨帧几何一致性 ──
    # v3: VGGT 点云漂移(主) + FlowSeek 光流时序一致性(辅) 联合评分
    #     VGGT 对相邻帧独立重建,漂移度量 3D 结构一致性 (60% 权重)
    #     FlowSeek 逐帧对光流 AEPE 的 CV 度量 2D 运动场时序稳定性 (40% 权重)
    #     加入逐帧定位信息: 记录每对帧的漂移/AEPE 值, 便于追溯
    def _eval_crossframe_geometry(self, frames) -> EvalResult:
        from scipy.spatial import cKDTree
        T = len(frames)
        rng = np.random.default_rng(0)
        drifts = []           # 逐帧对漂移值
        drift_frames = []     # 对应的帧对索引 (用于定位)
        H, W = frames.shape[1:3]
        img_diag = np.sqrt(H**2 + W**2)
        pc_ok = True
        try:
            for t in range(min(T - 1, 4)):
                p0 = self.vggt.reconstruct_pointcloud(frames[t]).points
                p1 = self.vggt.reconstruct_pointcloud(frames[t + 1]).points
                if len(p0) < 10 or len(p1) < 10:
                    continue
                n = min(500, len(p0), len(p1))
                a = p0[rng.choice(len(p0), n, replace=False)]
                b = p1[rng.choice(len(p1), n, replace=False)]
                scale = np.std(b, axis=0).mean() + 1e-8
                d = cKDTree(b).query(a, k=1)[0].mean() / scale
                drifts.append(d)
                drift_frames.append((t, t + 1))
        except Exception:
            pc_ok = False

        # FlowSeek 逐帧对光流时序一致性
        aepe = None
        aepe_per_pair = None
        flow_cv = None
        flow_score = None
        try:
            flow = self.flowseek.compute_flow(frames)
            flow_fwd = flow.flow_fwd  # [T-1, H, W, 2]
            # 每帧对的平均光流模长 + 归一化
            aepe_per_pair = [float(np.linalg.norm(flow_fwd[t].reshape(-1, 2), axis=-1).mean())
                             for t in range(len(flow_fwd))]
            aepe = float(np.mean(aepe_per_pair))
            # 逐帧对 AEPE 的变异系数: 低 CV = 光流量级稳定 = 跨帧几何一致
            if len(aepe_per_pair) >= 2:
                flow_cv = float(np.std(aepe_per_pair) / (np.mean(aepe_per_pair) + 1e-8))
                flow_score = score_metric("R19_flow_cv", flow_cv)
        except Exception:
            aepe = None

        if not drifts and aepe is None:
            return EvalResult.error("R19-跨帧几何一致性", "VGGT/FlowSeek 均不可用")

        avg_drift = float(np.mean(drifts)) if drifts else 0.0

        # 联合评分: VGGT 3D 漂移(60%) + FlowSeek 2D 时序一致性(40%)
        if drifts and flow_score is not None:
            drift_score = score_metric("R19_drift", avg_drift)
            score = int(np.clip(round(drift_score * 0.6 + flow_score * 0.4), 1, 5))
        elif drifts:
            score = score_metric("R19_drift", avg_drift)
        else:
            score = 3  # 仅有 FlowSeek 无 VGGT 时给中性分

        # 违例列表 (含帧级别定位)
        v = []
        if drifts:
            for i, d in enumerate(drifts):
                if d > THRESHOLDS["R19_drift"]["th"][1]:  # > 4分段阈值
                    frame_pair = drift_frames[i] if i < len(drift_frames) else (i, i + 1)
                    v.append(f"帧对{frame_pair}: 点云漂移={d:.3f}")
        if flow_score is not None and flow_score < 4:
            v.append(f"光流时序CV={flow_cv:.3f} (逐帧对AEPE={[round(x,1) for x in aepe_per_pair[:6]]}...)"
                     if aepe_per_pair and len(aepe_per_pair) > 6
                     else f"光流时序CV={flow_cv:.3f} (AEPE-per-pair={[round(x,1) for x in (aepe_per_pair or [])]})")

        return EvalResult("R19-跨帧几何一致性", score, v,
                          metrics={"avg_chamfer_drift": avg_drift,
                                   "drift_per_pair": {f"{a}-{b}": round(d, 4)
                                                      for (a, b), d in zip(drift_frames, drifts)},
                                   "optical_flow_AEPE": aepe,
                                   "flow_aepe_per_pair": aepe_per_pair,
                                   "flow_temporal_cv": flow_cv,
                                   "flow_score": flow_score},
                          status="proxy" if pc_ok else "degraded")

    # ── R20 运动连续性 v2: 帧级定位 ──
    def _eval_motion_continuity(self, frames, fps: float = 24.0) -> EvalResult:
        T = len(frames)
        if T < 3:
            return EvalResult.skipped("R20-运动连续性", "帧数<3")

        interp_error = None
        interp_per_frame = None
        try:
            out = self.vfimamba.interpolate_odd_frames(frames)
            interp = out.interpolated_frames
            target_idx = getattr(out, "quality_scores", None)
            if len(interp) > 0 and target_idx is not None:
                errs = []
                interp_per_frame = []
                for k in range(len(interp)):
                    ti = int(target_idx[k])
                    if 0 <= ti < T:
                        e = np.mean(np.abs(
                            interp[k].astype(np.float32) - frames[ti].astype(np.float32))) / 255.0
                        errs.append(float(e))
                        interp_per_frame.append({"target_frame": ti, "error": round(float(e), 4)})
                interp_error = float(np.mean(errs)) if errs else None
        except Exception as e:
            print(f"[R20] VFIMamba 插帧失败: {type(e).__name__}: {e}")
            interp_error = None

        mean_jerk = None
        jerk_per_obj = None
        try:
            traj = self.spatial_tracker.track_3d(frames, fps=fps)
            if traj.trajectories.shape[1] >= 4:
                acc = np.diff(traj.trajectories, n=2, axis=1)
                jerk = np.linalg.norm(np.diff(acc, axis=1), axis=-1)  # [N, T-3]
                mean_jerk = float(np.mean(jerk))
                # 逐物体记录最大 jerk 帧位置
                jerk_per_obj = []
                for i in range(min(jerk.shape[0], 5)):
                    max_t = int(np.argmax(jerk[i])) if jerk.shape[1] > 0 else -1
                    jerk_per_obj.append({
                        "obj": i, "max_jerk_frame": max_t + 1,
                        "max_jerk": round(float(jerk[i].max()) if jerk.shape[1] > 0 else 0, 4),
                        "mean_jerk": round(float(jerk[i].mean()) if jerk.shape[1] > 0 else 0, 4),
                    })
        except Exception as e:
            print(f"[R20] SpatialTracker jerk 失败: {type(e).__name__}: {e}")
            mean_jerk = None

        if interp_error is None and mean_jerk is None:
            return EvalResult.error("R20-运动连续性", "VFIMamba/SpatialTracker 均不可用")

        comps = []
        if interp_error is not None:
            comps.append(1.0 - min(interp_error * 10, 1.0))
        if mean_jerk is not None:
            comps.append(1.0 - min(mean_jerk, 1.0))
        combined = float(np.mean(comps))
        score = int(np.clip(combined * 4 + 1, 1, 5))
        v = []
        if interp_error is not None and interp_error > 0.05:
            worst = max(interp_per_frame, key=lambda x: x["error"]) if interp_per_frame else {}
            v.append(f"插帧误差={interp_error:.3f} (最差帧{worst.get('target_frame','?')}: "
                     f"{worst.get('error','?')})")
        if mean_jerk is not None and mean_jerk > 0.3:
            worst_obj = max(jerk_per_obj, key=lambda x: x["max_jerk"]) if jerk_per_obj else {}
            v.append(f"轨迹jerk={mean_jerk:.3f} (最差物体{worst_obj.get('obj','?')} "
                     f"帧{worst_obj.get('max_jerk_frame','?')}: {worst_obj.get('max_jerk','?')})")
        return EvalResult("R20-运动连续性", score, v,
                          metrics={"interp_error": interp_error,
                                   "interp_per_frame": interp_per_frame,
                                   "mean_jerk": mean_jerk,
                                   "jerk_per_object": jerk_per_obj},
                          status="proxy")

    # ── R21 物体恒存 v4: 连续模式 + 三阶段时间分镜模式 ──
    @staticmethod
    def _storyboard_phase_indices(storyboard: dict, n_frames: int) -> list[np.ndarray]:
        total = float(storyboard.get("total_seconds", 10.0) or 10.0)
        groups = []
        for phase in storyboard.get("phases") or []:
            start = int(np.floor(float(phase.get("start", 0.0)) / total * n_frames))
            end = int(np.ceil(float(phase.get("end", 0.0)) / total * n_frames))
            start = int(np.clip(start, 0, max(n_frames - 1, 0)))
            end = int(np.clip(end, start + 1, n_frames))
            groups.append(np.arange(start, end, dtype=int))
        return groups

    def _eval_storyboard_persistence(self, seg, storyboard: dict,
                                     frame_shape: tuple[int, int, int]) -> EvalResult:
        T, H, W = frame_shape
        if storyboard.get("mode") != "temporal_occlusion":
            return EvalResult.skipped(
                "R21-物体恒存", "分镜mode不是temporal_occlusion")
        groups = self._storyboard_phase_indices(storyboard, T)
        if len(groups) != 3:
            return EvalResult.skipped(
                "R21-物体恒存", f"时间分镜需要3阶段，实际{len(groups)}")
        masks = seg.masks
        if masks is None or masks.ndim != 4 or masks.shape[0] == 0:
            return EvalResult.skipped("R21-物体恒存", "分镜模式未获得目标mask轨迹")

        min_area = max(10, int(H * W * 0.0005))
        areas = masks.sum(axis=(2, 3))
        presence = areas > min_area
        initial, hidden, post = groups
        initial_ratios = presence[:, initial].mean(axis=1)
        initial_areas = areas[:, initial].mean(axis=1)
        target = int(max(range(len(initial_ratios)),
                         key=lambda i: (float(initial_ratios[i]), float(initial_areas[i]))))
        initial_visible = float(presence[target, initial].mean())
        hidden_visible = float(presence[target, hidden].mean())
        post_same = float(presence[target, post].mean())
        post_any = float(np.any(presence[:, post], axis=0).mean())
        hidden_absent = 1.0 - hidden_visible
        post_score = max(post_same, 0.6 * post_any)
        identity_score = post_same
        combined = (0.30 * initial_visible + 0.25 * hidden_absent
                    + 0.30 * post_score + 0.15 * identity_score)
        score = int(np.clip(round(1.0 + 4.0 * combined), 1, 5))

        violations = []
        if initial_visible < 0.5:
            violations.append(f"初始可见阶段目标召回不足={initial_visible:.0%}")
        if hidden_absent < 0.5:
            violations.append(f"遮挡阶段目标仍大面积可见={hidden_visible:.0%}")
        if post_any < 0.3:
            violations.append(f"远侧观察阶段未重新检测到目标={post_any:.0%}")
        elif post_same < 0.3:
            violations.append(f"远侧检测存在但原跟踪ID未恢复={post_same:.0%}")

        return EvalResult(
            "R21-物体恒存", score, violations,
            metrics={
                "mode": "storyboard_object_persistence",
                "target_track_index": target,
                "phase_frame_indices": [group.tolist() for group in groups],
                "initial_visible_ratio": round(initial_visible, 4),
                "occluded_visible_ratio": round(hidden_visible, 4),
                "occluded_absence_ratio": round(hidden_absent, 4),
                "post_same_track_ratio": round(post_same, 4),
                "post_any_target_ratio": round(post_any, 4),
                "min_mask_area": min_area,
                "storyboard": storyboard,
            },
            status="proxy")

    def _eval_continuous_occlusion(self, seg,
                                   frame_shape: tuple[int, int, int]) -> EvalResult:
        """无显式分镜的R21遮挡序列：自动寻找中段不可见区间和后段重检。"""
        T, H, W = frame_shape
        masks = seg.masks
        if masks is None or masks.ndim != 4 or masks.shape[0] == 0:
            return EvalResult.skipped("R21-物体恒存", "连续遮挡模式未获得目标mask轨迹")
        min_area = max(10, int(H * W * 0.0005))
        areas = masks.sum(axis=(2, 3))
        presence = areas > min_area
        first_end = max(1, int(round(T * 0.25)))
        last_start = min(T - 1, int(round(T * 0.75)))
        initial_ratios = presence[:, :first_end].mean(axis=1)
        initial_areas = areas[:, :first_end].mean(axis=1)
        target = int(max(range(len(initial_ratios)),
                         key=lambda i: (float(initial_ratios[i]), float(initial_areas[i]))))
        timeline = presence[target]
        search_start = max(1, int(round(T * 0.15)))
        search_end = min(T - 1, int(round(T * 0.85)))
        runs = []
        run_start = None
        for t in range(search_start, search_end):
            if not timeline[t] and run_start is None:
                run_start = t
            elif timeline[t] and run_start is not None:
                runs.append((run_start, t))
                run_start = None
        if run_start is not None:
            runs.append((run_start, search_end))
        min_run = max(2, int(round(T * 0.08)))
        runs = [run for run in runs if run[1] - run[0] >= min_run]
        longest = max(runs, key=lambda run: run[1] - run[0]) if runs else None
        initial_visible = float(timeline[:first_end].mean())
        final_same = float(timeline[last_start:].mean())
        final_any = float(np.any(presence[:, last_start:], axis=0).mean())
        violations = []
        if longest is None:
            score = 3 if initial_visible >= 0.5 and final_same >= 0.5 else 2
            violations.append("未检测到持续的中段完全遮挡区间")
            gap = None
        else:
            start, end = longest
            before = float(timeline[:start].mean()) if start > 0 else 0.0
            after_same = float(timeline[end:].mean()) if end < T else 0.0
            after_any = float(np.any(presence[:, end:], axis=0).mean()) if end < T else 0.0
            post = max(after_same, 0.6 * after_any)
            combined = 0.35 * before + 0.25 + 0.25 * post + 0.15 * after_same
            score = int(np.clip(round(1.0 + 4.0 * combined), 1, 5))
            gap = [int(start), int(end - 1)]
            if before < 0.5:
                violations.append(f"遮挡前目标可见率不足={before:.0%}")
            if after_any < 0.3:
                violations.append(f"遮挡后未重新检测到目标={after_any:.0%}")
            elif after_same < 0.3:
                violations.append(f"遮挡后检测存在但原跟踪ID未恢复={after_same:.0%}")

        return EvalResult(
            "R21-物体恒存", score, violations,
            metrics={
                "mode": "continuous_occlusion_proxy",
                "target_track_index": target,
                "initial_visible_ratio": round(initial_visible, 4),
                "final_same_track_ratio": round(final_same, 4),
                "final_any_target_ratio": round(final_any, 4),
                "detected_occlusion_gap": gap,
                "min_mask_area": min_area,
            },
            status="proxy")

    # v3 通用连续模式保留: ID/计数异常检测 + 帧级定位
    def _eval_object_persistence(self, frames, concepts=None,
                                 storyboard: Optional[dict] = None,
                                 evaluation_method: str = "default",
                                 prompt: str = "") -> EvalResult:
        T, H, W = frames.shape[:3]
        try:
            # ── 多帧候选找初始物体框, 再喂给视频追踪 ──
            # 首帧物体可能在画面边缘/被遮挡/太小, 单帧检测易漏。
            # 改为在前后多帧中选「检测到物体最清晰的一帧」做追踪锚点。
            from .scale_pose import ScalePoseEvaluator
            # 注意: SAM3 视频追踪的 box 初始 prompt 只允许一个 box
            # (sam3_video_inference.add_prompt 硬断言 size(0)==1), 不能传多框。
            # 用 single=True 返回面积最大的 1 个框; 多物体靠 text prompt 补充。
            best_idx, initial_boxes = ScalePoseEvaluator._initial_boxes_from_multiframe(
                self.sam3, frames, dimension="R21", concepts=concepts)
            seg = self.sam3.segment_video(frames, dimension="R21",
                                          concepts=concepts,
                                          initial_boxes=initial_boxes,
                                          prompt_frame_index=(best_idx or 0))
        except Exception as e:
            return EvalResult.error("R21-物体恒存", f"SAM3 视频追踪失败: {e}")
        if evaluation_method == "storyboard_object_persistence":
            return self._eval_storyboard_persistence(
                seg, storyboard or {}, (T, H, W))
        if evaluation_method == "continuous_object_persistence":
            return self._eval_continuous_occlusion(seg, (T, H, W))
        tids = seg.tracking_ids
        if tids is None or tids.size == 0:
            # VLM 二次确认: 目标物体未生成/不可见 → unobservable; 可见 → skipped
            obs, obs_reason = self.vlm_observable(
                frames[len(frames) // 2], prompt)
            if obs is False:
                return EvalResult.unobservable(
                    "R21-物体恒存",
                    f"SAM3 视频追踪无追踪ID, VLM 确认目标物体未生成/不可见: "
                    f"{obs_reason}")
            if obs is True:
                return EvalResult.skipped(
                    "R21-物体恒存",
                    f"无法获取追踪ID (VLM判断物体可见, 追踪能力不足: {obs_reason})")
            # obs is None: VLM 不可用, 无法二次确认
            return EvalResult.skipped(
                "R21-物体恒存", "无法获取追踪ID (VLM不可用, 无法二次确认)")

        N = tids.shape[0]  # 总 track 数
        v = []
        locations = {}     # 结构化定位信息

        # ── 1. ID 跳变: 同一 track 出现多个不同有效 ID ──
        id_switches = 0
        id_switch_tracks = []
        for i in range(N):
            uniq = np.unique(tids[i][tids[i] >= 0])
            if len(uniq) > 1:
                id_switches += 1
                # 记录第一次ID跳变的帧位置
                switch_frames = []
                prev_id = tids[i][tids[i] >= 0][0]
                for t in range(1, T):
                    if tids[i, t] >= 0 and tids[i, t] != prev_id and tids[i, t - 1] >= 0:
                        switch_frames.append(t)
                    if tids[i, t] >= 0:
                        prev_id = tids[i, t]
                id_switch_tracks.append({"track": i, "switch_at_frames": switch_frames[:5]})

        locations["id_switch_tracks"] = id_switch_tracks

        # ── 2. 物体计数分析 ──
        counts = None
        n_count_violations = 0

        if seg.masks is not None and seg.masks.ndim == 4:
            counts = np.sum(seg.masks.sum(axis=(2, 3)) > 10, axis=0)  # [T]
            counts = counts.astype(int)
            locations["count_per_frame"] = counts.tolist()

            # 2a: 任何相邻帧物体数差异 >= 2 (突变检测)
            diffs = np.abs(np.diff(counts.astype(float)))
            abrupt_jumps = np.where(diffs >= 2)[0]
            if len(abrupt_jumps) > 0:
                n_count_violations += 1
                jump_info = [(int(f), int(counts[f]), int(counts[f + 1]))
                             for f in abrupt_jumps[:5]]
                v.append(f"物体数突变: 帧{jump_info} (相邻帧差异≥2)")
                locations["count_abrupt_jumps"] = jump_info

            # 2b: 持续的小幅度计数变化
            count_mode = int(np.bincount(np.clip(counts, 0, 20)).argmax())
            deviant_frames = int(np.sum(counts != count_mode))
            deviant_ratio = deviant_frames / max(T, 1)
            if deviant_ratio > 0.1 and count_mode > 0:
                # 寻找持续性偏差段
                run_ranges = []  # [(start, end, length), ...]
                current_start = None
                for t in range(T):
                    if counts[t] != count_mode:
                        if current_start is None:
                            current_start = t
                    else:
                        if current_start is not None:
                            run_len = t - current_start
                            if run_len >= 3:
                                run_ranges.append((int(current_start), t - 1, run_len))
                            current_start = None
                if current_start is not None:
                    run_len = T - current_start
                    if run_len >= 3:
                        run_ranges.append((int(current_start), T - 1, run_len))

                if run_ranges:
                    n_count_violations += 1
                    range_str = ", ".join(f"帧{s}-{e}({l}帧)" for s, e, l in run_ranges[:3])
                    v.append(f"物体数偏离mode({count_mode}): {range_str}")
                locations["count_deviations"] = run_ranges[:5]

            # 2c: 物体数永久性变化
            if T >= 4:
                half = T // 2
                med_first = float(np.median(counts[:half]))
                med_second = float(np.median(counts[half:]))
                if abs(med_first - med_second) >= 1:
                    n_count_violations += 1
                    v.append(f"物体数永久变化: {int(med_first)}(帧0-{half-1}) → "
                             f"{int(med_second)}(帧{half}-{T-1})")

            # 2d: 滑动窗口最大-最小差异
            if T >= 6:
                window = min(5, T // 2)
                max_diffs = []
                for t in range(T - window + 1):
                    w = counts[t:t + window]
                    max_diffs.append(int(np.max(w)) - int(np.min(w)))
                max_diff_val = max(max_diffs, default=0)
                if max_diff_val >= 2:
                    n_count_violations += 1
                    if all(d < 2 for d in diffs):
                        worst_t = int(np.argmax(max_diffs))
                        v.append(f"滑动窗口内物体数波动≥2 "
                                 f"(最差窗口帧{worst_t}-{worst_t+window}, "
                                 f"max-min={max_diff_val})")

            # 2e: ID 总数变化
            active_per_frame = np.sum(tids >= 0, axis=0)
            locations["active_per_frame"] = active_per_frame.tolist()
            if T >= 4:
                med_active_first = float(np.median(active_per_frame[:half]))
                med_active_second = float(np.median(active_per_frame[half:]))
                if abs(med_active_first - med_active_second) >= 1:
                    v.append(f"活跃track数变化: "
                             f"{int(med_active_first)}(帧0-{half-1}) → "
                             f"{int(med_active_second)}(帧{half}-{T-1})")
                    n_count_violations += 1
        else:
            counts = np.array([N] * T)

        if id_switches > 0:
            track_list = [t["track"] for t in id_switch_tracks[:3]]
            v.append(f"{id_switches}/{N}个物体ID跳变 (track索引: {track_list})")

        # ── 打分 ──
        total_violations = (1 if id_switches > 0 else 0) + n_count_violations
        score = 5 if total_violations == 0 else max(1, 5 - total_violations)

        metrics = {"id_switches": id_switches,
                   "n_tracked": int(N),
                   "total_frames": T,
                   "count_mode": int(np.bincount(
                       np.clip(counts.astype(int), 0, 20)).argmax()
                   ) if counts is not None else -1,
                   "count_violations": n_count_violations}
        # 合并结构化定位信息 (仅存储可序列化的基本数据,避免嵌套过深)
        if locations:
            metrics["locations"] = locations
        return EvalResult("R21-物体恒存", score, v,
                          metrics=metrics, status="proxy")
