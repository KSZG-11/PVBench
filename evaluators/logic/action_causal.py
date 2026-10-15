"""
L1 逻辑层 — 动作因果效应评测器（全局 Benchmark）

评测方式: 反事实动作对 (Counterfactual Action Pairs)
- 对同一场景生成"施加动作"(action) 和"不施加动作"(control) 两个版本
- SpatialTracker 追踪两版本的物体3D轨迹
- 对比轨迹差异量化因果效应：位移增幅比、方向一致性、结果语义验证
- 无需外部物理仿真器
- 单视频降级模式：以首帧为静态基线近似评测
"""
from typing import Optional
import numpy as np
from PIL import Image
from ..base import BaseEvaluator, EvalResult, LayerResult


# ── 动作方向先验（VGGT坐标系近似） ──
_ACTION_DIRECTION_HINTS = {
    "push":   "远离施力者方向",
    "pull":   "靠近施力者方向",
    "drop":   "竖直向下",
    "kick":   "斜前方上升后下落",
    "throw":  "斜前方抛物线",
    "pour":   "竖直向下流动",
    "strike": "快速单方向",
    "cut":    "垂直于接触面的单方向",
    "roll":   "沿接触面滚动",
    "blow":   "远离吹气方向，轻质物体可能向上",
    "crumple": "向中心压缩/内聚",
}


class ActionCausalEvaluator(BaseEvaluator):
    """
    动作因果效应 全局 Benchmark

    反事实评测流程:
    1. SpatialTracker 追踪 action 视频中物体的3D轨迹
    2. 若有control视频：SpatialTracker 追踪 control 视频轨迹 → 对比因果效应比
    3. 若仅有action视频：以首帧为静态基线近似评测位移量
    4. VLM 验证视频结果是否符合预期的 action outcome
    5. 综合出分：位移量 + 因果增幅比 + 方向启发性 + 结果语义匹配
    """

    def __init__(self, vlm, spatial_tracker,
                 config: Optional[dict] = None):
        super().__init__("L1-Logic/ActionCausal", config)
        self.vlm = vlm
        self.spatial_tracker = spatial_tracker

    # ── 全局评测入口 ──

    def evaluate_global(self,
                        test_cases: list[dict],
                        videos: list
                        ) -> LayerResult:
        """
        全局评测：对全部反事实动作对聚合

        Args:
            test_cases: [{"id": "AC-001", "action_type": "push",
                          "expected_outcome": "...", ...}, ...]
            videos: 每项是 (action_frames, control_frames_or_None) 的二元组，
                    或 list[Image] 的单元组（单视频降级模式）
        """
        all_results = []
        scores = []

        for idx, (case, video_entry) in enumerate(zip(test_cases, videos)):
            if isinstance(video_entry, (list, tuple)) and len(video_entry) == 2:
                action_frames, control_frames = video_entry
            else:
                # 单视频降级: 整个 list 是 action_frames
                action_frames = video_entry if isinstance(video_entry, list) else [video_entry]
                control_frames = None

            if not action_frames:
                continue

            result = self._evaluate_case(idx, case, action_frames, control_frames)
            all_results.append(result)
            scores.append(result.score)

        mean_score = float(np.mean(scores)) if scores else 0.0
        return LayerResult(
            layer="L1-动作因果效应",
            results=all_results,
            aggregates={
                "mean_score": mean_score,
                "n_cases": len(all_results),
                "distribution": {
                    "1": scores.count(1), "2": scores.count(2),
                    "3": scores.count(3), "4": scores.count(4),
                    "5": scores.count(5),
                },
            },
            mean_scores={"AC-1_动作因果": mean_score},
        )

    def evaluate(self, frames: list[Image.Image],
                 prompt: str = "",
                 **kwargs) -> list[EvalResult]:
        """单案例评测（兼容旧接口：以首帧为静态基线）"""
        result = self._evaluate_case(0, {}, frames, control_frames=None)
        return [result]

    # ── 单案例评测核心 ──

    def _evaluate_case(self, idx: int, case: dict,
                       action_frames: list[Image.Image],
                       control_frames: Optional[list[Image.Image]] = None
                       ) -> EvalResult:
        """评测单个反事实动作对"""
        action_arrays = np.stack([np.array(f) for f in action_frames])

        violations: list[str] = []
        metrics: dict = {}

        # 1. SpatialTracker → action 视频轨迹
        try:
            act_traj = self.spatial_tracker.track_3d(action_arrays)
            act_pts = act_traj.trajectories  # [N, T, 3]
        except Exception as e:
            return EvalResult(f"AC-1-案例{idx}", 1,
                              [f"Action轨迹追踪失败: {e}"], status="error")

        if act_pts.shape[1] < 2:
            return EvalResult(f"AC-1-案例{idx}", 1,
                              ["帧数不足(<2)，无法计算位移"], status="skipped")

        # 各点首帧→末帧位移
        act_disp = act_pts[:, -1] - act_pts[:, 0]           # [N, 3]
        act_mag = np.linalg.norm(act_disp, axis=-1)          # [N]
        metrics["action_displacement_mean"] = float(np.mean(act_mag))
        metrics["action_displacement_max"] = float(np.max(act_mag))
        metrics["action_displacement_median"] = float(np.median(act_mag))
        metrics["n_tracked_points"] = int(act_pts.shape[0])

        # 2. 反事实对比（control 视频可用 → true counterfactual）
        has_control = control_frames is not None and len(control_frames) > 0
        causal_ratio = 1.0

        if has_control:
            try:
                ctrl_arrays = np.stack([np.array(f) for f in control_frames])
                ctrl_traj = self.spatial_tracker.track_3d(ctrl_arrays)
                ctrl_pts = ctrl_traj.trajectories                 # [M, T_c, 3]

                n_common = min(act_pts.shape[0], ctrl_pts.shape[0])
                T_c = min(act_pts.shape[1], ctrl_pts.shape[1])
                if T_c < 2 or n_common < 1:
                    raise ValueError("control视频帧数或点数不足")

                ctrl_disp = ctrl_pts[:n_common, -1] - ctrl_pts[:n_common, 0]
                ctrl_mag = np.linalg.norm(ctrl_disp, axis=-1)
                act_mag_aligned = act_mag[:n_common]

                ctrl_mean = float(np.mean(ctrl_mag)) + 1e-6
                causal_ratio = float(np.mean(act_mag_aligned) / ctrl_mean)

                metrics["causal_ratio"] = causal_ratio
                metrics["control_displacement_mean"] = float(np.mean(ctrl_mag))
                metrics["control_mode"] = "counterfactual_pair"

                # 有多少点 action 版本的位移显著大于 control（>1.5x）
                n_causal = int(np.sum(act_mag_aligned > ctrl_mag * 1.5))
                metrics["causal_points_ratio"] = float(n_causal / max(n_common, 1))

                if causal_ratio < 1.2:
                    violations.append(
                        f"因果效应弱: action位移约为control的{causal_ratio:.1f}倍(<1.2x)")

            except Exception as e:
                violations.append(f"Control轨迹对比失败: {str(e)[:80]}")
                metrics["control_error"] = str(e)[:100]
                metrics["control_mode"] = "fallback_to_single"
        else:
            metrics["control_mode"] = "single_video_baseline"

        # 3. 位移量 → 基础分 (control不可用时的主要信号)
        max_disp = metrics["action_displacement_max"]
        if max_disp < 0.5:
            base_score = 1          # 几乎无运动
        elif max_disp < 2.0:
            base_score = 2
        elif max_disp < 8.0:
            base_score = 3
        elif max_disp < 30.0:
            base_score = 4
        else:
            base_score = 5

        # 4. 反事实因果增幅 → 加分/扣分
        if has_control and "causal_ratio" in metrics:
            cr = metrics["causal_ratio"]
            if cr > 8.0:
                base_score = min(5, base_score + 2)    # 强因果
            elif cr > 4.0:
                base_score = min(5, base_score + 1)
            elif cr < 1.2:
                base_score = max(1, base_score - 1)    # 因果弱

        # 5. VLM 结果验证 (方案A单视频: expected_outcome=prompt, 作为主判据)
        outcome_score = self._verify_outcome_vlm(
            action_frames,
            case.get("expected_outcome", ""),
        )
        metrics["vlm_outcome_score"] = outcome_score
        if outcome_score >= 4:
            base_score = min(5, base_score + 1)
        elif outcome_score <= 2:
            base_score = max(1, base_score - 1)
            violations.append("VLM判定视频结果与预期outcome不匹配")

        # 6. 方向启发性检查
        action_type = case.get("action_type", "")
        direction_ok = self._check_direction_heuristic(
            act_pts, action_type
        )
        metrics["direction_heuristic_ok"] = direction_ok
        if not direction_ok and max_disp > 1.0:
            violations.append(
                f"运动主方向与'{action_type}'动作预期方向不一致")
            base_score = max(1, base_score - 1)

        # 7. 方案A(无action_type/无control): 位移量判分对微小动作不敏感,
        #    改用 VLM 动作因果判断为主判据 (避免全部误判为1分)。
        if not has_control and not action_type and outcome_score is not None \
                and outcome_score > 0:
            base_score = outcome_score

        score = max(1, min(5, base_score))
        return EvalResult(f"AC-1-案例{idx}", score, violations, metrics=metrics)

    # ── VLM 结果语义验证 ──

    def _verify_outcome_vlm(self, frames: list[Image.Image],
                            expected_outcome: str) -> int:
        """VLM 判断视频结果是否符合预期的动作结果"""
        if not expected_outcome:
            return 3
        try:
            sampled = self._sample_frames(frames, 4)
            prompt = f"""你是一个物理合理性评测专家。请观察视频帧，判断以下预期的动作结果是否在视频中正确呈现。

预期结果: {expected_outcome}

评分标准:
5: 视频完美呈现了预期结果，场景变化符合物理直觉
4: 基本符合预期，有微小偏差（如速度/时序略有不自然）
3: 部分符合，主要结果出现但细节错误
2: 严重偏离预期，关键结果未出现
1: 完全不符合，视频内容与预期结果无关

回复格式:
评分: X/5
原因: <一句话>"""
            response = self.vlm.chat(sampled, prompt, max_new_tokens=150)
            return self._extract_score(response)
        except Exception:
            return 3

    # ── 方向启发性检查 ──

    def _check_direction_heuristic(self,
                                    trajectories: np.ndarray,
                                    action_type: str) -> bool:
        """
        检查运动主方向是否与动作类型一致（简单启发式，不强依赖）。

        在 VGGT 坐标系近似下:
          y 轴 ≈ 竖直（重力方向）
          x/z 轴 ≈ 水平面
        """
        if not action_type or action_type not in _ACTION_DIRECTION_HINTS:
            return True  # 无先验，跳过

        disp = trajectories[:, -1] - trajectories[:, 0]   # [N, 3]
        mean_disp = np.mean(disp, axis=0)                   # [3]
        mag = np.linalg.norm(mean_disp)

        if mag < 0.3:
            return False  # 运动太微弱，无法判断

        # 归一化主方向
        d = mean_disp / mag   # [dx, dy, dz]
        dy, dh = abs(d[1]), np.sqrt(d[0]**2 + d[2]**2)  # 竖直分量 / 水平分量

        # 按动作类型检查主导方向
        if action_type in ("drop", "pour"):
            # 应主要为向下运动
            return d[1] < -0.25   # dy 显著为负（向下）
        elif action_type in ("blow",):
            # 轻质物体可能飘起，需要一定向上或水平分量
            return d[1] > -0.1    # 不应明显向下
        elif action_type == "crumple":
            # 应向内压缩 → 末期点间距小于初期
            init_spread = np.std(trajectories[:, 0, :], axis=0)
            final_spread = np.std(trajectories[:, -1, :], axis=0)
            return np.mean(final_spread) < np.mean(init_spread) * 1.05
        else:
            # push / kick / throw / strike / cut / roll / pull
            # 应有显著水平分量（不是纯竖直运动）
            return dh > 0.25

    # ── 工具方法 ──

    def _extract_score(self, text: str) -> int:
        import re
        for line in text.split("\n"):
            match = re.search(r"(\d+)\s*/\s*5", line)
            if match:
                return max(1, min(5, int(match.group(1))))
        return 3

    @staticmethod
    def _sample_frames(frames, n):
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]
