"""
L3 物理层 — 标度位姿组评测器
R04 绝对尺度 (DA3 + SAM3 + Qwen3-VL) + R09 相对尺度 (VGGT + SAM3) + R10 三维位姿 (Any6D + VGGT + Qwen3-VL)

评测思路 (来自 PHYSICS_TAXONOMY.md):
- R04: CI2V 直接米制误差 vs GT; T2V/I2V DA3 尺度先验 + Qwen3-VL 合理区间判断
- R09: 跨帧包围盒变异系数 CV; 物体尺寸比误差; 无GT VLM 合理性
- R10: Sim(3)对齐后 ATE + RPE; T2V/I2V prompt 指定已知位姿变化 → VLM 比对最终误差
"""
import re
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult
from ..thresholds import score_metric, score_lower_better

# R04 物体验证: 常见物理对象名词 (用于从 prompt 提取关键物体词)。
# ⚠️ 覆盖物理 prompt 库 (physics_taxonomy_prompts.yaml) 中反复出现的可分割实体名词,
#    否则 R04/R06/R07 等维度的物体验证会漏掉真正的物理物体 (如 sand/bucket/steel/clay),
#    反而把 hybrid prompt 里的 IF 元素 (如 gate/star marker) 误当成物理物体。
_PHYSICAL_OBJECT_PATTERNS = re.compile(
    r'\b(person|man|woman|child|ball|cup|mug|bottle|chair|table|desk|door|'
    r'window|car|bike|bicycle|book|phone|laptop|box|bag|shoe|hat|clock|'
    r'plant|tree|flower|dog|cat|bird|fish|ruler|pencil|pen|knife|fork|'
    r'spoon|plate|bowl|glass|pillow|blanket|towel|key|watch|ring|candle|'
    r'lamp|umbrella|backpack|helmet|glove|scarf|belt|wallet|brush|comb|'
    r'mirror|frame|vase|statue|toy|block|balloon|wheel|pole|stick|rod|'
    r'building|house|wall|floor|ceiling|stair|bridge|road|path|fence|'
    r'gate|doorway|hallway|corridor|room|kitchen|bathroom|bedroom|'
    # ── 物理 prompt 补充: 材料/流体/颗粒 ──
    r'sand|bucket|water|oil|steel|clay|wood|wooden|marble|napkin|cloth|'
    r'aluminum|paper|paperclip|foam|cardboard|brass|rubber|ice|chocolate|'
    # ── 物理 prompt 补充: 物体/器械 ──
    r'apple|fruit|truck|train|carriage|bust|drone|beanbag|cannon|cradle|'
    r'puck|ramp|plank|pot|bob|horseshoe|magnet|generator|stove|grill|'
    r'ember|meat|sphere|cone|cylinder|cube|coin|hammer|nail|sofa|couch|'
    r'tennis|soccer|basketball|bowling|skater|koi|sheet|spring|string|'
    r'blade|spoke|bench|'
    # ── 补充: Any6D mesh 生成(R10/R17) 等场景的物体名词 ──
    r'drill|chuck|satellite|pendulum|excavator|arch|polyhedron|lattice|'
    r'manipulator|solar|panel|package)\b',
    re.IGNORECASE
)


class ScalePoseEvaluator(BaseEvaluator):
    """标度位姿组评测器（R04 / R09 / R10）"""

    def __init__(self, da3, sam3, vggt, any6d, vlm=None, config: Optional[dict] = None):
        super().__init__("L3-Physics/ScalePose", config)
        self.da3 = da3
        self.sam3 = sam3
        self.vggt = vggt
        self.any6d = any6d
        self.vlm = vlm

    # ── Prompt 关键词提取 ──
    @staticmethod
    def _extract_prompt_nouns(prompt: str) -> set[str]:
        """从 prompt 文本中提取关键物理对象名词 (小写集合)。"""
        if not prompt:
            return set()
        found = _PHYSICAL_OBJECT_PATTERNS.findall(prompt.lower())
        return set(found)

    @staticmethod
    def _name_matches_prompt(class_name: str, prompt_nouns: set[str]) -> bool:
        """判断 SAM3 检测到的类别名是否与 prompt 提到的物体相关。

        SAM3 的 class_name 可能是短语 (如 "coffee mug" / "door frame"),
        而 prompt_nouns 是单词 (如 "mug" / "door"), 需做词包含匹配而非精确相等。
        """
        if not class_name or not prompt_nouns:
            return True  # 无类别名或无 prompt 名词时不过滤
        name_low = class_name.lower().strip()
        if name_low in prompt_nouns:
            return True
        # 词包含: "coffee mug" 含 "mug"; "door frame" 含 "door"
        name_words = set(re.findall(r"[a-z]+", name_low))
        return bool(name_words & prompt_nouns)

    # ── R04 物体验证 ──
    @staticmethod
    def _verify_objects(seg_class_names: Optional[list[str]],
                        prompt_nouns: set[str]) -> dict:
        """验证 SAM3 分割到的物体是否与 prompt 描述的场景匹配。

        Returns:
            dict with keys:
                matched: list[str]  检测到且与prompt匹配的物体名
                unmatched: list[str] 检测到但与prompt无关的物体名
                missing: list[str]   prompt提到但未检测到的物体名
                has_match: bool      是否有任何匹配
                match_ratio: float   匹配率 (matched / prompt_nouns)
        """
        detected = set()
        if seg_class_names:
            detected = {n.lower().strip() for n in seg_class_names if n}

        # 词包含匹配: SAM3 类别可能是短语 ("coffee mug"), 需与 prompt 单词 ("mug") 匹配
        matched = set()
        unmatched = set()
        for name in detected:
            if ScalePoseEvaluator._name_matches_prompt(name, prompt_nouns):
                matched.add(name)
            else:
                unmatched.add(name)
        missing = set()
        for noun in prompt_nouns:
            if not any(noun in re.findall(r"[a-z]+", m.lower()) for m in matched):
                missing.add(noun)

        match_ratio = len(matched) / max(len(prompt_nouns), 1)
        return {
            "matched": sorted(matched),
            "unmatched": sorted(unmatched),
            "missing": sorted(missing),
            "has_match": len(matched) > 0,
            "match_ratio": round(match_ratio, 3),
        }

    # ── R04 VLM 合理区间判断 ──
    def _eval_size_reasonability_vlm(self, sizes: dict,
                                      prompt: str) -> tuple:
        """用 Qwen3-VL 判断 DA3 估计的米制尺寸是否在合理区间内。

        表格要求: "用 Qwen3-VL 判断是否在合理区间内"

        Returns:
            (score: int 1-5, violations: list[str])
        """
        sizes_str = "\n".join(
            f"  - {name}: 高度 {s['height_m']:.3f}m, 宽度 {s['width_m']:.3f}m"
            for name, s in sizes.items()
        )
        vlm_prompt = (
            f"你是物理尺寸评估专家。以下是视频生成模型根据 prompt 生成的视频中，"
            f"检测到的物体及其估计尺寸（基于度量深度模型的尺度先验）。\n\n"
            f"【原始 prompt】\"{prompt}\"\n\n"
            f"【DA3估计的物体米制尺寸】\n{sizes_str}\n\n"
            f"【任务】判断每个物体的估计尺寸是否在物理合理区间内。"
            f"考虑该物体在真实世界中应有的典型尺寸范围，"
            f"例如：网球≈0.06-0.07m，人体≈1.5-2.0m，杯子≈0.08-0.15m，"
            f"桌子≈0.7-1.2m高，汽车≈1.4-1.8m高/4-5m长，建筑>2m等。\n\n"
            f"【评分】\n"
            f"5: 全部物体尺寸均在合理范围内\n"
            f"4: 大部分合理，个别轻微偏差（±30%内）\n"
            f"3: 半数合理，部分明显偏差\n"
            f"2: 多数物体尺寸异常\n"
            f"1: 几乎所有物体尺寸严重不合理\n\n"
            f"返回 JSON: {{\"score\": <1-5整数>, "
            f"\"violations\": [\"每个违例物体的描述\"]}}"
        )
        try:
            if hasattr(self.vlm, "chat_json"):
                data = self.vlm.chat_json(
                    [], vlm_prompt, max_new_tokens=512)  # 纯文本, 不需要图像帧
            else:
                return None, ["VLM不支持chat_json接口"]
        except Exception as e:
            return None, [f"VLM调用失败: {e}"]

        if data.get("_parse_error"):
            return None, [f"VLM输出解析失败: {data.get('_raw', '')[:100]}"]

        raw_score = data.get("score")
        try:
            score = int(round(float(raw_score)))
            score = max(1, min(5, score))
        except (TypeError, ValueError):
            return None, [f"VLM未返回有效score: {raw_score}"]

        violations = data.get("violations", [])
        if isinstance(violations, str):
            violations = [violations] if violations.strip() else []
        return score, violations

    # ── R10 VLM 位姿变化比对 ──
    def _eval_pose_reasonability_vlm(self, frames, positions,
                                       rotations, prompt: str) -> tuple:
        """用 Qwen3-VL 判断估计的位姿变化是否与 prompt 描述一致。

        表格要求: "prompt 指定已知物体位姿变化，比对最终角度误差、
        位置误差、转轴误差 如 '立方体绕竖直轴恰好转 90°'"

        Returns:
            (score: int 1-5, violations: list[str])
        """
        # 提取位姿变化的数值描述
        pos_start = positions[0, 0] if positions.ndim >= 3 else positions[0]
        pos_end = positions[0, -1] if positions.ndim >= 3 else positions[-1]
        pos_disp = float(np.linalg.norm(pos_end - pos_start))

        vlm_prompt = (
            f"你是三维位姿评估专家。以下是视频生成模型根据 prompt 生成的视频，"
            f"以及从视频中估计的物体三维位姿变化。\n\n"
            f"【原始 prompt】\"{prompt}\"\n\n"
            f"【估计的位姿变化】\n"
            f"  - 位置位移量: {pos_disp:.3f} (归一化单位)\n"
            f"  - 总帧数: {positions.shape[min(1, positions.ndim - 1)]}\n\n"
            f"【任务】判断视频中物体的位姿变化是否符合 prompt 的描述。\n"
            f"请特别关注:\n"
            f"1. prompt 是否指定了特定角度旋转（如\"转90度\"）？实际是否大致完成？\n"
            f"2. prompt 是否指定了特定位移/位置关系？实际是否体现？\n"
            f"3. 视频中物体是否确实发生了预期的位姿变化（非静止）？\n\n"
            f"【评分】\n"
            f"5: 位姿变化完全符合 prompt 描述（角度/方向/幅度均正确）\n"
            f"4: 大致符合，有轻微偏差\n"
            f"3: 部分符合，但有明显方向或幅度错误\n"
            f"2: 仅勉强相关，大部分不符\n"
            f"1: 完全不符合（如prompt说旋转但物体静止；方向错误等）\n\n"
            f"返回 JSON: {{\"score\": <1-5整数>, "
            f"\"violations\": [\"位姿违例描述\"]}}"
        )

        # 用中间关键帧辅助 VLM 判断
        T = len(frames)
        key_indices = sorted({0, T // 2, T - 1})
        key_frames = [frames[i] for i in key_indices]

        try:
            if hasattr(self.vlm, "chat_json"):
                data = self.vlm.chat_json(
                    key_frames, vlm_prompt, max_new_tokens=512)
            else:
                return None, ["VLM不支持chat_json接口"]
        except Exception as e:
            return None, [f"VLM调用失败: {e}"]

        if data.get("_parse_error"):
            return None, [f"VLM输出解析失败: {data.get('_raw', '')[:100]}"]

        raw_score = data.get("score")
        try:
            score = int(round(float(raw_score)))
            score = max(1, min(5, score))
        except (TypeError, ValueError):
            return None, [f"VLM未返回有效score: {raw_score}"]

        violations = data.get("violations", [])
        if isinstance(violations, str):
            violations = [violations] if violations.strip() else []
        return score, violations

    def evaluate(self, frames: list[Image.Image],
                 metric_scale: Optional[float] = None,
                 gt_sizes: Optional[dict] = None,
                 gt_poses: Optional[np.ndarray] = None,
                 prompt: str = "",
                 any6d_depths: Optional[np.ndarray] = None,
                 any6d_intrinsics: Optional[np.ndarray] = None,
                 any6d_masks: Optional[np.ndarray] = None,
                 any6d_mesh=None,
                 target_dimensions: Optional[list[str]] = None
                 ) -> list[EvalResult]:
        frame_arrays = [np.array(f) for f in frames]
        key_frame = frame_arrays[len(frames) // 2]
        all_frames = np.stack(frame_arrays)
        prompt_nouns = self._extract_prompt_nouns(prompt)
        # 维度级按需执行: 只算 target_dimensions 声明的维度, 避免 R10(Any6D)等
        # 重型维度被白跑 (之前「模块级执行 + 最后过滤」导致 R10 在只标 R09 的
        # prompt 上也执行, 甚至因 Any6D 退化点云段错误崩溃)。
        want = set(target_dimensions) if target_dimensions else None
        def _need(prefix: str) -> bool:
            if want is None:
                return True
            return any(d.startswith(prefix) for d in want)
        results = []
        if _need("R04"):
            results.append(self._eval_absolute_scale(key_frame, gt_sizes, prompt_nouns, prompt))
        if _need("R09"):
            results.append(self._eval_relative_scale(all_frames, gt_sizes, prompt))
        if _need("R10"):
            results.append(self._eval_pose(
                all_frames, gt_poses, prompt,
                any6d_depths=any6d_depths,
                any6d_intrinsics=any6d_intrinsics,
                any6d_masks=any6d_masks,
                any6d_mesh=any6d_mesh,
            ))
        return results

    # ── R04 绝对尺度 ──
    def _eval_absolute_scale(self, frame, gt_sizes, prompt_nouns=None, prompt: str = "") -> EvalResult:
        try:
            depth_map = self.da3.estimate_metric_depth(frame).depth_map  # 米制
            # 从 prompt 筛选 R04 相关概念词(如 model car/calibration board),
            # 否则用 R04 完整概念词会召回不到这些物体 → 分割失败 → 无有效深度。
            concepts = self._prompt_concepts(prompt, "R04")
            seg = self.sam3.segment_frame(frame, dimension="R04",
                                          concepts=concepts)
        except Exception as e:
            return EvalResult.error("R04-绝对尺度", f"DA3/SAM3 推理失败: {e}")

        masks = seg.masks
        if masks is None or len(masks) == 0:
            # VLM 二次确认: 物体不可见(模型没生成) → unobservable; 可见 → skipped
            obs, obs_reason = self.vlm_observable(frame, prompt)
            if obs is False:
                return EvalResult.unobservable(
                    "R04-绝对尺度",
                    f"SAM3 未分割到物体, VLM 确认物体不可见/未生成: {obs_reason}")
            if obs is True:
                return EvalResult.skipped(
                    "R04-绝对尺度",
                    f"未分割到物体 (VLM判断物体可见, 检测失败: {obs_reason})")
            return EvalResult.skipped(
                "R04-绝对尺度", "未分割到物体 (VLM不可用, 无法二次确认)")
        bboxes = self.sam3.get_bounding_boxes(masks)  # [N,4] xyxy
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, -1)

        H, W = depth_map.shape
        focal = max(H, W)  # 粗略焦距 (无相机标定时的近似)
        sizes = {}
        for i, bbox in enumerate(bboxes):
            name = seg.class_names[i] if seg.class_names else f"obj_{i}"
            # ── 类别过滤: 若 prompt 明确提到了物体, 则只对 prompt 相关的物体评分 ──
            # 避免 SAM3 开放词汇分割误检的类别 (如把手误判为 person) 进入评分,
            # 导致 "prompt 里根本没有 person 却报 person 身高不合理" 的假阴性。
            if prompt_nouns and not self._name_matches_prompt(name, prompt_nouns):
                continue
            x0, y0, x1, y1 = map(int, bbox[:4])
            m = masks[i] > 0.5 if i < len(masks) else np.ones((H, W), bool)
            # ⚠️ SAM3 mask 是原始帧分辨率, DA3 深度图可能 resize 到别的尺寸
            #    (如 294×504), 两者不一致时 depth_map[m] 会索引错位 → 深度全无效
            #    → R04 报「无有效物体深度」skipped。须先把 mask resize 到深度图尺寸。
            if m.shape != depth_map.shape:
                import cv2
                m = cv2.resize(m.astype(np.float32),
                               (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
            obj_d = depth_map[m]
            obj_d = obj_d[np.isfinite(obj_d) & (obj_d > 0)]
            if len(obj_d) == 0:
                # ── 兜底: mask 内深度全无效时, 退到 bbox 区域取深度 ──
                # DA3 对细长机械臂/小模型车等物体的深度在精确 mask 内可能
                # 全为 0/NaN (深度估计空洞), 但 bbox 内(含周围背景上下文)
                # 往往有有效深度。用 bbox 中位数作为该物体的近似深度。
                y0c = max(0, y0); y1c = min(H, y1 + 1)
                x0c = max(0, x0); x1c = min(W, x1 + 1)
                if y1c > y0c and x1c > x0c:
                    bbox_d = depth_map[y0c:y1c, x0c:x1c]
                    obj_d = bbox_d[np.isfinite(bbox_d) & (bbox_d > 0)]
            if len(obj_d) == 0:
                continue
            med = float(np.median(obj_d))
            sizes[name] = {"height_m": (y1 - y0) * med / focal,
                           "width_m": (x1 - x0) * med / focal}

        if not sizes:
            # VLM 二次确认: 物体不可见 → unobservable; 可见但深度无效 → skipped
            obs, obs_reason = self.vlm_observable(frame, prompt)
            if obs is False:
                return EvalResult.unobservable(
                    "R04-绝对尺度",
                    f"DA3 在物体 mask 内无有效深度, VLM 确认物体不可见/未生成: {obs_reason}")
            if obs is True:
                return EvalResult.skipped(
                    "R04-绝对尺度",
                    f"无有效物体深度 (VLM判断物体可见, 深度估计失败: {obs_reason})")
            return EvalResult.skipped(
                "R04-绝对尺度", "无有效物体深度 (VLM不可用, 无法二次确认)")

        if gt_sizes is not None:
            errs = []
            for name, s in sizes.items():
                if name in gt_sizes:
                    errs.append(abs(s["height_m"] - gt_sizes[name].get("height_m", s["height_m"])))
                    errs.append(abs(s["width_m"] - gt_sizes[name].get("width_m", s["width_m"])))
            if errs:
                avg = float(np.mean(errs))
                score = score_metric("R04_size_err_m", avg)
                return EvalResult("R04-绝对尺度", score,
                                  [] if score >= 4 else [f"平均尺寸误差={avg:.3f}m"],
                                  metrics={"avg_error_m": avg, "sizes": sizes}, status="ok")

        # T2V/I2V: Qwen3-VL 合理区间判断 (替代硬阈值 0.01-20m)
        # 表格要求: "用 Qwen3-VL 判断是否在合理区间内"
        v = []
        vlm_score = None
        if self.vlm and getattr(self.vlm, "is_loaded", False):
            try:
                vlm_score, vlm_violations = self._eval_size_reasonability_vlm(
                    sizes, prompt if prompt else "")
                if vlm_violations:
                    v.extend(vlm_violations)
            except Exception as e:
                v.append(f"VLM合理性判断失败: {e}")
        if vlm_score is not None:
            score = vlm_score
        else:
            # VLM 不可用时的硬阈值兜底 (仅作为最后手段)
            for name, s in sizes.items():
                for dn, val in [("height", s["height_m"]), ("width", s["width_m"])]:
                    if val < 0.01:
                        v.append(f"{name} {dn}={val:.4f}m 过小")
                    elif val > 20:
                        v.append(f"{name} {dn}={val:.2f}m 过大")
            score = 5 if not v else max(1, 5 - len(v))

        # ── 物体验证: 检测到的物体是否匹配 prompt 描述 ──
        content_mismatch = False
        content_warnings = []
        verify_extra = {}
        if prompt_nouns:
            seg_names = seg.class_names if seg.class_names else None
            verify = self._verify_objects(seg_names, prompt_nouns)
            verify_extra = {
                "prompt_objects": sorted(prompt_nouns),
                "matched_objects": verify["matched"],
                "unmatched_objects": verify["unmatched"],
                "missing_objects": verify["missing"],
                "object_match_ratio": verify["match_ratio"],
            }
            # 若 prompt 提到了关键物体但 SAM3 一个都没检测到 → 内容不匹配
            if prompt_nouns and not verify["has_match"]:
                content_mismatch = True
                content_warnings.append(
                    f"prompt提及物体{prompt_nouns}, "
                    f"但SAM3仅检测到{verify['unmatched'] or '无'}"
                )
            # 若匹配率过低 → 大部分关键物体缺失
            elif verify["match_ratio"] < 0.3 and len(prompt_nouns) >= 2:
                content_mismatch = True
                content_warnings.append(
                    f"prompt物体匹配率仅{verify['match_ratio']:.0%}: "
                    f"匹配{verify['matched']}, 缺失{verify['missing']}"
                )

        metrics = {"estimated_sizes": sizes}
        metrics.update(verify_extra)

        return EvalResult("R04-绝对尺度", score, v,
                          metrics=metrics, status="proxy",
                          content_mismatch=content_mismatch,
                          content_warnings=content_warnings)

    @staticmethod
    def _prompt_concepts(prompt: str, dimension: str) -> Optional[list[str]]:
        """从样本prompt中筛选该维度真正出现的SAM3概念词。

        维度默认概念把多个互斥场景的词拼在一起（ball/balloon/person等），
        对开放词汇视频Detector会稀释召回。只传prompt中实际出现的具体词；
        若没有命中返回None，让SAM3使用维度默认词。
        """
        try:
            from models.sam3 import DIMENSION_CONCEPTS
        except Exception:
            return None
        low = (prompt or "").lower()
        concepts = DIMENSION_CONCEPTS.get(dimension, [])
        matched = []
        for concept in concepts:
            words = [w for w in re.findall(r"[a-z]+", concept.lower())
                     if len(w) > 2]
            # 允许常见复数/词形变化: "car" 匹配 "cars", "ring" 匹配 "rings" 等
            if words and all(
                    re.search(r"\b" + re.escape(w) + r"(?:s|es)?\b", low)
                    for w in words):
                matched.append(concept)
        return matched[:3] or None

    @staticmethod
    def _is_multi_object_scale(prompt: str) -> bool:
        """检测 R09 是否为「多物体比例」场景（目标②）。

        目标②的物理语义：多个物体一起运动时，它们之间的尺寸比例应保持
        稳定（真实尺寸不变）。触发词：two/identical/both/pair/rings/parallel
        等表示「多个同类物体」的词。
        """
        if not prompt:
            return False
        low = prompt.lower()
        multi_hints = (
            "two ", "identical", "both ", "a pair", "pair of",
            "interlocking rings", "two rings", "parallel lanes",
            "two identical", "both cars", "two cars",
        )
        return any(h in low for h in multi_hints)

    @staticmethod
    def _initial_boxes_from_frame(sam3, frame, dimension: str,
                                  concepts=None,
                                  single: bool = True) -> Optional[np.ndarray]:
        """用单帧分割找初始物体框, 转成归一化 [xmin,ymin,width,height] (0~1)。

        单帧分割含宽泛词兜底, 召回强于视频追踪的纯 text PCS, 用于给
        segment_video 的 box prompt 提供初始目标, 规避具体词召回差的问题。

        single=True (默认): SAM3 视频追踪的 box prompt 只允许**一个 box**,
        故只返回面积最大的主目标框 [1,4]。
        single=False: 返回全部有效框 [N,4] (供其他用途)。
        """
        try:
            seg = sam3.segment_frame(frame, dimension=dimension,
                                     concepts=concepts)
        except Exception:
            return None
        if seg.masks is None or len(seg.masks) == 0:
            return None
        bboxes = sam3.get_bounding_boxes(seg.masks)  # [N,4] xyxy (像素)
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, -1)
        H, W = frame.shape[:2]
        # xyxy (像素) → 归一化 xywh [xmin, ymin, width, height]
        norm = np.zeros((len(bboxes), 4), dtype=np.float32)
        norm[:, 0] = bboxes[:, 0] / W
        norm[:, 1] = bboxes[:, 1] / H
        norm[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / W
        norm[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / H
        # 截断到 [0,1] 防越界
        norm = np.clip(norm, 0.0, 1.0)
        # 过滤退化框 (宽或高为 0)
        valid = (norm[:, 2] > 0.005) & (norm[:, 3] > 0.005)
        norm = norm[valid]
        if len(norm) == 0:
            return None
        if single:
            # 只保留面积最大的主目标 (box prompt 限制单 box)
            areas = norm[:, 2] * norm[:, 3]
            return norm[np.argmax(areas)][None]  # [1,4]
        return norm

    @staticmethod
    def _initial_boxes_from_multiframe(sam3, frames, dimension: str,
                                       concepts=None,
                                       candidate_indices=None,
                                       single: bool = True):
        """多帧候选初始框: 在多个候选帧中分别单帧分割, 选「检测到物体状态最佳」
        的一帧作为视频追踪的检测锚点, 规避首帧物体未就位/太小/被遮挡导致的漏检。

        返回 (best_frame_index, boxes) 或 (None, None)。boxes 为归一化
        [xmin,ymin,width,height] (0~1), 与 segment_video 的 prompt_frame_index 配合使用。

        Args:
            frames: [T,H,W,3] numpy array
            candidate_indices: 候选帧索引列表; None 则取 [0, T//4, T//2, 3T//4, T-1]
                去重后的有序列表 (覆盖视频前后段, 覆盖物体不同状态)。
            single: 每帧只保留面积最大的主目标 (视频追踪 box prompt 限单 box)。
        """
        T = len(frames)
        if candidate_indices is None:
            # 覆盖前/中/后段的关键帧, 应对物体「后出现/由远及近/先遮挡后现身」
            candidate_indices = sorted(set(
                [0, max(0, T // 4), T // 2, max(0, 3 * T // 4), max(0, T - 1)]
            ))
        else:
            candidate_indices = sorted(set(
                i for i in candidate_indices if 0 <= i < T))

        best_idx = None
        best_boxes = None
        best_score = -1.0   # 用「主目标面积占比」衡量该帧检测质量

        for idx in candidate_indices:
            try:
                seg = sam3.segment_frame(frames[idx], dimension=dimension,
                                         concepts=concepts)
            except Exception:
                continue
            if seg.masks is None or len(seg.masks) == 0:
                continue
            bboxes = sam3.get_bounding_boxes(seg.masks)  # [N,4] xyxy 像素
            if bboxes.ndim == 1:
                bboxes = bboxes.reshape(1, -1)
            if bboxes.shape[0] == 0:
                continue
            H, W = frames[idx].shape[:2]
            # 归一化 xywh
            norm = np.zeros((len(bboxes), 4), dtype=np.float32)
            norm[:, 0] = bboxes[:, 0] / W
            norm[:, 1] = bboxes[:, 1] / H
            norm[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / W
            norm[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / H
            norm = np.clip(norm, 0.0, 1.0)
            # 过滤退化框 (放宽: 允许贴边物体, 只剔除宽高几乎为0的)
            valid = (norm[:, 2] > 0.01) & (norm[:, 3] > 0.01)
            norm = norm[valid]
            if len(norm) == 0:
                continue
            # 该帧的「检测质量分」= 最大物体面积占比 (物体越大越清晰越好追踪)
            areas = norm[:, 2] * norm[:, 3]
            score = float(areas.max())
            if single:
                frame_boxes = norm[int(np.argmax(areas))][None]
            else:
                frame_boxes = norm
            # 选检测质量最高的帧
            if score > best_score:
                best_score = score
                best_idx = idx
                best_boxes = frame_boxes

        if best_idx is None:
            return None, None
        print(f"[SAM3] 多帧候选: 在帧{best_idx}/{T}检测到主目标"
              f"(面积占比={best_score:.3f}), 候选帧={candidate_indices}")
        return best_idx, best_boxes

    # ── R09 相对尺度 ──
    def _eval_relative_scale(self, frames, gt_sizes, prompt: str = "") -> EvalResult:
        T = len(frames)
        multi_object = self._is_multi_object_scale(prompt)
        try:
            concepts = self._prompt_concepts(prompt, "R09")
            # ── 多帧候选找初始物体框, 再喂给视频追踪 ──
            # 首帧物体可能在画面远端(面积小)/边缘/被遮挡, 单帧检测易漏。
            # 改为在前后多帧中选「检测到物体最清晰的一帧」做追踪锚点,
            # 规避首帧锚点脆弱导致的「未检测到物体」。
            # ⚠️ SAM3 视频追踪 box 初始 prompt 只允许一个 box
            #    (sam3_video_inference 硬断言 size(0)==1)。
            #    目标②(多物体)不能用 box 追踪多物体 → 改走 text prompt(多概念词),
            #    让 SAM3 的 Detector 召回并追踪全部匹配物体。
            if multi_object:
                # 多物体: 不传 box, 用 text prompt 追踪全部匹配物体
                seg = self.sam3.segment_video(frames, dimension="R09",
                                              concepts=concepts,
                                              initial_boxes=None,
                                              prompt_frame_index=0)
            else:
                best_idx, initial_boxes = self._initial_boxes_from_multiframe(
                    self.sam3, frames, dimension="R09", concepts=concepts,
                    single=True)
                seg = self.sam3.segment_video(frames, dimension="R09",
                                              concepts=concepts,
                                              initial_boxes=initial_boxes,
                                              prompt_frame_index=(best_idx or 0))
        except Exception as e:
            return EvalResult.error("R09-相对尺度", f"SAM3 视频追踪失败: {e}")

        masks = seg.masks  # [N_objects, T, H, W]
        tids = seg.tracking_ids  # [N_objects, T]
        if masks is None or masks.ndim != 4 or masks.shape[0] == 0:
            # prompt 声明了运动/远近变化但追踪不到物体 → 先关键词预筛, 再 VLM 确认
            mov_kw = ["travel", "approach", "move", "walk", "roll",
                      "drift", "fly", "rise", "descend", "recede",
                      "toward", "away", "closer", "farther",
                      "运动", "走近", "远离", "由远及近"]
            low = (prompt or "").lower()
            if any(kw in low for kw in mov_kw):
                obs, obs_reason = self.vlm_observable(
                    frames[len(frames) // 2], prompt)
                if obs is False:
                    return EvalResult.unobservable(
                        "R09-相对尺度",
                        f"SAM3 未追踪到运动物体, VLM 确认物体未生成/不可见: "
                        f"{obs_reason}")
            return EvalResult.skipped("R09-相对尺度", "未检测到物体")

        # get_bounding_boxes 支持 [N, T, H, W] → [N, T, 4]
        bboxes = self.sam3.get_bounding_boxes(masks)  # [N, T, 4] xyxy
        if bboxes.ndim == 3 and bboxes.shape[1] == 1 and bboxes.shape[0] > 1:
            bboxes = bboxes[:, 0, :]  # 兼容 [N, 1, 4] 降维

        # ── 过滤微小跟踪碎片 (SAM3 过分割产生的噪声 track) ──
        # 用「峰值面积」而非「平均面积」判断: R09 场景物体常「由远及近」,
        # 远端帧面积极小会拉低平均值, 导致整条 track 被误过滤。
        # ⚠️ 阈值 0.5% (原 2%): R09 任务设计就是远景小物体 (走廊尽头小球/
        #    升空氦气球/远处玩具车), 832×480 下 2%≈8000px² 永远达不到 →
        #    52% skipped。0.5%(~2000px²) 仍高于噪声碎片量级(<0.2%)。
        N = masks.shape[0]
        H_img, W_img = frames[0].shape[:2]
        min_area = int(H_img * W_img * 0.005)
        valid_objects = []
        for j in range(N):
            valid_t = (np.where(tids[j] >= 0)[0] if tids is not None
                       else np.arange(T))
            if len(valid_t) < 3:
                continue
            areas = []
            for t in valid_t:
                if t < bboxes.shape[1]:
                    b = bboxes[j, t]
                    w, h = b[2] - b[0], b[3] - b[1]
                    if w > 1 and h > 1:
                        areas.append(w * h)
            if not areas:
                continue
            peak_area = float(np.max(areas))   # 峰值面积 (近端最大)
            if peak_area >= min_area:
                valid_objects.append((j, peak_area, valid_t))

        if not valid_objects:
            # 关键词预筛 + VLM 确认
            mov_kw = ["travel", "approach", "move", "walk", "roll",
                      "drift", "fly", "rise", "toward", "away",
                      "运动", "走近", "远离", "由远及近"]
            low = (prompt or "").lower()
            if any(kw in low for kw in mov_kw):
                obs, obs_reason = self.vlm_observable(
                    frames[len(frames) // 2], prompt)
                if obs is False:
                    return EvalResult.unobservable(
                        "R09-相对尺度",
                        f"无有效运动物体, VLM 确认物体未生成/不可见: {obs_reason}")
            return EvalResult.skipped(
                "R09-相对尺度",
                "无有效物体(峰值面积<0.5%图像或跟踪帧数<3)")

        # 按面积排序, 取前 5 个 (最多)
        valid_objects.sort(key=lambda x: x[1], reverse=True)
        max_objects = min(5, len(valid_objects))
        top_objects = valid_objects[:max_objects]

        # ── 目标②: 多物体「尺寸比例」稳定性 ──
        # 物理语义: 多个同类物体一起运动时, 它们的真实尺寸不变, 故两两
        # 之间的尺寸比(对角线比)应跨帧保持稳定。ratio CV 越小 = 比例越合理。
        if multi_object and len(top_objects) >= 2:
            ratio_cvs = []
            pairs = []
            for a in range(len(top_objects)):
                for b in range(a + 1, len(top_objects)):
                    ja, jb = top_objects[a][0], top_objects[b][0]
                    ta, tb = top_objects[a][2], top_objects[b][2]
                    common_t = sorted(set(ta) & set(tb))
                    if len(common_t) < 3:
                        continue
                    ratios = []
                    for t in common_t:
                        ba = bboxes[ja, t]
                        bb = bboxes[jb, t]
                        wa, ha = ba[2] - ba[0], ba[3] - ba[1]
                        wb, hb = bb[2] - bb[0], bb[3] - bb[1]
                        da = np.sqrt(wa * wa + ha * ha)
                        db = np.sqrt(wb * wb + hb * hb)
                        if da > 1 and db > 1:
                            ratios.append(da / db)
                    if len(ratios) < 3:
                        continue
                    ratios = np.array(ratios)
                    if ratios.mean() > 1e-6:
                        ratio_cvs.append(
                            float(ratios.std() / (ratios.mean() + 1e-8)))
                        pairs.append((ja, jb))
            if ratio_cvs:
                cv_avg = float(np.mean(ratio_cvs))
                score = score_metric("R09_cv", cv_avg)
                return EvalResult(
                    "R09-相对尺度", score,
                    [] if score >= 4 else [f"尺寸比CV={cv_avg:.3f}"],
                    metrics={"cv_avg": round(cv_avg, 4),
                             "ratio_cvs": ratio_cvs,
                             "n_tracked": N, "n_kept": max_objects,
                             "object_pairs": pairs,
                             "method": "multi_object_ratio",
                             "video_tracking": True},
                    status="proxy")
            # 多物体但配对失败 → 回退单物体 CV

        # ── 目标①: 单物体跨帧尺寸 CV ──
        # 单物体场景用 single=True 追踪 1 个主目标; 取 max CV(等价该物体 CV)。
        # 多物体场景若目标②配对失败也回退到这里, 此时 max CV 取运动最显著物体。
        cvs = []
        for j, _, valid_t in top_objects:
            diags = []
            for t in valid_t:
                if t >= bboxes.shape[1]:
                    continue
                b = bboxes[j, t]
                w, h = b[2] - b[0], b[3] - b[1]
                if w > 1 and h > 1:
                    diags.append(np.sqrt(w * w + h * h))
            if len(diags) < 2:
                continue
            diags = np.array(diags)
            if diags.mean() > 1e-6:
                cvs.append(float(diags.std() / (diags.mean() + 1e-8)))

        if not cvs:
            return EvalResult.skipped("R09-相对尺度", "无法计算尺寸CV")
        # 取运动最显著(尺寸 CV 最大)的物体, 而非所有物体平均
        cv_sel = float(max(cvs))
        score = score_metric("R09_cv", cv_sel)
        return EvalResult("R09-相对尺度", score,
                          [] if score >= 4 else [f"跨帧CV={cv_sel:.3f}"],
                          metrics={"cv_avg": round(cv_sel, 4), "cvs": cvs,
                                   "n_tracked": N, "n_kept": max_objects,
                                   "method": "single_object_cv",
                                   "video_tracking": True},
                          status="proxy")

    # ── R10 三维位姿 ──
    def _eval_pose(self, frames, gt_poses, prompt: str = "",
                   any6d_depths=None, any6d_intrinsics=None,
                   any6d_masks=None, any6d_mesh=None) -> EvalResult:
        # 官方Any6D需要显式RGB-D/mask/K/mesh；缺任一输入时走VGGT降级。
        positions, rotations, status, source = None, None, "ok", "any6d"
        try:
            po = self.any6d.estimate_pose_sequence(
                frames,
                masks=any6d_masks,
                depths=any6d_depths,
                intrinsics=any6d_intrinsics,
                mesh=any6d_mesh,
            )
            positions, rotations = po.positions, po.rotations
        except Exception as e:
            # 降级: VGGT 相机外参 (相机位姿代替物体位姿, 表格定义为 degraded)
            print(f"[R10] Any6D 位姿估计失败, 降级 VGGT: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            try:
                extr = self.vggt.estimate_camera_poses(frames)  # [T,3,4]
                positions = extr[:, :, 3][None]  # [1,T,3]
                rotations = extr[:, :3, :3][None]  # [1,T,3,3]
                status, source = "degraded", "vggt_camera"
            except Exception as e2:
                return EvalResult.error("R10-三维位姿", f"Any6D/VGGT 位姿失败: {e2}")

        if gt_poses is not None and status == "ok":
            gt = np.asarray(gt_poses)
            # 简化 ATE (位置)
            n = min(positions.shape[0], gt.shape[0]) if gt.ndim >= 2 else 0
            if n == 0:
                return EvalResult.skipped("R10-三维位姿", "GT位姿维度不匹配")
            ate = float(np.mean(np.linalg.norm(positions[:n] - gt[:n], axis=-1)))
            score = score_metric("R10_ate_m", ate)
            return EvalResult("R10-三维位姿", score,
                              [] if score >= 4 else [f"ATE={ate:.3f}m"],
                              metrics={"ATE": ate, "source": source}, status="ok")

        # T2V/I2V 无GT: prompt 指定已知位姿变化 → VLM 比对
        # 表格要求: "prompt 指定已知物体位姿变化，比对最终角度误差、位置误差、转轴误差"
        vlm_score = None
        vlm_violations = []
        if self.vlm and getattr(self.vlm, "is_loaded", False) and prompt:
            try:
                vlm_score, vlm_violations = self._eval_pose_reasonability_vlm(
                    frames, positions, rotations, prompt)
            except Exception as e:
                vlm_violations.append(f"VLM位姿判断失败: {e}")

        if vlm_score is not None:
            score = vlm_score
            status = "proxy"  # VLM 代理评测, 非 GT
        else:
            # VLM 不可用时: 帧间跳变作为底线兜底 (非表格推荐方案)
            if positions.shape[1] < 2:
                return EvalResult.skipped("R10-三维位姿", "帧数不足且VLM不可用")
            jumps = np.linalg.norm(np.diff(positions, axis=1), axis=-1)
            max_jump = float(jumps.max())
            score = score_lower_better(max_jump, [1.0, 2.0, 5.0, 10.0])
            status = "proxy" if status == "ok" else status

        return EvalResult("R10-三维位姿", score,
                          vlm_violations,
                          metrics={"source": source}, status=status)
