"""
L3 物理层 — 几何组评测器
R01 深度 (VGGT + SAM3) + R02 表面法线 (StableNormal + VGGT) + R03 形状 (VGGT)

评测思路 (依 PHYSICS_TAXONOMY.md 表格):
- R01: 有GT → AbsRel / δ₁δ₂δ₃ / RMSE; 无GT → SAM3分割物体→检验遮挡区域深度序正确率
- R02: 有GT → Mean/Median 角度误差; 无GT → 法线-深度梯度一致性
- R03: 有GT → Sim(3)对齐后 Chamfer-L1; 无GT → 跨帧形状自洽(Kabsch对齐+Chamfer)
"""
import re
import numpy as np
from typing import Optional
from PIL import Image

from ..base import BaseEvaluator, EvalResult
from ..thresholds import score_metric, score_lower_better


class GeometryEvaluator(BaseEvaluator):
    """几何组评测器（R01 深度 / R02 法线 / R03 形状）"""

    def __init__(self, vggt, stable_normal, sam3=None, vlm=None,
                 config: Optional[dict] = None):
        super().__init__("L3-Physics/Geometry", config)
        self.vggt = vggt
        self.stable_normal = stable_normal
        self.sam3 = sam3  # R01/R03 需要 SAM3 分割
        self.vlm = vlm    # R03 非标准几何体按表格使用 Qwen3-VL

    def evaluate(self, frames: list[Image.Image],
                 depth_gt: Optional[np.ndarray] = None,
                 normal_gt: Optional[np.ndarray] = None,
                 pointcloud_gt: Optional[np.ndarray] = None,
                 camera_poses: Optional[np.ndarray] = None,
                 prompt: str = "",
                 evaluator_context: Optional[dict] = None,
                 target_dimensions: Optional[list[str]] = None
                 ) -> list[EvalResult]:
        frame_arrays = np.stack([np.array(f) for f in frames])
        key_frame = frame_arrays[len(frames) // 2]

        # 维度级按需执行: 只算 target_dimensions 声明的维度
        want = set(target_dimensions) if target_dimensions else None
        def _need(prefix: str) -> bool:
            return want is None or any(d.startswith(prefix) for d in want)

        # 只调用一次 VGGT; 深度图复用给 R01 和 R02
        need_depth = _need("R01") or _need("R02")
        try:
            if need_depth:
                depth_output = self.vggt.estimate_depth(key_frame)
                depth_map = depth_output.depth_map
            else:
                depth_map = None
        except Exception as e:
            depth_map = None
            depth_err = EvalResult.error("R01-深度", f"VGGT 深度推理失败: {e}")
            normal_err = EvalResult.error("R02-表面法线", f"VGGT 深度推理失败(法线-深度一致性需要): {e}")
            results = []
            if _need("R01"):
                results.append(depth_err)
            if _need("R02"):
                results.append(normal_err)
            if _need("R03"):
                results.append(self._eval_shape(key_frame, frame_arrays, pointcloud_gt, prompt))
            return results

        results = []
        if _need("R01"):
            results.append(self._eval_depth(key_frame, depth_map, depth_gt, prompt,
                                            frames=frame_arrays))
        if _need("R02"):
            results.append(self._eval_normals(key_frame, depth_map, normal_gt, camera_poses))
        if _need("R03"):
            results.append(self._eval_shape(key_frame, frame_arrays, pointcloud_gt, prompt,
                                            evaluator_context=evaluator_context))
        return results

    @staticmethod
    def _prompt_concepts(prompt: str, dimension: str) -> Optional[list[str]]:
        """从当前样本prompt筛选真实出现的维度概念，避免跨场景概念污染。"""
        try:
            from models.sam3 import DIMENSION_CONCEPTS
        except Exception:
            return None
        low = (prompt or "").lower()
        selected = []
        for concept in DIMENSION_CONCEPTS.get(dimension, []):
            words = [w for w in re.findall(r"[a-z]+", concept.lower())
                     if len(w) > 2]
            if words and all(re.search(r"\b" + re.escape(w) + r"\b", low)
                             for w in words):
                selected.append(concept)
        return selected or None

    @staticmethod
    def _extract_depth_order(prompt: str) -> list[tuple[str, str]]:
        """从 prompt 提取「A before B / A in front of B / B behind A」前后关系对。

        返回 [(front_phrase, back_phrase), ...]，front 声明更靠前（更浅）。
        R01 深度序的正确判据应是「prompt 声明的前后关系」，而非强制物体投影重叠。

        短语边界：名词短语用「贪婪匹配连续单词，但遇关系词/介词/标点即截断」，
        避免吞入 "with separate contours" 之类修饰从句。
        """
        if not prompt:
            return []
        # 关系词/介词/标点边界（后面接这些就停止短语匹配）
        BOUND = (r"(?:before|behind|in\s+front\s+of|with|at|of|and|against|"
                 r"beside|on|,|\.|;|from|seconds)")
        # 名词短语: 冠词(可选) + 连续单词（非贪婪，遇边界词停止）
        NP = r"(?:(?:[AaNn]|[Tt]he)\s+)?[a-z][\w\-]*(?:\s+[a-z][\w\-]*)*?"
        pairs = []
        # 模式1: "X before Y" / "X directly before Y"
        for m in re.finditer(
                rf"({NP})\s+(?:directly\s+)?before\s+({NP})"
                rf"(?=\s*{BOUND}|\s*$)", prompt):
            front, back = m.group(1).strip(), m.group(2).strip()
            front = re.sub(r"^(a|an|the)\s+", "", front, flags=re.I)
            back = re.sub(r"^(a|an|the)\s+", "", back, flags=re.I)
            if front and back and front.lower() != back.lower():
                pairs.append((front, back))
        # 模式2: "X in front of Y"
        for m in re.finditer(
                rf"({NP})\s+in\s+front\s+of\s+({NP})"
                rf"(?=\s*{BOUND}|\s*$)", prompt):
            front, back = m.group(1).strip(), m.group(2).strip()
            front = re.sub(r"^(a|an|the)\s+", "", front, flags=re.I)
            back = re.sub(r"^(a|an|the)\s+", "", back, flags=re.I)
            if front and back and front.lower() != back.lower():
                pairs.append((front, back))
        # 模式3: "Y behind X"（等价 X before Y）
        for m in re.finditer(
                rf"({NP})\s+behind\s+({NP})"
                rf"(?=\s*{BOUND}|\s*$)", prompt):
            back, front = m.group(1).strip(), m.group(2).strip()
            front = re.sub(r"^(a|an|the)\s+", "", front, flags=re.I)
            back = re.sub(r"^(a|an|the)\s+", "", back, flags=re.I)
            if front and back and front.lower() != back.lower():
                pairs.append((front, back))
        # 模式4: "X is seen partly before Y" / "X partly before Y"
        # (被动/部分遮挡句式, hybrid 场景: "The principal target is seen
        #  partly before a large blue reference panel")
        for m in re.finditer(
                rf"({NP})\s+(?:is\s+|are\s+)?(?:seen\s+)?"
                rf"(?:partly|partially)\s+before\s+({NP})"
                rf"(?=\s*{BOUND}|\s*$)", prompt):
            front, back = m.group(1).strip(), m.group(2).strip()
            front = re.sub(r"^(a|an|the)\s+", "", front, flags=re.I)
            back = re.sub(r"^(a|an|the)\s+", "", back, flags=re.I)
            if front and back and front.lower() != back.lower():
                pairs.append((front, back))
        return pairs

    @staticmethod
    def _match_declared_objects(declared_pairs, obj_names, obj_depths):
        """把 prompt 声明的 (front, back) 短语匹配到 SAM3 分割出的物体。

        返回 [(front_name, back_name, front_depth, back_depth), ...]。
        匹配规则: 短语的每个实词都须出现在物体名中（忽略冠词/颜色词做模糊容错），
        且 front 与 back 须匹配到不同物体。
        """
        matched = []
        for front_phrase, back_phrase in declared_pairs:
            # 短语 → 实词集合（>2 字母，去掉颜色/冠词等弱词）
            def _words(phrase):
                stop = {"the", "a", "an", "with", "and", "of", "in", "at",
                        "large", "small", "matte", "pale", "dark", "bright"}
                return {w for w in re.findall(r"[a-z]+", phrase.lower())
                        if len(w) > 2 and w not in stop}

            fw, bw = _words(front_phrase), _words(back_phrase)
            if not fw or not bw:
                continue
            front_idx = back_idx = None
            for idx, name in enumerate(obj_names):
                nl = set(re.findall(r"[a-z]+", str(name).lower()))
                if front_idx is None and fw & nl:
                    front_idx = idx
                if back_idx is None and bw & nl and idx != front_idx:
                    back_idx = idx
            # 指代兜底: "the principal target / main object / primary subject"
            # 等指代性短语无法按词匹配时, 映射为面积最大的物体 (场景主目标)。
            _DEICTIC = {"principal", "target", "main", "subject", "primary",
                        "object", "central"}
            def _largest(exclude=None):
                best, best_area = None, 0
                for k, (_, _, m) in enumerate(obj_depths):
                    if exclude is not None and k == exclude:
                        continue
                    area = int(m.sum()) if m is not None else 0
                    if area > best_area:
                        best, best_area = k, area
                return best
            if front_idx is None and fw and fw <= _DEICTIC:
                front_idx = _largest()
            if back_idx is None and bw and bw <= _DEICTIC:
                back_idx = _largest(exclude=front_idx)
            if front_idx is None or back_idx is None:
                continue
            # obj_depths[i] = (mask_idx, median_depth, mask)
            front_depth = obj_depths[front_idx][1]
            back_depth = obj_depths[back_idx][1]
            matched.append((obj_names[front_idx], obj_names[back_idx],
                            front_depth, back_depth))
        return matched

    @staticmethod
    def _extract_spatial_relation(prompt: str) -> list[tuple[str, str, str]]:
        """从 prompt 提取「非前后」空间关系对（R01 无 before 时的降级判据）。

        返回 [(relation, obj_a_phrase, obj_b_phrase), ...]，relation ∈
        {"on", "beside", "against"}：
        - on:      A 支撑在 B 上（B 应更靠后/更下方，深度接近）
        - beside:  A 与 B 并置（两物体深度应接近，同一空间层）
        - against: A 靠在一个大背景/参照物前（A 应更靠前/更浅）
        """
        if not prompt:
            return []
        NP = r"(?:(?:[AaNn]|[Tt]he)\s+)?[a-z][\w\-]*(?:\s+[a-z][\w\-]*)*?"
        BOUND = (r"(?:on|beside|against|before|behind|with|at|of|and|"
                 r"in|,|\.|;|from|seconds)")
        rels = []
        # 模式: "A on B"
        for m in re.finditer(
                rf"({NP})\s+on\s+({NP})(?=\s*{BOUND}|\s*$)", prompt):
            a, b = m.group(1).strip(), m.group(2).strip()
            a = re.sub(r"^(a|an|the)\s+", "", a, flags=re.I)
            b = re.sub(r"^(a|an|the)\s+", "", b, flags=re.I)
            if a and b and a.lower() != b.lower():
                rels.append(("on", a, b))
        # 模式: "A beside B"
        for m in re.finditer(
                rf"({NP})\s+beside\s+({NP})(?=\s*{BOUND}|\s*$)", prompt):
            a, b = m.group(1).strip(), m.group(2).strip()
            a = re.sub(r"^(a|an|the)\s+", "", a, flags=re.I)
            b = re.sub(r"^(a|an|the)\s+", "", b, flags=re.I)
            if a and b and a.lower() != b.lower():
                rels.append(("beside", a, b))
        # 模式: "A against B"
        for m in re.finditer(
                rf"({NP})\s+against\s+({NP})(?=\s*{BOUND}|\s*$)", prompt):
            a, b = m.group(1).strip(), m.group(2).strip()
            a = re.sub(r"^(a|an|the)\s+", "", a, flags=re.I)
            b = re.sub(r"^(a|an|the)\s+", "", b, flags=re.I)
            if a and b and a.lower() != b.lower():
                rels.append(("against", a, b))
        return rels

    def _depth_unobservable_or_skipped(self, n_found: int, prompt: str,
                                       sam3_unavailable: bool,
                                       frame=None) -> EvalResult:
        """R01 分割物体不足时的结果判定。

        关键区分:
        - prompt 声明了「前后/空间关系」(before/on/beside/against), 但 SAM3
          只分割出 <2 个物体 → 用 VLM 看关键帧确认「物体是否真的被完全遮挡/
          不可见」→ 是则 unobservable(模型未满足可观测性约束), 否则 skipped。
        - prompt 未声明多物体关系, 或 SAM3 本身不可用 → 保持 skipped(评测器
          侧问题, 非模型问题)。
        """
        declared = (GeometryEvaluator._extract_depth_order(prompt)
                    or GeometryEvaluator._extract_spatial_relation(prompt))
        if declared and not sam3_unavailable:
            # VLM 二次确认 (有 frame 才做, 无 frame 直接判 unobservable)
            if frame is not None:
                obs, obs_reason = self.vlm_observable(frame, prompt)
                if obs is True:
                    # VLM 说物体可见 → 是 SAM3 检测能力不足, 保持 skipped
                    return EvalResult.skipped(
                        "R01-深度",
                        f"SAM3 分割物体数={n_found}(<2), 但 VLM 判断物体可见"
                        f"(检测能力不足): {obs_reason}")
            return EvalResult.unobservable(
                "R01-深度",
                f"prompt 声明了{len(declared)}个前后/空间关系对, 但SAM3仅分割出"
                f"{n_found}个物体(<2): 目标物体被完全遮挡/不可观测, "
                f"模型未满足「遮挡时保持轮廓可见」约束")
        return EvalResult.skipped(
            "R01-深度",
            f"SAM3 分割物体数={n_found}(<2), 无法检验深度序"
            + ("; SAM3 不可用" if sam3_unavailable else ""))

    # ── R01 深度 ──
    def _eval_depth(self, frame, depth_map, depth_gt, prompt: str = "",
                    frames=None) -> EvalResult:
        if depth_gt is not None:
            # ── CI2V: 定量误差 vs 仿真 GT ──
            gt = np.asarray(depth_gt)
            if gt.ndim == 3:
                gt = gt[len(gt) // 2]
            pred = depth_map
            if gt.shape != pred.shape:
                import cv2
                pred = cv2.resize(pred.astype(np.float32),
                                  (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            mask = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
            if mask.sum() < 100:
                return EvalResult.skipped("R01-深度", "有效GT像素不足")
            scale = np.median(gt[mask]) / (np.median(pred[mask]) + 1e-8)
            ps = pred * scale
            absrel = float(np.mean(np.abs(ps[mask] - gt[mask]) / (gt[mask] + 1e-8)))
            ratio = np.maximum(ps[mask] / (gt[mask] + 1e-8), gt[mask] / (ps[mask] + 1e-8))
            delta1 = float(np.mean(ratio < 1.25))
            delta2 = float(np.mean(ratio < 1.25 ** 2))
            delta3 = float(np.mean(ratio < 1.25 ** 3))
            rmse = float(np.sqrt(np.mean((ps[mask] - gt[mask]) ** 2)))
            score = score_metric("R01_absrel", absrel)
            return EvalResult("R01-深度", score,
                              [] if score >= 4 else [f"AbsRel={absrel:.3f}, δ₁={delta1:.3f}"],
                              metrics={"AbsRel": absrel, "delta1": delta1,
                                       "delta2": delta2, "delta3": delta3,
                                       "RMSE": rmse, "scale_alignment": "median"},
                              status="ok")

        # ── T2V/I2V proxy: 深度序正确率（表格方案：通过物体遮挡检查深度排序是否合理）──
        # 对 SAM3 分割的物体对（可见 mask 的 bbox 有投影重叠），检查 VGGT
        # 深度排序是否一致。可见实例 mask 本身通常互斥，不能要求 mask 直接相交。
        pred = depth_map
        sam3_seg = None
        if self.sam3 is not None:
            concepts = self._prompt_concepts(prompt, "R01")
            try:
                sam3_seg = self.sam3.segment_frame(
                    frame, dimension="R01", concepts=concepts)
            except Exception:
                sam3_seg = None

        masks = sam3_seg.masks if sam3_seg is not None else None
        if masks is None or len(masks) < 2:
            # ── 多帧重试: 前后遮挡场景里, 被遮挡物体(如 statue)在关键帧
            #    可能被前景物体(如旋转的 vase)完全遮住 → SAM3 只分割出 1 个物体。
            #    遍历其他采样帧, 找「能分割出 ≥2 个物体」的帧, 用该帧重跑 VGGT
            #    深度, 避免因单帧遮挡误 skip。
            if (frames is not None and len(frames) > 1
                    and self.sam3 is not None and self.vggt is not None):
                retried = False
                # 优先试首帧/末帧(物体通常错开), 再均匀扫中间帧
                try_order = [0, len(frames) - 1] + list(
                    range(1, len(frames) - 1, max(1, len(frames) // 8)))
                for idx in try_order:
                    if idx == len(frames) // 2:
                        continue  # 跳过已试过的关键帧
                    try:
                        seg2 = self.sam3.segment_frame(
                            frames[idx], dimension="R01", concepts=concepts)
                        if seg2.masks is not None and len(seg2.masks) >= 2:
                            depth2 = self.vggt.estimate_depth(
                                frames[idx]).depth_map
                            sam3_seg = seg2
                            pred = depth2
                            masks = seg2.masks
                            retried = True
                            print(f"[R01] 关键帧仅{len(masks) if masks is not None else 0}物体, "
                                  f"改在帧{idx}分割出{len(seg2.masks)}物体重试")
                            break
                    except Exception:
                        continue
                if not retried:
                    n_found = 0 if masks is None else len(masks)
                    return self._depth_unobservable_or_skipped(
                        n_found, prompt, self.sam3 is None, frame=frame)
            else:
                n_found = 0 if masks is None else len(masks)
                return self._depth_unobservable_or_skipped(
                    n_found, prompt, self.sam3 is None, frame=frame)

        # ── 遮挡深度序分析 ──
        # 表格方案: "prompt 指定前后遮挡关系，检验重建的深度序正确率"
        # 物理判据: 两物体掩码重叠区域的**可见像素属于前景物体**, 故该区域深度应
        # 显著更接近前景物体的深度分布, 而不是背景物体的。若 VGGT 在重叠区给出的
        # 深度反而更接近背景物体 → 深度序错误(把被遮挡物画到了前面)。
        N = len(masks)
        H, W = pred.shape
        obj_depths = []
        obj_names = []

        for i in range(N):
            m = masks[i] > 0.5
            # 将 mask 缩放到深度图分辨率
            if m.shape != pred.shape:
                import cv2
                m = cv2.resize(m.astype(np.float32),
                               (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
            valid = m & np.isfinite(pred) & (pred > 0)
            obj_d = pred[valid]
            if len(obj_d) < 50:
                continue
            obj_depths.append((i, float(np.median(obj_d)), m))
            name = sam3_seg.class_names[i] if sam3_seg.class_names else f"obj_{i}"
            obj_names.append(name)

        if len(obj_depths) < 2:
            return self._depth_unobservable_or_skipped(
                len(obj_depths), prompt, self.sam3 is None, frame=frame)

        # ── 主路径: prompt 声明的「前后关系」直接判深度序 ──
        # 深度序的本质判据是 prompt 声明 "A before B"(A 在前), 而不是强制物体
        # 投影重叠。只要 SAM3 分割出 A、B 两个物体, 就可用各自专属区中位深度
        # 直接比较: A 的深度应显著小于 B (A 更靠前/更浅)。
        # 这消除了「物体并排/无重叠」导致的误 skip。
        declared_pairs = self._extract_depth_order(prompt)
        if declared_pairs:
            matched = self._match_declared_objects(
                declared_pairs, obj_names, obj_depths)
            if matched:
                n_ok = 0
                n_bad = 0
                viol = []
                details = []
                for (front_name, back_name, front_d, back_d) in matched:
                    # front 应更浅(深度更小)
                    ok = front_d <= back_d
                    if ok:
                        n_ok += 1
                    else:
                        n_bad += 1
                        viol.append(
                            f"深度序矛盾: {front_name}(d={front_d:.3f})应在前,"
                            f"但{back_name}(d={back_d:.3f})深度反而更小")
                    details.append({
                        "front": front_name, "back": back_name,
                        "front_depth": round(front_d, 4),
                        "back_depth": round(back_d, 4),
                        "consistent": bool(ok),
                    })
                acc = n_ok / (n_ok + n_bad)
                if acc >= 1.0:
                    score = 5
                elif acc >= 0.75:
                    score = 4
                elif acc >= 0.5:
                    score = 3
                elif acc >= 0.25:
                    score = 2
                else:
                    score = 1
                return EvalResult(
                    "R01-深度", score, viol,
                    metrics={
                        "n_declared_pairs": len(matched),
                        "n_consistent": n_ok,
                        "n_inconsistent": n_bad,
                        "depth_order_accuracy": round(acc, 3),
                        "object_names": [d["front"] + ">" + d["back"]
                                         for d in details],
                        "pair_details": details,
                        "method": "depth_order_declared"},
                    status="proxy")

        # ── 降级判据: 无 before 关系时, 检验「空间关系合理性」 ──
        # 若 prompt 声明了 on/beside/against 等非前后关系, 用对应物理判据:
        #   on/against: 支撑物/背景应更靠后(深度更大), 被支撑物/物体更靠前
        #   beside:     并置物体深度应接近(同一空间层)
        spatial_rels = self._extract_spatial_relation(prompt)
        if spatial_rels:
            # 匹配空间关系对到分割物体
            rel_pairs = []
            for rel, a_phrase, b_phrase in spatial_rels:
                idx_a = idx_b = None
                def _words(phrase):
                    stop = {"the", "a", "an", "with", "and", "of", "in", "at",
                            "large", "small", "matte", "pale", "dark", "bright",
                            "background"}
                    return {w for w in re.findall(r"[a-z]+", phrase.lower())
                            if len(w) > 2 and w not in stop}
                aw, bw = _words(a_phrase), _words(b_phrase)
                for idx, name in enumerate(obj_names):
                    nl = set(re.findall(r"[a-z]+", str(name).lower()))
                    if idx_a is None and aw & nl:
                        idx_a = idx
                    if idx_b is None and bw & nl and idx != idx_a:
                        idx_b = idx
                if idx_a is None or idx_b is None:
                    continue
                da = obj_depths[idx_a][1]
                db = obj_depths[idx_b][1]
                rel_pairs.append((rel, obj_names[idx_a], obj_names[idx_b],
                                  da, db, idx_a, idx_b))
            if rel_pairs:
                n_ok = 0
                n_bad = 0
                viol = []
                details = []
                for (rel, na, nb, da, db, ia, ib) in rel_pairs:
                    if rel in ("on", "against"):
                        # A 支撑在 B 上 / A 靠在 B 前: A 应更浅(深度更小)
                        ok = da <= db
                        expected = f"{na} 应比 {nb} 更靠前"
                    else:  # beside: 深度应接近
                        denom = max(abs(da), abs(db)) + 1e-8
                        ok = abs(da - db) / denom < 0.30
                        expected = f"{na} 与 {nb} 深度应接近"
                    if ok:
                        n_ok += 1
                    else:
                        n_bad += 1
                        viol.append(
                            f"空间关系矛盾({rel}): {expected}, "
                            f"但{na}(d={da:.3f})/{nb}(d={db:.3f})")
                    details.append({
                        "relation": rel, "obj_a": na, "obj_b": nb,
                        "depth_a": round(da, 4), "depth_b": round(db, 4),
                        "consistent": bool(ok),
                    })
                acc = n_ok / (n_ok + n_bad)
                if acc >= 1.0:
                    score = 5
                elif acc >= 0.75:
                    score = 4
                elif acc >= 0.5:
                    score = 3
                elif acc >= 0.25:
                    score = 2
                else:
                    score = 1
                return EvalResult(
                    "R01-深度", score, viol,
                    metrics={
                        "n_spatial_pairs": len(rel_pairs),
                        "n_consistent": n_ok,
                        "n_inconsistent": n_bad,
                        "spatial_accuracy": round(acc, 3),
                        "pair_details": details,
                        "method": "spatial_relation"},
                    status="proxy")

        # ── 兜底: 无声明关系时, 用遮挡重叠对检验深度序 ──
        # 检查所有有重叠的物体对: 重叠区深度是否与"前景物体"一致
        n_pairs = 0
        n_consistent = 0
        n_inconsistent = 0
        violations = []
        pair_details = []

        for a in range(len(obj_depths)):
            for b in range(a + 1, len(obj_depths)):
                ia, da, ma = obj_depths[a]
                ib, db, mb = obj_depths[b]
                # 实例分割的可见面 mask 通常天然互斥：被遮挡物在重叠
                # 投影区没有可见像素，因此不能只用 ma & mb 判断“投影重叠”。
                # 先用两个可见 mask 的 bbox 交集判断投影重叠，再在交集内
                # 使用可见 union 取深度，兼容前景遮挡背景的正常实例分割。
                ya, xa = np.where(ma)
                yb, xb = np.where(mb)
                if not len(xa) or not len(xb):
                    continue
                x0, x1 = max(xa.min(), xb.min()), min(xa.max(), xb.max())
                y0, y1 = max(ya.min(), yb.min()), min(ya.max(), yb.max())
                if x1 < x0 or y1 < y0:
                    continue
                bbox_overlap = np.zeros_like(ma, dtype=bool)
                bbox_overlap[y0:y1 + 1, x0:x1 + 1] = True
                bbox_area = int(bbox_overlap.sum())
                if bbox_area < max(50, H * W * 0.005):
                    continue
                overlap = ((ma | mb) & bbox_overlap
                           & np.isfinite(pred) & (pred > 0))
                if overlap.sum() < max(20, int(bbox_area * 0.10)):
                    continue
                overlap_kind = "visible_mask_union_in_bbox"

                # 两物体各自的"非重叠专属区域"深度(作为各自深度的可靠参考)
                excl_a = ma & ~mb & np.isfinite(pred) & (pred > 0)
                excl_b = mb & ~ma & np.isfinite(pred) & (pred > 0)
                if excl_a.sum() < 30 or excl_b.sum() < 30:
                    continue

                ref_a = float(np.median(pred[excl_a]))
                ref_b = float(np.median(pred[excl_b]))
                # 深度差异需足够显著才构成可判定的遮挡关系
                denom = max(abs(ref_a), abs(ref_b)) + 1e-8
                if abs(ref_a - ref_b) / denom < 0.10:
                    continue

                n_pairs += 1
                ov_med = float(np.median(pred[overlap]))
                # 前景 = 深度更小者(深度=沿光轴距离)
                front_ref, back_ref = ((ref_a, ref_b) if ref_a < ref_b
                                       else (ref_b, ref_a))
                front_name = (obj_names[a] if ref_a < ref_b else obj_names[b])
                back_name = (obj_names[b] if ref_a < ref_b else obj_names[a])
                # 重叠区深度应更靠近前景
                d_front = abs(ov_med - front_ref)
                d_back = abs(ov_med - back_ref)
                ok = d_front <= d_back
                if ok:
                    n_consistent += 1
                else:
                    n_inconsistent += 1
                    if len(violations) < 3:
                        violations.append(
                            f"深度序矛盾: {front_name}(d={front_ref:.3f})应遮挡"
                            f"{back_name}(d={back_ref:.3f}), 但重叠区深度"
                            f"{ov_med:.3f}更接近后者")
                pair_details.append({
                    "front": front_name, "back": back_name,
                    "overlap_median": round(ov_med, 4),
                    "front_ref": round(front_ref, 4),
                    "back_ref": round(back_ref, 4),
                    "bbox_overlap_pixels": bbox_area,
                    "overlap_pixels": int(overlap.sum()),
                    "overlap_kind": overlap_kind,
                    "consistent": bool(ok),
                })

        if n_pairs == 0:
            return EvalResult.skipped(
                "R01-深度",
                f"{len(obj_depths)}个物体间无「可判定遮挡关系」的重叠对"
                f"(需掩码重叠且深度差>10%)")

        consistency = n_consistent / n_pairs
        if consistency > 0.95:
            score = 5
        elif consistency > 0.8:
            score = 4
        elif consistency > 0.6:
            score = 3
        elif consistency > 0.4:
            score = 2
        else:
            score = 1

        if n_inconsistent > 0 and len(violations) < 1:
            violations.append(
                f"深度序不一致: {n_inconsistent}/{n_pairs}对物体重叠区深度排序矛盾")

        return EvalResult("R01-深度", score, violations,
                          metrics={"n_objects": len(obj_depths),
                                   "n_overlap_pairs": n_pairs,
                                   "n_consistent": n_consistent,
                                   "n_inconsistent": n_inconsistent,
                                   "depth_order_accuracy": round(consistency, 3),
                                   "object_names": obj_names[:10],
                                   "pair_details": pair_details[:8],
                                   "method": "depth_order_occlusion"},
                          status="proxy")

    # ── R02 法线 ──
    def _eval_normals(self, frame, depth_map, normal_gt,
                      camera_poses: Optional[np.ndarray] = None) -> EvalResult:
        try:
            pred = np.asarray(self.stable_normal.estimate(frame).normal_map,
                              dtype=np.float32)  # [H,W,3], camera coordinates
        except Exception as e:
            return EvalResult.error("R02-表面法线", f"StableNormal 推理失败: {e}")

        if normal_gt is not None:
            # ── CI2V: 外部 GT 为逐帧 WorldNormal，转换到预测的相机坐标 ──
            gt = np.asarray(normal_gt, dtype=np.float32)
            if gt.ndim == 4:
                gt = gt[len(gt) // 2]
            if gt.ndim != 3 or gt.shape[-1] != 3:
                return EvalResult.skipped("R02-表面法线",
                                         f"GT法线期望[H,W,3]或[T,H,W,3]，实际{gt.shape}")
            if gt.shape[:2] != pred.shape[:2]:
                import cv2
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

            # CI2V_DATA_BUILD 规定 normals 为 WorldNormal；camera_poses 是
            # world→camera 外参。法向量只乘旋转，不乘平移。
            camera_rotation_applied = False
            if camera_poses is not None:
                poses = np.asarray(camera_poses)
                if poses.ndim == 3 and poses.shape[1:] == (3, 4):
                    R_wc = poses[len(poses) // 2, :, :3].astype(np.float32)
                    gt = gt @ R_wc.T
                    camera_rotation_applied = True

            pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
            gt_norm = np.linalg.norm(gt, axis=-1, keepdims=True)
            valid = (np.isfinite(pred).all(axis=-1)
                     & np.isfinite(gt).all(axis=-1)
                     & (pred_norm[..., 0] > 1e-6)
                     & (gt_norm[..., 0] > 1e-6))
            if int(valid.sum()) < 100:
                return EvalResult.skipped("R02-表面法线", "有效GT法线像素不足")
            pred = pred / np.maximum(pred_norm, 1e-8)
            gt = gt / np.maximum(gt_norm, 1e-8)
            dot = np.clip(np.sum(pred * gt, axis=-1), -1, 1)
            ang = np.degrees(np.arccos(dot))[valid]
            mean_ang = float(np.mean(ang))
            med_ang = float(np.median(ang))
            r1125 = float(np.mean(ang < 11.25))
            score = score_metric("R02_mean_angle", mean_ang)
            return EvalResult("R02-表面法线", score,
                              [] if score >= 4 else [f"Mean角误差={mean_ang:.1f}°"],
                              metrics={"mean_angle_deg": mean_ang,
                                       "median_angle_deg": med_ang,
                                       "ratio_11_25": r1125,
                                       "valid_pixels": int(valid.sum()),
                                       "gt_coordinate_system": "world",
                                       "prediction_coordinate_system": "camera",
                                       "camera_rotation_applied": camera_rotation_applied},
                              status="ok")

        # ── T2V/I2V proxy: 法线-深度梯度一致性（表格方案）──
        # 将 depth_map 缩放到法线图尺寸
        import cv2
        H_n, W_n = pred.shape[:2]
        if depth_map.shape[:2] != (H_n, W_n):
            depth_resized = cv2.resize(depth_map.astype(np.float32),
                                       (W_n, H_n), interpolation=cv2.INTER_LINEAR)
        else:
            depth_resized = depth_map

        # 由深度梯度反投影派生法线
        gy, gx = np.gradient(depth_resized.astype(np.float32))
        derived = np.stack([-gx, -gy, np.ones_like(depth_resized)], axis=-1)
        norm = np.linalg.norm(derived, axis=-1, keepdims=True) + 1e-8
        derived /= norm

        # 计算角度差
        dot = np.clip(np.sum(pred * derived, axis=-1), -1, 1)
        consistency_ang = float(np.degrees(np.mean(np.arccos(dot))))
        score = score_lower_better(consistency_ang, [15, 25, 35, 50])

        # 如果一致性角度过大（>60°），说明 VGGT 深度和 StableNormal 法线之间存在系统性矛盾
        # 不是评测器问题，而是视频本身几何不一致 → 低分合理
        if consistency_ang > 60:
            score = 1

        return EvalResult("R02-表面法线", score,
                          [] if score >= 4 else [f"法线-深度梯度差={consistency_ang:.1f}°"],
                          metrics={"normal_depth_consistency_deg": round(consistency_ang, 1)},
                          status="proxy")

    # ── R03 形状 ──
    _CANONICAL_SHAPES = ("triangular prism", "cylinder", "sphere", "cube", "cone")

    @classmethod
    def _shape_types_from_prompt(cls, prompt: str) -> list[str]:
        low = (prompt or "").lower()
        return [shape for shape in cls._CANONICAL_SHAPES if shape in low]

    @staticmethod
    def _canonical_surface(shape: str, n: int = 3000) -> np.ndarray:
        """生成单位尺度的理想形状表面点云（固定seed，可复现）。"""
        rng = np.random.default_rng(0)
        if shape == "sphere":
            z = rng.uniform(-1.0, 1.0, n)
            phi = rng.uniform(0.0, 2.0 * np.pi, n)
            r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
        if shape == "cube":
            pts = rng.uniform(-1.0, 1.0, (n, 3))
            face = rng.integers(0, 6, n)
            axis, sign = face // 2, np.where(face % 2 == 0, -1.0, 1.0)
            pts[np.arange(n), axis] = sign
            return pts
        if shape == "cylinder":
            pts = np.empty((n, 3), dtype=np.float64)
            part = rng.random(n)
            side = part < 0.7
            theta = rng.uniform(0, 2 * np.pi, n)
            pts[side, 0] = np.cos(theta[side])
            pts[side, 1] = np.sin(theta[side])
            pts[side, 2] = rng.uniform(-1.0, 1.0, side.sum())
            cap = ~side
            rr = np.sqrt(rng.random(cap.sum()))
            pts[cap, 0] = rr * np.cos(theta[cap])
            pts[cap, 1] = rr * np.sin(theta[cap])
            pts[cap, 2] = np.where(part[cap] < 0.85, -1.0, 1.0)
            return pts
        if shape == "cone":
            pts = np.empty((n, 3), dtype=np.float64)
            side = rng.random(n) < 0.8
            z = rng.uniform(-1.0, 1.0, side.sum())
            theta = rng.uniform(0, 2 * np.pi, side.sum())
            radius = (1.0 - z) / 2.0
            pts[side] = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
            base = ~side
            rr = np.sqrt(rng.random(base.sum()))
            th = rng.uniform(0, 2 * np.pi, base.sum())
            pts[base] = np.stack([rr * np.cos(th), rr * np.sin(th),
                                  np.full(base.sum(), -1.0)], axis=1)
            return pts
        if shape == "triangular prism":
            # 三角柱的三条侧面 + 两个三角端面。
            v = np.array([[-1.0, -0.8], [1.0, -0.8], [0.0, 0.9]])
            pts = []
            for _ in range(n):
                q = rng.random()
                z = rng.uniform(-1.0, 1.0)
                if q < 0.6:
                    edge = rng.integers(0, 3)
                    t = rng.random()
                    xy = (1 - t) * v[edge] + t * v[(edge + 1) % 3]
                    pts.append([xy[0], xy[1], z])
                else:
                    a, b = rng.random(2)
                    if a + b > 1:
                        a, b = 1 - a, 1 - b
                    xy = v[0] + a * (v[1] - v[0]) + b * (v[2] - v[0])
                    pts.append([xy[0], xy[1], -1.0 if q < 0.8 else 1.0])
            return np.asarray(pts)
        raise ValueError(f"未定义canonical shape: {shape}")

    @staticmethod
    def _align_canonical_to_observation(canonical: np.ndarray,
                                        observed: np.ndarray,
                                        iterations: int = 10) -> np.ndarray:
        """用Sim(3)+ICP把理想模型对齐到观测点云。

        单视图只看到可见表面，所以最终误差使用 observed→canonical 的单向
        Chamfer，避免不可见背面给被测视频施加不可能的惩罚。
        """
        from scipy.spatial import cKDTree
        src = canonical.astype(np.float64).copy()
        dst = observed.astype(np.float64)
        src_center, dst_center = src.mean(0), dst.mean(0)
        src_scale = np.sqrt(np.mean(np.sum((src - src_center) ** 2, axis=1))) + 1e-8
        dst_scale = np.sqrt(np.mean(np.sum((dst - dst_center) ** 2, axis=1))) + 1e-8
        src = (src - src_center) * (dst_scale / src_scale) + dst_center
        tree = cKDTree(dst)
        for _ in range(iterations):
            _, idx = tree.query(src, k=1)
            matched = dst[idx]
            sc, mc = src.mean(0), matched.mean(0)
            H = (src - sc).T @ (matched - mc)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            src = (src - sc) @ R.T + mc
        return src

    def _eval_shape_proxy(self, key_frame, all_frames, prompt: str,
                          evaluator_context: Optional[dict] = None) -> EvalResult:
        """R03 T2V/I2V：canonical Chamfer 或 VLM 形状完整性代理。"""
        known = self._shape_types_from_prompt(prompt)
        if known:
            if self.sam3 is None:
                return EvalResult.skipped("R03-形状", "SAM3未加载，无法分割已知形状物体")
            try:
                # 只传当前prompt中实际出现的标准形状，不能使用R03维度的
                # 全部概念词，否则R03-001会额外分割不存在的物体。
                seg = self.sam3.segment_frame(
                    key_frame, dimension="R03", concepts=known)
            except Exception as exc:
                return EvalResult.error("R03-形状", f"SAM3形状分割失败: {exc}")
            if seg.masks is None or len(seg.masks) == 0:
                return EvalResult.skipped("R03-形状", "SAM3未分割到已知形状物体")

            from scipy.spatial import cKDTree
            errs, evaluated, violations = [], [], []
            names = [str(x).lower() for x in (seg.class_names or [])]
            detected_shapes, missing_shapes = [], []
            for shape in known:
                candidates = [i for i, name in enumerate(names) if shape in name]
                if not candidates:
                    missing_shapes.append(shape)
                    continue
                detected_shapes.append(shape)
                # 同概念可能有多个实例；仅取最高置信度实例，避免重复计权。
                i = max(candidates, key=lambda j: float(seg.confidences[j])
                        if seg.confidences is not None and j < len(seg.confidences) else 0.0)
                observed = self.vggt.reconstruct_masked_pointcloud(
                    key_frame, seg.masks[i]).points
                if observed is None or len(observed) < 100:
                    missing_shapes.append(shape)
                    continue
                canonical = self._canonical_surface(shape)
                aligned_canonical = self._align_canonical_to_observation(canonical, observed)
                # 只评估可见观测表面到理想表面的距离，符合单视图几何约束。
                scale = np.sqrt(np.mean(np.sum((observed - observed.mean(0)) ** 2, axis=1))) + 1e-8
                err = float(cKDTree(aligned_canonical).query(observed, k=1)[0].mean() / scale)
                errs.append(err)
                evaluated.append({"shape": shape, "chamfer_visible_norm": round(err, 4),
                                  "n_points": int(len(observed))})
                if err >= 0.10:
                    violations.append(f"{shape} 可见表面Chamfer/scale={err:.3f}")

            shape_recall = len(evaluated) / max(len(known), 1)
            if missing_shapes:
                violations.insert(0, f"canonical形状召回不足: 缺失 {missing_shapes}")
            if not errs:
                return EvalResult.skipped(
                    "R03-形状",
                    f"未获得prompt已知形状{known}的有效SAM3掩码点云")
            avg = float(np.mean(errs))
            chamfer_score = score_metric("R03_chamfer_norm", avg)
            recall_score = (5 if shape_recall >= 1.0 else
                            4 if shape_recall >= 0.75 else
                            3 if shape_recall >= 0.5 else
                            2 if shape_recall > 0 else 1)
            score = min(chamfer_score, recall_score)
            return EvalResult(
                "R03-形状", score,
                violations if score < 4 else (violations if missing_shapes else []),
                metrics={"canonical_shapes": evaluated,
                         "expected_shapes": known,
                         "detected_shapes": detected_shapes,
                         "missing_shapes": missing_shapes,
                         "shape_recall": round(shape_recall, 3),
                         "chamfer_visible_norm": avg,
                         "chamfer_score": chamfer_score,
                         "recall_score": recall_score,
                         "method": "canonical_visible_chamfer"},
                status="proxy")

        # 表格对非标准几何体要求用VLM判形状完整/无扭曲。
        if self.vlm is None or not getattr(self.vlm, "is_loaded", False):
            return EvalResult.skipped("R03-形状", "非标准形状需要Qwen3-VL，但VLM未加载")
        idx = np.linspace(0, len(all_frames) - 1, min(4, len(all_frames)), dtype=int)
        images = [Image.fromarray(all_frames[i].astype(np.uint8)) for i in idx]
        judge = (
            "你是三维几何评估专家。观察这些同一视频的关键帧。原始场景描述为："
            f"\"{prompt}\"。\n"
            "判断主体物体的三维形状是否完整、各部分比例是否协调、跨视角是否出现"
            "塌陷、拉伸、断裂或自相矛盾的几何。不要评价光照、材质或动作。\n"
            "评分：5=形状完整且跨视角一致；3=主体正确但局部比例/连续性异常；"
            "1=明显塌陷、扭曲或部件不一致。\n"
            "返回 JSON: {\"score\": <1-5整数>, \"violations\": [\"...\"]}"
        )
        try:
            data = self.vlm.chat_json(images, judge, max_new_tokens=512)
        except Exception as exc:
            return EvalResult.error("R03-形状", f"VLM形状判定失败: {exc}")
        if data.get("_parse_error"):
            return EvalResult.error("R03-形状", "VLM形状判定JSON解析失败")
        try:
            score = max(1, min(5, int(round(float(data.get("score"))))))
        except (TypeError, ValueError):
            return EvalResult.error("R03-形状", "VLM未返回有效score")
        violations = data.get("violations", [])
        if isinstance(violations, str):
            violations = [violations] if violations else []
        context = evaluator_context or {}
        metrics = {
            "method": "vlm_shape_integrity",
            "frames_checked": len(images),
        }
        for key in ("ci2v_sample_id", "label", "expected_violation", "scenario"):
            if context.get(key) is not None:
                metrics[key] = context[key]
        return EvalResult("R03-形状", score, [str(x)[:200] for x in violations],
                          metrics=metrics, status="proxy")

    @staticmethod
    def _select_gt_pointcloud(pointcloud_gt: np.ndarray) -> np.ndarray:
        """规范 CI2V R03 点云为[N,3]；逐帧点云取中间帧。"""
        gt = np.asarray(pointcloud_gt, dtype=np.float64)
        if gt.ndim == 3 and gt.shape[-1] == 3:
            gt = gt[gt.shape[0] // 2]
        if gt.ndim != 2 or gt.shape[-1] != 3:
            return np.zeros((0, 3), dtype=np.float64)
        return gt[np.isfinite(gt).all(axis=1)]

    @staticmethod
    def _align_pointcloud_to_gt(source: np.ndarray, target: np.ndarray,
                                iterations: int = 10) -> np.ndarray:
        """用尺度初始化+最近邻ICP完成Sim(3)对齐，禁止反射。"""
        src = np.asarray(source, dtype=np.float64).copy()
        dst = np.asarray(target, dtype=np.float64)
        src_c, dst_c = src.mean(0), dst.mean(0)
        src_s = np.sqrt(np.mean(np.sum((src - src_c) ** 2, axis=1))) + 1e-8
        dst_s = np.sqrt(np.mean(np.sum((dst - dst_c) ** 2, axis=1))) + 1e-8
        src = (src - src_c) * (dst_s / src_s) + dst_c
        from scipy.spatial import cKDTree
        tree = cKDTree(dst)
        for _ in range(iterations):
            _, idx = tree.query(src, k=1)
            matched = dst[idx]
            sc, mc = src.mean(0), matched.mean(0)
            H = (src - sc).T @ (matched - mc)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            src = (src - sc) @ R.T + mc
        return src

    def _eval_shape(self, key_frame, all_frames, pointcloud_gt,
                    prompt: str = "",
                    evaluator_context: Optional[dict] = None) -> EvalResult:
        # ── 有 GT 路径：单帧预测 vs 外部仿真点云 ──
        if pointcloud_gt is not None:
            gt_pts = self._select_gt_pointcloud(pointcloud_gt)
            if len(gt_pts) < 100:
                return EvalResult.skipped("R03-形状", "有效GT点云点数不足")
            try:
                pred_pts = np.asarray(
                    self.vggt.reconstruct_pointcloud(key_frame).points,
                    dtype=np.float64)
            except Exception as e:
                return EvalResult.error("R03-形状", f"VGGT 点云重建失败: {e}")
            pred_pts = pred_pts[np.isfinite(pred_pts).all(axis=1)]
            if len(pred_pts) < 100:
                return EvalResult.skipped("R03-形状", "重建点云点数不足")
            from scipy.spatial import cKDTree
            rng = np.random.default_rng(0)
            n = min(4000, len(pred_pts), len(gt_pts))
            p = pred_pts[rng.choice(len(pred_pts), n, replace=False)]
            g = gt_pts[rng.choice(len(gt_pts), n, replace=False)]
            p_aligned = self._align_pointcloud_to_gt(p, g)
            gt_scale = np.sqrt(np.mean(np.sum((g - g.mean(0)) ** 2, axis=1))) + 1e-8
            d1 = cKDTree(g).query(p_aligned, k=1)[0].mean()
            d2 = cKDTree(p_aligned).query(g, k=1)[0].mean()
            chamfer_norm = float((d1 + d2) / 2 / gt_scale)
            score = score_metric("R03_chamfer_norm", chamfer_norm)
            return EvalResult("R03-形状", score,
                              [] if score >= 4 else [f"Chamfer/scale={chamfer_norm:.3f}"],
                              metrics={"chamfer_l1_norm": chamfer_norm,
                                       "n_pred_points": int(len(pred_pts)),
                                       "n_gt_points": int(len(gt_pts)),
                                       "alignment": "sim3_icp",
                                       "gt_pointcloud_layout": "frame_first_to_mid"},
                              status="ok")

        # ── T2V/I2V proxy（严格按表格） ──
        # 已知标准形状：mask筛选重建点云 vs canonical理想模型 Chamfer。
        # 非标准形状：Qwen3-VL 判完整性/无扭曲。两条均来自PHYSICS_TAXONOMY。
        return self._eval_shape_proxy(key_frame, all_frames, prompt,
                                       evaluator_context=evaluator_context)

        # ── 保留旧跨帧实现供后续canonical路径开发（当前不可达） ──
        T = len(all_frames)
        n_sample = min(6, T)
        if n_sample < 2:
            try:
                pred_pts = self.vggt.reconstruct_pointcloud(key_frame).points
            except Exception as e:
                return EvalResult.error("R03-形状", f"VGGT 点云重建失败: {e}")
            if pred_pts is None or len(pred_pts) < 100:
                return EvalResult.skipped("R03-形状", "重建点云点数不足")
            extents = pred_pts.max(0) - pred_pts.min(0)
            min_ext = float(extents.min())
            score = 5 if min_ext > 0.01 else (3 if min_ext > 0.001 else 1)
            return EvalResult("R03-形状", score,
                              [] if score >= 4 else [f"点云退化 min_extent={min_ext:.4f}"],
                              metrics={"n_points": len(pred_pts), "extent": extents.tolist(),
                                       "frames_checked": 1},
                              status="proxy")

        indices = np.linspace(0, T - 1, n_sample, dtype=int)
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(0)

        pc_list = []
        for idx in indices:
            try:
                pc = self.vggt.reconstruct_pointcloud(all_frames[idx]).points
                if pc is not None and len(pc) >= 100:
                    pc_list.append(pc)
                else:
                    pc_list.append(None)
            except Exception:
                pc_list.append(None)

        valid_indices = [i for i, pc in enumerate(pc_list) if pc is not None]
        if len(valid_indices) < 2:
            return EvalResult.skipped("R03-形状", f"有效点云帧数不足 ({len(valid_indices)}/{n_sample})")

        def kabsch_rotation(P, Q):
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

        def icp_align(source, target, iterations=8):
            """无点对应关系时用最近邻ICP，而不是把两帧随机点强行一一配对。

            旧实现对两帧独立随机采样后直接Kabsch；采样点并不对应，旋转和
            Chamfer会被随机配对污染，R03在三条视频上全部1很可能由此造成。
            """
            src = source.astype(np.float64).copy()
            dst = target.astype(np.float64)
            # 初始只做质心平移，之后每轮以最近邻建立临时对应
            src += dst.mean(axis=0) - src.mean(axis=0)
            tree = cKDTree(dst)
            for _ in range(iterations):
                _, nn = tree.query(src, k=1)
                matched = dst[nn]
                R = kabsch_rotation(src, matched)
                src = (src - src.mean(axis=0)) @ R.T + matched.mean(axis=0)
            return src

        pair_drifts = []
        n_pairs = 0
        n_large_drift = 0
        violations = []

        for a in range(len(valid_indices) - 1):
            for b in range(a + 1, len(valid_indices)):
                ia, ib = valid_indices[a], valid_indices[b]
                Pa = pc_list[ia]
                Pb = pc_list[ib]
                n_pts = min(500, len(Pa), len(Pb))
                pa = Pa[rng.choice(len(Pa), n_pts, replace=False)]
                pb = Pb[rng.choice(len(Pb), n_pts, replace=False)]

                pb_aligned = icp_align(pb, pa)

                scale = float(np.std(pa, axis=0).mean()) + 1e-8
                d1 = cKDTree(pb_aligned).query(pa, k=1)[0].mean()
                d2 = cKDTree(pa).query(pb_aligned, k=1)[0].mean()
                drift = float((d1 + d2) / 2 / scale)
                pair_drifts.append(drift)
                n_pairs += 1

                if drift > 0.08:
                    n_large_drift += 1
                    if n_large_drift <= 3:
                        violations.append(
                            f"帧{indices[ia]}→帧{indices[ib]}: 对齐后Chamfer={drift:.3f}"
                            f" (形状变化, 非纯旋转)")

        avg_drift = float(np.mean(pair_drifts)) if pair_drifts else 0.0
        max_drift = float(max(pair_drifts)) if pair_drifts else 0.0

        ref_pc = pc_list[valid_indices[0]]
        extents = ref_pc.max(axis=0) - ref_pc.min(axis=0)
        min_ext = float(np.min(extents))

        if min_ext < 0.001:
            score = 1
            violations.insert(0, f"点云退化 min_extent={min_ext:.4f}")
        elif n_large_drift == 0:
            score = 5
        elif n_large_drift <= n_pairs * 0.2:
            score = 4
        elif n_large_drift <= n_pairs * 0.4:
            score = 3
        elif n_large_drift <= n_pairs * 0.6:
            score = 2
        else:
            score = 1

        return EvalResult("R03-形状", score, violations,
                          metrics={
                              "n_points": len(ref_pc),
                              "extent": extents.tolist(),
                              "frames_checked": n_sample,
                              "n_pairs": n_pairs,
                              "n_large_drift_pairs": n_large_drift,
                              "avg_drift": round(avg_drift, 4),
                              "max_drift": round(max_drift, 4),
                          },
                          status="proxy")
