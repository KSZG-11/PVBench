"""
物体级 3D 轨迹构建 — 修正 SpatialTracker「网格点 ≠ 物体」的语义错位

背景问题 (P0)
--------------
`SpatialTracker.track_3d()` 默认在首帧铺 grid_size×grid_size=400 个均匀网格查询点，
返回 `trajectories` 形状 [N=400, T, 3]。但PHYSICS_TAXONOMY.md 对 R22/R23/R24/R15/R16
的要求全部是**物体级**物理量：

- R22 ②「同重力场**不同质量物体**自由下落应同速」
- R23 「**物体**碰撞前后速度」「Σmᵢvᵢ = Σmᵢvᵢ′ 解质量比」
- R24 「**物体**速度、高度」→ KE+PE
- R16 「针对匀速运动，速度的变异系数」（背景静止点会稀释 CV）

若把 400 个网格点当物体，会导致:
1. `N < 2` 之类的保护永不生效（N≈400 恒成立）；
2. R23 的 O(N²)=79800 对「碰撞」多为同一物体上两个网格点靠近 → 质量比无物理意义；
3. 重力方向估计被大量静止背景点稀释 → 落回硬编码 [0,-1,0]；
4. R24逐「物体」找弹跳峰值 → 违例轻易累积到≥4 → 分数被系统性压到 1。

本模块的解法（表格内方案，不引入表格未提及的退化路径）
-------------------------------------------------------
表格为 R22/R23/R24 指定的模型是 SpatialTracker V2；为 R09/R11/R12/R21 指定了 SAM 3。
两者组合即可得到物体级轨迹，无需引入新模型：

  SAM3 分割首帧 → 每个物体掩码内采样 K 个查询点 → 一次性喂给 SpatialTracker
  → 按物体分组聚合每个物体的质心轨迹 [N_obj, T, 3]

同时保留网格点轨迹用于「场景级」量（如 R19跨帧几何、R17 Kabsch 旋转），
这两者本来就该用稠密点而非物体质心。
"""
import numpy as np
from typing import Optional


# 每个物体在其掩码内采样的查询点数 (足够估质心，又不过度膨胀 N)
POINTS_PER_OBJECT = 24
# 物体掩码最小面积占比 (过滤 SAM3 过分割碎片)
MIN_MASK_AREA_RATIO = 0.005
# 最多跟踪的物体数 (按面积降序取前N，控制 R23 的 O(N²) 成本)
MAX_OBJECTS = 8


class ObjectTracks:
    """物体级轨迹容器，字段语义与 TrajectoryOutput 对齐但 N 是**物体数**。

    Attributes:
        trajectories: [N_obj, T, 3] 每个物体的质心世界坐标
        velocities:   [N_obj, T-1, 3] m/s (或归一化单位/s)
        accelerations:[N_obj, T-2, 3]
        object_names: list[str] 物体名 (来自 SAM3 class_names)
        point_tracks: [N_points, T, 3] 原始逐点轨迹 (供刚体旋转估计等使用)
        point_owner:  [N_points] 每个点所属物体索引
        source:       "sam3_object_centroid" | "grid_fallback"
        n_objects:    int
    """

    def __init__(self, trajectories, velocities=None, accelerations=None,
                 object_names=None, point_tracks=None, point_owner=None,
                 source="sam3_object_centroid", confidences=None):
        self.trajectories = trajectories
        self.velocities = velocities
        self.accelerations = accelerations
        self.object_names = object_names or []
        self.point_tracks = point_tracks
        self.point_owner = point_owner
        self.source = source
        self.confidences = confidences

    @property
    def n_objects(self) -> int:
        return 0 if self.trajectories is None else int(self.trajectories.shape[0])


def _sample_points_in_mask(mask: np.ndarray, k: int,
                           rng: np.random.Generator) -> Optional[np.ndarray]:
    """在掩码内均匀采样 k 个 (x, y) 像素坐标。

    优先取靠掩码内部的点（腐蚀一圈避免边界点跟丢），不足则退回全掩码。
    """
    m = mask > 0.5
    if m.sum() < 4:
        return None

    inner = m
    try:
        import cv2
        er = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
        if er.sum() >= max(4, k // 2):
            inner = er > 0
    except Exception:
        pass

    ys, xs = np.nonzero(inner)
    if len(xs) == 0:
        return None
    if len(xs) <= k:
        idx = np.arange(len(xs))
    else:
        idx = rng.choice(len(xs), k, replace=False)
    return np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)  # [k,2]


def _motion_reference_mask(reference_masks: np.ndarray,
                           target_only: bool = False):
    """从外部CI2V实例mask中选择要追踪的对象mask。

    R15/R16 的GT mask是整数实例图 [T,H,W]；其中可能同时包含桌面、地面和
    运动物体。R15 的目标是运动物体，不应把最大面积的静态支撑物喂给
    SpatialTracker。这里按实例质心的时间位移选择运动量最大的实例，
    只返回首帧掩码供query_points使用。

    Returns:
        bool mask [H,W]，或 None。
    """
    m = np.asarray(reference_masks)
    if m.ndim == 4:
        # [N,T,H,W] bool/instance masks → 每个实例的 OR/逐实例序列
        candidates = []
        for i in range(m.shape[0]):
            candidates.append(m[i] > 0.5)
    elif m.ndim == 3:
        # [T,H,W] integer instance labels
        labels = [x for x in np.unique(m) if x != 0]
        candidates = [(m == x) for x in labels]
    else:
        return None
    if not candidates:
        return None

    def motion_score(seq):
        centers = []
        for t in range(seq.shape[0]):
            ys, xs = np.nonzero(seq[t])
            centers.append([float(xs.mean()), float(ys.mean())]
                           if len(xs) else [np.nan, np.nan])
        c = np.asarray(centers)
        valid = np.isfinite(c).all(axis=1)
        if valid.sum() < 2:
            return -1.0
        idx = np.where(valid)[0]
        # 用首尾位移，并加入相邻帧路径长度，避免周期运动首尾恰好重合
        displacement = float(np.linalg.norm(c[idx[-1]] - c[idx[0]]))
        path = float(np.linalg.norm(np.diff(c[idx], axis=0), axis=1).sum())
        return displacement + 0.25 * path

    scores = [motion_score(c) for c in candidates]
    chosen = None
    if target_only:
        # 若场景静止，所有运动分数都为0，不能因列表顺序选中大桌面。
        # R15 的目标物体是可移动小物体，因此平分时选择首帧面积最小的实例；
        # 有运动时仍优先运动分数最大的实例。
        max_score = max(scores)
        if max_score <= 1e-6:
            areas = [int(c[0].sum()) for c in candidates]
            chosen = int(np.argmin(areas))
        else:
            chosen = int(np.argmax(scores))
    if chosen is not None:
        print(f"[ObjectTracks] external target candidate={chosen}, "
              f"scores={[round(float(s), 3) for s in scores]}, "
              f"areas={[int(c[0].sum()) for c in candidates]}")
        return candidates[chosen][0].astype(bool)
    # 非target-only：合并各实例首帧，作为多个物体的整体查询区域
    return np.logical_or.reduce([c[0] for c in candidates]).astype(bool)


def build_object_tracks(spatial_tracker, sam3, frames: np.ndarray,
                        fps: float = 24.0,
                        dimension: str = "object_tracks",
                        max_objects: int = MAX_OBJECTS,
                        reference_masks: Optional[np.ndarray] = None,
                        target_only: bool = False,
                        prompt: str = "") -> ObjectTracks:
    """构建物体级 3D 轨迹。

    Args:
        spatial_tracker: SpatialTracker V2 封装 (需track_3d / compute_velocity_accel)
        sam3: SAM3 封装 (需 segment_frame)。为 None 或分割失败 → 网格降级
        frames: [T,H,W,3] uint8
        fps: 帧率
        dimension: 传给 SAM3 的维度标签(影响其文本 prompt)
        max_objects: 最多跟踪物体数
        reference_masks: CI2V外部实例mask；提供时优先于SAM3，避免目标错配
        target_only: 只追踪reference_masks中运动量最大的实例（R15使用）

    Returns:
        ObjectTracks。若 SAM3 不可用则 source="grid_fallback"，此时
        trajectories 退化为「全体网格点质心」的单物体轨迹 [1,T,3]，
        并保留 point_tracks 供场景级分析——**绝不**再把 400 点当 400 个物体。

    Raises:
        RuntimeError: SpatialTracker 推理失败 (由调用方转成 status=error)
    """
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[np.newaxis, ...]
    T, H, W = frames.shape[:3]
    rng = np.random.default_rng(0)

    # ── 1) 优先使用CI2V外部实例mask；无GT时才使用SAM3 ──
    masks, names = None, []
    mask_source = "sam3"
    if reference_masks is not None:
        ref = _motion_reference_mask(reference_masks, target_only=target_only)
        if ref is not None:
            masks = np.asarray(ref)[None, ...]
            names = ["gt_motion_target" if target_only else "gt_reference"]
            mask_source = "external_gt_mask"
    if masks is None and sam3 is not None:
        # 多帧候选: 首帧物体可能未就位/太小/被遮挡, 单帧分割易漏检。
        # 在前后多帧中选「检测到物体最多」的一帧做分割 (R22/R23 需要 ≥2 个物体,
        # 碰撞/多物体场景物体可能在不同帧才全部出现)。
        try:
            best_masks, best_names, best_count = None, [], -1
            candidate_idx = sorted(set(
                [0, max(0, T // 4), T // 2, max(0, 3 * T // 4), max(0, T - 1)]
            ))
            for idx in candidate_idx:
                seg = sam3.segment_frame(frames[idx], dimension=dimension,
                                         prompt=prompt)
                if seg.masks is not None and len(seg.masks) > 0:
                    if len(seg.masks) > best_count:
                        best_count = len(seg.masks)
                        best_masks = seg.masks
                        best_names = (list(seg.class_names)
                                      if seg.class_names else [])
            if best_masks is not None:
                masks = best_masks
                names = best_names
        except Exception:
            masks = None

    # 过滤碎片 + 按面积降序取前 max_objects
    kept = []
    if masks is not None:
        # 外部GT mask已经是实例真值，不能用通用SAM3的面积阈值过滤小目标。
        # 当前cue_ball面积约807px，而0.5%图像阈值约1996px，会把真实目标过滤掉。
        # ⚠️ SAM3 的 0.5% 阈值对「俯拍/远景小物体」(如气桌冰球 puck~800px、
        #   弹珠 marble) 也过严 → R23 报「有效物体数=1」skipped。改为
        #   max(100px, 0.1%图像) 的宽松下限: 保留小目标, 仍滤除 <100px 碎片。
        if mask_source == "external_gt_mask":
            min_area = 100.0
        else:
            min_area = max(100.0, H * W * 0.001)
        for i in range(len(masks)):
            m = masks[i] > 0.5
            area = float(m.sum())
            if area >= min_area:
                kept.append((i, area))
        kept.sort(key=lambda x: x[1], reverse=True)
        kept = kept[:max_objects]

    # ── 2) 构造查询点: 每个物体掩码内采样 K 点，记录归属 ──
    query_xy, owner = [], []
    for slot, (mi, _area) in enumerate(kept):
        pts = _sample_points_in_mask(masks[mi], POINTS_PER_OBJECT, rng)
        if pts is None:
            continue
        query_xy.append(pts)
        owner.append(np.full(len(pts), slot, dtype=np.int32))

    if not query_xy:
        # ── 降级: 无可用物体掩码 → 用默认网格，但只产出「场景级单体」轨迹 ──
        traj = spatial_tracker.track_3d(frames, fps=fps)
        pt = np.asarray(traj.trajectories)  # [N_grid,T,3]
        finite = np.isfinite(pt).all(axis=(1, 2))
        pts_ok = pt[finite] if finite.any() else pt
        centroid = np.nanmedian(pts_ok, axis=0)[None]  # [1,T,3] 用中位数抗离群
        vt = spatial_tracker.compute_velocity_accel(centroid, fps=fps)
        return ObjectTracks(
            trajectories=centroid,
            velocities=vt.velocities,
            accelerations=vt.accelerations,
            object_names=["scene"],
            point_tracks=pt,
            point_owner=None,
            source="grid_fallback",
            confidences=getattr(traj, "confidences", None),
        )

    query_xy_all = np.concatenate(query_xy, axis=0)          # [N_pts,2]
    owner_all = np.concatenate(owner, axis=0)                # [N_pts]
    # SpatialTracker 需要 [N,3] = (t, x, y)，t=0 表示首帧发起追踪
    query_xyt = np.concatenate(
        [np.zeros((len(query_xy_all), 1), np.float32), query_xy_all], axis=1)
    print(f"[ObjectTracks] mask_source={mask_source}, target_only={target_only}, "
          f"kept={len(kept)}, query_points={query_xyt.shape}")

    # ── 3) 一次推理拿到所有点的3D 轨迹 ──
    traj = spatial_tracker.track_3d(frames, query_points=query_xyt, fps=fps)
    point_tracks = np.asarray(traj.trajectories)             # [N_pts,T,3]
    point_conf = getattr(traj, "confidences", None)
    if point_conf is not None:
        point_conf = np.asarray(point_conf)
        # SpatialTracker 的vis在不同封装路径可能是 [T,N,1]、[T,N] 或 [N,T]。
        # 统一成 [N,T]，否则Fail Rate会被静默禁用。
        if point_conf.ndim == 3 and point_conf.shape[-1] == 1:
            point_conf = point_conf[..., 0]
        if point_conf.ndim == 2:
            if point_conf.shape == (point_tracks.shape[1], point_tracks.shape[0]):
                point_conf = point_conf.T
            elif point_conf.shape != point_tracks.shape[:2]:
                point_conf = None
        else:
            point_conf = None
    print(f"[ObjectTracks] returned_tracks={point_tracks.shape}, "
          f"requested_points={len(owner_all)}, "
          f"finite_ratio={np.isfinite(point_tracks).all(axis=-1).mean():.3f}, "
          f"confidence_available={point_conf is not None}")

    # SpatialTracker 内部会把图像 crop/resize 到 518；若其返回点数与输入不一致，
    # 说明查询点未被原样透传，此时物体归属失效 → 退回场景级单体，避免错误归组。
    if point_tracks.shape[0] != len(owner_all):
        finite = np.isfinite(point_tracks).all(axis=(1, 2))
        pts_ok = point_tracks[finite] if finite.any() else point_tracks
        centroid = np.nanmedian(pts_ok, axis=0)[None]
        vt = spatial_tracker.compute_velocity_accel(centroid, fps=fps)
        return ObjectTracks(
            trajectories=centroid, velocities=vt.velocities,
            accelerations=vt.accelerations, object_names=["scene"],
            point_tracks=point_tracks, point_owner=None,
            source="grid_fallback", confidences=getattr(traj, "confidences", None),
        )

    # ── 4) 按物体聚合质心轨迹 (中位数抗跟丢离群点) ──
    n_obj = len(kept)
    obj_traj, obj_names, obj_confidences = [], [], []
    for slot in range(n_obj):
        sel = owner_all == slot
        if sel.sum() < 3:
            continue
        pts = point_tracks[sel]                              # [k,T,3]
        finite_point_ratio = float(np.isfinite(pts).all(axis=-1).mean())
        with np.errstate(invalid="ignore"):
            c = np.nanmedian(pts, axis=0)                    # [T,3]
        # 少量 query point 丢失是正常遮挡/低置信度，不应把整个目标降级为
        # grid_fallback。只要物体质心有足够时间点，就做时间插值/前后填充。
        c = _interpolate_track(c)
        if c is None:
            print(f"[ObjectTracks] object slot={slot} discarded: "
                  f"finite_point_ratio={finite_point_ratio:.3f}")
            continue
        print(f"[ObjectTracks] object slot={slot} aggregated: "
                  f"finite_point_ratio={finite_point_ratio:.3f}")
        obj_traj.append(c)
        if point_conf is not None:
            obj_confidences.append(np.nanmedian(point_conf[sel], axis=0))
        mi = kept[slot][0]
        obj_names.append(names[mi] if mi < len(names) else f"obj_{slot}")

    if not obj_traj:
        finite = np.isfinite(point_tracks).all(axis=(1, 2))
        pts_ok = point_tracks[finite] if finite.any() else point_tracks
        centroid = np.nanmedian(pts_ok, axis=0)[None]
        vt = spatial_tracker.compute_velocity_accel(centroid, fps=fps)
        return ObjectTracks(
            trajectories=centroid, velocities=vt.velocities,
            accelerations=vt.accelerations, object_names=["scene"],
            point_tracks=point_tracks, point_owner=owner_all,
            source="grid_fallback", confidences=getattr(traj, "confidences", None),
        )

    obj_traj = np.stack(obj_traj)                            # [N_obj,T,3]
    vt = spatial_tracker.compute_velocity_accel(obj_traj, fps=fps)
    return ObjectTracks(
        trajectories=obj_traj,
        velocities=vt.velocities,
        accelerations=vt.accelerations,
        object_names=obj_names,
        point_tracks=point_tracks,
        point_owner=owner_all,
        source=("external_gt_object_centroid" if mask_source == "external_gt_mask"
                else "sam3_object_centroid"),
        confidences=(np.stack(obj_confidences).astype(np.float32)
                     if len(obj_confidences) == len(obj_traj) else None),
    )


def _interpolate_track(arr: np.ndarray, min_valid_ratio: float = 0.5):
    """对[T,3]质心轨迹插值，允许部分query点/帧不可见。

    返回None仅当有效时间点不足；这比“任意NaN就退回网格轨迹”更符合
    物体追踪语义。端点使用最近有效点，内部使用逐维线性插值。
    """
    out = np.asarray(arr, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 3:
        return None
    valid = np.isfinite(out).all(axis=1)
    if valid.mean() < min_valid_ratio or valid.sum() < 2:
        return None
    x = np.arange(len(out))
    for d in range(3):
        col = out[:, d]
        good = np.isfinite(col)
        if good.sum() < 2:
            return None
        out[:, d] = np.interp(x, x[good], col[good])
    return out.astype(np.float32)


def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """对 [T,3] 轨迹沿时间做前向再后向填充，保留旧接口。"""
    out = arr.copy()
    T = out.shape[0]
    for d in range(out.shape[1]):
        col = out[:, d]
        # 前向
        last = np.nan
        for t in range(T):
            if np.isfinite(col[t]):
                last = col[t]
            elif np.isfinite(last):
                col[t] = last
        # 后向
        nxt = np.nan
        for t in range(T - 1, -1, -1):
            if np.isfinite(col[t]):
                nxt = col[t]
            elif np.isfinite(nxt):
                col[t] = nxt
        out[:, d] = col
    return out
