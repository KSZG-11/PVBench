"""
评测阈值集中管理 — 所有 1-5 分映射阈值统一在此定义，可追溯、可校准。

设计:
- 每个物理量的阈值以 (metric_name, thresholds, direction) 三元组描述。
- direction="lower_better": 值越小越好 (误差类)，thresholds 升序，落入第 i 段得 (5-i) 分。
- direction="higher_better": 值越大越好 (准确率类)，thresholds 降序。
- 阈值来源尽量对齐领域惯例 (如深度 AbsRel<0.1 为优、法线角度误差<11.25°) 并在注释标注。

score_from_thresholds(value, thresholds, lower_better) 是统一映射入口。
"""
from typing import List


def score_lower_better(value: float, thresholds: List[float]) -> int:
    """值越小越好。thresholds 升序 [t1<t2<t3<t4]。
    value<t1→5, <t2→4, <t3→3, <t4→2, 否则1。"""
    for i, t in enumerate(thresholds):
        if value < t:
            return 5 - i
    return 1


def score_higher_better(value: float, thresholds: List[float]) -> int:
    """值越大越好。thresholds 降序 [t1>t2>t3>t4]。
    value>t1→5, >t2→4, >t3→3, >t4→2, 否则1。"""
    for i, t in enumerate(thresholds):
        if value > t:
            return 5 - i
    return 1


# ── 阈值表 (metric → thresholds) ──
# 注释标注依据来源。lower_better 用升序，higher_better 用降序。
THRESHOLDS = {
    # R01 深度: AbsRel 越小越好。0.1 为优 (KITTI/NYU 领域惯例)
    "R01_absrel":        {"th": [0.10, 0.20, 0.30, 0.50], "lower_better": True},
    # R02 法线: 平均角度误差(度)。11.25° 是法线评测标准阈 (DSINE/StableNormal)
    "R02_mean_angle":    {"th": [10.0, 15.0, 20.0, 30.0], "lower_better": True},
    # R03 形状: 归一化 Chamfer-L1 (除以点云尺度)
    "R03_chamfer_norm":  {"th": [0.02, 0.05, 0.10, 0.20], "lower_better": True},
    # R04 绝对尺度: 米制尺寸平均误差(米)
    "R04_size_err_m":    {"th": [0.10, 0.30, 0.50, 1.00], "lower_better": True},
    # R05 形变: 归一化 Chamfer / 刚体距离变化率
    # VGGT 点云重建本身有 ~0.08-0.15 的随机噪声(尤其在部分遮挡时),
    # 阈值需高于噪声基线才能避免将重建噪声误判为形变。
    # 0.15: 噪声基线以下=5分(刚体); 0.25: 轻微形变=4; 0.40: 中等形变=3
    # 0.60: 明显形变=2; >0.60=1分(剧烈形变)
    "R05_deform":        {"th": [0.15, 0.25, 0.40, 0.60], "lower_better": True},
    # R09 相对尺度: 跨帧包围盒变异系数 CV
    "R09_cv":            {"th": [0.05, 0.10, 0.20, 0.40], "lower_better": True},
    # R10 位姿: ATE(米)
    "R10_ate_m":         {"th": [0.10, 0.20, 0.50, 1.00], "lower_better": True},
    # R15 位移: Worldline L2(米)
    "R15_worldline_l2":  {"th": [0.10, 0.30, 0.50, 1.00], "lower_better": True},
    # R16 速度: 速度误差 (m/s), 以 24fps 为基准的典型运动量级
    #   <0.5 m/s=5分(十分精确), <1.0=4, <2.0=3, <5.0=2, ≥5.0=1
    "R16_vel_err":       {"th": [0.50, 1.00, 2.00, 5.00], "lower_better": True},
    # R17 角速度: 误差 (rad/s), 以 24fps 为基准
    #   <0.1 rad/s=5分, <0.3=4, <0.5=3, <1.0=2, ≥1.0=1
    "R17_angvel_err":    {"th": [0.10, 0.30, 0.50, 1.00], "lower_better": True},
    # R17 角速度自洽: 旋转 CV (自由旋转应 ω 恒定)
    "R17_rot_cv":        {"th": [0.05, 0.10, 0.20, 0.50], "lower_better": True},
    # R18/R13 光影: 光源方向跨帧角度变化(度)
    "R2x_light_angle":   {"th": [5.0, 10.0, 20.0, 45.0], "lower_better": True},
    # R19 跨帧几何: 点云跨帧 Chamfer 漂移(归一化)
    # VGGT 对同一场景相邻帧独立重建时存在 ~0.05-0.20 的随机漂移
    # (相机运动、纹理抖动等导致), 原始 0.02/0.05 阈过于严苛。
    # 放宽适配 VGGT 重建噪声基线: 0.10 内=5分, 0.25 内=4分
    "R19_drift":         {"th": [0.10, 0.25, 0.45, 0.70], "lower_better": True},
    # R19 FlowSeek 光流时序一致性: 逐帧对 AEPE 变异系数 CV
    # 低 CV = 逐帧光流量级稳定 → 跨帧几何一致; 高 CV = 帧间光流跳动 → 几何不稳定
    # 阈值: <0.10=5(极稳定), <0.20=4, <0.35=3, <0.55=2, ≥0.55=1
    "R19_flow_cv":       {"th": [0.10, 0.20, 0.35, 0.55], "lower_better": True},
    # R22/R23/R24 GT 轨迹/动量/能量误差
    "R22_traj_err":      {"th": [0.10, 0.30, 0.50, 1.00], "lower_better": True},
    "R23_mom_err":       {"th": [0.30, 0.60, 1.00, 2.00], "lower_better": True},
    "R24_energy_err":    {"th": [0.50, 1.00, 2.00, 5.00], "lower_better": True},

    # ── L2 画质层 (FQ/TQ) ──
    # FQ-1 主体一致性: DINOv3 相邻帧余弦相似度 (higher_better)。
    #   相邻帧天然相似(0.85~0.99), 不能线性 *5 映射(会全 5 分)。
    #   在 0.85~0.99 区间细分: 越接近 1 越一致。
    # ⚠️ 2026-09-15 按 MiniMax-H3 分布校准 (n=42): 实测全部 >0.98 (旧阈值
    #    失效全 5)。新阈值=分布 p75/p50/p25/p10, 在该分布上各档约 25%。
    #    跨数据分布使用时应重新校准 (metrics 保留原始 sim 可溯)。
    "FQ1_subject_cons":  {"th": [0.9964, 0.9940, 0.9910, 0.9881], "lower_better": False},
    # TQ-1 运动平滑度: 光流加速度平滑度 (higher_better, 0~1)。
    #   motion_smoothness 返回 max(0, 1-avg_acc/2), 通常 0.9~1.0。
    #   在 0.9~1.0 区间细分。
    "TQ1_smoothness":    {"th": [0.98, 0.95, 0.90, 0.80], "lower_better": False},
    # TQ-4 背景一致性: CLIP 相邻帧图像特征余弦相似度 (higher_better)。
    #   通常 0.5~0.9, 需区分背景是否漂移。
    # ⚠️ 2026-09-15 按 MiniMax-H3 分布校准 (n=48): CLIP 修复 (5.x 返回类型)
    #    后实测全部 >0.90 (旧阈值失效全 5)。新阈值=分布 p75/p50/p25/p10,
    #    各档约 25%; 跨数据分布使用时应重新校准 (metrics 保留原始值可溯)。
    "TQ4_bg_cons":       {"th": [0.9912, 0.9836, 0.9741, 0.9640], "lower_better": False},
}


def score_metric(key: str, value: float) -> int:
    """按 THRESHOLDS 表给单个指标打分。"""
    if key not in THRESHOLDS:
        raise KeyError(f"未定义阈值: {key}")
    spec = THRESHOLDS[key]
    if spec["lower_better"]:
        return score_lower_better(value, spec["th"])
    return score_higher_better(value, spec["th"])
