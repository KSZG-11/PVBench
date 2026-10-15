"""连续物理维度编号及旧数据兼容映射。"""

CONTIGUOUS_PHYSICS_DIMENSIONS = {
    "R01": "深度",
    "R02": "表面法线",
    "R03": "形状",
    "R04": "绝对尺度",
    "R05": "当前形变状态",
    "R06": "材料属性",
    "R07": "热辐射",
    "R08": "多视图几何一致性",
    "R09": "相对尺度",
    "R10": "三维位姿",
    "R11": "空间关系",
    "R12": "稳定性",
    "R13": "光影空间一致性",
    "R14": "磁电作用关系",
    "R15": "位移",
    "R16": "速度与加速度",
    "R17": "角速度",
    "R18": "光影时间一致性",
    "R19": "跨帧几何一致性",
    "R20": "运动连续性",
    "R21": "物体恒存",
    "R22": "力的动态效应",
    "R23": "动量守恒",
    "R24": "能量守恒",
    "R25": "相变",
}

LEGACY_TO_CONTIGUOUS = {
    f"R{legacy:02d}": f"R{index:02d}"
    for index, legacy in enumerate(
        (2, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 19, 20, 21, 22,
         24, 26, 27, 28, 30, 32, 34, 38, 41, 43),
        1,
    )
}


def normalize_dimension_id(dimension: str) -> str:
    """将旧Manifest中的R编号转换为当前连续编号。"""
    return LEGACY_TO_CONTIGUOUS.get(dimension, dimension)


PHYSICS_DIMENSIONS = tuple(CONTIGUOUS_PHYSICS_DIMENSIONS)
