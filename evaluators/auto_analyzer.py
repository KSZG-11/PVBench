"""
多物理自动分析器 — 根据 prompt/视频内容自动判断应评测哪些物理维度

设计思路:
- 每个物理维度对应一组关键词 (中英文)
- 从 prompt 文本 + 可选的 VLM 首帧快速扫描中匹配
- 返回应启用的模块列表 + 维度列表
- 不匹配的维度自动跳过, 节省计算资源

使用方式:
    1. 仅基于 prompt 关键词 (快速, 适合批处理):
       analyzer = PhysicsAutoAnalyzer()
       enabled_modules = analyzer.analyze_from_prompt(prompt_text)

    2. 基于 prompt + VLM 首帧 (更准确, 需要额外 VLM 调用):
       analyzer = PhysicsAutoAnalyzer(vlm=vlm_instance)
       enabled_modules, dimensions = analyzer.analyze_full(prompt_text, first_frame)
"""

import re
from typing import Optional

# ── 每个物理维度的关键词映射 ──
# 方案: prompt 里包含这些词 → 大概率该维度有评测意义
DIMENSION_KEYWORDS = {
    # 几何组
    "R01": {
        "module": "geometry",
        "zh": ["深度", "距离", "前景", "背景", "近", "远", "地平线", "透视", "消失点"],
        "en": ["depth", "distance", "foreground", "background", "horizon",
               "perspective", "vanishing", "near", "far", "meters away",
               "road", "corridor"],
    },
    "R02": {
        "module": "geometry",
        "zh": ["法线", "曲面", "光照方向", "明暗", "表面", "光照", "石膏", "布料", "褶皱"],
        "en": ["normal", "surface", "shading", "specular", "diffuse",
               "bust", "cloth", "draped", "lighting", "plaster"],
    },
    "R03": {
        "module": "geometry",
        "zh": ["形状", "几何", "点云", "模型", "旋转", "多视角", "转盘", "三维"],
        "en": ["shape", "geometric", "point cloud", "turntable", "rotating",
               "sphere", "cube", "cone", "cylinder", "wooden", "model"],
    },
    # 标度位姿组
    "R04": {
        "module": "scale_pose",
        "zh": ["绝对尺度", "尺寸", "厘米", "米", "高度", "长度", "尺", "参照"],
        "en": ["scale", "size", "centimeter", "meter", "ruler", "height",
               "reference", "measurement"],
    },
    "R09": {
        "module": "scale_pose",
        "zh": ["相对尺度", "远近", "由远及近", "远离", "靠近", "尺寸变化", "上升", "气球"],
        "en": ["relative", "approaching", "receding", "closer", "further",
               "balloon", "rolling toward", "from far", "corridor"],
    },
    "R10": {
        "module": "scale_pose",
        "zh": ["位姿", "轨迹", "路径", "无人机", "行走", "定位", "相机位姿", "圆形"],
        "en": ["pose", "trajectory", "drone", "walking", "circular",
               "L-path", "L-shape", "ATE", "position"],
    },
    # 形变空间组
    "R05": {
        "module": "deformation_spatial",
        "zh": ["形变", "变形", "挤压", "压入", "弹性", "软体", "刚体", "黏土", "橡胶"],
        "en": ["deformation", "deform", "squeeze", "press", "elastic",
               "clay", "rubber", "stress ball", "compress"],
    },
    "R11": {
        "module": "deformation_spatial",
        "zh": ["空间关系", "堆叠", "遮挡", "穿过", "上面", "下面", "旁边", "层级", "火车"],
        "en": ["spatial", "stack", "on top", "behind", "occlusion",
               "layer", "tower", "cube", "train", "carriages"],
    },
    # 静力学光影组
    "R12": {
        "module": "statics_lighting",
        "zh": ["稳定", "静止", "不动", "积木", "塔", "桌子", "放置", "静态"],
        "en": ["stability", "stable", "static", "stationary",
               "Jenga", "tower", "still", "still life", "tripod"],
    },
    "R13": {
        "module": "statics_lighting",
        "zh": ["光影", "光源", "阴影", "光照一致", "灯光", "太阳", "影子"],
        "en": ["lighting", "shadow", "lamp", "light source", "sun",
               "shading", "matte sphere", "illumination"],
    },
    "R18": {
        "module": "statics_lighting",
        "zh": ["光影时间", "时间光照", "持续", "蜡烛", "静物", "无风"],
        "en": ["temporal lighting", "candle", "still life",
               "continuous", "draft-free", "fixed light"],
    },
    # VLM 评测组
    "R06": {
        "module": "material",
        "zh": ["材料", "材质", "弹性", "硬度", "玻璃", "陶瓷", "钢铁", "木头", "橡胶", "弹跳"],
        "en": ["material", "glass", "ceramic", "steel", "wood", "rubber",
               "bounce", "marble", "elasticity", "stiffness", "drop"],
    },
    "R07": {
        "module": "thermal",
        "zh": ["热", "温度", "火焰", "高温", "烧", "辐射", "热浪", "熔", "锻", "炭火"],
        "en": ["thermal", "heat", "flame", "forge", "glowing", "radiation",
               "charcoal", "grill", "embers", "steam", "sizzling"],
    },
    "R14": {
        "module": "electromag",
        "zh": ["磁", "电", "磁铁", "静电", "电荷", "吸引", "排斥", "火花", "回形针"],
        "en": ["magnet", "electric", "electrostatic", "magnetic",
               "Van de Graaff", "paperclip", "spark", "charge"],
    },
    "R25": {
        "module": "phase_change",
        "zh": ["相变", "融化", "熔化", "凝固", "蒸发", "沸腾", "冰", "水", "巧克力"],
        "en": ["phase change", "melt", "melting", "freeze", "boil",
               "ice", "chocolate", "evaporate", "liquid", "solid"],
    },
    # 运动学组
    "R15": {
        "module": "kinematics",
        "zh": ["位移", "运动轨迹", "滚动", "摆动", "路径", "台球", "节拍器", "单摆"],
        "en": ["displacement", "trajectory", "roll", "billiard",
               "pendulum", "metronome", "path", "swing"],
    },
    "R16": {
        "module": "kinematics",
        "zh": ["速度", "加速度", "加速", "减速", "斜面", "下滑", "起步", "匀速"],
        "en": ["velocity", "acceleration", "accelerate", "decelerate",
               "ramp", "incline", "speed", "starting", "cruising"],
    },
    "R17": {
        "module": "kinematics",
        "zh": ["角速度", "旋转", "转动", "转", "风扇", "轮子", "自旋", "电动机"],
        "en": ["angular velocity", "rotation", "rotate", "spin",
               "fan", "wheel", "ceiling fan", "bicycle wheel"],
    },
    # 运动持续性组
    "R19": {
        "module": "motion_persistence",
        "zh": ["跨帧", "连续帧", "漂移", "推轨", "环绕", "柱廊", "雕像", "相机运动"],
        "en": ["cross-frame", "drift", "dolly", "orbital",
               "corridor", "columns", "statue", "camera movement"],
    },
    "R20": {
        "module": "motion_persistence",
        "zh": ["运动连续", "连续运动", "滑冰", "游", "流畅", "螺旋", "曲线", "无跳帧"],
        "en": ["motion continuity", "skater", "swim", "figure-eight",
               "loop", "glide", "smooth", "continuous", "spiral"],
    },
    "R21": {
        "module": "motion_persistence",
        "zh": ["恒存", "遮挡", "消失", "重现", "经过", "穿过", "后面", "沙发", "树"],
        "en": ["permanence", "occlusion", "behind", "emerge",
               "sofa", "tree trunk", "pass behind", "disappear", "reappear"],
    },
    # 动力学组
    "R22": {
        "module": "dynamics",
        "zh": ["力", "推", "拉", "弹簧", "发射", "施力", "受力", "因果", "弹射"],
        "en": ["force", "push", "pull", "spring", "launch",
               "cannon", "dynamic", "causal", "pushing"],
    },
    "R23": {
        "module": "dynamics",
        "zh": ["动量", "碰撞", "守恒", "牛顿摆", "冰球", "撞击", "对撞", "相碰"],
        "en": ["momentum", "collision", "conservation",
               "Newton's cradle", "puck", "head-on", "strike", "impact"],
    },
    "R24": {
        "module": "dynamics",
        "zh": ["能量", "守恒", "势能", "动能", "单摆", "弹跳", "下落", "周期", "振荡"],
        "en": ["energy", "conservation", "potential", "kinetic",
               "pendulum", "bounce", "drop", "oscillation", "cycle"],
    },
}


class PhysicsAutoAnalyzer:
    """根据 prompt 文本自动判断应启用的物理评测维度。

    核心原理: 每条 prompt 描述物理实验场景, 关键词直接关联所需的物理量。
    例如 "Jenga tower" → R12 静力学; "Newton's cradle" → R23 动量。

    返回的数据可用于:
    - PhysicsPipeline.run(..., modules=..., target_dimensions=...)
    - 显存分组优化 (只加载需要的模型)
    """

    def __init__(self, default_enable: bool = True):
        """
        Args:
            default_enable: 无关键词匹配时的行为。
                True  = 全部启用 (保守模式, 不遗漏)
                False = 全部跳过 (严格模式, 仅评测明确相关的)
        """
        self.default_enable = default_enable
        # 预处理: 构建从 module → dimensions 的映射
        self._module_dims: dict[str, list[str]] = {}
        for dim, info in DIMENSION_KEYWORDS.items():
            mod = info["module"]
            self._module_dims.setdefault(mod, []).append(dim)

    def analyze_from_prompt(self, prompt: str) -> dict:
        """从 prompt 文本自动判断应启用的模块和维度。

        Args:
            prompt: T2V prompt 文本

        Returns:
            {
                "enabled_modules": set of module names,
                "enabled_dimensions": set of dimension IDs (e.g., "R01", "R15"),
                "matched": {dim: [matched_keywords, ...]},
                "unmatched": set of dimensions  (未匹配, 按 default_enable 决定启用与否)
            }
        """
        text_lower = prompt.lower()
        matched_dims: dict[str, list[str]] = {}
        matched_modules: set[str] = set()
        all_dims = set(DIMENSION_KEYWORDS.keys())

        for dim, info in DIMENSION_KEYWORDS.items():
            found = []
            for kw in info["zh"]:
                if kw in text_lower:
                    found.append(f"zh:{kw}")
            for kw in info["en"]:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                    found.append(f"en:{kw}")
            if found:
                matched_dims[dim] = found[:5]  # 最多保留5个
                matched_modules.add(info["module"])

        unmatched_dims = all_dims - set(matched_dims.keys())

        if self.default_enable:
            # 保守: 未匹配的维度也启用 (担心漏掉)
            enabled_dims = all_dims
            enabled_modules = set(info["module"] for info in DIMENSION_KEYWORDS.values())
        else:
            enabled_dims = set(matched_dims.keys())
            enabled_modules = matched_modules

        return {
            "enabled_modules": sorted(enabled_modules),
            "enabled_dimensions": sorted(enabled_dims),
            "matched": {d: kw for d, kw in matched_dims.items()},
            "unmatched": sorted(unmatched_dims),
        }

    def analyze_with_confidence(self, prompt: str) -> dict:
        """带置信度的分析 — 对每维度输出匹配强度 (0.0~1.0)。

        用于更精细的决策, 例如:
        - confidence >= 0.7: 必定启用
        - confidence 0.3~0.7: 启用但标记 low_confidence
        - confidence < 0.3: 跳过
        """
        text_lower = prompt.lower()
        results = {}
        all_dims = set(DIMENSION_KEYWORDS.keys())

        for dim, info in DIMENSION_KEYWORDS.items():
            zh_matches = sum(1 for kw in info["zh"] if kw in text_lower)
            en_matches = sum(
                1 for kw in info["en"]
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower)
            )
            # 置信度: 中文匹配权重稍高 (T2V prompt 多为英文)
            total_kw = len(info["zh"]) + len(info["en"])
            weighted = (zh_matches * 0.4 + en_matches * 0.6)
            confidence = min(1.0, weighted / max(total_kw * 0.15, 1))
            results[dim] = {
                "confidence": round(confidence, 3),
                "zh_matches": zh_matches,
                "en_matches": en_matches,
            }

        high_conf = {d for d, r in results.items() if r["confidence"] >= 0.5}
        low_conf = {d for d, r in results.items() if 0.1 <= r["confidence"] < 0.5}
        skipped = {d for d, r in results.items() if r["confidence"] < 0.1}

        enabled_modules = set(
            DIMENSION_KEYWORDS[d]["module"]
            for d in (high_conf | (low_conf if self.default_enable else set()))
        )

        return {
            "enabled_modules": sorted(enabled_modules),
            "high_confidence_dims": sorted(high_conf),
            "low_confidence_dims": sorted(low_conf),
            "skipped_dims": sorted(skipped),
            "per_dimension": results,
        }

    def get_modules_for_dimensions(self, dimensions: list[str]) -> list[str]:
        """根据维度列表反推需要启用的模块。"""
        modules = set()
        for dim in dimensions:
            if dim in DIMENSION_KEYWORDS:
                modules.add(DIMENSION_KEYWORDS[dim]["module"])
        return sorted(modules)

    @property
    def all_modules(self) -> list[str]:
        return sorted(set(info["module"] for info in DIMENSION_KEYWORDS.values()))

    @property
    def all_dimensions(self) -> list[str]:
        return sorted(DIMENSION_KEYWORDS.keys())
