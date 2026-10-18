#!/usr/bin/env python3
"""
物理 3D/4D WAM Benchmark — 入口 2: 物理层评测（逐案例分析）

评测单个视频中特定物理现象的保真度。
每个物理量使用 物理量分类体系.xlsx D列 指定的专用模型。

用法:
    # 评测单个视频的全部物理量
    python run_physics.py --video falling_glass.mp4

    # 指定评测模块
    python run_physics.py --video bouncing_ball.mp4 --modules kinematics dynamics

    # 指定场景类型(影响动力学评测)
    python run_physics.py --video free_fall.mp4 --scene-type projectile

    # 带 ground truth 评测
    python run_physics.py --video simulation.mp4 --gt-depth depth.npy --gt-normals normals.npy
"""
import argparse
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CORE_DIR)

from pipeline_physics import PhysicsPipeline

# 所有支持的物理评测模块
AVAILABLE_MODULES = [
    "geometry",            # R01深度+R02法线+R03形状 (VGGT-Omega + StableNormal)
    "scale_pose",          # R04绝对尺度+R09相对尺度+R10位姿 (DA3+SAM3+Any6D+VGGT+Qwen3.8-27B)
    "deformation_spatial", # R05形变+R11空间关系 (VGGT+SpatialTracker+SAM3)
    "statics_lighting",    # R12稳定性+R13光影空间+R18光影时间
    "material",            # R06材料属性 (Qwen3.8-27B)
    "thermal",             # R07热辐射 (Qwen3.8-27B)
    "electromag",          # R14磁电 (Qwen3.8-27B)
    "phase_change",        # R25相变 (Qwen3.8-27B)
    "kinematics",          # R15位移+R16速度加速度+R17角速度
    "motion_persistence",  # R19跨帧几何+R20运动连续性+R21物体恒存
    "dynamics",            # R22力效应+R23动量+R24能量
    "multiview",           # R08多视图几何 (需要多视角输入)
]


def main():
    parser = argparse.ArgumentParser(
        description="物理 3D/4D WAM 物理层评测 — 逐案例物理量分析"
    )
    parser.add_argument("--video", "-v", required=True,
                        help="视频文件路径")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--modules", nargs="+", default=None,
                        choices=AVAILABLE_MODULES,
                        help="指定评测模块，默认全部启用")
    parser.add_argument("--target-dimensions", nargs="+", default=None,
                        help="只保留指定维度结果，例如 --target-dimensions R10")
    parser.add_argument("--scene-type", default="unspecified",
                        choices=["unspecified", "free_fall", "projectile",
                                 "collision", "incline", "pendulum",
                                 "heating", "phase_change", "magnetic"],
                        help="场景类型(影响动力学/相变/磁电评测的假设)")
    parser.add_argument("--expected-materials", nargs="+", default=None,
                        help="期望的材料列表，如 '玻璃 水'")
    parser.add_argument("--prompt", default=None,
                        help="视频的 prompt 文本 (用于 R04 物体验证等需要 prompt 的维度; 不传则相关过滤失效)")
    parser.add_argument("--evaluation-method", default=None,
                        help="评测方法 (R08: temporal_multiview_proxy / continuous_view_sequence_proxy; "
                             "R21: storyboard_object_persistence / continuous_object_persistence)")
    parser.add_argument("--storyboard-json", default=None,
                        help="storyboard JSON 字符串 (如 '{\"enabled\":true,\"mode\":\"temporal_multiview\","
                             "\"total_seconds\":10,\"phase_count\":3,\"phases\":[...]}')")

    # Ground truth 输入 (用于有GT的CI2V场景)
    parser.add_argument("--gt-depth", default=None,
                        help="GT深度图 .npy 文件")
    parser.add_argument("--gt-normals", default=None,
                        help="GT法向量 .npy 文件")
    parser.add_argument("--gt-trajectories", default=None,
                        help="GT轨迹 .npy 文件 [N, T, 3]")
    parser.add_argument("--gt-poses", default=None,
                        help="GT位姿 .npy 文件")
    parser.add_argument("--gt-sizes", default=None,
                        help="GT物体尺寸 JSON 文件")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: 视频文件未找到: {args.video}")
        sys.exit(1)

    # 构建 kwargs
    kwargs = {"scene_type": args.scene_type}
    if args.expected_materials:
        kwargs["expected_materials"] = args.expected_materials
    if args.evaluation_method:
        kwargs["evaluation_method"] = args.evaluation_method
    if args.storyboard_json:
        try:
            import json as _json
            kwargs["storyboard"] = _json.loads(args.storyboard_json)
        except Exception as e:
            print(f"  [Warn] --storyboard-json 解析失败, 忽略: {e}")

    # 加载 GT 数据
    gt_files = {
        "depth_gt": args.gt_depth,
        "normal_gt": args.gt_normals,
        "gt_trajectories": args.gt_trajectories,
        "gt_poses": args.gt_poses,
    }
    for key, path in gt_files.items():
        if path and os.path.exists(path):
            if path.endswith(".npy"):
                kwargs[key] = np.load(path)
            elif path.endswith(".json"):
                import json
                with open(path) as f:
                    kwargs[key] = json.load(f)
            print(f"  Loaded GT: {key} from {path}")

    # 初始化管线
    modules = args.modules
    if modules is None:
        print(f"[Physics] 评测全部 {len(AVAILABLE_MODULES)} 个物理模块")
    else:
        print(f"[Physics] 评测指定模块: {modules}")

    pipeline = PhysicsPipeline(config_path=args.config)

    try:
        pipeline.initialize()
    except Exception as e:
        # 单模型加载失败不中断；相关维度会记为 error。
        print(f"模型初始化部分失败(软失败, 相关维度将记 status=error): {e}")

    # 运行
    result = pipeline.run(
        video_path=args.video,
        modules=modules,
        target_dimensions=args.target_dimensions,
        prompt=args.prompt,
        **kwargs
    )


if __name__ == "__main__":
    main()
