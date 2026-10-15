#!/usr/bin/env python3
"""
向后兼容：请使用新的入口脚本

- 全局 Benchmark (逻辑 + 画质): python run_global.py --video-dir <dir>
- 物理层评测: python run_physics.py --video <path>

用法:
    python run.py --video <path> --prompt "..."       # 物理评测
    python run.py --video-dir <dir>                    # 全局评测
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_global import GlobalBenchmarkPipeline
from pipeline_physics import PhysicsPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Physical 3D/4D WAM Benchmark — 兼容入口"
    )
    parser.add_argument("--video", "-v", default=None, help="视频文件路径（物理评测）")
    parser.add_argument("--video-dir", default=None, help="视频目录（全局评测）")
    parser.add_argument("--prompt", "-p", default="", help="T2V prompt")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    parser.add_argument("--modules", nargs="+", default=None, help="物理评测模块")
    parser.add_argument("--skip-init", action="store_true")

    args = parser.parse_args()

    if args.video_dir:
        # 全局模式
        glob_pattern = os.path.join(args.video_dir, "*")
        import glob
        videos = sorted(glob.glob(glob_pattern))
        videos = [v for v in videos if v.endswith(('.mp4', '.avi', '.mov', '.webm'))]
        mapping = [
            (os.path.splitext(os.path.basename(v))[0], v)
            for v in videos
        ]
        pipeline = GlobalBenchmarkPipeline(config_path=args.config)
        if not args.skip_init:
            pipeline.initialize()
        pipeline.run(prompt_video_mapping=mapping)

    elif args.video:
        # 物理模式
        pipeline = PhysicsPipeline(config_path=args.config)
        if not args.skip_init:
            pipeline.initialize()
        pipeline.run(video_path=args.video, modules=args.modules)

    else:
        print("请指定 --video 或 --video-dir\n")
        print("推荐使用新入口:")
        print("  python run_global.py --video-dir <dir>     # 全局Benchmark")
        print("  python run_physics.py --video <path>        # 物理评测")


if __name__ == "__main__":
    main()
