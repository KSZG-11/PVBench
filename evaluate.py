#!/usr/bin/env python3
"""Public PVBench evaluation entry point.

Examples:
  python evaluate.py --video /path/to/video.mp4 --config configs/config.yaml
  python evaluate.py --video-dir /path/to/videos --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))



def main() -> None:
    parser = argparse.ArgumentParser(description="PVBench video evaluation")
    parser.add_argument("--video", "-v", help="evaluate one video with the physical layer")
    parser.add_argument("--video-dir", help="evaluate all videos in a directory with global metrics")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="YAML configuration path")
    parser.add_argument("--prompt", "-p", default="", help="prompt associated with the video")
    parser.add_argument("--modules", nargs="+", default=None, help="physical modules to run")
    parser.add_argument("--skip-init", action="store_true", help="skip model initialization")
    args = parser.parse_args()

    if args.video_dir:
        from pipeline_global import GlobalBenchmarkPipeline
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = sorted(
            str(path) for path in Path(args.video_dir).iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        mapping = [(Path(video).stem, video) for video in videos]
        pipeline = GlobalBenchmarkPipeline(config_path=args.config)
        if not args.skip_init:
            pipeline.initialize()
        pipeline.run(prompt_video_mapping=mapping)
        return

    if args.video:
        from pipeline_physics import PhysicsPipeline
        pipeline = PhysicsPipeline(config_path=args.config)
        if not args.skip_init:
            pipeline.initialize()
        pipeline.run(video_path=args.video, modules=args.modules, prompt=args.prompt)
        return

    parser.error("provide either --video or --video-dir")


if __name__ == "__main__":
    main()
