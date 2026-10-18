#!/usr/bin/env python3
"""Public PVBench evaluation entry point.

Examples:
  python evaluate.py --video /path/to/video.mp4 --config configs/config.yaml
  python evaluate.py --video-dir /path/to/videos --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))


def main() -> None:
    parser = argparse.ArgumentParser(description="PVBench video evaluation")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", "-v", help="evaluate one video with the physical layer")
    inputs.add_argument("--video-dir", help="evaluate all videos in a directory with global metrics")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="YAML configuration path")
    parser.add_argument("--prompt", "-p", default="", help="prompt associated with the video")
    parser.add_argument("--modules", nargs="+", default=None, help="physical modules to run")
    args = parser.parse_args()

    if args.video_dir:
        video_dir = Path(args.video_dir)
        if not video_dir.is_dir():
            parser.error(f"video directory does not exist: {video_dir}")
        from pipeline_global import GlobalBenchmarkPipeline
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = sorted(
            str(path) for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not videos:
            parser.error(f"no supported videos found in: {video_dir}")
        mapping = [(Path(video).stem, video) for video in videos]
        pipeline = GlobalBenchmarkPipeline(config_path=args.config)
        pipeline.initialize()
        pipeline.run(prompt_video_mapping=mapping)
        return

    video_path = Path(args.video)
    if not video_path.is_file():
        parser.error(f"video file does not exist: {video_path}")
    from pipeline_physics import PhysicsPipeline
    pipeline = PhysicsPipeline(config_path=args.config)
    pipeline.initialize()
    pipeline.run(video_path=str(video_path), modules=args.modules, prompt=args.prompt)


if __name__ == "__main__":
    main()
