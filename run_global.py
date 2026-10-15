#!/usr/bin/env python3
"""
物理 3D/4D WAM Benchmark — 入口 1: 全局 Benchmark（逻辑 + 画质）

评测视频生成模型的整体能力，支持 T2V / I2V / CI2V 三种模态.

用法:
    # T2V: 预生成的视频文件 + prompt 测试集
    python run_global.py --video-dir ./generated_videos/ --config config.yaml

    # I2V/CI2V: 含参考图的映射文件
    python run_global.py --mapping mapping.json --config config.yaml

    # 只跑逻辑层
    python run_global.py --video-dir ./videos/ --layers logic

mapping.json 格式 (T2V):
[
    {"prompt": "a red car driving on a mountain road", "video": "videos/001.mp4"},
    ...
]

mapping.json 格式 (I2V):
[
    {"prompt": "a dog running in a park", "video": "videos/001.mp4",
     "ref_image": "refs/scene_001.jpg", "modality": "I2V"},
    ...
]

mapping.json 格式 (CI2V):
[
    {"prompt": "the character waving hands", "video": "videos/001.mp4",
     "ref_image": "refs/char_001.png", "modality": "CI2V"},
    ...
]
"""
import argparse
import json
import os
import sys
import glob
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_global import GlobalBenchmarkPipeline


def find_videos(directory: str) -> list[str]:
    """在目录中查找视频文件"""
    video_exts = {"*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.gif"}
    videos = []
    for ext in video_exts:
        videos.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(videos)


def build_mapping_from_dir(video_dir: str,
                           prompts_file: Optional[str] = None
                           ) -> list:
    """从目录构建评测映射，并在 JSONL 模式保留 sample_id。

    JSONL 记录应包含 ``sample_id`` 和 ``t2v_prompt``（或 ``prompt``）；
    视频文件名包含同一个 sample_id 时按 ID 对齐，避免按目录排序错配。
    普通文本文件继续按行与视频排序对齐。
    """
    videos = find_videos(video_dir)
    if not videos:
        print(f"Error: 在 {video_dir} 中未找到视频文件")
        return []

    if prompts_file and os.path.exists(prompts_file) and prompts_file.lower().endswith(".jsonl"):
        records = []
        with open(prompts_file, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{prompts_file}:{line_no}: JSONL解析失败: {exc}") from exc
                sample_id = str(item.get("sample_id") or item.get("prompt_id") or "")
                prompt = item.get("t2v_prompt") or item.get("prompt")
                dims = item.get("dimensions") or []
                if isinstance(dims, str):
                    dims = [d.strip() for d in dims.split("|") if d.strip()]
                if sample_id and prompt:
                    records.append((sample_id, str(prompt), list(dims)))
        by_id = {
            sample_id: video for video in videos
            for sample_id in [os.path.basename(video).split("_", 1)[0]]
        }
        mapping = []
        for sample_id, prompt, dims in records:
            video = by_id.get(sample_id)
            if video:
                mapping.append({
                    "prompt_id": sample_id,
                    "prompt": prompt,
                    "video": video,
                    "modality": "T2V",
                    "dimensions": dims,
                })
        return mapping

    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    else:
        prompts = [os.path.splitext(os.path.basename(v))[0] for v in videos]

    n = min(len(prompts), len(videos))
    return [(prompts[i], videos[i]) for i in range(n)]


def main():
    parser = argparse.ArgumentParser(
        description="物理 3D/4D WAM 全局 Benchmark — 评测模型整体能力 (T2V/I2V/CI2V)"
    )
    parser.add_argument("--video-dir", default=None,
                        help="生成视频所在目录")
    parser.add_argument("--mapping", default=None,
                        help="JSON 映射文件 (支持 prompt/video/ref_image/modality 字段)")
    parser.add_argument("--prompts-file", default=None,
                        help="Prompt文本文件（一行一个）或含sample_id/t2v_prompt的JSONL")
    parser.add_argument("--canonical-ref-dir", default=None,
                        help="T2V Canonical Reference Proxy参考图目录，文件名按prompt_id.png")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--layers", default="all",
                        choices=["all", "logic", "quality"],
                        help="只跑指定层 (默认全部)")
    parser.add_argument("--skip-init", action="store_true",
                        help="跳过模型初始化")
    args = parser.parse_args()

    # 构建映射
    if args.mapping:
        with open(args.mapping, "r") as f:
            data = json.load(f)
        # 支持新格式 (dict with ref_image/modality) 和旧格式
        mapping = []
        canonical_dir = args.canonical_ref_dir
        for item in data:
            if isinstance(item, dict):
                item = dict(item)
                if canonical_dir and item.get("prompt_id"):
                    canonical_path = os.path.join(canonical_dir, f"{item['prompt_id']}.png")
                    if os.path.exists(canonical_path):
                        item["canonical_ref_image"] = canonical_path
                        item["reference_role"] = "t2v_canonical"
                mapping.append(item)
            elif isinstance(item, list):
                mapping.append(tuple(item))
        n_refs = sum(
            bool(m.get("ref_image") or m.get("canonical_ref_image"))
            if isinstance(m, dict) else bool(len(m) >= 3 and m[2])
            for m in mapping
        )
        print(f"[Global Benchmark] 映射加载: {len(mapping)} 条, "
              f"其中含参考图: {n_refs} 条")
    elif args.video_dir:
        mapping = build_mapping_from_dir(args.video_dir, args.prompts_file)
    else:
        print("Error: 请指定 --video-dir 或 --mapping")
        sys.exit(1)

    if not mapping:
        print("Error: 未找到任何 (prompt, video) 对")
        sys.exit(1)

    print(f"[Global Benchmark] 共 {len(mapping)} 个评测案例")

    # 初始化管线
    pipeline = GlobalBenchmarkPipeline(config_path=args.config)

    if not args.skip_init:
        try:
            pipeline.initialize()
        except Exception as e:
            print(f"模型初始化失败: {e}")
            print("继续使用 stub 模式...")

    # 运行
    run_logic = args.layers in ("all", "logic")
    run_quality = args.layers in ("all", "quality")

    result = pipeline.run(
        prompt_video_mapping=mapping,
        run_logic=run_logic,
        run_quality=run_quality,
    )

    print(f"\nFinal Score: {result.overall_score():.2f} / 5.00")


if __name__ == "__main__":
    main()
