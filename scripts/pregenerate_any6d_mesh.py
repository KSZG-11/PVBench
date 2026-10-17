#!/usr/bin/env python3
"""
预生成 Any6D 所需的 InstantMesh mesh（独立进程，避免显存争抢）。

背景：
  主评测循环里 Any6D(R10/R17) 的 prepare_auto_inputs 会临时加载 InstantMesh
  的 Zero123++ 扩散模型（~15-30GB），与常驻的 VLM(27GB 8bit) 等模型叠加会 OOM。
  且 VLM 8bit 无法干净 offload（bitsandbytes<0.48），反复卸载会显存泄漏。

方案：
  在独立进程里（只加载 SAM3 + DA3 + Any6D，不加载 VLM），把所有含 R10/R17
  维度的视频的 mesh 提前生成并缓存到 results/physics/any6d_auto/<video>/mesh_auto.obj。
  主循环遇到已缓存的 mesh 时直接复用，不再触发 InstantMesh。

用法:
  python scripts/pregenerate_any6d_mesh.py \
      --manifest data/manifest_minimax_h3_600.json \
      [--only-missing] [--limit N]
"""
import os
import sys
import json
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))

_hf_cache = os.environ.get("HF_HOME") or os.path.join(PROJECT_ROOT, "hf_cache")
os.environ["HF_HOME"] = _hf_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = _hf_cache

from sampler import FrameSampler
from models.loader import ModelLoader
import yaml


def load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    validated = []
    for item in manifest:
        if "prompt" not in item or "video" not in item:
            continue
        validated.append(item)
    return validated


def needs_mesh(item: dict) -> bool:
    """该视频是否含 R10/R17 维度（需要 Any6D mesh）。"""
    dims = item.get("dimensions", [])
    return any(d in ("R10", "R17") for d in dims)


def mesh_cache_path(config, video_path: str) -> str:
    output_dir = config.get("global", {}).get("output_dir", "./results")
    # ⚠️ 缓存 key 必须区分视频来源目录：T2V(minimax_h3_outputs_480p) 与
    #    I2V(minimax_h3_nf4_i2v_outputs) 的视频文件名相同(都是 U46V2-XXXX)，
    #    若只用 basename 会导致 I2V 误复用 T2V 的 mesh(首帧内容不同)。
    #    故 key 用「父目录名 + 文件名」组合。
    parent = os.path.basename(os.path.dirname(video_path))
    base = os.path.splitext(os.path.basename(video_path))[0]
    key = f"{parent}_{base}"
    return os.path.join(output_dir, "physics", "any6d_auto", key, "mesh_auto.obj")


def main():
    parser = argparse.ArgumentParser(description="预生成 Any6D InstantMesh mesh")
    parser.add_argument("--manifest", "-m", required=True,
                        help="manifest JSON 文件路径")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--only-missing", action="store_true",
                        help="仅生成缺失的 mesh（跳过已缓存的）")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 个需要 mesh 的视频")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest = load_manifest(args.manifest)
    targets = [x for x in manifest if needs_mesh(x)]
    if args.limit:
        targets = targets[:args.limit]

    print(f"[PreGen] 需要 mesh 的视频: {len(targets)} 个")

    # 统计缺失
    missing = []
    for x in targets:
        mp = mesh_cache_path(config, x["video"])
        if not (os.path.isfile(mp) and os.path.getsize(mp) > 0):
            missing.append(x)
    print(f"[PreGen] 缺失 mesh: {len(missing)} 个")
    if args.only_missing:
        targets = missing
        print(f"[PreGen] 仅处理缺失: {len(targets)} 个")

    if not targets:
        print("[PreGen] 无需要生成的目标，退出")
        return

    # 只加载 SAM3 + DA3 + Any6D（不加载 VLM 等，显存充足）
    print("[PreGen] 加载 SAM3 + DA3 + Any6D ...")
    loader = ModelLoader(config)
    sam3 = loader.get_sam3()
    da3 = loader.get_da3()
    any6d = loader.get_any6d()
    if not getattr(any6d, "is_loaded", False):
        print("[PreGen] Any6D 未加载，退出")
        return

    sampler = FrameSampler(config.get("sampling", {}))

    ok = 0
    failed = []
    for i, item in enumerate(targets):
        video = item["video"]
        pid = item.get("prompt_id", "?")
        mp = mesh_cache_path(config, video)
        if os.path.isfile(mp) and os.path.getsize(mp) > 0:
            print(f"[{i+1}/{len(targets)}] {pid}: mesh 已缓存，跳过")
            ok += 1
            continue

        print(f"[{i+1}/{len(targets)}] {pid}: 生成 mesh ...")
        frames, _ = sampler.sample_from_video(video)
        if not frames:
            print(f"  [SKIP] 无法采样帧: {video}")
            failed.append((pid, "采样失败"))
            continue

        output_dir = os.path.dirname(mp)
        try:
            auto = any6d.prepare_auto_inputs(
                np.stack([np.asarray(f) for f in frames]),
                sam3=sam3,
                da3=da3,
                dimension="R10",
                mesh_path=config.get("models", {}).get("physics", {})
                          .get("any6d", {}).get("mesh_path"),
                output_dir=output_dir,
                prompt=item.get("prompt"),
            )
            print(f"  [OK] mesh={auto.get('any6d_mesh')}")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {pid}: {type(e).__name__}: {e}")
            failed.append((pid, f"{type(e).__name__}: {e}"))

    print(f"\n[PreGen] 完成: 成功 {ok}, 失败 {len(failed)}")
    for pid, reason in failed:
        print(f"  - {pid}: {reason}")


if __name__ == "__main__":
    main()
