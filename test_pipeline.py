#!/usr/bin/env python3
"""
管线鲁棒性验证脚本 — 验证"无 stub / 软失败"设计

在**未部署任何模型权重**的机器上运行时:
- 每个 evaluator 都应优雅地返回 EvalResult(status="error") 而非崩溃;
- pipeline 应能跑完全部模块并生成报告;
- 聚合分数只统计 status ∈ {ok, proxy, degraded} 的维度。

这验证的是"框架健壮性"，不是"评测正确性"(后者需真实模型+视频)。
"""
import os, sys, yaml, tempfile, argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_test_video(output_path: str, n_frames: int = 12, size: tuple = (256, 256)):
    import cv2
    H, W = size
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 8.0, (W, H))
    for t in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :] = [int(128 + 127 * np.sin(t * 0.5 + k)) for k in (4.2, 2.1, 0)]
        cx = int(W * 0.3 + W * 0.4 * t / n_frames)
        cy = int(H * 0.5 + H * 0.3 * np.sin(t * 0.8))
        cv2.circle(frame, (cx, cy), 30, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    print(f"  [OK] 生成测试视频: {output_path} ({n_frames} frames)")


def test_sampler():
    from sampler import FrameSampler
    print("\n[Test 1] FrameSampler ...")
    sampler = FrameSampler({"max_frames": 8, "uniform": True})
    vp = tempfile.mktemp(suffix=".mp4")
    generate_test_video(vp, n_frames=16)
    frames = sampler.sample_from_video(vp)
    assert len(frames) > 0
    print(f"  [PASS] 采样 {len(frames)} 帧, 尺寸={frames[0].size}")
    os.unlink(vp)


def test_base_status():
    """验证 EvalResult.status 与聚合逻辑"""
    from evaluators.base import EvalResult, LayerResult
    print("\n[Test 2] status 聚合逻辑 ...")
    layer = LayerResult(layer="test", results=[
        EvalResult("A", 5, status="ok"),
        EvalResult("B", 3, status="proxy"),
        EvalResult.error("C", "模型缺失"),
        EvalResult.skipped("D", "帧数不足"),
    ])
    avg = layer.compute_avg()
    assert len(layer.scored_results()) == 2, "应只统计 ok+proxy"
    assert abs(avg - 4.0) < 1e-6, f"均分应为4.0(=(5+3)/2), 实际{avg}"
    print(f"  [PASS] 4维中2维计分, avg={avg:.2f} (error/skipped 已排除)")


def test_soft_fail_pipeline():
    """核心: 无模型权重时 pipeline 不崩溃, 全部记 error 并出报告"""
    print("\n[Test 3] 软失败管线 (未部署模型) ...")
    tmp = tempfile.mkdtemp()
    vp = os.path.join(tmp, "test.mp4")
    generate_test_video(vp, n_frames=12)

    from pipeline_physics import PhysicsPipeline
    pipeline = PhysicsPipeline(config_path="config.yaml")
    pipeline.initialize()  # 模型加载失败会被 _safe_load 捕获
    result = pipeline.run(video_path=vp)

    print(f"\n  Pipeline 完成, 层数={len(result.layers)}")
    n_error = n_scored = 0
    for layer in result.layers:
        for r in layer.results:
            if r.status == "error":
                n_error += 1
            elif r.counts_toward_score:
                n_scored += 1
        print(f"    [{layer.layer}] results={len(layer.results)} "
              f"scored={len(layer.scored_results())} avg={layer.compute_avg():.2f}")
    print(f"\n  总计: {n_scored} 维计分, {n_error} 维 error(模型缺失)")
    print("  [PASS] 管线未崩溃, 结果格式合法")
    os.unlink(vp)
    os.rmdir(tmp)


def main():
    parser = argparse.ArgumentParser(description="管线鲁棒性验证")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Physical World Benchmark — 管线鲁棒性验证")
    print("(无 stub: 未部署模型时应优雅记 error, 而非假数据)")
    print("=" * 60)

    test_sampler()
    test_base_status()
    if not args.quick:
        test_soft_fail_pipeline()

    print("\n" + "=" * 60)
    print("验证完成。真实评测需按 DEPLOYMENT.md 部署模型 + 提供真实视频。")
    print("=" * 60)


if __name__ == "__main__":
    main()
