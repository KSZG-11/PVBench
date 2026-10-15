"""
向后兼容：请使用新版 pipeline

- 全局 Benchmark (逻辑 + 画质): pipeline_global.py → run_global.py
- 物理层评测: pipeline_physics.py → run_physics.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 重新导出新 pipeline 的类（向后兼容）
from pipeline_global import GlobalBenchmarkPipeline
from pipeline_physics import PhysicsPipeline

# 旧的 BenchmarkPipeline 类名映射
BenchmarkPipeline = GlobalBenchmarkPipeline

__all__ = ["BenchmarkPipeline", "GlobalBenchmarkPipeline", "PhysicsPipeline"]
