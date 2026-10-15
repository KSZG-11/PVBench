from .base import BaseEvaluator, EvalResult, LayerResult, BenchmarkResult
from .logic import InstructionFollowingEvaluator, ActionCausalEvaluator, CommonsenseEvaluator
from .quality import FrameQualityEvaluator, TemporalQualityEvaluator
from .physics import (
    GeometryEvaluator,
    ScalePoseEvaluator,
    DeformationSpatialEvaluator,
    StaticsLightingEvaluator,
    MaterialEvaluator,
    ThermalEvaluator,
    ElectromagEvaluator,
    PhaseChangeEvaluator,
    KinematicsEvaluator,
    MotionPersistenceEvaluator,
    DynamicsEvaluator,
    MultiviewEvaluator,
)

__all__ = [
    # Base
    "BaseEvaluator", "EvalResult", "LayerResult", "BenchmarkResult",
    # Logic (L1) — 指令遵循+语义对齐已合并为9维
    "InstructionFollowingEvaluator", "ActionCausalEvaluator", "CommonsenseEvaluator",
    # Quality (L2) — 仅纯画质指标
    "FrameQualityEvaluator", "TemporalQualityEvaluator",
    # Physics (L3) - 12 evaluators
    "GeometryEvaluator",
    "ScalePoseEvaluator",
    "DeformationSpatialEvaluator",
    "StaticsLightingEvaluator",
    "MaterialEvaluator",
    "ThermalEvaluator",
    "ElectromagEvaluator",
    "PhaseChangeEvaluator",
    "KinematicsEvaluator",
    "MotionPersistenceEvaluator",
    "DynamicsEvaluator",
    "MultiviewEvaluator",
]
