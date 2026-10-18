from .geometry import GeometryEvaluator
from .scale_pose import ScalePoseEvaluator
from .deformation_spatial import DeformationSpatialEvaluator
from .statics_lighting import StaticsLightingEvaluator
from .material import MaterialEvaluator
from .thermal import ThermalEvaluator
from .electromag import ElectromagEvaluator
from .phase_change import PhaseChangeEvaluator
from .kinematics import KinematicsEvaluator
from .motion_persistence import MotionPersistenceEvaluator
from .dynamics import DynamicsEvaluator
from .multiview import MultiviewEvaluator

__all__ = [
    "GeometryEvaluator",           # R01+R02+R03
    "ScalePoseEvaluator",          # R04+R09+R10
    "DeformationSpatialEvaluator", # R05+R11
    "StaticsLightingEvaluator",     # R12+R13+R18
    "MaterialEvaluator",           # R06 (Qwen3.8-27B)
    "ThermalEvaluator",            # R07 (Qwen3.8-27B)
    "ElectromagEvaluator",         # R14 (Qwen3.8-27B)
    "PhaseChangeEvaluator",        # R25 (Qwen3.8-27B)
    "KinematicsEvaluator",         # R15+R16+R17
    "MotionPersistenceEvaluator",  # R19+R20+R21
    "DynamicsEvaluator",           # R22+R23+R24
    "MultiviewEvaluator",          # R08
]
