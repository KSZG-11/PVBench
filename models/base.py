"""
专用模型统一基类
所有物理评测用到的模型都继承此基类，确保统一接口。
"""
from abc import ABC, abstractmethod
from typing import Optional, Any
import os
import glob
import torch
import numpy as np
from dataclasses import dataclass, field


_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".pkl", ".ckpt")


def patch_torchvision_nms_cpu_fallback():
    """torchvision CUDA 算子 (nms / roi_align) 无 CUDA kernel 时的 CPU 兜底。

    torch-base(H20) 环境的 torchvision 是无 CUDA 编译版 (只注册了 CPU/autocast kernel,
    没有 cuda/*.cu)。CUDA 张量上调用会报:
      'Could not run torchvision::{nms,roi_align} with arguments from the CUDA backend'
    → DA3(R04) / SAM3(R09/R11/R12/R21) 全崩。
    monkeypatch: 把算子落到 CPU 执行再搬回原 device。
      - nms: box 小算子, CPU 无性能损失。
      - roi_align: SAM3 decoder(RoIAlign 类)/geometry_encoder 用; 逐帧特征图上, CPU 可接受。
    ⚠️ 这些模型常在 torch.autocast(bfloat16) 下调用 → 张量可能是 bf16, 而 CPU kernel 多不支持
       半精度, 故 CPU 计算前统一转 float32, 结果再转回原 dtype+device。
    幂等; torchvision 有 CUDA kernel 时探测通过直接跳过 (走原生)。
    """
    try:
        import sys
        import torch
        import torchvision
        import torchvision.ops as _ops
        import torchvision.ops.boxes as _boxes
    except Exception:
        return
    if getattr(torchvision, "_nms_cpu_patched", False):
        return
    # 探测: CUDA 上原生 torchvision::nms 是否可用。可用(如官方 cu12 torchvision)则
    # 不 patch, 走原生更快; 仅当缺 CUDA kernel(无 CUDA 编译的 torchvision)才兜底。
    # nms 与 roi_align 同属一个 torchvision 编译产物, 用 nms 探测即可代表两者。
    if torch.cuda.is_available():
        try:
            _b = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda")
            _s = torch.tensor([1.0], device="cuda")
            _ops.nms(_b, _s, 0.5)
            return  # 原生 CUDA nms 正常, 无需 patch
        except Exception:
            pass

    # ── nms CPU 兜底 ──
    _orig_nms = _ops.nms

    def _nms_cpu(boxes, scores, iou_threshold):
        if getattr(boxes, "is_cuda", False):
            dev = boxes.device
            r = _orig_nms(boxes.detach().float().cpu(),
                          scores.detach().float().cpu(), iou_threshold)
            return r.to(dev)
        return _orig_nms(boxes, scores, iou_threshold)

    # 同时替换 torchvision.ops.nms 与 torchvision.ops.boxes.nms:
    # batched_nms 内部按模块全局名 `nms` 调用, 故 boxes.nms 必须也替换。
    _ops.nms = _nms_cpu
    _boxes.nms = _nms_cpu

    # ── roi_align CPU 兜底 ──
    _orig_roi_align = _ops.roi_align

    def _roi_align_cpu(input, boxes, *args, **kwargs):
        if getattr(input, "is_cuda", False):
            dev, dt = input.device, input.dtype
            cpu_in = input.detach().float().cpu()
            if isinstance(boxes, (list, tuple)):
                cpu_boxes = [b.detach().float().cpu() if hasattr(b, "detach") else b
                             for b in boxes]
            else:
                cpu_boxes = boxes.detach().float().cpu() if hasattr(boxes, "detach") else boxes
            out = _orig_roi_align(cpu_in, cpu_boxes, *args, **kwargs)
            return out.to(device=dev, dtype=dt)
        return _orig_roi_align(input, boxes, *args, **kwargs)

    # functional 命名空间 (geometry_encoders 用 torchvision.ops.roi_align(...)):
    _ops.roi_align = _roi_align_cpu
    # ⚠️ RoIAlign 类的 forward 调用其所在子模块 torchvision.ops.roi_align 的全局函数
    #    (非 ops 命名空间), 故必须替换子模块内的 roi_align (decoder.py 的 RoIAlign 才生效)。
    _roi_submod = sys.modules.get("torchvision.ops.roi_align")
    if _roi_submod is not None and hasattr(_roi_submod, "roi_align"):
        _roi_submod.roi_align = _roi_align_cpu

    torchvision._nms_cpu_patched = True
    print("[patch] torchvision.nms + roi_align → CPU 兜底 (torchvision 无 CUDA 编译)")


def patch_diffusers_controlnet_compat():
    """diffusers >=0.31 将 diffusers.models.controlnet 子模块重构为 controlnets。
    StableNormal 的自定义 controlnet 权重文件 (hf_cache/stable_normal/*/) 内含有:
      - `from diffusers.models.controlnet import ControlNetOutput`
      - `from diffusers.models import ControlNetModel`  (间接依赖 controlnet 子模块解析)
      - `from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline`
    这些在新版 diffusers (>=0.31, 实际 0.33+) 均会 ImportError。
    补丁: 创建假模块 + 假子包兼容上述所有路径。
    幂等; 若模块已存在或 diffusers 未安装则跳过。
    """
    import sys
    _cn_key = "diffusers.models.controlnet"
    _pl_key = "diffusers.pipelines.controlnet"
    if _cn_key in sys.modules and _pl_key in sys.modules:
        return
    try:
        import diffusers
        import diffusers.models
        import diffusers.pipelines
    except ImportError:
        return

    import types

    # ── 1) diffusers.models.controlnet 兼容 ──
    # diffusers 0.29.x: 真实模块 diffusers.models.controlnet 包含 ControlNetOutput
    #   + ControlNetModel; 仅需注入 ControlNetModel 到顶层, 不创建假模块。
    # diffusers >=0.31: 重构为 diffusers.models.controlnets (复数); 旧路径不存在,
    #   需创建假模块兼容 StableNormal 自定义权重内的 pickle 引用。
    if _cn_key not in sys.modules:
        _CNO = None
        _CNM = None
        _has_real_module = False

        # 优先尝试真实旧模块 (diffusers < 0.31)
        try:
            from diffusers.models.controlnet import \
                ControlNetOutput as _CNO, ControlNetModel as _CNM
            _has_real_module = True
        except ImportError:
            pass

        if not _has_real_module:
            # 旧模块不存在 → 试新版 controlnets 惰性加载 (diffusers >= 0.31)
            if _CNO is None:
                try:
                    _CNO = getattr(diffusers.models, "ControlNetOutput", None)
                except Exception:
                    pass
            if _CNO is None:
                try:
                    from diffusers.models.controlnets.controlnet import \
                        ControlNetOutput as _CNO
                except ImportError:
                    pass
            if _CNM is None:
                try:
                    _CNM = getattr(diffusers.models, "ControlNetModel", None)
                except Exception:
                    pass
            if _CNM is None:
                try:
                    from diffusers.models.controlnets.controlnet import \
                        ControlNetModel as _CNM
                except ImportError:
                    pass

            # 创建假模块仅在新版 diffusers (真实模块不存在时)
            _m = types.ModuleType("diffusers.models.controlnet")
            if _CNO is not None:
                _m.ControlNetOutput = _CNO
            if _CNM is not None:
                _m.ControlNetModel = _CNM
            sys.modules[_cn_key] = _m
            setattr(diffusers.models, "controlnet", _m)

        # 注入到 diffusers.models 顶层 (新旧版都需要:
        # controlnetvae.py 内部 `from diffusers.models import ControlNetModel`)
        if _CNM is not None:
            setattr(diffusers.models, "ControlNetModel", _CNM)
        print("[patch] diffusers.models.controlnet "
              f"(real={_has_real_module} "
              f"ControlNetOutput={_CNO is not None} "
              f"ControlNetModel={_CNM is not None})")

    # ── 1.6) transformers.utils.FLAX_WEIGHTS_NAME 兼容 ──
    # diffusers 0.29.2 的 pipeline_controlnet 内部 `from transformers.utils import
    # FLAX_WEIGHTS_NAME`, 但 transformers>=4.57 已将其从 utils.__init__ 移除。
    # 需在导入 pipeline 之前注入该常量 (值="flax_model.msgpack")。
    try:
        import transformers.utils as _tf_utils
        if not hasattr(_tf_utils, "FLAX_WEIGHTS_NAME"):
            setattr(_tf_utils, "FLAX_WEIGHTS_NAME", "flax_model.msgpack")
            print("[patch] transformers.utils.FLAX_WEIGHTS_NAME 已注入")
    except ImportError:
        pass

    # ── 2) diffusers.pipelines.controlnet (假子包, 兼容旧导入路径) ──
    if _pl_key not in sys.modules:
        try:
            import diffusers.pipelines.controlnet as _real_pkg
            _cn_pipeline = getattr(_real_pkg, "StableDiffusionControlNetPipeline", None)
        except (ImportError, AttributeError):
            _cn_pipeline = None
        # 若旧路径不存在, 尝试新版路径
        if _cn_pipeline is None:
            try:
                from diffusers.pipelines.controlnet.pipeline_controlnet \
                    import StableDiffusionControlNetPipeline as _cn_pipeline
            except ImportError:
                try:
                    from diffusers.pipelines.controlnet.pipeline_controlnet_sd3 \
                        import StableDiffusion3ControlNetPipeline as _cn_pipeline
                except ImportError:
                    pass
        if _cn_pipeline is not None:
            _pp = types.ModuleType("diffusers.pipelines.controlnet")
            _pp.StableDiffusionControlNetPipeline = _cn_pipeline
            sys.modules[_pl_key] = _pp
            print("[patch] diffusers.pipelines.controlnet "
                  "→ 兼容 StableDiffusionControlNetPipeline")


def patch_tied_weights_keys_compat():
    """transformers 4.57+ 的 PreTrainedModel._move_missing_keys_from_meta_to_device 会
    访问 self.all_tied_weights_keys (不带下划线), 但 InternVL 系列 (ReasonX/FineVQ)
    的建模代码只定义了 _tied_weights_keys → AttributeError。

    幂等补丁: monkeypatch PreTrainedModel.__init__, init 后把 _tied_weights_keys
    同步到 all_tied_weights_keys (并容错 list → dict)。
    """
    import transformers.modeling_utils as _tmu
    if getattr(_tmu, "_tied_weights_keys_patched", False):
        return
    _orig_init = _tmu.PreTrainedModel.__init__

    def _safe_init(self, config, *args, **kwargs):
        _orig_init(self, config, *args, **kwargs)
        val = getattr(self, "_tied_weights_keys", None)
        if val is not None:
            if isinstance(val, list):
                val = {k: True for k in val}
            object.__setattr__(self, "_tied_weights_keys", val)
            object.__setattr__(self, "all_tied_weights_keys", val)
        if not hasattr(self, "all_tied_weights_keys"):
            object.__setattr__(self, "all_tied_weights_keys", {})

    _tmu.PreTrainedModel.__init__ = _safe_init
    _tmu._tied_weights_keys_patched = True
    print("[patch] PreTrainedModel 权重绑定兼容补丁已应用 (all_tied_weights_keys)")


def patch_transformers_generation_compat():
    """transformers >=4.46 将 GreedySearchDecoderOnlyOutput / SampleDecoderOnlyOutput
    等旧输出类重命名/移除 → VFIMamba/mamba-ssm 导入链报 ImportError。
    补丁: 缺失的类统一映射到新版 GenerateDecoderOnlyOutput 或创建 stub。
    幂等; 若存在或 transformers 未安装则跳过。
    """
    import sys
    import dataclasses
    _patched_flag = "_generation_compat_patched"
    if "transformers.generation" not in sys.modules:
        try:
            import transformers.generation as _tg
        except ImportError:
            return
    else:
        _tg = sys.modules["transformers.generation"]
    if getattr(_tg, _patched_flag, False):
        return

    # 别名列表: 旧名 → 首选新类 (GenerateDecoderOnlyOutput)
    _MISSING_CLASSES = [
        "GreedySearchDecoderOnlyOutput",
        "GreedySearchEncoderDecoderOutput",
        "SampleDecoderOnlyOutput",
        "SampleEncoderDecoderOutput",
        "BeamSearchDecoderOnlyOutput",
        "BeamSearchEncoderDecoderOutput",
        "BeamSampleDecoderOnlyOutput",
        "BeamSampleEncoderDecoderOutput",
        "ContrastiveSearchDecoderOnlyOutput",
        "ContrastiveSearchEncoderDecoderOutput",
    ]
    _fallback = None
    for _cand in ("GenerateDecoderOnlyOutput", "GenerateEncoderDecoderOutput"):
        if hasattr(_tg, _cand):
            _fallback = getattr(_tg, _cand)
            break

    patched_any = False
    for _name in _MISSING_CLASSES:
        if hasattr(_tg, _name):
            continue
        if _fallback is not None:
            setattr(_tg, _name, _fallback)
        else:
            # 终极兜底: stub dataclass (from_pretrained 不崩即可)
            @dataclasses.dataclass
            class _StubOutput:
                sequences: object = None
                scores: object = None
                logits: object = None
                attentions: object = None
                hidden_states: object = None
                past_key_values: object = None
            setattr(_tg, _name, _StubOutput)
        patched_any = True
    if patched_any:
        print(f"[patch] transformers.generation 旧输出类→新版别名 "
              f"(fallback={_fallback.__name__ if _fallback else 'stub'})")
    _tg._generation_compat_patched = True


def _dir_has_model(d: str) -> bool:
    """目录内是否含 config.json 或任意权重文件 (判定为可离线加载的模型目录)。"""
    if not os.path.isdir(d):
        return False
    if os.path.isfile(os.path.join(d, "config.json")):
        return True
    for f in os.listdir(d):
        if f.endswith(_WEIGHT_EXTS):
            return True
    return False


def resolve_local_model(model_id: str, cache_dir: Optional[str] = None) -> str:
    """离线优先解析: 若本地已下好权重, 返回本地目录路径供 from_pretrained 直接加载;
    否则原样返回 model_id (在线下载)。

    统一解决"魔搭 flat 格式 (hf_cache/models/{org}--{name}) 与 HF snapshot 格式
    (models--{org}--{name}/snapshots/*/) 与 transformers 默认查找路径不一致"导致的
    离线找不到权重问题。被 Qwen3.5 / DA3 / SAM3 / ReasonX / VGGT-Omega / SpatialTracker 等复用。
    """
    if not model_id or "/" not in str(model_id):
        # 已是本地路径或无 org 前缀
        if model_id and os.path.isdir(model_id):
            return model_id
        # 仍尝试按裸名在 cache 里找
    if not cache_dir:
        return model_id
    name = str(model_id).split("/")[-1]
    # 1) flat 目录 (魔搭): org--name / AI-ModelScope--name / name
    variants = [str(model_id).replace("/", "--"), f"AI-ModelScope--{name}", name]
    for base in (os.path.join(cache_dir, "models"), cache_dir):
        for v in variants:
            d = os.path.join(base, v)
            if _dir_has_model(d):
                return d
            # flat 目录可能再套 snapshots/*/
            for sd in sorted(glob.glob(os.path.join(d, "snapshots", "*"))):
                if _dir_has_model(sd):
                    return sd
    # 2) HF snapshot 格式: (hub/)models--*--name/snapshots/*/
    for pat in (
        os.path.join(cache_dir, f"models--*--{name}", "snapshots", "*"),
        os.path.join(cache_dir, "hub", f"models--*--{name}", "snapshots", "*"),
    ):
        for d in sorted(glob.glob(pat)):
            if _dir_has_model(d):
                return d
    return model_id


@dataclass
class DepthOutput:
    """深度图输出"""
    depth_map: np.ndarray          # [H, W] 相对深度
    metric_depth: Optional[float] = None  # 如果有度量深度
    confidence: Optional[np.ndarray] = None  # [H, W] 置信度


@dataclass
class PointCloudOutput:
    """3D点云输出"""
    points: np.ndarray             # [N, 3] 点云坐标
    colors: Optional[np.ndarray] = None  # [N, 3] RGB颜色
    camera_pose: Optional[np.ndarray] = None  # [4, 4] 相机位姿


@dataclass
class NormalOutput:
    """法向量输出"""
    normal_map: np.ndarray         # [H, W, 3] 法向量图 (值域[-1,1])
    confidence: Optional[np.ndarray] = None


@dataclass
class TrajectoryOutput:
    """3D轨迹输出"""
    trajectories: np.ndarray       # [N_objects, T, 3] 物体3D轨迹
    velocities: Optional[np.ndarray] = None  # [N_objects, T-1, 3] 速度
    accelerations: Optional[np.ndarray] = None  # [N_objects, T-2, 3] 加速度
    angular_velocities: Optional[np.ndarray] = None  # [N_objects, T-1, 3]
    object_ids: Optional[list] = None  # 物体标识
    confidences: Optional[np.ndarray] = None  # [N_objects, T]


@dataclass
class SegmentationOutput:
    """分割输出"""
    masks: np.ndarray              # [N_objects, T, H, W] 或 [N_objects, H, W]
    class_names: Optional[list] = None
    confidences: Optional[np.ndarray] = None
    tracking_ids: Optional[np.ndarray] = None  # [N_objects, T] 跨帧ID


@dataclass
class PoseOutput:
    """6D位姿输出"""
    positions: np.ndarray          # [N_objects, T, 3] 位置
    rotations: np.ndarray          # [N_objects, T, 3, 3] 旋转矩阵
    scales: Optional[np.ndarray] = None  # [N_objects, T, 3] 尺寸
    metric: bool = False           # 是否为米制


@dataclass
class ShadingOutput:
    """光影/着色分析输出"""
    shading_map: np.ndarray        # [T, H, W, 3] 着色图
    light_directions: Optional[np.ndarray] = None  # [T, 3] 光源方向
    shadow_mask: Optional[np.ndarray] = None  # [T, H, W] 阴影遮罩
    specular_mask: Optional[np.ndarray] = None  # [T, H, W] 高光遮罩


@dataclass
class OpticalFlowOutput:
    """光流输出"""
    flow_fwd: np.ndarray           # [T-1, H, W, 2] 前向光流
    flow_bwd: Optional[np.ndarray] = None  # [T-1, H, W, 2] 后向光流
    occlusion_mask: Optional[np.ndarray] = None


@dataclass
class InterpolationOutput:
    """插帧输出"""
    interpolated_frames: np.ndarray  # [N_interp, H, W, 3] 插值帧
    quality_scores: Optional[np.ndarray] = None


@dataclass
class SimulationOutput:
    """物理仿真输出"""
    rollout_trajectories: np.ndarray  # [N_objects, T, 3] 仿真轨迹
    rollout_states: dict = field(default_factory=dict)  # 额外状态（速度/力等）
    contact_events: Optional[list] = None  # 碰撞接触事件


class BaseSpecialistModel(ABC):
    """所有专用物理评测模型的抽象基类。

    统一生命周期接口 (开源标准):
      - load(**kwargs) / unload() / is_loaded   : 加载/释放/状态
      - to_cpu() / to_cuda()                     : 显存错峰 (默认 no-op, 可按需覆写)
      - metadata                                  : 模型来源/版本/本地路径/降级路径 (复现必需)

    推理方法 (estimate_* / segment_* / chat 等) 保持各模型语义化命名, 不做强制统一。
    """

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self._model: Any = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self, **kwargs) -> None:
        """加载模型权重"""
        ...

    @abstractmethod
    def unload(self) -> None:
        """释放模型显存"""
        ...

    # ── 统一生命周期扩展 (开源标准) ──
    def to_cpu(self) -> None:
        """把模型挪到 CPU 释放显存 (通用实现：遍历所有 nn.Module 子属性)。

        供「显存错峰」使用：低频/非必需模型在评测某重型模块（如 Any6D 的
        InstantMesh）前 offload 到 CPU，用完 to_cuda 搬回，避免单卡 OOM。
        """
        import torch
        moved = False
        for _name, attr in vars(self).items():
            if isinstance(attr, torch.nn.Module):
                try:
                    attr.to("cpu")
                    moved = True
                except Exception:
                    pass
        if moved:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def to_cuda(self) -> None:
        """把模型搬回 GPU (通用实现：遍历所有 nn.Module 子属性)。"""
        import torch
        for _name, attr in vars(self).items():
            if isinstance(attr, torch.nn.Module):
                try:
                    attr.to(self.device)
                except Exception:
                    pass

    @property
    def metadata(self) -> dict:
        """模型来源/版本/本地路径等 provenance 信息 (复现必需)。

        子类应覆写以提供: {model_id, source, checkpoint, device, ...}
        """
        return {
            "class": type(self).__name__,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()
