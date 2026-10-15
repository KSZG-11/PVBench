"""
统一模型加载器 — 管理所有模型的加载/卸载/显存优化

v3 (软失败策略):
- 移除 stub。所有模型 load() 失败会抛异常。
- loader 用 _safe_load 捕获异常并记录, 仍返回模型对象 (is_loaded=False)。
- evaluator 调用其推理方法时会再次抛 RuntimeError, 由 evaluator 捕获记 status=error。
- 这样单个模型缺失不会中断整个 benchmark, 只使相关维度记为 error/跳过。
"""
from typing import Optional, Dict, Any
from .vlm import VLMWrapper
from .vggt import VGGTModel
from .spatial_tracker import SpatialTrackerModel
from .sam3 import SAM3Model
from .stable_normal import StableNormalModel
from .depth_metric import DA3Model
from .reasonx import ReasonXModel
from .any6d import Any6DModel
from .flowseek import FlowSeekModel
from .vfimamba import VFIMambaModel
from .finevq import FineVQModel
# Genesis 已移除：AC-1 动作因果改用反事实动作对评测
from .specialist import SpecialistModels


def _safe_load(model, name: str):
    """尝试加载模型; 失败仅记录不抛出 (软失败)。返回模型对象。"""
    try:
        model.load()
    except Exception as e:
        print(f"[ModelLoader] {name} 加载失败 ({type(e).__name__}: {e}); "
              f"相关维度将记为 status=error")
    return model


class ModelLoader:
    """统一加载器：按需加载、显存管理、单例缓存、软失败"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cache: Dict[str, Any] = {}

        # torchvision::nms CUDA kernel 缺失兜底 (DA3/SAM3 依赖 nms)。全局幂等 patch。
        try:
            from .base import patch_torchvision_nms_cpu_fallback
            patch_torchvision_nms_cpu_fallback()
        except Exception as e:
            print(f"[ModelLoader] torchvision nms patch 跳过: {e}")
        # diffusers>=0.31 ControlNetOutput 路径变更 → StableNormal 兼容
        try:
            from .base import patch_diffusers_controlnet_compat
            patch_diffusers_controlnet_compat()
        except Exception as e:
            print(f"[ModelLoader] diffusers controlnet patch 跳过: {e}")
        # transformers>=4.46 GreedySearchDecoderOnlyOutput 移除 → VFIMamba 兼容
        try:
            from .base import patch_transformers_generation_compat
            patch_transformers_generation_compat()
        except Exception as e:
            print(f"[ModelLoader] transformers generation patch 跳过: {e}")

        global_cfg = config.get("global", {})
        self.hf_cache_dir = global_cfg.get("hf_cache_dir")
        if self.hf_cache_dir:
            import os
            os.environ["HF_HOME"] = self.hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(self.hf_cache_dir, "hub")
            # StableNormal/DINOv2 和部分 torchvision 模型通过 torch.hub
            # 读取 TORCH_HOME；不能依赖调用者是否提前 source env.sh。
            os.environ.setdefault(
                "TORCH_HOME", os.path.join(self.hf_cache_dir, "torch_hub"))
            print(f"[ModelLoader] HF Cache: {self.hf_cache_dir}")
            print(f"[ModelLoader] Torch Hub Cache: {os.environ['TORCH_HOME']}")

        model_cfg = config.get("models", {})
        self.vlm_config = model_cfg.get("vlm", {})
        self.quality_config = model_cfg.get("quality", {})
        self.physics_config = model_cfg.get("physics", {})

    def get_vlm(self, layer: str = "default") -> VLMWrapper:
        # layer只覆盖default中的参数，避免physics/logic仅配置max_new_tokens时
        # 丢失Qwen3.5的model_name而意外回退到Qwen3-VL。
        default_cfg = dict(self.vlm_config.get("default", {}))
        layer_cfg = dict(self.vlm_config.get(layer, {}))
        vlm_cfg = {**default_cfg, **layer_cfg}
        model_name = vlm_cfg.get("model_name", "Qwen/Qwen3.5-9B")
        # 环境变量覆盖: PWB_VLM_MODEL 可临时切换 VLM (如 Qwen3.8-27B 对比 Qwen3.5),
        # 不改 config 即可跑两遍做 baseline 对比。
        import os as _os
        env_override = _os.environ.get("PWB_VLM_MODEL")
        if env_override:
            model_name = env_override
        # ⚠️ 按 model_name 缓存 (而非 layer): default/logic/physics 用的是同一个 Qwen3-VL-8B,
        #    若按 layer 各加载一份, A100-40GB 会 OOM (两份 8B ≈ 32GB)。共享单实例即可,
        #    max_new_tokens 差异由各 evaluator 调用时按需传入。
        key = f"vlm::{model_name}"
        if key not in self._cache:
            m = VLMWrapper(
                model_name=model_name,
                device=vlm_cfg.get("device", "cuda"),
                load_in_4bit=vlm_cfg.get("load_in_4bit", False),
                load_in_8bit=vlm_cfg.get("load_in_8bit", False),
                max_new_tokens=vlm_cfg.get("max_new_tokens", 512),
            )
            self._cache[key] = _safe_load(m, f"VLM({layer})")
        else:
            # 若之前被 offload 到 CPU 腾显存, 使用前搬回 GPU。
            # 8bit/4bit offload 是 unload() 释放 (is_loaded=False), 需无条件
            # 调 to_cuda() 让其内部重新 load(); 非量化模型则 to_cuda() 搬回 GPU。
            vlm = self._cache[key]
            if hasattr(vlm, "to_cuda"):
                vlm.to_cuda()
        return self._cache[key]

    def offload_vlm(self):
        """把已加载的 VLM 临时挪到 CPU 释放显存 (供不需要 VLM 的评测层调用)。"""
        for key, obj in self._cache.items():
            if key.startswith("vlm::") and getattr(obj, "is_loaded", False) \
                    and hasattr(obj, "to_cpu"):
                obj.to_cpu()

    def offload_except(self, keep: set):
        """把「除 keep 之外的」已加载模型临时 offload 到 CPU，释放显存。

        用于显存错峰：评测某个重型模块（如 Any6D 的 InstantMesh 临时加载
        Zero123++）前，把用不到的模型 offload 腾空间；之后由 get_xxx() 触发
        to_cuda 搬回（或显式调用 restore_except_offload）。

        keep: 需保留在 GPU 的模型 key 集合（如 {"any6d", "sam3", "da3"}）。
        """
        for key, obj in self._cache.items():
            base = key.split("::")[-1]  # vlm::xxx → xxx；其他 key 不变
            if base in keep:
                continue
            if getattr(obj, "is_loaded", False) and hasattr(obj, "to_cpu"):
                try:
                    obj.to_cpu()
                    print(f"[ModelLoader] offload 到 CPU: {key}")
                except Exception as e:
                    print(f"[ModelLoader] offload {key} 失败(忽略): {e}")

    def restore_offloaded(self):
        """把之前 offload 到 CPU 的模型搬回 GPU（to_cuda）。

        注意: 8bit/4bit 量化模型在 offload 时是 unload() 彻底释放 (is_loaded=False),
        此处不能用 is_loaded 过滤, 否则它们永远不会被恢复。改为: 只要有 to_cuda
        方法就调用 (to_cuda 内部会判断是否需要重新 load)。
        """
        for key, obj in self._cache.items():
            if hasattr(obj, "to_cuda"):
                try:
                    obj.to_cuda()
                except Exception as e:
                    print(f"[ModelLoader] 搬回 {key} 失败(忽略): {e}")

    def get_specialist(self) -> SpecialistModels:
        if "specialist" not in self._cache:
            dino_cfg = self.quality_config.get("dino", {})
            m = SpecialistModels(device=dino_cfg.get("device"),
                                 dino_model_id=dino_cfg.get("model_id"))
            try:
                m.load_all()
            except Exception as e:
                print(f"[ModelLoader] Specialist 加载失败: {e}")
            self._cache["specialist"] = m
        return self._cache["specialist"]

    def get_finevq(self) -> FineVQModel:
        """获取 FineVQ 视频美学评分模型 (FQ-2)。懒加载 + 缓存。"""
        if "finevq" not in self._cache:
            cfg = self.quality_config.get("finevq", {})
            m = FineVQModel(model_path=cfg.get("model_path"),
                            device=cfg.get("device", "cuda"),
                            hf_cache_dir=self.hf_cache_dir,
                            num_segments=cfg.get("num_segments", 8))
            self._cache["finevq"] = _safe_load(m, "FineVQ")
        return self._cache["finevq"]

    def get_vggt(self) -> VGGTModel:
        if "vggt" not in self._cache:
            cfg = self.physics_config.get("vggt", {})
            m = VGGTModel(model_path=cfg.get("model_path"),
                          device=cfg.get("device", "cuda"),
                          model_version=cfg.get("model_version", "omega"),
                          image_resolution=cfg.get("image_resolution", 512))
            self._cache["vggt"] = _safe_load(m, "VGGT")
        return self._cache["vggt"]

    def get_spatial_tracker(self) -> SpatialTrackerModel:
        if "spatial_tracker" not in self._cache:
            cfg = self.physics_config.get("spatial_tracker", {})
            m = SpatialTrackerModel(model_path=cfg.get("model_path"),
                                    device=cfg.get("device", "cuda"),
                                    hf_cache_dir=self.hf_cache_dir,
                                    vo_points=cfg.get("vo_points", 756))
            self._cache["spatial_tracker"] = _safe_load(m, "SpatialTracker V2")
        return self._cache["spatial_tracker"]

    def get_sam3(self):
        """获取分割模型（统一使用 SAM3）。

        SAM3 加载失败 → 抛异常，调用方捕获后记 status=error。
        不再有 SAM2 降级路径（按表格设计，所有分割维度均使用 SAM3）。
        """
        if "sam3" not in self._cache:
            cfg = self.physics_config.get("sam3", {})
            model = SAM3Model(model_type=cfg.get("model_type", "vit_h"),
                               checkpoint_path=cfg.get("checkpoint_path"),
                               device=cfg.get("device", "cuda"),
                               confidence_threshold=cfg.get("confidence_threshold", 0.1))
            loaded = _safe_load(model, "SAM3")
            if loaded.is_loaded:
                print("[ModelLoader] SAM3 ✓")
            else:
                print("[ModelLoader] SAM3 加载失败, 分割相关维度将记 status=error")
            self._cache["sam3"] = loaded
        return self._cache["sam3"]

    def get_stable_normal(self) -> StableNormalModel:
        if "stable_normal" not in self._cache:
            cfg = self.physics_config.get("stable_normal", {})
            m = StableNormalModel(model_path=cfg.get("model_path"),
                                  device=cfg.get("device", "cuda"),
                                  use_turbo=cfg.get("use_turbo", False))
            self._cache["stable_normal"] = _safe_load(m, "StableNormal")
        return self._cache["stable_normal"]

    def get_da3(self) -> DA3Model:
        if "da3" not in self._cache:
            cfg = self.physics_config.get("da3", {})
            m = DA3Model(model_path=cfg.get("model_path"),
                         device=cfg.get("device", "cuda"))
            self._cache["da3"] = _safe_load(m, "DA3Metric-Large")
        return self._cache["da3"]

    def get_reasonx(self) -> ReasonXModel:
        if "reasonx" not in self._cache:
            cfg = self.physics_config.get("reasonx", {})
            m = ReasonXModel(model_path=cfg.get("model_path"),
                             device=cfg.get("device", "cuda"),
                             use_judge=cfg.get("use_judge", True))
            self._cache["reasonx"] = _safe_load(m, "ReasonX")
        return self._cache["reasonx"]

    def get_any6d(self) -> Any6DModel:
        if "any6d" not in self._cache:
            cfg = self.physics_config.get("any6d", {})
            m = Any6DModel(model_path=cfg.get("model_path"),
                           device=cfg.get("device", "cuda"),
                           hf_cache_dir=self.hf_cache_dir,
                           mesh_path=cfg.get("mesh_path"))
            self._cache["any6d"] = _safe_load(m, "Any6D")
        return self._cache["any6d"]

    def get_flowseek(self) -> FlowSeekModel:
        if "flowseek" not in self._cache:
            cfg = self.physics_config.get("flowseek", {})
            m = FlowSeekModel(model_path=cfg.get("model_path"),
                              device=cfg.get("device", "cuda"),
                              hf_cache_dir=self.hf_cache_dir)
            self._cache["flowseek"] = _safe_load(m, "FlowSeek")
        return self._cache["flowseek"]

    def get_vfimamba(self) -> VFIMambaModel:
        if "vfimamba" not in self._cache:
            cfg = self.physics_config.get("vfimamba", {})
            m = VFIMambaModel(model_path=cfg.get("model_path"),
                              device=cfg.get("device", "cuda"),
                              hf_cache_dir=self.hf_cache_dir)
            self._cache["vfimamba"] = _safe_load(m, "VFIMamba")
        return self._cache["vfimamba"]

    def unload_all(self):
        for model in self._cache.values():
            if hasattr(model, "unload"):
                try:
                    model.unload()
                except Exception:
                    pass
        self._cache.clear()

    def unload_except(self, keep: set):
        """卸载「除 keep 之外」的所有已加载模型，彻底释放显存。

        用于评测阶段切换：物理层(L3)跑完后，全局层(L1+L2)开始前，卸载
        VGGT/SAM3/Any6D/DA3/StableNormal/ReasonX/FlowSeek/VFIMamba 等
        物理专用模型，给 VLM + Specialist(DINO) + FineVQ 腾显存。

        keep: 需保留的模型 key 集合（如 {"vlm", "spatial_tracker"}）。
              匹配规则与 offload_except 一致: key.split("::")[-1]。
        与 offload_except 的区别: 这里是 unload() 彻底释放并清缓存,
        而非 .to("cpu") 搬运(后者对 8bit 模型无效且会占系统 RAM)。
        """
        import torch
        for key in list(self._cache.keys()):
            # vlm::<model_name> 的 key 前缀是 "vlm::", 需识别为 "vlm" 以便 keep 匹配;
            # 其他 key (vggt/sam3/any6d/...) 无 "::" 前缀, base 即 key 本身。
            if "::" in key:
                base = key.split("::")[0]
            else:
                base = key
            if base in keep:
                continue
            obj = self._cache[key]
            if hasattr(obj, "unload"):
                try:
                    obj.unload()
                    print(f"[ModelLoader] 卸载: {key}")
                except Exception as e:
                    print(f"[ModelLoader] 卸载 {key} 失败(忽略): {e}")
            del self._cache[key]
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def print_status(self):
        print("\n=== 已加载模型 ===")
        for name, obj in self._cache.items():
            loaded = getattr(obj, "is_loaded", "?")
            mark = "✓" if loaded is True else ("✗" if loaded is False else "?")
            print(f"  [{name}] loaded={loaded} {mark}")
        print("==================\n")

    def collect_metadata(self) -> dict:
        """收集所有已加载模型的 provenance 元数据 (统一 metadata 接口)。

        用于复现说明/开源报告: 每个模型返回 {model_id, source, checkpoint, device, ...}。
        """
        out = {}
        for name, obj in self._cache.items():
            try:
                # metadata 是 property (getattr 触发 getter 返回 dict)
                out[name] = getattr(obj, "metadata", {})
            except Exception as e:
                out[name] = {"error": str(e)}
        return out
