"""
GPU 后端检测工具 — 统一支持 NVIDIA CUDA / AMD ROCm / CPU

用途：所有模型加载逻辑使用此模块检测后端，
      避免硬编码 CUDA 特定的错误提示和安装命令。
"""

import os
import subprocess
import torch
from typing import Tuple


# ── 兼容垫片: torch<2.3 无 torch.compiler.is_compiling, 但 transformers>=4.57
#    (Qwen3-VL 必需) 在 generate/forward 时会调用它 → 缺失会 AttributeError。
#    系统 torch 锁死 2.2.2+cu118 (driver 470 只兼容 cu11x, 不能升级), 故打垫片。 ──
try:
    import torch.compiler as _torch_compiler
    if not hasattr(_torch_compiler, "is_compiling"):
        _torch_compiler.is_compiling = lambda *a, **k: False
except Exception:
    pass


def _detect_amd_hardware() -> bool:
    """检测是否有 AMD GPU 硬件（即使 PyTorch 是 CUDA 编译版）

    通过以下方式检测：
    1. /dev/kfd 是否存在（AMD ROCm 内核驱动）
    2. /opt/rocm/ 目录是否存在
    3. rocm-smi 命令是否可用
    """
    # AMD ROCm KFD (Kernel Fusion Driver) 设备节点
    if os.path.exists("/dev/kfd"):
        return True
    # ROCm 安装目录
    if os.path.exists("/opt/rocm"):
        return True
    # rocm-smi 命令
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "AMD" in result.stdout.upper():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return False


def detect_backend() -> Tuple[str, str]:
    """检测 GPU 后端类型和 GPU 名称

    Returns:
        (backend, device_name):
            - backend 为 "ROCm" / "CUDA" / "AMD_HW_WRONG_PYTORCH" / "CPU"
            - AMD_HW_WRONG_PYTORCH: 硬件是 AMD GPU，但 PyTorch 是 CUDA 编译版
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if any(kw in device_name.upper() for kw in ("AMD", "MI", "RADEON", "INSTINCT")):
            return "ROCm", device_name
        else:
            return "CUDA", device_name

    # PyTorch 检测不到 GPU，但可能硬件是 AMD 而 PyTorch 是 CUDA 编译版
    if _detect_amd_hardware():
        return "AMD_HW_WRONG_PYTORCH", "AMD GPU (PyTorch CUDA 编译版，需重装为 ROCm 版)"

    return "CPU", "CPU"


def is_rocm() -> bool:
    """是否为 AMD ROCm 环境（PyTorch 为 ROCm 编译版）"""
    return detect_backend()[0] in ("ROCm", "AMD_HW_WRONG_PYTORCH")


def is_cuda() -> bool:
    """是否为 NVIDIA CUDA 环境"""
    return detect_backend()[0] == "CUDA"


def get_rocm_version() -> str:
    """获取已安装的 ROCm 主版本号（major.minor），用于构造 PyTorch wheel URL

    检测优先级：
    1. /opt/rocm/.info/version 文件 (ROCm 官方方式，如 "7.0.2" -> "7.0")
    2. /opt/rocm-*/ 目录名称
    3. 默认 "6.2"
    """
    import re as _re
    import glob as _glob

    # 方法1: 读取 /opt/rocm/.info/version
    version_file = "/opt/rocm/.info/version"
    if os.path.isfile(version_file):
        try:
            with open(version_file, "r") as f:
                full_ver = f.read().strip()
            # 从 "7.0.2" 提取 "7.0"
            m = _re.match(r"(\d+\.\d+)", full_ver)
            if m:
                return m.group(1)
        except (OSError, UnicodeDecodeError):
            pass

    # 方法2: 检查 /opt/rocm-* 目录名中的版本号
    for p in sorted(_glob.glob("/opt/rocm-*"), reverse=True):
        if os.path.isdir(p):
            m = _re.search(r"(\d+\.\d+)", os.path.basename(p))
            if m:
                return m.group(1)

    return "6.2"  # 默认假设


def get_pytorch_install_cmd() -> str:
    """根据当前 GPU 后端返回正确的 PyTorch 安装命令"""
    backend, _ = detect_backend()
    if backend in ("ROCm", "AMD_HW_WRONG_PYTORCH"):
        rocm_ver = get_rocm_version()
        return (
            f"pip install torch torchvision --index-url "
            f"https://download.pytorch.org/whl/rocm{rocm_ver}"
        )
    elif backend == "CUDA":
        return (
            "pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu126"
        )
    else:
        return "pip install torch torchvision"


def get_gpu_check_cmd() -> str:
    """根据当前后端返回 GPU 状态检查命令"""
    backend, _ = detect_backend()
    if backend in ("ROCm", "AMD_HW_WRONG_PYTORCH"):
        return "rocm-smi"
    elif backend == "CUDA":
        return "nvidia-smi"
    else:
        return "echo 'No GPU detected'"


def get_safe_dtype() -> torch.dtype:
    """返回当前 GPU 安全可用的最佳 dtype

    MI300X+ 支持 bfloat16，但老款 AMD GPU (MI100/MI50) 不支持。
    统一使用 float16 以确保兼容性；如果连 float16 都不支持则回退到 float32。
    """
    backend, _ = detect_backend()
    if backend in ("CPU", "AMD_HW_WRONG_PYTORCH"):
        # AMD_HW_WRONG_PYTORCH: PyTorch 是 CUDA 编译版，GPU 不可用，回退 float32
        return torch.float32

    # 测试 bfloat16 支持
    try:
        t = torch.zeros(1, dtype=torch.bfloat16)
        t = t.to("cuda:0")
        del t
        torch.cuda.empty_cache()
        return torch.bfloat16
    except Exception:
        pass

    # 回退到 float16
    try:
        t = torch.zeros(1, dtype=torch.float16)
        t = t.to("cuda:0")
        del t
        torch.cuda.empty_cache()
        return torch.float16
    except Exception:
        return torch.float32


def bitsandbytes_available() -> bool:
    """bitsandbytes 是否可用

    bitsandbytes 是 NVIDIA CUDA 专属的量化库，在 AMD ROCm 上不可用。
    ROCm 用户需使用 torch.float16 + device_map 替代 4-bit 量化。
    """
    if is_rocm():
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def verify_pytorch_rocm() -> None:
    """诊断并打印当前环境信息（首次加载时调用）

    如果检测到 AMD 硬件但 PyTorch 是 CUDA 编译版，会给出明确的重装指引。
    """
    backend, info = detect_backend()
    print(f"[Backend] GPU 后端: {backend} | {info}")
    print(f"[Backend] PyTorch 版本: {torch.__version__}")

    if backend == "AMD_HW_WRONG_PYTORCH":
        print("=" * 65)
        print("[Backend] ⚠️  检测到 AMD GPU 硬件，但 PyTorch 是 CUDA 编译版！")
        print(f"[Backend]     PyTorch: {torch.__version__} (CUDA 版)")
        print("[Backend]     AMD GPU 需要 ROCm 编译版的 PyTorch 才能使用。")
        print(f"[Backend]     请重装: {get_pytorch_install_cmd()}")
        print("[Backend]     重装后记得重启 Python 进程。")
        print("=" * 65)
    elif backend == "ROCm":
        print(f"[Backend] ✅ AMD ROCm 环境已就绪")
    elif backend == "CUDA":
        print(f"[Backend] ✅ NVIDIA CUDA 环境已就绪")
    else:
        print(f"[Backend] ⚠️  未检测到 GPU，将使用 CPU 模式")
