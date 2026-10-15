"""
全局随机种子 — 保证 benchmark 可复现。

在 pipeline 初始化时调用 set_global_seed(seed)，统一设定
python random / numpy / torch (含 CUDA) 的种子。
"""
import os
import random


def set_global_seed(seed: int = 42) -> None:
    """设定全局随机种子 (可复现)。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 追求可复现时可开启(会牺牲部分性能)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"[Seed] 全局随机种子已设定: {seed}")
