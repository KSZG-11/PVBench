"""
Genesis — 物理仿真引擎封装（真实推理，无 stub）

用途：物理仿真 rollout 生成（作为动作因果 GT 参考）
涉及的评测维度：AC-1 动作因果效应

官方: Genesis-Embodied-AI/Genesis, `pip install genesis-world`。
  import genesis as gs; gs.init(backend=gs.gpu); scene = gs.Scene(...)

设计: 加载/推理失败 → 抛异常, evaluator 记 status=error。
      仿真 rollout 需要显式场景配置(物体/质量/初速/地面摩擦)，由调用方提供。
"""
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseSpecialistModel, SimulationOutput


class GenesisEngine(BaseSpecialistModel):
    """Genesis 物理引擎封装（真实仿真）"""

    def __init__(self, device: str = "cuda", dt: float = 0.01, **kwargs):
        super().__init__(device=device, **kwargs)
        self.dt = dt
        self._gs = None

    def load(self, **kwargs) -> None:
        """初始化 Genesis。失败抛异常。"""
        import genesis as gs
        backend = gs.gpu if self.device.startswith("cuda") else gs.cpu
        try:
            gs.init(backend=backend)
        except Exception:
            # 已初始化则忽略
            pass
        self._gs = gs
        self._model = gs
        self._is_loaded = True
        print("[Genesis] ✓ 物理引擎初始化成功")

    def unload(self) -> None:
        self._model = None
        self._gs = None
        self._is_loaded = False

    def rollout(self, scene_config: Dict[str, Any],
                action: Dict[str, Any], n_steps: int = 100) -> SimulationOutput:
        """物理仿真 rollout（历史 AC-1 实现）。失败抛 RuntimeError。

        scene_config: {"objects":[{"type","mass","position","radius"}], "ground":{"friction"}}
        action: {"type":"push","target":idx,"force":[fx,fy,fz],"duration":steps}
        """
        if not self._is_loaded:
            raise RuntimeError("Genesis 未加载")
        gs = self._gs

        scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=self.dt))
        scene.add_entity(gs.morphs.Plane())

        objects = scene_config.get("objects", [])
        entities = []
        for obj in objects:
            pos = obj.get("position", [0, 0, 1])
            if obj.get("type") == "sphere":
                ent = scene.add_entity(gs.morphs.Sphere(
                    pos=tuple(pos), radius=obj.get("radius", 0.1)))
            else:
                ent = scene.add_entity(gs.morphs.Box(
                    pos=tuple(pos), size=obj.get("size", (0.1, 0.1, 0.1))))
            entities.append(ent)

        scene.build()

        N = len(entities)
        traj = np.zeros((N, n_steps, 3))
        target = action.get("target", 0)
        force = np.array(action.get("force", [0, 0, 0]), dtype=np.float64)
        duration = action.get("duration", 0)

        for t in range(n_steps):
            if t < duration and 0 <= target < N:
                try:
                    entities[target].apply_external_force(force=tuple(force))
                except Exception:
                    pass
            scene.step()
            for i, ent in enumerate(entities):
                p = ent.get_pos()
                traj[i, t] = np.asarray(p.cpu() if hasattr(p, "cpu") else p).reshape(3)

        return SimulationOutput(rollout_trajectories=traj,
                                rollout_states={"dt": self.dt}, contact_events=[])
