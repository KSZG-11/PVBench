"""
SAM3 — 分割与视频追踪模型封装（官方源码版，真实推理，无 stub）

用途：实例分割、视频目标追踪、掩码生成
涉及的物理量：R04绝对尺度, R09相对尺度, R11空间关系, R12稳定性, R21物体恒存

架构 (v4 原生视频传播):
  segment_frame(): Sam3Processor (图像模型) → 单帧概念分割(PCS)
  segment_video():  Sam3VideoPredictor → 原生 propagate_in_video
                    (Detector+T检测+Tracker记忆库传播, 无需手写IoU匹配/重识别)

官方源码 API:
  # 图像模式 (segment_frame)
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor
  model = build_sam3_image_model(checkpoint_path=<本地 sam3.pt>, load_from_HF=False, device="cuda")
  processor = Sam3Processor(model, device="cuda", confidence_threshold=0.5)
  state = processor.set_image(pil_image)
  state = processor.set_text_prompt("object", state)
  # state["masks"]: bool [N,1,H,W], state["boxes"]: [N,4] xyxy, state["scores"]: [N]

  # 视频模式 (segment_video) — v4 新增
  from sam3.model.sam3_video_predictor import Sam3VideoPredictor
  predictor = Sam3VideoPredictor(checkpoint_path=ckpt)
  resp = predictor.handle_request({"type":"start_session", "resource_path": [pil_frames]})
  predictor.handle_request({"type":"add_prompt", "session_id": sid, "frame_index":0, "text":"ball. sofa"})
  for out in predictor.handle_stream_request({"type":"propagate_in_video", "session_id": sid}):
      masks = out["out_binary_masks"]  # {obj_id: tensor(1,H,W)} — ID由tracker维护

文本引导分割策略:
  每个评测维度对应一组针对性概念词 (DIMENSION_CONCEPTS)。
  segment_frame → 逐词PCS + IoU去重 (多物体场景)
  segment_video → 合并前几个具体词为单个 text prompt, 喂入 propagate_in_video

设计: 加载/推理失败 → 抛异常, 调用方捕获后记 status=error。
      需 pip install -e vendor/sam3_src (纯 python 依赖, 无需编译)。
"""
import os
import glob
import re
import numpy as np
import torch
from typing import Optional, List

from .base import BaseSpecialistModel, SegmentationOutput


# SAM3 概念分割需 text prompt。未指定时用通用开放词汇逐个尝试召回场景物体。
DEFAULT_CONCEPTS = ["object", "thing", "item"]

# 按评测维度的针对性文本提示词。
# 每个维度列出该维度 physics prompt 中出现的典型物体类别，
# SAM3 用这些词做概念分割 (PCS) 可以精准定位评测关心的物体，
# 避免通用词 "object/thing/item" 召回背景干扰物或漏掉核心物体。
# 设计原则:
#   短标签优先 (SAM3 对 1-3 词的短语召回最好)。
#   按面积/显著性排序 (前面的词优先被匹配)。
#   覆盖 physics.yaml 对应 prompt 中描述的主要物体。
DIMENSION_CONCEPTS = {
    # ── GeometryEvaluator ──
    "R01": [
        "red cube", "blue cube", "yellow cylinder", "green sphere",
        "red rectangular box", "blue cylinder", "car", "truck", "street lamp",
        "person", "doorway", "mountain", "bookshelf", "book", "potted plant",
        "wall clock",
        # ⚠️ unified_46d_v2 的 R01 含更多样的前后遮挡物体对,
        #    概念词须覆盖, 否则 SAM3 只分割出 1 个物体 → R01 全 skipped。
        "robot arm", "storage bin", "vase", "statue",
        "bicycle wheel", "screen", "sphere",
        # ⚠️ unified_46d_v2 R01 含 torus-knot sculpture(圆环结雕塑) / rectangular package
        #    / purple vase / black bicycle wheel 等
        "torus-knot sculpture", "torus knot", "rectangular package",
        "package", "purple vase", "black bicycle wheel",
        "ball", "cue ball",
        # ⚠️ unified_46d_v2 R01 的 hybrid 场景含 "large blue reference panel"
        #    (参照面板) —— 前后关系声明的 back 物体, 缺失会导致声明对匹配失败
        "reference panel", "blue panel", "panel",
    ],
    "R03": [
        "sphere", "cube", "cone", "cylinder", "triangular prism",
        "sedan", "car", "torus knot sculpture",
        # ⚠️ unified_46d_v2 R03 含 asymmetric toy excavator(玩具挖掘机) / stone arch(石拱)
        #    / lattice polyhedron(网格多面体) / wooden chair(木椅) / toy train 等
        "toy excavator", "excavator", "stone arch", "arch",
        "lattice polyhedron", "polyhedron", "wooden chair", "chair",
        "toy train", "model car", "ball",
    ],

    # ── ScalePoseEvaluator ──
    "R04": [
        # PHY-R04-001: 黄色直尺+马克杯+智能手机+铅笔 俯拍
        # PHY-R04-002: 人+门框 正面
        "ruler", "coffee mug", "smartphone", "pencil",
        "person", "doorway", "door frame",
        # ⚠️ unified_46d_v2 的 R04 含「网球/足球/篮球 + 标尺」等球类参照场景。
        #    球类概念词必须放在宽泛词(measurable/foreground)之前:
        #    _segment_one 遍历到宽泛词且已召回≥2个具体物体时会 break,
        #    放后面会导致球类词永远遍历不到 → 球检测不到 → R04 skipped。
        "tennis ball", "soccer ball", "basketball", "ball", "hand",
        # ⚠️ unified_46d_v2 的 R04 含「model car + calibration board」和
        #    「robot/manipulator + model car」等具身尺度场景。这些词必须在
        #    宽泛词之前, 否则 SAM3 召回不到 → 无有效深度 → R04 skipped。
        "model car", "toy car", "calibration board", "metric board",
        "robot", "robot arm", "manipulator", "apparatus",
        # ⚠️ unified_46d_v2 R04 含 stone arch(石拱) / toolbox(工具箱) / adult(成人) 等
        "stone arch", "arch", "toolbox", "tool box", "adult",
        "toy train", "train",
        "measurable object", "foreground object",
    ],
    "R09": [
        # PHY-R09-001: 红色游乐球 由远及近
        # PHY-R09-002: 氦气球 上升缩小
        # PHY-R09-003: 两辆玩具车 远离
        # PHY-R09-004: 人走远 (unified_46d_v2 扩充)
        # ⚠️ unified_46d_v2 的 R09 含大量 hybrid 具身样本 (drone/ring/robot),
        #    概念词须覆盖实际 prompt 中的物体, 否则 SAM3 视频追踪无召回 → 全 skipped。
        "red ball", "playground ball", "party balloon", "balloon",
        "toy car", "car", "drone", "ring", "person",
        "robot", "wheeled robot", "sphere",
        "moving object", "main object",
    ],
    # R10 三维位姿 (Any6D)：原缺失此键 → 回退宽泛词 object/thing/item，
    # 对「椅子/钻头/立方体」等具体物体召回差 → SAM3 首帧检测不到 → mesh 生成失败。
    # 概念词须覆盖 unified_46d_v2 R10 实际场景物体。
    "R10": [
        "asymmetric cube", "colored cube", "cube",
        "drone", "quadcopter",
        "toy excavator", "excavator",
        "chair",
        "rectangular package", "package", "box",
        "toy train", "train",
    ],

    # ── DeformationSpatialEvaluator ──
    "R11": [
        # PHY-R11-001: 三色立方体堆叠 (红/蓝/黄)
        # PHY-R11-002: 玩具火车+彩色车厢+车站
        "red cube", "blue cube", "yellow cube",
        "wooden block", "cube", "stacked block",
        "toy train", "train engine", "train carriage",
        "station building", "colored carriage",
        # ⚠️ unified_46d_v2 的 R11 含 green apple(青苹果) 等场景
        "green apple", "apple", "small box",
        "foreground object",
    ],

    # ── StaticsLightingEvaluator ──
    "R12": [
        # PHY-R12-001: 15层叠叠乐积木塔
        # PHY-R12-002: 餐桌上盘子/水杯/叉子/餐巾/盐瓶+放汤碗
        "wooden block", "Jenga block", "stacked block",
        "ceramic plate", "water glass", "metal fork",
        "cloth napkin", "salt shaker", "bowl", "bowl of soup",
        # ⚠️ unified_46d_v2 R12 含 stack of books(书堆) / long board(长木板) 等
        "book", "stack of books", "board", "long board",
        "stationary object", "tabletop object",
        "plate", "glass", "fork", "napkin",
    ],

    # ── MotionPersistenceEvaluator ──
    "R21": [
        # PHY-R21-001: 黄色网球滚过沙发后被遮挡→重现
        # PHY-R21-002: 人走过大树后被遮挡→重现
        # ⚠️ unified_46d_v2 的 R21 含更多样的遮挡目标 (toy train/cart/fish/cardboard box
        #    及 hybrid 具身样本 robot), 概念词须覆盖, 否则无召回 → 全 skipped。
        "yellow tennis ball", "tennis ball", "ball",
        "sofa", "couch", "large sofa",
        "person", "person walking",
        "oak tree", "tree trunk", "large tree",
        "toy train", "train", "tunnel", "cart", "wall",
        "fish", "reef", "robot", "cardboard box",
        "occluded object", "moving object",
    ],

    # ── MultiviewEvaluator ──
    "R08": [
        # PHY-R08-001: 环绕大理石马雕像 (台座)
        # PHY-R08-002: 桌上 cube/sphere/cone 环绕
        "marble statue", "horse statue", "statue", "pedestal",
        "cube", "sphere", "cone", "table",
        # ⚠️ unified_46d_v2 R08 含 drone / green apple / stone arch / manipulator / house model 等
        "drone", "green apple", "apple", "stone arch", "arch",
        "manipulator", "robot arm", "house model", "model house",
        "ball",
    ],

    # ── StaticsLightingEvaluator (R13 光影空间一致性) ──
    "R13": [
        # PHY-R13-001: 台灯照亮白球/白立方/白圆柱
        # PHY-R13-002: 五根等距杆 + 阴影
        "desk lamp", "lamp", "white sphere", "white cube", "white cylinder",
        "sphere", "cube", "cylinder",
        "vertical pole", "pole", "shadow",
        # ⚠️ unified_46d_v2 R13 含 asymmetric cube / metallic sphere / ceramic sphere / toy car
        "asymmetric cube", "metallic sphere", "ceramic sphere", "toy car",
    ],

    # ── KinematicsEvaluator ──
    "R15": [
        # PHY-R15-001: 白色台球滚过绿色台呢
        # PHY-R15-002: 配重杆(节拍器)摆动
        "cue ball", "white ball", "billiard ball", "ball",
        "weighted rod", "rod", "metronome",
        # ⚠️ unified_46d_v2 的 R15/R16 共用此键(dimension="R15")，物体远不止台球/节拍器，
        #    须覆盖实际场景，否则 SAM3 召回失败 → 物体轨迹缺失 → 降级/跳帧。
        #   R15: asymmetric cube / beanbag / drone / cardboard box / quadruped robot
        #        / fish / wheeled cart / cue ball
        #   R16: bowling ball / car / cart / metal ball / pendulum bob / toy train
        "asymmetric cube", "cube",
        "beanbag", "bean bag",
        "drone", "quadcopter",
        "cardboard box", "box",
        "quadruped robot", "robot",
        "fish", "striped fish",
        "wheeled cart", "cart",
        "bowling ball", "metal ball", "steel ball",
        "car", "pendulum bob", "pendulum", "toy train", "train",
    ],
    "R17": [
        # PHY-R17-001: 三叶吊扇 (一叶贴色标)
        # PHY-R17-002: 自行车轮 (36辐条)
        "ceiling fan", "fan", "fan blade",
        "bicycle wheel", "wheel", "spoke",
        # ⚠️ unified_46d_v2 的 R17 含更多旋转物体，须覆盖否则 SAM3 召回失败：
        #   drill chuck(钻夹头) / satellite model(卫星) / pendulum bob(钟摆)
        #   / asymmetric cube on motorized axle(电机轴上的非对称立方体)。
        "drill chuck", "drill", "chuck",
        "satellite model", "satellite", "solar panel",
        "pendulum bob", "pendulum", "brass bob",
        "asymmetric cube", "cube",
        "mobile manipulator", "robot arm", "manipulator",
    ],

    # ── DynamicsEvaluator ──
    # ⚠️ dynamics.py 的 R22/R23/R24 共用一次 build_object_tracks(dimension="R22"),
    #    故 R22 的概念词必须覆盖三个维度的全部场景物体, 否则 R23(冰球/弹珠)等
    #    场景会因概念词缺失 → SAM3 召回失败 → grid_fallback → 有效物体数=1 skipped。
    "R22": [
        # PHY-R22-001: 推纸箱滑动 / 002: 弹簧炮弹泡沫球 / 003: 球落地弹跳
        # 004: 木块斜面下滑 / 005: 绳球圆周 / 006: 钢球+塑料球同落
        "cardboard box", "box", "hand",
        "foam ball", "toy cannon", "cannon",
        "wooden block", "block", "plank",
        "ball", "steel ball", "plastic ball",
        # R23 动量守恒: 牛顿摆五钢球 / 两冰球对碰 / 大弹珠撞小弹珠
        "newton's cradle", "metal ball",
        "puck", "hockey puck", "air table",
        "marble", "grooved track", "track",
        # ⚠️ unified_46d_v2 的 R24(能量守恒)复用 R22 概念词(dimension="R22")，
        #    须覆盖 R24 实际物体: brass pendulum bob(铜钟摆) / cart(推车) / tennis ball(网球)。
        "pendulum bob", "pendulum", "brass bob",
        "cart", "tennis ball",
    ],
    "R23": [
        # PHY-R23-001: 牛顿摆五钢球 / 002: 两冰球对碰 / 003: 大弹珠撞小弹珠
        # (保留独立键供将来 R23 单独调用时使用)
        "newton's cradle", "steel ball", "metal ball",
        "puck", "air table", "hockey puck",
        "marble", "grooved track", "track",
        "ball",
    ],
}


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    union = a.sum() + b.sum() - inter
    return inter / max(union, 1)


class SAM3Model(BaseSpecialistModel):
    """SAM3 封装（官方源码 build_sam3_image_model + Sam3Processor）

    支持按评测维度使用针对性文本 prompt:
      segment_frame(frame, dimension="R04") → 用 R04 特化概念词
      segment_video(frames, dimension="R21") → 用 R21 特化概念词
    未传 dimension 时回退到通用 DEFAULT_CONCEPTS。
    """

    def __init__(self, model_type: str = "vit_h",
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 concepts: Optional[list] = None,
                 confidence_threshold: float = 0.5,
                 **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_id = checkpoint_path or "facebook/sam3"
        self.concepts = concepts or DEFAULT_CONCEPTS
        self.confidence_threshold = confidence_threshold
        self._processor = None
        self._video_predictor = None  # 惰性加载, 仅 segment_video() 触发

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": "facebook/sam3",
            "checkpoint": self.model_id,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

    def _resolve_checkpoint(self) -> Optional[str]:
        """定位本地 sam3.pt。找不到返回 None (由 build 决定是否联网下载)。"""
        from .base import resolve_local_model
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        # checkpoint_path 若直接指向 .pt 文件
        if self.model_id.endswith(".pt") and os.path.isfile(self.model_id):
            return self.model_id
        src = resolve_local_model("facebook/sam3", cache_dir)
        if src != "facebook/sam3" and os.path.isdir(src):
            cands = sorted(glob.glob(os.path.join(src, "**", "sam3.pt"), recursive=True)) \
                or sorted(glob.glob(os.path.join(src, "**", "*.pt"), recursive=True))
            if cands:
                return cands[0]
        return None

    def load(self, **kwargs) -> None:
        """加载 SAM3 官方图像模型。失败抛异常。"""
        # vendor/sam3_src 已 pip install -e → import sam3 可用
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        ckpt = self._resolve_checkpoint()
        load_from_HF = ckpt is None   # 本地无 .pt 才允许联网(离线环境应恒为本地)
        print(f"[SAM3] 构建官方图像模型 (checkpoint={'本地 ' + os.path.basename(ckpt) if ckpt else 'HF下载'})")
        model = build_sam3_image_model(
            checkpoint_path=ckpt,
            load_from_HF=load_from_HF,
            device=self.device,
            eval_mode=True,
        )
        self._model = model
        self._processor = Sam3Processor(
            model, device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        self._is_loaded = True
        print(f"[SAM3] ✓ 官方源码模型加载成功 (concepts={self.concepts})")

    def _load_video_predictor(self) -> None:
        """惰性加载 SAM3 视频预测器 (仅 segment_video() 触发)。

        Sam3VideoPredictor 内部调用 build_sam3_video_model 构建
        Detector + Tracker 双模型, 并用 propagate_in_video 原生传播。
        与 segment_frame() 用的图像模型独立, 内存额外 ≈5-6GB。
        """
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        ckpt = self._resolve_checkpoint()
        print(f"[SAM3] 构建视频预测器 "
              f"(checkpoint={'本地 ' + os.path.basename(ckpt) if ckpt else 'HF下载'})")
        # Sam3VideoPredictor 内部硬编码 device="cuda", 不接受 device 参数。
        # 在 CPU-only 环境下 build_sam3_video_model 会自动 fallback。
        self._video_predictor = Sam3VideoPredictor(
            checkpoint_path=ckpt,
            async_loading_frames=False,  # 传 PIL list, 无需异步文件加载
            apply_temporal_disambiguation=True,
            strict_state_dict_loading=False,  # 相同 checkpoint, 允许 tracker 侧多出 key
        )
        print(f"[SAM3] ✓ 视频预测器加载成功")

    def _build_video_text_prompt(self, dimension: Optional[str] = None,
                                 concepts: Optional[List[str]] = None) -> str:
        """从维度概念词构建 propagate_in_video 的 text prompt。

        concepts由上层根据当前样本prompt筛选时，优先使用样本中真实出现的
        具体物体词，避免把互不相关的"ball/balloon/sofa/person"拼成一个
        SAM3文本查询导致Detector无召回。
        """

        # 只取前几个具体概念词，用句号连接；过多概念会稀释PCS召回。
        if concepts is None:
            concepts = DIMENSION_CONCEPTS.get(dimension, None) if dimension else None
        if concepts is None:
            concepts = self.concepts or DEFAULT_CONCEPTS
        # 过滤宽泛兜底词, 只保留具体概念
        specific = [c for c in concepts if not self._is_broad_concept(c)]
        if not specific:
            specific = concepts[:4]
        # 取前5个最具体概念 (太多会稀释 PCS 打分)
        specific = specific[:5]
        return ". ".join(specific)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._video_predictor = None
        self._is_loaded = False
        torch.cuda.empty_cache()

    # 宽泛/兜底概念词标记（此类词召回任何物体，容易与前面具体词重复）
    _BROAD_CONCEPT_PATTERNS = (
        "foreground", "background", "object", "item", "thing",
        "main", "movable", "measurable", "stationary", "tabletop",
        "occluded", "moving",
    )

    @classmethod
    def _is_broad_concept(cls, concept: str) -> bool:
        """判断是否为宽泛兜底概念词（不应优先于具体词召回结果）。"""
        low = concept.lower()
        return any(p in low for p in cls._BROAD_CONCEPT_PATTERNS)

    @staticmethod
    def prompt_concepts(prompt: str, dimension: str) -> Optional[List[str]]:
        """从当前样本 prompt 筛选该维度真实出现的概念词。

        维度默认概念把多个互斥场景的词拼在一起，对开放词汇 Detector 会稀释召回。
        只返回 prompt 中实际出现的具体词；无命中返回 None（让 SAM3 用维度默认词）。
        """
        if not prompt:
            return None
        low = prompt.lower()
        selected = []
        for concept in DIMENSION_CONCEPTS.get(dimension, []):
            words = [w for w in re.findall(r"[a-z]+", concept.lower())
                     if len(w) > 2]
            # 允许常见复数/词形变化: "car" 匹配 "cars", "ring" 匹配 "rings"
            if words and all(
                    re.search(r"\b" + re.escape(w) + r"(?:s|es)?\b", low)
                    for w in words):
                selected.append(concept)
        return selected or None

    def _segment_one(self, frame: np.ndarray,
                     override_concepts: Optional[List[str]] = None,
                     max_results: Optional[int] = None) -> List[dict]:
        """单帧概念分割, 返回 [{segmentation(bool HxW), score, bbox(xyxy)}]。

        v2.4: 遍历所有概念词并合并结果 + IoU 去重（不再贪心 break），
              支持多物体场景（如 R11 三色立方体堆叠、R12 餐桌多物品）。

        Args:
            frame: [H,W,3] uint8 numpy array
            override_concepts: 临时覆盖 self.concepts 的概念词列表;
                               None 则使用 self.concepts (默认通用词)
        """
        from PIL import Image as _Image
        import contextlib
        concepts = override_concepts if override_concepts is not None else self.concepts
        image = _Image.fromarray(frame.astype(np.uint8))
        _amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                if (torch.cuda.is_available() and str(self.device).startswith("cuda"))
                else contextlib.nullcontext())
        all_results = []                # 所有未去重结果 [{seg, score, bbox, area}]
        _debug_printed = getattr(self, '_sam3_debug_done', False)
        _n_specific = 0                 # 从具体（非兜底）词累计召回的物体数
        with torch.inference_mode(), _amp:
            for concept in concepts:
                # 若已从具体词召回了 ≥2 个物体且当前词是兜底宽泛词 → 跳过防噪声
                # （≤1 个物体时仍需兜底词补召回，避免漏检）
                is_broad = self._is_broad_concept(concept)
                if _n_specific >= 2 and is_broad:
                    if not _debug_printed:
                        print(f"[SAM3] 已有 {_n_specific} 个具体物体，跳过兜底词 "
                              f"'{concept}' 及后续")
                    break

                state = self._processor.set_image(image)
                # 先以 thresh=0.0 获取全部 pred, 再手动按正常阈值过滤
                _orig_thresh = self._processor.confidence_threshold
                self._processor.confidence_threshold = 0.0
                state = self._processor.set_text_prompt(concept, state)
                self._processor.confidence_threshold = _orig_thresh

                masks_all = state.get("masks")   # [N,1,H,W] bool tensor
                scores_all = state.get("scores") # [N]
                n_raw = len(masks_all) if masks_all is not None else 0
                if hasattr(scores_all, "cpu"):
                    sc_np = scores_all.float().cpu().numpy()
                else:
                    sc_np = np.asarray(scores_all) if scores_all is not None else np.array([])
                n_keep = int((sc_np > self.confidence_threshold).sum()) if len(sc_np) else 0

                # 仅首帧打印一次摘要 (避免每帧刷屏)
                if not _debug_printed or n_keep > 0:
                    top_scores = sorted([float(x) for x in sc_np], reverse=True)[:3]
                    mark = " [兜底]" if is_broad else ""
                    print(f"[SAM3] concept='{concept}'{mark} n_raw={n_raw} "
                          f"top3_scores={[round(s,3) for s in top_scores]} "
                          f"conf_thresh={self.confidence_threshold} n_keep={n_keep}")

                if masks_all is None or len(masks_all) == 0:
                    continue

                # 手动按正常阈值过滤
                keep_idx = np.where(sc_np > self.confidence_threshold)[0]
                if len(keep_idx) == 0:
                    continue

                masks = masks_all
                if hasattr(masks, "cpu"):
                    masks = masks[torch.as_tensor(keep_idx, device=masks.device)]
                    masks = masks.float().cpu().numpy()
                else:
                    masks = np.asarray(masks)[keep_idx]
                sc = sc_np[keep_idx]
                # R01只需要少量高置信度候选来寻找可判定遮挡对。
                # street lamp/book等开放词汇会返回大量低质量实例；保留每个
                # concept的top-K，避免候选爆炸后所有物体对都不可判定。
                if max_results is not None and len(keep_idx) > max_results:
                    order = np.argsort(sc_np[keep_idx])[::-1][:max_results]
                    keep_idx = keep_idx[order]
                    masks = masks_all
                    if hasattr(masks, "cpu"):
                        masks = masks[torch.as_tensor(keep_idx, device=masks.device)]
                        masks = masks.float().cpu().numpy()
                    else:
                        masks = np.asarray(masks)[keep_idx]
                    sc = sc_np[keep_idx]

                if masks.ndim == 4:      # [N,1,H,W]
                    masks = masks[:, 0]
                elif masks.ndim == 2:    # [H,W] 单实例
                    masks = masks[None]
                n_before = len(all_results)
                for i in range(masks.shape[0]):
                    mask_i = masks[i].astype(bool)
                    if mask_i.sum() < 50:
                        continue
                    ys, xs = np.where(mask_i)
                    if len(xs) == 0:
                        continue
                    all_results.append({
                        "segmentation": mask_i,
                        "score": float(sc[i]) if i < len(sc) else 1.0,
                        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                        "area": int(mask_i.sum()),
                        "_concept": concept,
                    })
                # 统计从具体词召回的物体数（兜底词不算，用于判断何时跳过兜底词）
                if not is_broad:
                    _n_specific += len(all_results) - n_before
        self._sam3_debug_done = True

        # ── IoU 去重: 仅在同一概念词内部去重 ──
        # 关键: 不同概念词召回的两个物体即使 mask 大面积投影重叠
        # (如 "vase before statue" 前后遮挡场景), 也代表两个语义不同的物体,
        # 绝不能跨概念词 IoU 去重, 否则「前后遮挡」的两个物体会被误合并成 1 个
        # → R01 报「分割物体数=1(<2)」skipped。
        if len(all_results) <= 1:
            return all_results

        # 按概念词分组, 组内按 score 降序做 IoU 去重(去掉同一概念的碎片重复)
        by_concept: dict = {}
        for r in all_results:
            by_concept.setdefault(r.get("_concept", ""), []).append(r)
        kept = []
        for concept, group in by_concept.items():
            group.sort(key=lambda r: r["score"], reverse=True)
            for r in group:
                dup = False
                for k in kept:
                    if k.get("_concept") != concept:
                        continue  # 跨概念不去重(不同语义物体)
                    if _mask_iou(r["segmentation"], k["segmentation"]) > 0.7:
                        dup = True
                        break
                if not dup:
                    kept.append(r)
        kept.sort(key=lambda r: r["area"], reverse=True)

        n_before = len(all_results)
        if n_before != len(kept):
            print(f"[SAM3] IoU去重(组内): {n_before} → {len(kept)} 个物体 "
                  f"(移除 {n_before - len(kept)} 个重复)")
        return kept

    def segment_frame(self, frame: np.ndarray,
                      dimension: Optional[str] = None,
                      concepts: Optional[List[str]] = None,
                      prompt: Optional[str] = None) -> SegmentationOutput:
        """单帧概念分割 (R04/R09/R11/R12)。

        Args:
            frame: [H,W,3] uint8 numpy array
            dimension: 评测维度 ID (如 "R04","R09","R11","R12")。
                      传入时使用 DIMENSION_CONCEPTS 中的针对性文本提示词；
                      None 则使用通用 DEFAULT_CONCEPTS。
            concepts: 显式概念词列表(优先级最高)。None 时若传了 prompt,
                      则从 prompt 自动筛选该维度的真实概念词。
            prompt: 样本 prompt 文本, 用于自动筛选概念词(避免把维度默认词
                    中未出现的互斥场景词喂给 Detector 稀释召回)。
                      失败抛 RuntimeError。
        """
        if not self._is_loaded:
            raise RuntimeError("SAM3 未加载")
        if concepts is None and prompt:
            concepts = self.prompt_concepts(prompt, dimension)
        concepts = concepts or (DIMENSION_CONCEPTS.get(dimension, None) if dimension else None)
        # R01/R03只需有限的候选物体；R03必须只分割当前prompt中的标准形状，
        # 不能把维度库中未出现在样本中的形状（如torus knot）带进来。
        max_results = 4 if dimension in {"R01", "R03"} else None
        results = self._segment_one(
            frame, override_concepts=concepts, max_results=max_results)
        if dimension in {"R01", "R03"}:
            # 同一语义概念可能产生多个候选实例；R01/R03的物体级几何
            # 比对不能把同一概念的重复mask当成多个物体。每个概念只保留
            # 置信度最高的一个，再施加全局上限。
            unique = {}
            for result in results:
                concept = result.get("_concept", "")
                if concept not in unique or result["score"] > unique[concept]["score"]:
                    unique[concept] = result
            results = list(unique.values())
        if max_results is not None and len(results) > max_results:
            results = sorted(results, key=lambda r: r["score"], reverse=True)[:max_results]
        if not results:
            H, W = frame.shape[:2]
            return SegmentationOutput(masks=np.zeros((0, H, W), dtype=np.float32),
                                      class_names=[], confidences=np.array([]))
        masks = np.stack([r["segmentation"] for r in results]).astype(np.float32)
        return SegmentationOutput(
            masks=masks,
            class_names=[r.get("_concept", f"object_{i}")
                         for i, r in enumerate(results)],
            confidences=np.array([r["score"] for r in results]),
        )

    def segment_video(self, frames: np.ndarray,
                      dimension: Optional[str] = None,
                      concepts: Optional[List[str]] = None,
                      initial_boxes: Optional[np.ndarray] = None,
                      prompt_frame_index: int = 0,
                      prompt: Optional[str] = None) -> SegmentationOutput:
        """原生视频传播 (R09/R15/R21)。

        v4: 使用 SAM3 官方 Sam3VideoPredictor.propagate_in_video, 替换手写
        IoU 匹配 + ghost tracks + ReID。Detector+Tracker 联合推理,
        memory bank 保证跨帧 ID 一致性, 内置遮挡处理+重识别。

        Args:
            frames: [T,H,W,3] uint8 numpy array
            dimension: 评测维度 ID (如 "R21")。
                      传入时使用 DIMENSION_CONCEPTS 构建 text prompt；
                      None 则使用默认概念词。
            concepts: 显式概念词列表(优先级最高)。None 且传了 prompt 时,
                      从 prompt 自动筛选该维度真实概念词。
            prompt: 样本 prompt 文本, 用于自动筛选概念词。
            initial_boxes: 可选 [N,4] 归一化 [xmin,ymin,width,height] (0~1) 的初始框。
                      提供时用 box prompt 替代 text prompt 做初始检测 ——
                      规避 SAM3 Detector 对具体词(如 "red ball"/"toy car")
                      PCS 召回差导致视频追踪无目标的问题。
            prompt_frame_index: 注入初始框/文本的帧索引 (默认0)。
                      配合多帧候选: 当首帧物体未就位/太小/被遮挡时,
                      可改用后续帧做检测锚点 (SAM3 从该帧向前后双向传播)。
                      失败抛 RuntimeError。
        """
        if not self._is_loaded:
            raise RuntimeError("SAM3 未加载")
        if self._video_predictor is None:
            self._load_video_predictor()

        T, H, W = frames.shape[:3]

        # numpy → PIL 列表 (Sam3VideoPredictor.start_session 原生支持)
        from PIL import Image as _Image
        pil_frames = [_Image.fromarray(frames[t].astype(np.uint8)) for t in range(T)]

        use_boxes = initial_boxes is not None and len(initial_boxes) > 0
        if concepts is None and prompt and not use_boxes:
            concepts = self.prompt_concepts(prompt, dimension)
        text_prompt = self._build_video_text_prompt(dimension, concepts=concepts)
        if use_boxes:
            print(f"[SAM3 video] 用 {len(initial_boxes)} 个初始框替代 text_prompt "
                  f"(text='{text_prompt}' 仅作兜底) T={T}")
        else:
            print(f"[SAM3 video] text_prompt='{text_prompt}' T={T}")

        import contextlib
        _amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                if (torch.cuda.is_available() and str(self.device).startswith("cuda"))
                else contextlib.nullcontext())

        with torch.inference_mode(), _amp:
            # 1. 创建 session (传 PIL 列表, 无需写临时文件)
            resp = self._video_predictor.handle_request({
                "type": "start_session",
                "resource_path": pil_frames,
            })
            sid = resp["session_id"]

            # 2. 在 prompt_frame_index 帧注入初始 prompt (box 优先, 否则 text)
            #    多帧候选时用「物体状态最佳」的帧做锚点, 而非硬编码首帧。
            add_req = {
                "type": "add_prompt",
                "session_id": sid,
                "frame_index": prompt_frame_index,
                "text": text_prompt,
            }
            if use_boxes:
                # SAM3 add_prompt 的 boxes 要求归一化 [xmin,ymin,width,height] (0~1)
                boxes = np.asarray(initial_boxes, dtype=np.float32)
                add_req["bounding_boxes"] = boxes
                # ⚠️ box label 语义: 1=positive box(追踪该物体), 0=negative box(排除)。
                #    之前误用 list(range(...)) 生成 [0,1,...], 第一个 box 是 0(negative),
                #    导致 SAM3 警告 "A negative box is added as a visual prompt" 且
                #    该 negative box 不追踪任何物体 → 有效物体帧=0 → 全 skipped。
                #    必须全 1 (positive box)。
                add_req["bounding_box_labels"] = [1] * len(boxes)
            self._video_predictor.handle_request(add_req)

            # 3. 原生传播: Tracker memory bank 维护跨帧 ID 一致性
            # ⚠️ 输出结构: predictor 层 yield {"frame_index":..., "outputs": {...}},
            #    其中 outputs 含 "out_binary_masks"([N,H,W] bool) 和 "out_obj_ids"([N])。
            #    不能直接 out.get("out_binary_masks") —— 那会恒为空 dict 导致全 skipped。
            all_frame_masks = []   # [{obj_id: numpy(H,W) bool}, ...] 每帧
            n_yielded = 0
            for out in self._video_predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": sid,
            }):
                outputs = out.get("outputs") or {}
                binary_masks = outputs.get("out_binary_masks")
                obj_ids = outputs.get("out_obj_ids")
                frame_masks = {}
                if binary_masks is not None and obj_ids is not None \
                        and len(binary_masks) > 0:
                    for i in range(len(obj_ids)):
                        oid = int(obj_ids[i])
                        frame_masks[oid] = np.asarray(binary_masks[i]).astype(bool)
                all_frame_masks.append(frame_masks)
                n_yielded += 1

            # 4. 关闭 session
            self._video_predictor.handle_request({
                "type": "close_session",
                "session_id": sid,
            })

        print(f"[SAM3 video] propagate 完成: {n_yielded}/{T} 帧产出, "
              f"有效物体帧={sum(1 for fm in all_frame_masks if fm)}")

        # ── 转换为 SegmentationOutput ──
        # propagate_in_video 可能产出少于 T 帧 (通常不会, 但防御)
        n_out = min(len(all_frame_masks), T)
        if n_out == 0:
            return SegmentationOutput(
                masks=np.zeros((0, T, H, W), dtype=np.float32),
                class_names=[], tracking_ids=np.zeros((0, T), dtype=int))

        # 收集所有跨帧出现过的 obj_id
        all_obj_ids = set()
        for fm in all_frame_masks[:n_out]:
            all_obj_ids.update(fm.keys())
        if not all_obj_ids:
            return SegmentationOutput(
                masks=np.zeros((0, T, H, W), dtype=np.float32),
                class_names=[], tracking_ids=np.zeros((0, T), dtype=int))

        sorted_ids = sorted(all_obj_ids)
        id_to_idx = {oid: i for i, oid in enumerate(sorted_ids)}
        N_obj = len(sorted_ids)

        masks = np.zeros((N_obj, T, H, W), dtype=np.float32)
        tracking_ids = np.full((N_obj, T), -1, dtype=int)

        for t, fm in enumerate(all_frame_masks[:n_out]):
            for oid, mask_np in fm.items():
                if oid not in id_to_idx:
                    continue
                idx = id_to_idx[oid]
                # mask_np: (H, W) bool numpy → float32
                masks[idx, t] = mask_np.astype(np.float32)
                tracking_ids[idx, t] = int(oid)

        return SegmentationOutput(
            masks=masks,
            class_names=[f"object_{i}" for i in range(N_obj)],
            tracking_ids=tracking_ids,
        )

    @staticmethod
    def get_bounding_boxes(masks: np.ndarray) -> np.ndarray:
        """从掩码提取 2D 包围盒 [x0,y0,x1,y1]。masks [N,H,W] 或 [N,T,H,W]。"""
        if masks is None or (hasattr(masks, "shape") and masks.shape[0] == 0):
            return np.zeros((0, 4))
        multi_t = masks.ndim == 4
        arr = masks if multi_t else masks[:, np.newaxis, :, :]
        N, T = arr.shape[:2]
        bboxes = np.zeros((N, T, 4))
        for i in range(N):
            for t in range(T):
                ys, xs = np.where(arr[i, t] > 0.5)
                if len(xs) > 0:
                    bboxes[i, t] = [xs.min(), ys.min(), xs.max(), ys.max()]
        return bboxes if multi_t else bboxes[:, 0, :]
