"""
FineVQ — 视频质量/美学评分模型封装（FQ-2 美学质量，真实推理，无 stub）

用途：L2 画质层 FQ-2 美学质量评分，替代旧的 LAION Aesthetic Predictor。

模型：FineVQ (CVPR 2025 Highlight, IntMeGroup/FineVQ)
  - 权重: IntMeGroup/FineVQ_score (8B, InternVL2-8B base + 回归头, BF16 Safetensors)
  - 架构: internvl_chat_infer 自定义代码 (InternVLChatModel, trust_remote_code=True)
  - 输入: 视频帧(稀疏8帧 + 密集运动帧) + 固定 prompt
  - 输出: score1 标量 (MOS 连续分数, 约 1-5 区间)

依赖:
  - git clone https://github.com/IntMeGroup/FineVQ → third_party/FineVQ (取 internvl 代码)
  - 权重 snapshot 下载到本地 (HF_HOME/hub/models--IntMeGroup--FineVQ_score/snapshots/*)
  - decord + imageio-ffmpeg 读/写视频

设计: 加载/推理失败 → 抛异常, evaluator 记 status=error。
      score() 返回连续 MOS 分数, 由 evaluator 做 1-5 整数映射 (round)。
      只在 L2 画质层加载, 用 unload()/to_cpu()/to_cuda() 与物理层模型错峰。

实现：复用 FineVQ 仓库的 internvl 自定义建模代码 (internvl_chat_infer) 与
      LazySupervisedDataset.video_get_item 预处理管线, 保证与官方推理完全一致。
"""
import os
import sys
import tempfile
from typing import Optional, List

import numpy as np
import torch
from PIL import Image

from .base import BaseSpecialistModel


_FINEVQ_DEFAULTS = {
    "force_image_size": 448,
    "max_dynamic_patch": 6,
    "down_sample_ratio": 0.5,
    "drop_path_rate": 0.1,
    "vision_select_layer": -1,
    "ps_version": "v2",
    "use_thumbnail": True,
    "dynamic_image_size": True,
    "num_segments": 8,
    "conv_style": "internlm2-chat",
    "max_seq_length": 4096,
}


class FineVQModel(BaseSpecialistModel):
    """FineVQ 视频质量/美学评分封装。"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda",
                 hf_cache_dir: Optional[str] = None,
                 num_segments: int = 8,
                 **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_path = model_path or "IntMeGroup/FineVQ_score"
        self.hf_cache_dir = hf_cache_dir
        self.num_segments = num_segments

        self._model = None
        self._tokenizer = None
        self._dataset_cls = None
        self._img_context_token_id = None

    @property
    def metadata(self) -> dict:
        return {
            "class": type(self).__name__,
            "model_id": "IntMeGroup/FineVQ_score",
            "checkpoint": self.model_path,
            "device": self.device,
            "num_segments": self.num_segments,
            "is_loaded": self._is_loaded,
        }

    # ── 内部工具 ──
    def _ensure_internvl_path(self) -> str:
        finevq_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "third_party", "FineVQ"))
        if not os.path.isdir(finevq_root):
            raise FileNotFoundError(
                f"FineVQ 源码目录不存在: {finevq_root}\n"
                f"请执行: git clone https://github.com/IntMeGroup/FineVQ {finevq_root}")
        for sub in (finevq_root, os.path.join(finevq_root, "internvl")):
            if sub not in sys.path:
                sys.path.insert(0, sub)
        return finevq_root

    def _resolve_model_src(self) -> str:
        from .base import resolve_local_model
        cache_dir = self.hf_cache_dir or os.environ.get("HF_HOME") \
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
        return resolve_local_model(self.model_path, cache_dir)

    def load(self, **kwargs) -> None:
        """加载 FineVQ。失败抛异常。"""
        self._ensure_internvl_path()

        # ⚠️ InternVL 建模代码只有 _tied_weights_keys, transformers 4.57+ 会访问
        #    all_tied_weights_keys → AttributeError (与 ReasonX 相同)。公共补丁。
        from .base import patch_tied_weights_keys_compat
        patch_tied_weights_keys_compat()

        from internvl.model.internvl_chat_infer import (
            InternVisionConfig, InternVisionModel,
            InternVLChatConfig, InternVLChatModel)
        from transformers import AutoTokenizer

        src = self._resolve_model_src()
        is_local = os.path.isdir(src)
        # ⚠️ fp32 加载 (FQ-2 NaN 根因修复): FineVQ 的 vit_embeds 范数 (~58) 与
        #    文本 embeds (~0.87) 悬殊是其固有设计; LoRA 增量把 wqkv 输出推大到
        #    ~184, RoPE 后 QK^T ≈ 3000 → bf16 溢出 inf → softmax 全 NaN →
        #    nan_to_num 抹平 → score 恒定。fp32 下该量级不溢出, 可正常推理。
        #    (显存: 8B fp32 ≈ 32GB, H20-95GB 充裕)
        load_kw = {"torch_dtype": torch.float32, "trust_remote_code": True}
        if is_local:
            load_kw["local_files_only"] = True
            print(f"[FineVQ] 使用本地权重目录: {src}")

        # ⚠️ FineVQ 的 InternVLChatConfig.__init__ 硬编码 `open('config.json')` (相对 CWD),
        #    故加载 config 时须临时 chdir 到权重目录 (否则报 FileNotFoundError: config.json)。
        old_cwd = os.getcwd()
        if is_local:
            os.chdir(src)
        try:
            # 1) config + 模型
            config = InternVLChatConfig.from_pretrained(src, trust_remote_code=True)
        finally:
            os.chdir(old_cwd)
        config.vision_config.drop_path_rate = _FINEVQ_DEFAULTS["drop_path_rate"]
        config.template = _FINEVQ_DEFAULTS["conv_style"]
        config.select_layer = _FINEVQ_DEFAULTS["vision_select_layer"]
        config.dynamic_image_size = _FINEVQ_DEFAULTS["dynamic_image_size"]
        config.use_thumbnail = _FINEVQ_DEFAULTS["use_thumbnail"]
        config.ps_version = _FINEVQ_DEFAULTS["ps_version"]
        config.max_dynamic_patch = _FINEVQ_DEFAULTS["max_dynamic_patch"]
        # ⚠️ 不强制 flash_attention_2 (FQ-2 恒定输出根因): FineVQ 权重的
        #    llm_config.attn_implementation="flash_attention_2", 其 2024 自定义
        #    flash varlen 路径与 flash-attn 2.7.4 不兼容 → 第 1 个 block 输出
        #    100% NaN → nan_to_num 抹平 → score 恒定 0.009。
        #    注意: InternVLChatModel.__init__ 会按 has_flash_attn 硬编码覆盖
        #    attn_implementation (第三方代码 142-144 行), 改 config 无效;
        #    正确做法 = 构造时传 use_flash_attn=False (kwargs 透传到 __init__,
        #    其分支会把 attn_implementation 设为 'eager')。
        if isinstance(config.llm_config, dict):
            config.llm_config["attn_implementation"] = "eager"
        else:
            config.llm_config.attn_implementation = "eager"

        self._model = InternVLChatModel.from_pretrained(
            src, config=config, use_flash_attn=False, **load_kw)

        # 2) tokenizer + 特殊 token
        from internvl.train.constants import (
            IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
            QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
            REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN)
        tok_kw = {"trust_remote_code": True, "use_fast": False}
        if is_local:
            tok_kw["local_files_only"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(
            src, add_eos_token=False, **tok_kw)
        self._tokenizer.tokenizer_path = src
        self._tokenizer.model_max_length = _FINEVQ_DEFAULTS["max_seq_length"]

        token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                      QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                      REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
        num_new_tokens = self._tokenizer.add_tokens(token_list, special_tokens=True)
        self._img_context_token_id = self._tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN)
        self._model.img_context_token_id = self._img_context_token_id

        # 3) force_image_size 调整 + num_image_token
        patch_size = self._model.config.vision_config.patch_size
        fs = _FINEVQ_DEFAULTS["force_image_size"]
        if self._model.config.vision_config.image_size != fs:
            self._model.vision_model.resize_pos_embeddings(
                old_size=self._model.config.vision_config.image_size,
                new_size=fs, patch_size=patch_size)
            self._model.config.vision_config.image_size = fs
        self._model.config.force_image_size = fs
        self._model.num_image_token = int(
            (fs // patch_size) ** 2 *
            (_FINEVQ_DEFAULTS["down_sample_ratio"] ** 2))

        # 4) resize embeddings (新 token)
        if num_new_tokens > 0:
            self._model.language_model.resize_token_embeddings(len(self._tokenizer))
            out_emb = self._model.language_model.get_output_embeddings().weight.data
            avg = out_emb[:-num_new_tokens].mean(dim=0, keepdim=True)
            out_emb[-num_new_tokens:] = avg
            self._model.config.llm_config.vocab_size = len(self._tokenizer)
            self._model.language_model.config.vocab_size = len(self._tokenizer)

        self._model.language_model.config.use_cache = False

        # ── LoRA 合并 (FQ-2 NaN 根因修复) ──
        # 该环境 (peft + torch2.7.1 + bf16) 下 language_model 的 LoraLayer
        # 动态分支前向产生 NaN (诊断: 禁用 LoRA 后 NaN 全消; layer0.attention
        # 起全 32 层 100% NaN → nan_to_num 抹平 → score 恒定 0.009)。
        # merge_and_unload 把 LoRA 权重离线合并进基座 (数学等价), 之后推理
        # 走纯 Linear, 绕开 LoraLayer 的 NaN 分支, 语义保持不变。
        _lm = self._model.language_model
        if hasattr(_lm, "merge_and_unload") and callable(
                getattr(_lm, "merge_and_unload", None)):
            try:
                self._model.language_model = _lm.merge_and_unload()
                print("[FineVQ] ✓ LoRA 已合并 (merge_and_unload, 修复 LoraLayer NaN)")
            except Exception as e:
                print(f"[FineVQ] ⚠️ LoRA 合并失败(保留原样, FQ-2 可能仍异常): {e}")

        # ── 修复 slowfast_model 权重 no-op (meta device) ──
        # FineVQ 的 motion 分支依赖 pytorchvideo 的 slowfast_r50 预训练权重。
        # modeling_internvl_chat.py 里 `slowfast_r50(pretrained=True)` 会尝试联网下载
        # SLOWFAST_8x8_R50.pkl, 离线环境下载失败 → slowfast 参数落在 meta device。
        # 而 checkpoint 里其实已含完整的 slowfast_model.* 权重 (660 keys, 在
        # model-00004-of-00004.safetensors), 但 transformers 的 from_pretrained 对
        # meta 参数做 no-op copy, 导致这 660 个权重全部没加载 → motion 特征为垃圾,
        # mlpscore 回归头输出退化为固定值 (FQ-2 全 5 分, MOS 恒定 4.594)。
        # 修复: 从 checkpoint shard 单独读回 slowfast 权重, 用 assign=True 填充 meta 参数。
        self._reload_slowfast_weights(src)

        self._model = self._model.eval().to(self.device)
        self._is_loaded = True
        print(f"[FineVQ] ✓ 加载成功 ({src}, {load_kw['torch_dtype']}, device={self.device})")

    def _reload_slowfast_weights(self, src: str) -> None:
        """把 checkpoint 里的 slowfast_model 权重用 assign=True 填回 meta 参数。

        transformers from_pretrained 对 meta device 参数做 no-op copy (不改权重),
        导致 slowfast 分支用未初始化权重。此处从 safetensors shard 手动读回
        slowfast_model.* 并 assign 到模型, 修复 motion 特征 → FQ-2 分数固定问题。
        """
        try:
            import json
            from safetensors.torch import load_file as _load_safetensors

            idx_path = os.path.join(src, "model.safetensors.index.json")
            if not os.path.isfile(idx_path):
                # 无 index (单文件权重), 尝试直接加载
                st_path = os.path.join(src, "model.safetensors")
                if not os.path.isfile(st_path):
                    return
                shards = {st_path}
            else:
                with open(idx_path, "r") as f:
                    idx = json.load(f)
                wm = idx.get("weight_map", {})
                slow_keys = [k for k in wm if k.startswith("slowfast_model.")]
                if not slow_keys:
                    return
                shards = {os.path.join(src, wm[k]) for k in slow_keys}

            slowfast_module = getattr(self._model, "slowfast_model", None)
            if slowfast_module is None:
                return

            # 收集所有 slowfast_model.* 的权重 (可能跨多个 shard)
            state = {}
            for shard in shards:
                if not os.path.isfile(shard):
                    continue
                shard_state = _load_safetensors(shard)
                for k, v in shard_state.items():
                    if k.startswith("slowfast_model."):
                        state[k] = v

            if not state:
                return

            # 去掉 "slowfast_model." 前缀, 转成 slowfast 子模块的 state_dict
            sub_state = {k[len("slowfast_model."):]: v for k, v in state.items()}

            # 检查是否有 meta 参数 (no-op 的证据)
            has_meta = False
            try:
                p = next(slowfast_module.parameters())
                has_meta = getattr(p, "is_meta", False)
            except StopIteration:
                pass

            if has_meta:
                # meta 参数须用 assign=True 就地填充 (否则仍是 no-op)
                _res = slowfast_module.load_state_dict(sub_state, assign=True)
            else:
                # 正常参数, 常规加载
                _res = slowfast_module.load_state_dict(sub_state, strict=False)

            # ⚠️ strict=False 会静默丢弃不匹配的 key → 权重可能没装上。
            #    打印 missing/unexpected, 便于诊断 FQ-2 恒定输出。
            _miss = list(getattr(_res, "missing_keys", []) or [])
            _unexp = list(getattr(_res, "unexpected_keys", []) or [])
            print(f"[FineVQ] ✓ slowfast_model 权重已重新加载 "
                  f"({len(sub_state)} 项, meta={has_meta}, "
                  f"missing={len(_miss)}, unexpected={len(_unexp)})")
            if _miss:
                print(f"[FineVQ] ⚠️ missing_keys 示例: {_miss[:5]}")
            if _unexp:
                print(f"[FineVQ] ⚠️ unexpected_keys 示例: {_unexp[:5]}")
        except Exception as e:
            import traceback
            print(f"[FineVQ] ⚠️ slowfast_model 权重重载失败(忽略): {e}")
            traceback.print_exc()

    def to_cpu(self) -> None:
        if self._model is None:
            return
        try:
            if next(self._model.parameters()).device.type == "cuda":
                self._model.to("cpu")
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FineVQ] offload 到 CPU 失败(忽略): {e}")

    def to_cuda(self) -> None:
        if self._model is None:
            return
        try:
            if next(self._model.parameters()).device.type != "cuda":
                self._model.to(self.device)
        except Exception as e:
            print(f"[FineVQ] 搬回 GPU 失败(忽略): {e}")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._dataset_cls = None
        self._is_loaded = False
        torch.cuda.empty_cache()

    # ── 推理 ──
    @torch.no_grad()
    def score(self, frames: List[Image.Image],
              prompt: str = "How would you rate the overall quality of this video?",
              ) -> float:
        """对视频做质量/美学评分, 返回连续 MOS 分数 (约 1-5)。

        复用 FineVQ 的 LazySupervisedDataset.video_get_item 预处理管线。
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("FineVQ 未加载")
        if len(frames) < 2:
            raise ValueError("FineVQ 需要至少 2 帧视频")

        # 懒加载 dataset 类 (避免在 import 阶段触发 inference.py 的 dist 副作用,
        # 只 import 纯 dataset 类定义所在模块的依赖)
        if self._dataset_cls is None:
            from internvl.train.dataset import (
                build_transform, dynamic_preprocess)
            self._build_transform = build_transform
            self._dynamic_preprocess = dynamic_preprocess

        # 写临时视频, 用 decord 读取 (与官方 load_video 一致)
        tmpdir = tempfile.mkdtemp(prefix="finevq_")
        video_path = os.path.join(tmpdir, "input.mp4")
        try:
            self._save_frames_to_mp4(frames, video_path)
            item = self._build_item(video_path, prompt)
        except Exception as e:
            import traceback
            print(f"[FineVQ] _build_item 失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
        finally:
            try:
                os.remove(video_path)
                os.rmdir(tmpdir)
            except OSError:
                pass

        # ⚠️ _build_item 返回的张量都在 CPU (transform/tokenizer 输出), 模型在 CUDA,
        #    须统一 .to(device) + bf16, 否则 forward 里 index_select 报 device 不匹配。
        dev = self.device
        try:
            output = self._model(
                pixel_values2=item["pixel_values2"].to(dev).to(self._model.dtype),
                pixel_values=item["pixel_values"].to(dev).to(self._model.dtype),
                input_ids=item["input_ids"].to(dev),
                attention_mask=item["attention_mask"].to(dev),
                image_flags=item["image_flags"].to(dev),
                labels=item["labels"].to(dev),
            )
        except Exception as e:
            import traceback
            print(f"[FineVQ] forward 失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
        if not isinstance(output, dict) or "score1" not in output:
            raise RuntimeError(f"FineVQ 输出缺少 score1: {type(output)}")
        return float(output["score1"].item())

    def _build_item(self, video_path: str, prompt: str) -> dict:
        """复刻 LazySupervisedDataset.video_get_item 的输入构造 (单视频)。"""
        from decord import VideoReader, cpu
        from internvl.train.constants import (
            IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN)
        from internvl.train.dataset import (
            dynamic_preprocess, build_transform, preprocess_internlm)

        fs = _FINEVQ_DEFAULTS["force_image_size"]
        max_num = 1  # 对应官方 load_video(max_num=1)

        # 稀疏帧 (8) + 密集帧 (运动特征)
        image_list = self._load_video(video_path, num_segments=self.num_segments)
        image_list2 = self._load_video(video_path, dense=True)

        transform = build_transform(is_train=False, input_size=fs,
                                    normalize_type="imagenet")

        # 构造 prompt: Frame{i}:<image> + Motion Feature:<image>
        special_tokens = "\n".join(
            [f"Frame{i + 1}: <image>" for i in range(len(image_list))])
        special_tokens += "\nMotion Feature: <image>"
        question = special_tokens + "\n" + prompt

        pixel_values = torch.stack([transform(im) for im in image_list])
        pixel_values2 = torch.stack([transform(im) for im in image_list2])

        num_patches = pixel_values.size(0) + 1
        num_image_tokens = [self._model.num_image_token] * num_patches
        num_image_tokens[-1] = 1

        conversations = [{
            "from": "human", "value": question,
        }, {
            "from": "ai", "value": "The overall quality of the video is excellent.",
        }]
        ret = preprocess_internlm(
            _FINEVQ_DEFAULTS["conv_style"], [conversations],
            self._tokenizer, num_image_tokens,
            group_by_length=False, ds_name="finevq", num_image=num_patches)

        # ⚠️ 关键修复 (FQ-2 恒定输出根因): preprocess_internlm 把序列 pad 到
        #    max_seq_length(4096), 而官方推理 batch=1 经 collate 不 pad。
        #    模型 forward 写死取 last_hidden_states[:, -4, :] 算 score1 ——
        #    pad 序列下 -4 落在 pad token (恒定, 无内容信息) → 输出恒定 4.59375。
        #    按 attention_mask 截到真实长度, 使 -4 落在真实回答区 (与官方一致)。
        _L = int(ret["attention_mask"].sum().item())
        return {
            # ⚠️ 保留 batch 维 [1, L] (forward 里 get_input_embeddings 需要 [B, N] 输入
            #    才能出 [B, N, C]; 官方 inference.py 用 DataLoader batch=1 自动加 batch 维)。
            "input_ids": ret["input_ids"][:, :_L],
            "labels": ret["labels"][:, :_L],
            "attention_mask": ret["attention_mask"][:, :_L],
            "pixel_values": pixel_values,
            "pixel_values2": pixel_values2,
            "image_flags": torch.tensor([1] * (num_patches - 1), dtype=torch.long),
        }

    def _load_video(self, video_path: str, num_segments: int = 8,
                    dense: bool = False) -> List[Image.Image]:
        """复刻 FineVQ load_video/load_video2: decord 读帧 + dynamic_preprocess。"""
        from decord import VideoReader, cpu
        from internvl.train.dataset import dynamic_preprocess

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        if dense:
            n = 32 if len(vr) < 32 else len(vr)
        else:
            n = num_segments

        seg_size = float(max_frame) / n
        indices = [int(seg_size / 2 + round(seg_size * i)) for i in range(n)]
        images = []
        for idx in indices:
            img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
            patches = dynamic_preprocess(
                img, image_size=_FINEVQ_DEFAULTS["force_image_size"],
                use_thumbnail=True, max_num=1)
            images.append(patches[0])
        return images

    @staticmethod
    def _save_frames_to_mp4(frames: List[Image.Image], out_path: str) -> None:
        import imageio
        writer = imageio.get_writer(out_path, fps=8, codec="libx264")
        for f in frames:
            writer.append_data(np.array(f.convert("RGB")))
        writer.close()
