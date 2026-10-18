"""
Qwen 视觉语言模型推理封装（真实推理，无 stub）

使用 Qwen3.8-27B 的原生多模态接口。

官方 API (已查证, 需 transformers>=4.57):
  # 通用: AutoModelForImageTextToText 覆盖 Qwen3_5ForConditionalGeneration。
  from transformers import AutoModelForImageTextToText, AutoProcessor
  model = AutoModelForImageTextToText.from_pretrained(id, dtype="auto")
  processor = AutoProcessor.from_pretrained(id)
  messages = [{"role":"user","content":[{"type":"image","image":pil},{"type":"text","text":prompt}]}]
  inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt").to(device)
  out = model.generate(**inputs, max_new_tokens=512)

设计: 加载/推理失败 → 抛异常, evaluator 记 status=error。
      chat_json() 提供结构化 JSON 输出解析(benchmark 可复现)。
"""
import json
import re
from typing import Optional, Union
from PIL import Image
import torch
from .backend_utils import bitsandbytes_available, get_safe_dtype


class VLMWrapper:
    """Qwen3.8-27B 统一推理接口（真实推理）"""

    def __init__(self, model_name: str,
                 device: Optional[str] = None,
                 load_in_4bit: bool = False,
                 load_in_8bit: bool = False,
                 max_new_tokens: int = 512,
                 **kwargs):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def metadata(self) -> dict:
        """模型 provenance (与 BaseSpecialistModel 统一接口对齐)。"""
        return {
            "class": type(self).__name__,
            "model_id": self.model_name,
            "device": self.device,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "is_loaded": self._is_loaded,
        }

    def load(self, **kwargs) -> None:
        """Load the configured Qwen3.8-27B VLM; raise on failure."""
        import os as _os
        from transformers import AutoProcessor
        from .base import resolve_local_model

        cache_dir = _os.environ.get("HF_HOME") or _os.environ.get("HUGGINGFACE_HUB_CACHE")
        # 离线优先: 把 model_name 解析成本地权重目录 (魔搭 flat / HF snapshot)
        src = resolve_local_model(self.model_name, cache_dir)
        is_local = src != self.model_name
        load_kwargs = {}
        if cache_dir and not is_local:
            load_kwargs["cache_dir"] = cache_dir
        if is_local:
            load_kwargs["local_files_only"] = True
            print(f"[VLM] 使用本地权重目录: {src}")

        if (self.load_in_4bit or self.load_in_8bit) and not bitsandbytes_available():
            print("[VLM] bitsandbytes 不可用，禁用量化，改用半精度")
            self.load_in_4bit = False
            self.load_in_8bit = False

        if self.load_in_4bit:
            # 用 BitsAndBytesConfig 显式指定量化 (transformers 5.x 对 load_in_4bit
            # 旧参数透传到 __init__ 会报 unexpected keyword, 需走 quantization_config)。
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=get_safe_dtype(),
            )
            load_kwargs["device_map"] = "auto"
        elif self.load_in_8bit:
            # 8bit 量化 (LLM.int8): 精度接近 fp16, 显存约 fp16 的 ~1/2。
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            # ⚠️ 不用 device_map="auto": accelerate 会给模型挂 hook/锁定设备,
            #    导致后续无法 .to("cpu") 做按层 offload (A100-40GB 装不下全部模型,
            #    SpatialTracker 等推理时需把闲置的 8B VLM 临时挪到 CPU 腾显存)。
            #    改为显式加载到 CPU 再手动搬到目标设备, 保留自由搬迁能力。
            load_kwargs["dtype"] = "auto"

        proc_kwargs = {k: v for k, v in load_kwargs.items() if k in ("cache_dir", "local_files_only")}
        self.processor = AutoProcessor.from_pretrained(src, **proc_kwargs)
        # Try the generic image-text loader first, then the multimodal fallback
        # for transformers versions that register the model under that class.
        from transformers import AutoModelForImageTextToText
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(src, **load_kwargs)
        except Exception:
            from transformers import AutoModelForMultimodalLM
            self.model = AutoModelForMultimodalLM.from_pretrained(src, **load_kwargs)
        if not self.load_in_4bit and not self.load_in_8bit:
            self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        print(f"[VLM] ✓ {self.model_name} 加载成功 (4bit={self.load_in_4bit})")

    def to_cpu(self) -> None:
        """把模型挪到 CPU 释放显存。

        注意: 8bit 量化模型 (bitsandbytes LLM.int8 + device_map="auto")
        无法用 .to("cpu") 跨设备移动 (需 bitsandbytes>=0.48.0)，且 accelerator
        挂钩会锁设备。这种情况下直接 unload() 彻底释放权重归还显存；
        之后由 to_cuda() 检测到未加载时重新 load() 恢复。
        """
        if self.model is None:
            return
        if self.load_in_8bit:
            # 8bit 模型无法跨设备移动 → 彻底卸载释放显存
            self.unload()
            return
        if self.load_in_4bit:
            # 4bit 模型同样无法 .to("cpu") 移动 → 卸载
            self.unload()
            return
        try:
            if next(self.model.parameters()).device.type == "cuda":
                self.model.to("cpu")
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[VLM] offload 到 CPU 失败(忽略): {e}")

    def to_cuda(self) -> None:
        """把模型搬回 GPU (若不在 GPU 上)。

        若之前因 8bit/4bit offload 被 unload() 释放，则重新 load() 恢复。
        """
        if self.model is None or not self._is_loaded:
            # 之前被 unload() 释放 (8bit/4bit offload) → 重新加载
            try:
                self.load()
                print("[VLM] 重新加载完成 (恢复 offload 前状态)")
            except Exception as e:
                print(f"[VLM] 重新加载失败(忽略): {e}")
            return
        if self.load_in_4bit or self.load_in_8bit:
            # 量化模型已由 load() 用 device_map 定位, 无需手动 .to()
            return
        try:
            if next(self.model.parameters()).device.type != "cuda":
                self.model.to(self.device)
        except Exception as e:
            print(f"[VLM] 搬回 GPU 失败(忽略): {e}")

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self._is_loaded = False
        torch.cuda.empty_cache()

    @torch.no_grad()
    def chat(self, images: Union[Image.Image, list],
             prompt: str, max_new_tokens: int = 512,
             temperature: float = 0.0) -> str:
        """多模态对话推理。失败抛 RuntimeError。"""
        if not self._is_loaded or self.model is None:
            raise RuntimeError("VLM 未加载")
        if isinstance(images, Image.Image):
            images = [images]

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        model_device = next(self.model.parameters()).device
        # Disable reasoning when supported so structured JSON remains parseable.
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                enable_thinking=False,
            ).to(model_device)
        except TypeError:
            # Older processors may not accept enable_thinking.
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model_device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        gen = self.model.generate(**inputs, **generation_kwargs)
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """从 VLM 原始输出中稳健提取一个 JSON 对象。

        处理推理模型的 reasoning 前缀、markdown 代码块、
        前后缀说明文字等污染。策略:
        1. 先剥掉 markdown 代码块围栏 (```json ... ```)。
        2. 优先从**最后一个** '{' 位置尝试 raw_decode (答案 JSON 通常在思考过程之后)，
           失败再正向扫描所有 '{' 位置。
        3. 容错: 去掉尾随逗号后再试。
        """
        if not text:
            return None
        # 剥掉 ```json ... ``` 或 ``` ... ``` 围栏 (优先取最长的匹配)
        fences = list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
        candidate = fences[-1].group(1) if fences else text

        decoder = json.JSONDecoder()
        braces = [m.start() for m in re.finditer(r"\{", candidate)]
        if not braces:
            return None
        # 优先从最后一个 '{' 往前试 (答案 JSON 通常在末尾)
        positions = sorted(set(braces), reverse=True)
        for idx in positions:
            for try_clean in (False, True):
                s = candidate[idx:]
                if try_clean:
                    s = re.sub(r",\s*([}\]])", r"\1", s)
                try:
                    obj, _ = decoder.raw_decode(s)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
        return None

    @torch.no_grad()
    def chat_json(self, images, prompt: str,
                  max_new_tokens: int = 512) -> dict:
        """要求模型输出 JSON 并解析。用于结构化评分(可复现)。

        prompt 内需明确要求 JSON 格式。解析失败返回 {"_parse_error": True}，
        由调用方处理。
        """
        # 追加 JSON 约束: 显式禁止思考过程, 只输出裸 JSON
        json_prompt = (prompt + '\n\n只输出一个 JSON 对象，不要任何额外文字、'
                       '解释或思考过程。直接以 "{" 开头输出。')
        text = self.chat(images, json_prompt, max_new_tokens=max_new_tokens)
        obj = self._extract_json(text)
        if obj is not None:
            return obj
        return {"_raw": text[:500], "_parse_error": True}
