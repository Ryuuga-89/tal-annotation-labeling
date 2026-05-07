"""Local multimodal VLM inference via transformers (Gemma/Qwen compatible)."""
from __future__ import annotations

import gc
import re
from typing import Any

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


class QwenLocalVLM:
    """Load once, run ``describe`` with a list of PIL images and a text instruction."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda:0",
        torch_dtype: str | torch.dtype = "auto",
        trust_remote_code: bool = True,
        attn_implementation: str | None = "sdpa",
    ) -> None:
        # Inspect config to branch on model family (Gemma4-E4B, etc.)
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model_type = getattr(cfg, "model_type", "")
        if isinstance(model_type, str) and "qwen2_5_vl" in model_type:
            raise ValueError(
                f"model_id={model_id!r} は Qwen2.5-VL 系 (model_type={model_type!r}) で、"
                "現在のパイプラインでは未対応です。Gemma4-E4B などの Gemma 系 VLM を指定してください。"
            )

        td: str | torch.dtype = torch_dtype
        if isinstance(td, str) and td not in ("auto",):
            td = getattr(torch, td)

        self._device_str = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        load_kw: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": td,
            "device_map": {"": device},
        }
        if attn_implementation:
            load_kw["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        except TypeError:
            load_kw.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        self.model.eval()

    @torch.inference_mode()
    def describe(
        self,
        images: list[Image.Image],
        instruction: str,
        *,
        max_new_tokens: int = 256,
    ) -> str:
        if not images:
            return ""

        # Chat template path (works for Gemma multimodal chat templates).
        try:
            messages = [{
                "role": "user",
                "content": (
                    [{"type": "image", "image": im} for im in images]
                    + [{"type": "text", "text": instruction}]
                ),
            }]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[prompt],
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
        except Exception:
            # Fallback for processors that do not support chat template.
            inputs = self.processor(
                text=[instruction],
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

        generated = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            gen_tokens = generated[0, input_ids.shape[1]:]
            out = self.processor.decode(gen_tokens, skip_special_tokens=True)
            return out.strip()
        out = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return out.strip()

    @torch.inference_mode()
    def describe_groups(
        self,
        image_groups: list[list[Image.Image]],
        instruction: str,
        *,
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Run one-shot inference for multiple frame groups.

        Output format expected from the model:
            G1: ...
            G2: ...
            ...
        """
        n = len(image_groups)
        if n == 0:
            return []

        content: list[dict[str, Any]] = []
        content.append(
            {
                "type": "text",
                "text": (
                    "以下に複数グループの画像列を与える。各グループを短い日本語1文で説明せよ。"
                    "必ず次の形式のみで出力すること: "
                    + " ".join([f"G{i+1}: <説明>" for i in range(n)])
                    + "。時刻情報は書かない。"
                ),
            }
        )
        for i, imgs in enumerate(image_groups, start=1):
            content.append({"type": "text", "text": f"グループ{i}:"})
            for im in imgs:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": instruction})

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt],
            images=[im for grp in image_groups for im in grp],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        generated = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            gen_tokens = generated[0, input_ids.shape[1]:]
            out = self.processor.decode(gen_tokens, skip_special_tokens=True).strip()
        else:
            out = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # Parse "Gk: ..." lines.
        parsed = [""] * n
        for line in out.splitlines():
            m = re.match(r"^\s*G(\d+)\s*[:：]\s*(.+?)\s*$", line)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                parsed[idx] = m.group(2).strip()
        return parsed

    @torch.inference_mode()
    def describe_batch(
        self,
        image_groups: list[list[Image.Image]],
        instructions: list[str],
        *,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Independent batched inference: one sample per segment."""
        if not image_groups:
            return []
        if len(image_groups) != len(instructions):
            raise ValueError("image_groups and instructions must have same length")

        messages = []
        for imgs, inst in zip(image_groups, instructions):
            content: list[dict[str, Any]] = []
            for im in imgs:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": inst})
            messages.append([{"role": "user", "content": content}])

        prompts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        # Preferred batched path.
        try:
            inputs = self.processor(
                text=prompts,
                images=image_groups,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            generated = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
            input_ids = inputs.get("input_ids", None)
            if input_ids is not None:
                outs = []
                for i in range(generated.shape[0]):
                    gen_tokens = generated[i, input_ids.shape[1]:]
                    outs.append(self.processor.decode(gen_tokens, skip_special_tokens=True).strip())
                return outs
            return [x.strip() for x in self.processor.batch_decode(generated, skip_special_tokens=True)]
        except Exception:
            # Fallback: keep correctness if processor/model does not support this batch shape.
            outs = []
            for imgs, inst in zip(image_groups, instructions):
                outs.append(self.describe(imgs, inst, max_new_tokens=max_new_tokens))
            return outs

    def unload(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
