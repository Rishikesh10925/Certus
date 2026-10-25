"""Generative VLM wrapper for S2 (roadmap Section 3.3). Scores how likely
the model is to say a fixed assertion about an image, teacher-forced, rather
than parsing free-form generations -- the same image always gets the same
number, and the score is comparable across images because the assertion
text never changes.

Default model: Qwen2-VL-2B-Instruct in bf16 on the GPU (~4.5 GB VRAM). There
is no fine-tuned Certus VLM (Week 4 logbook), so this is off the shelf.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

DEFAULT_GENERATIVE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MAX_PIXELS = 448 * 448  # caps visual tokens per image (~256) for speed


class GenerativeVLM:
    def __init__(
        self, model_name: str = DEFAULT_GENERATIVE_MODEL, device: str | None = None
    ) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=MAX_PIXELS)
        self.processor.tokenizer.padding_side = "left"
        self.model = (
            Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def assertion_logprob(
        self, image_paths: list[Path], question: str, assertion: str, batch_size: int = 8
    ) -> npt.NDArray[np.float64]:
        """Mean per-token log-probability of `assertion` as the model's
        answer to `question` about each image."""
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        tok = self.processor.tokenizer
        assertion_ids = tok(assertion, add_special_tokens=False).input_ids
        n = len(assertion_ids)

        out: list[float] = []
        for i in range(0, len(image_paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in image_paths[i : i + batch_size]]
            inputs = self.processor(
                text=[prompt + assertion] * len(images),
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            ids = inputs["input_ids"]
            # The assertion must tokenize identically inside the full prompt,
            # or we'd be scoring the wrong tokens.
            if ids[0, -n:].tolist() != assertion_ids:
                raise RuntimeError("assertion tokens don't line up with the end of the prompt")
            logits = self.model(**inputs).logits[:, -n - 1 : -1].float()
            logp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, -n:, None]).squeeze(-1)
            out.extend(logp.mean(dim=1).cpu().tolist())
        return np.array(out, dtype=np.float64)

    @torch.no_grad()
    def generate(
        self, image_paths: list[Path], question: str, max_new_tokens: int = 8, batch_size: int = 8
    ) -> list[str]:
        """Greedy answers to `question` about each image (for S3)."""
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        out: list[str] = []
        for i in range(0, len(image_paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in image_paths[i : i + batch_size]]
            inputs = self.processor(
                text=[prompt] * len(images), images=images, return_tensors="pt", padding=True
            ).to(self.device)
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new = gen[:, inputs["input_ids"].shape[1] :]
            answers: list[str] = self.processor.batch_decode(new, skip_special_tokens=True)
            out.extend(answers)
        return out
