"""Frozen VLM backbone wrapper (LoRA weights, eval mode, no gradient) used
for batch inference over images to produce pooled visual features and/or
generated defect assertions. This is the one and only place inference on
the underlying vision-language model happens -- every nonconformity score
reads from here rather than re-running the model. See roadmap Section 3.1.

There is no fine-tuned Certus VLM checkpoint in this repo yet -- the
default model is a frozen, off-the-shelf CLIP vision encoder, used as a
stand-in wherever "frozen VLM features" are needed before the real model is
wired in (e.g. the Week 2 shift diagnostic). Swap `DEFAULT_MODEL_NAME` for
the fine-tuned checkpoint once it exists; nothing downstream should need to
change, since every caller goes through `FrozenVLM.image_features`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


class FrozenVLM:
    """Frozen CLIP encoder. No gradients, eval mode, GPU when available."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def image_features(
        self, image_paths: list[Path], batch_size: int = 32
    ) -> npt.NDArray[np.float32]:
        """L2-normalised pooled image embeddings, shape (len(image_paths), d), in input order."""
        chunks: list[npt.NDArray[np.float32]] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            chunks.append(self.pil_features([Image.open(p).convert("RGB") for p in batch_paths]))
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def pil_features(self, images: list[Image.Image]) -> npt.NDArray[np.float32]:
        """`image_features` for one batch of already-loaded images."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        out: npt.NDArray[np.float32] = features.cpu().numpy()
        return out

    @torch.no_grad()
    def text_features(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """L2-normalised text embeddings, shape (len(texts), d), same space
        as `image_features` -- for zero-shot scoring."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        out: npt.NDArray[np.float32] = features.cpu().numpy()
        return out
