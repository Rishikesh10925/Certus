"""Open-vocabulary object detector for S4 (roadmap Section 3.3). There is
no trained CBRI detector (Week 4 logbook), so S4 uses OWLv2 zero-shot:
text queries name the defect classes, and the detector scores every
candidate box against every query.

Two speed fixes, both checked against the stock pipeline (same scores to
~3 dp, see tests and the Week 6 logbook):
- fp16 on the GPU. OWLv2 runs at 960x960; fp32 at batch 8 needs 11 GB,
  which on an 8 GB card spills to system memory (~11 s/image).
- Image preprocessing on the GPU. The stock Owlv2ImageProcessor resizes
  with scipy.ndimage on the CPU, which left the GPU idle ~90% of the time.
  `_pixel_values` reproduces its steps -- scale to [0, 1], pad bottom/right
  to a square with 0.5, bilinear resize to 960 (anti-aliased only when
  shrinking), clip, normalise -- with torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_DETECTOR = "google/owlv2-base-patch16-ensemble"


class OpenVocabDetector:
    def __init__(self, model_name: str = DEFAULT_DETECTOR, device: str | None = None) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        ip = self.processor.image_processor
        self.size = int(ip.size["height"])
        self.mean = torch.tensor(ip.image_mean, device=self.device).view(3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=self.device).view(3, 1, 1)

    def _pixel_values(self, images: list[Image.Image]) -> torch.Tensor:
        out = []
        for img in images:
            t = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).to(self.device)
            t = t.permute(2, 0, 1).float() / 255.0
            h, w = t.shape[1:]
            side = max(h, w)
            t = F.pad(t, (0, side - w, 0, side - h), value=0.5)
            t = F.interpolate(
                t[None],
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
                antialias=side > self.size,
            )[0].clamp(0.0, 1.0)
            out.append((t - self.mean) / self.std)
        return torch.stack(out).to(self.dtype)

    @torch.no_grad()
    def max_box_confidence(
        self, image_paths: list[Path], queries: list[str], batch_size: int = 4
    ) -> npt.NDArray[np.float64]:
        """Per image: the highest sigmoid confidence over all boxes and all
        queries -- how sure the detector is that at least one defect box is
        there."""
        text = self.processor(text=[queries], return_tensors="pt").to(self.device)
        out: list[float] = []
        for i in range(0, len(image_paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in image_paths[i : i + batch_size]]
            n = len(images)
            logits = self.model(
                input_ids=text["input_ids"].repeat(n, 1),
                attention_mask=text["attention_mask"].repeat(n, 1),
                pixel_values=self._pixel_values(images),
            ).logits.float()  # (batch, boxes, queries)
            out.extend(torch.sigmoid(logits).amax(dim=(1, 2)).cpu().tolist())
        return np.array(out, dtype=np.float64)
