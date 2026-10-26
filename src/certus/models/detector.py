"""Open-vocabulary object detector for S4 (roadmap Section 3.3). There is
no trained CBRI detector (Week 4 logbook), so S4 uses OWLv2 zero-shot:
text queries name the defect classes, and the detector scores every
candidate box against every query.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

DEFAULT_DETECTOR = "google/owlv2-base-patch16-ensemble"


class OpenVocabDetector:
    def __init__(self, model_name: str = DEFAULT_DETECTOR, device: str | None = None) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on the GPU: OWLv2 runs at 960x960, and fp32 at batch 8 needs
        # 11 GB -- on an 8 GB card that spills to system memory and runs at
        # ~11 s/image. fp16 at batch 2 is ~0.3 s/image, same scores to 3 dp.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def max_box_confidence(
        self, image_paths: list[Path], queries: list[str], batch_size: int = 2
    ) -> npt.NDArray[np.float64]:
        """Per image: the highest sigmoid confidence over all boxes and all
        queries -- how sure the detector is that at least one defect box is
        there."""
        out: list[float] = []
        for i in range(0, len(image_paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in image_paths[i : i + batch_size]]
            inputs = self.processor(
                text=[queries] * len(images), images=images, return_tensors="pt"
            ).to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            logits = self.model(**inputs).logits.float()  # (batch, boxes, queries)
            out.extend(torch.sigmoid(logits).amax(dim=(1, 2)).float().cpu().tolist())
        return np.array(out, dtype=np.float64)
