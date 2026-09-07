"""추론 — 512 원본 해상도에서 96×256×256 슬라이딩 윈도우, z는 32의 배수로 zero-pad 후 잘라냄."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from .data import normalize, PATCH


def _ceil32(n: int) -> int: return int(np.ceil(n / 32) * 32)


@torch.no_grad()
def predict_volume(model: torch.nn.Module, image_u8: np.ndarray, device, patch=PATCH, overlap: float = 0.5,
                   amp: bool = True, sw_batch: int = 1) -> tuple[np.ndarray, torch.Tensor]:
    """image_u8 (D,H,W) → (label map uint8 (D,H,W), softmax probs (C,D,H,W) on CPU float16)."""
    model.eval()
    D, H, W = image_u8.shape; Dp = max(_ceil32(D), patch[0])
    x = torch.from_numpy(normalize(image_u8))[None, None].to(device)
    if Dp != D: x = F.pad(x, (0, 0, 0, 0, 0, Dp - D))

    def _net(t):
        o = model(t); return o[0] if isinstance(o, (list, tuple)) else o
    with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
        logits = sliding_window_inference(x, roi_size=patch, sw_batch_size=sw_batch, predictor=_net, overlap=overlap, mode="gaussian")
    probs = logits[0, :, :D].float().softmax(0)
    return probs.argmax(0).to(torch.uint8).cpu().numpy(), probs.half().cpu()
