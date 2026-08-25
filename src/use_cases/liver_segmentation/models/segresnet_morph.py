"""
SegResNet with Morphological Descriptor (FedMorph model)
========================================================

Architecture:
  MONAI SegResNet (init_filters=8, ~1M params)
  + MorphologicalDescriptor: differentiable vol_ratios / centroids / compactness
"""

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class MorphologicalDescriptor(nn.Module):
    """Differentiable morphological features from predicted segment masks.

    Outputs (morph_vector, vol_ratios):
      morph_vector = [vol_ratios | centroids_flat | compactness]  dim = 5*C
      vol_ratios   = per-segment volume ratios                    dim = C
    """

    def __init__(self, num_segments: int = 9):
        super().__init__()
        self.num_segments = num_segments

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, seg_logits):
        seg_logits = seg_logits.float()
        seg_probs = torch.sigmoid(seg_logits)
        B, C, D, H, W = seg_probs.shape
        device = seg_probs.device

        volumes = seg_probs.sum(dim=(2, 3, 4))
        total_vol = volumes.sum(dim=1, keepdim=True).clamp(min=1.0)
        vol_ratios = volumes / total_vol

        grid_d = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1)
        grid_h = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
        grid_w = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)
        vol_safe = volumes.clamp(min=1.0)
        cent_d = (seg_probs * grid_d).sum(dim=(2, 3, 4)) / vol_safe
        cent_h = (seg_probs * grid_h).sum(dim=(2, 3, 4)) / vol_safe
        cent_w = (seg_probs * grid_w).sum(dim=(2, 3, 4)) / vol_safe
        centroids = torch.stack([cent_d, cent_h, cent_w], dim=2)

        dx = (seg_probs[:, :, 1:] - seg_probs[:, :, :-1]).abs().sum(dim=(2, 3, 4))
        dy = (seg_probs[:, :, :, 1:] - seg_probs[:, :, :, :-1]).abs().sum(
            dim=(2, 3, 4)
        )
        dz = (seg_probs[:, :, :, :, 1:] - seg_probs[:, :, :, :, :-1]).abs().sum(
            dim=(2, 3, 4)
        )
        surface = dx + dy + dz
        compactness = surface / vol_safe.pow(2.0 / 3.0).clamp(min=1e-3)
        comp_max = compactness.max(dim=1, keepdim=True)[0].clamp(min=1e-3)
        compactness = compactness / comp_max

        morph = torch.cat([vol_ratios, centroids.flatten(1), compactness], dim=1)
        morph = torch.nan_to_num(morph, nan=0.0, posinf=1.0, neginf=0.0)
        vol_ratios = torch.nan_to_num(vol_ratios, nan=0.0)
        return morph, vol_ratios


class SegResNetMorph(SegResNet):
    """SegResNet + MorphologicalDescriptor.

    Returns: (seg_logits, morph_feats, vol_ratios)
    """

    def __init__(self, num_segments: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.morph_desc = MorphologicalDescriptor(num_segments)

    def forward(self, x):
        x_enc, down_x = self.encode(x)
        down_x.reverse()
        seg_logits = self.decode(x_enc, down_x)
        morph_feats, vol_ratios = self.morph_desc(seg_logits)
        return seg_logits, morph_feats, vol_ratios


def build_model(config: dict, device: torch.device) -> SegResNetMorph:
    """Build SegResNetMorph from config dict."""
    num_classes = config["num_classes"]
    init_filters = config["init_filters"]
    blocks_down = tuple(config["blocks_down"])
    blocks_up = tuple(config["blocks_up"])

    model = SegResNetMorph(
        num_segments=num_classes,
        spatial_dims=3,
        init_filters=init_filters,
        in_channels=1,
        out_channels=num_classes,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        norm=("GROUP", {"num_groups": min(8, init_filters)}),
        dropout_prob=None,
        upsample_mode="nontrainable",
    ).to(device)

    n_p = sum(p.numel() for p in model.parameters())
    print(
        f"  SegResNet params: {n_p:,} "
        f"(init_filters={init_filters}, blocks_down={blocks_down})"
    )
    return model
