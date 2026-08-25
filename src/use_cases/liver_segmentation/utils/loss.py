"""Loss functions for liver segmentation with morphological consistency."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_fn(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5):
    """Soft Dice loss (operates on logits)."""
    p = pred.sigmoid()
    inter = (p * target).sum(dim=(2, 3, 4))
    union = p.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    return (1.0 - (2 * inter + smooth) / (union + smooth)).mean()


def morph_consistency_loss(pred_vol_ratios: torch.Tensor, gt_masks: torch.Tensor):
    """MSE between predicted and ground-truth volume ratios."""
    gt_volumes = gt_masks.sum(dim=(2, 3, 4))
    gt_total = gt_volumes.sum(dim=1, keepdim=True).clamp(min=1e-6)
    gt_ratios = gt_volumes / gt_total
    return F.mse_loss(pred_vol_ratios, gt_ratios)


def compute_loss(
    seg_logits: torch.Tensor,
    vol_ratios: torch.Tensor,
    masks: torch.Tensor,
    morph_coeff: float,
) -> tuple[torch.Tensor, float, float]:
    """Combined segmentation + morphological consistency loss.

    Returns (total_loss, seg_loss_val, morph_loss_val).
    """
    bce_seg = nn.BCEWithLogitsLoss()(seg_logits, masks)
    dl = dice_loss_fn(seg_logits, masks)
    seg_loss = bce_seg + dl

    m_loss = morph_consistency_loss(vol_ratios, masks)

    total = seg_loss + morph_coeff * m_loss
    return total, seg_loss.item(), m_loss.item()
