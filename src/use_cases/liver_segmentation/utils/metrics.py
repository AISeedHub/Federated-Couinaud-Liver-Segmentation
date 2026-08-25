"""Evaluation metrics for liver segmentation."""

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


@torch.no_grad()
def compute_per_segment_dice(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 9,
) -> np.ndarray:
    """Per-segment Dice scores used for FedMorph quality-weighted aggregation."""
    model.eval()
    dm = DiceMetric(include_background=True, reduction="mean")
    totals = np.zeros(num_classes)
    counts = np.zeros(num_classes)

    use_amp = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]
        with torch.amp.autocast("cuda", enabled=use_amp):
            seg_logits, _, _ = model(images)
        seg_pred = (seg_logits.sigmoid().cpu() > 0.5).float()
        B = images.shape[0]
        for b in range(B):
            for c in range(num_classes):
                gt = masks[b, c]
                if gt.sum() == 0:
                    continue
                dm.reset()
                dm(
                    y_pred=seg_pred[b, c].unsqueeze(0).unsqueeze(0),
                    y=gt.unsqueeze(0).unsqueeze(0),
                )
                totals[c] += dm.aggregate().item()
                counts[c] += 1

    for c in range(num_classes):
        if counts[c] > 0:
            totals[c] /= counts[c]
    return totals


@torch.no_grad()
def compute_morph_diversity(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> float:
    """Volume-ratio variance used for FedMorph diversity-weighted aggregation."""
    model.eval()
    all_vr: list[torch.Tensor] = []
    use_amp = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, _, vol_ratios = model(images)
        all_vr.append(vol_ratios.cpu())
    if not all_vr:
        return 0.0
    vr = torch.cat(all_vr, dim=0)
    return float(vr.var(dim=0).mean().item())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 9,
) -> tuple:
    """Full evaluation: per-segment Dice/HD95 and VR error.

    Returns (dice_per_class, hd95_per_class, mean_vr_err).
    """
    model.eval()
    dm = DiceMetric(include_background=True, reduction="mean")
    hm = HausdorffDistanceMetric(
        include_background=True, percentile=95.0, reduction="mean"
    )

    pcd: list[list[float]] = [[] for _ in range(num_classes)]
    phd: list[list[float]] = [[] for _ in range(num_classes)]
    vr_errors: list[torch.Tensor] = []

    use_amp = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]

        with torch.amp.autocast("cuda", enabled=use_amp):
            seg_logits, _morph, vol_ratios = model(images)

        seg_pred = (seg_logits.sigmoid().cpu() > 0.5).float()
        pred_vr = vol_ratios.cpu()
        gt_vol = masks.sum(dim=(2, 3, 4))
        gt_total = gt_vol.sum(dim=1, keepdim=True).clamp(min=1e-6)
        gt_vr = gt_vol / gt_total
        vr_errors.append((pred_vr - gt_vr).abs().mean(dim=1))

        B = images.shape[0]
        for b in range(B):
            for c in range(num_classes):
                gt = masks[b, c]
                if gt.sum() == 0:
                    continue
                pred = seg_pred[b, c]
                g5 = gt.unsqueeze(0).unsqueeze(0)
                p5 = pred.unsqueeze(0).unsqueeze(0)
                dm.reset()
                dm(y_pred=p5, y=g5)
                pcd[c].append(dm.aggregate().item())
                try:
                    hm.reset()
                    hm(y_pred=p5, y=g5)
                    phd[c].append(hm.aggregate().item())
                except Exception:
                    pass

    dv = torch.full((num_classes,), float("nan"))
    hv = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        if pcd[c]:
            dv[c] = np.nanmean(pcd[c])
        if phd[c]:
            hv[c] = np.nanmean(phd[c])

    mean_vr_err = float("nan")
    if vr_errors:
        mean_vr_err = float(torch.cat(vr_errors).mean().item())

    return dv, hv, mean_vr_err
