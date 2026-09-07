"""손실·지표 — 10클래스 소프트맥스(배경+9분절), 8분절 라벨은 4a/4b 확률 합산으로 부분 감독.

merge_4ab(logits→probs): 10클래스 확률에서 4a(4)+4b(5)를 합쳐 9클래스 확률로 만든다.
8분절 케이스의 라벨은 공용 전처리 규약상 ch4(=S4 전체)에 들어 있으므로, 9클래스 인덱스
[0,1,2,3, 4(S4), 5(S5), 6(S6), 7(S7), 8(S8)] 로 매핑해 CE+Dice를 계산한다.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

NC = 10
# 10클래스 라벨 → 8분절(9클래스) 라벨: 4a(4),4b(5)→4, S5(6)→5, S6(7)→6, S7(8)→7, S8(9)→8
_MAP10TO9 = torch.tensor([0, 1, 2, 3, 4, 4, 5, 6, 7, 8])


def probs_9(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(1)
    return torch.cat([p[:, :4], p[:, 4:6].sum(1, keepdim=True), p[:, 6:]], dim=1)


def soft_dice(probs: torch.Tensor, onehot: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dims = (0, 2, 3, 4)
    inter = (probs * onehot).sum(dims); denom = probs.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0                      # 배치에 없는 클래스는 제외
    d = (2 * inter + eps) / (denom + eps)
    return 1 - d[1:][present[1:]].mean() if present[1:].any() else probs.sum() * 0


def seg_loss(logits: torch.Tensor, label: torch.Tensor, seg8: torch.Tensor, w_ce: float = 1.0, w_dice: float = 1.0):
    """logits (B,10,D,H,W), label (B,D,H,W) long, seg8 (B,) bool — 샘플별로 9/10클래스 감독."""
    total = logits.new_zeros(()); n = 0
    for b in range(logits.shape[0]):
        lg, lb = logits[b:b + 1], label[b:b + 1]
        if bool(seg8[b]):
            p = probs_9(lg); lb9 = _MAP10TO9.to(lb.device)[lb]
            ce = F.nll_loss(torch.log(p.clamp_min(1e-7)), lb9)
            oh = F.one_hot(lb9, 9).permute(0, 4, 1, 2, 3).float()
        else:
            p = lg.softmax(1); ce = F.cross_entropy(lg, lb)
            oh = F.one_hot(lb, NC).permute(0, 4, 1, 2, 3).float()
        total = total + w_ce * ce + w_dice * soft_dice(p, oh); n += 1
    return total / max(n, 1)


@torch.no_grad()
def dice_per_class(pred: torch.Tensor, label: torch.Tensor, nc: int = NC) -> torch.Tensor:
    """pred/label (D,H,W) 정수 라벨맵 → (nc,) Dice, GT·예측 모두 없는 클래스는 nan."""
    out = torch.full((nc,), float("nan"))
    for c in range(nc):
        p = pred == c; g = label == c; s = p.sum() + g.sum()
        if g.sum() > 0 or p.sum() > 0: out[c] = (2 * (p & g).sum() / s).float()
    return out


def merge_pred_9(pred: torch.Tensor) -> torch.Tensor:
    """10클래스 예측 라벨맵 → 8분절 평가용 9클래스 라벨맵(4a/4b→4)."""
    return _MAP10TO9.to(pred.device)[pred.long()]
