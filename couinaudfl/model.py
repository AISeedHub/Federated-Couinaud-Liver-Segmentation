"""모델 — TotalSegmentator(nnU-Net v2, PlainConvUNet 3d_fullres)와 동일 구조 + 퀴노 10클래스 헤드.

TS task 291(total, CT) / 730(total_mr) plans.json 기준 (2026-09-02 확인):
  PlainConvUNet, base 32, 6 stages, features [32,64,128,256,320,320], conv 2/stage,
  kernel 3, strides [1,2,2,2,2,2] (등방), InstanceNorm3d(affine), LeakyReLU, deep supervision.
TS 체크포인트(checkpoint_final.pth)의 'network_weights'에서 seg 헤드(decoder.seg_layers.*)만 제외하고 이식한다.
z 방향은 4mm(면내 ~0.7mm)로 비등방이지만 인코더 가중치는 그대로 두고 파인튜닝으로 흡수한다.
"""
from __future__ import annotations
import os, glob
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

NUM_CLASSES = 10
TS_FEATURES = [32, 64, 128, 256, 320, 320]
TS_STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


def build_model(num_classes: int = NUM_CLASSES, deep_supervision: bool = True, in_channels: int = 1) -> PlainConvUNet:
    return PlainConvUNet(
        input_channels=in_channels, n_stages=6, features_per_stage=TS_FEATURES, conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6, strides=TS_STRIDES, n_conv_per_stage=[2] * 6, num_classes=num_classes,
        n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=deep_supervision)


def find_ts_checkpoint(root: str) -> str:
    c = sorted(glob.glob(os.path.join(root, "**", "checkpoint_final.pth"), recursive=True))
    if not c: raise FileNotFoundError(f"checkpoint_final.pth not under {root}")
    return c[0]


def load_ts_init(model: nn.Module, ckpt_path: str, verbose: bool = True) -> dict:
    """TS 가중치 이식. seg 헤드(클래스 수 다름)는 제외. → 통계 dict"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("network_weights", ck)
    own = model.state_dict(); new = {}; skipped = []
    for k, v in sd.items():
        k2 = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        if k2 in own and own[k2].shape == v.shape and not k2.startswith("decoder.seg_layers"):
            new[k2] = v
        else:
            skipped.append(k2)
    missing = [k for k in own if k not in new]
    model.load_state_dict(new, strict=False)
    st = {"transferred": len(new), "model_tensors": len(own), "skipped_from_ckpt": skipped, "not_initialized": missing}
    if verbose:
        print(f"[TS init] {len(new)}/{len(own)} tensors transferred from {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}; "
              f"not initialized: {len(missing)} ({', '.join(missing[:4])}{' …' if len(missing) > 4 else ''})")
    return st


def count_params(m: nn.Module) -> int: return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/data/datasets/couinaud_public/checkpoints/ts/291"
    m = build_model(); print(f"params {count_params(m)/1e6:.2f}M")
    st = load_ts_init(m, find_ts_checkpoint(root)); print("skipped:", st["skipped_from_ckpt"])
    m = m.cuda(); x = torch.randn(1, 1, 80, 256, 256, device="cuda")
    with torch.no_grad(): y = m(x)
    print("out", [tuple(t.shape) for t in (y if isinstance(y, (list, tuple)) else [y])])
