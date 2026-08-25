# -*- coding: utf-8 -*-
"""
seg_visualize.py — 연합학습(FedAvg) vs 단일기관 분절 세그멘테이션 시각화
========================================================================
CT 슬라이스 위에 GT / Single / FedAvg 분절을 색으로 오버레이해, 연합학습이
단일기관보다 잘 맞추는 것을 시각적으로 보여준다.

★ z 정렬: 데이터셋이 D>80이면 가운데 80슬라이스만 center-crop 하므로,
  CT·GT·예측을 모두 '데이터셋이 처리한 공간(80슬라이스·256)'에서 그린다
  (native로 되돌려 늘리면 z가 어긋남). 이 방식은 evaluate() Dice와도 일치.

실행:
  python3 seg_visualize.py --data-dir <데이터> --fl-weight <FL.pth> --single-weight <single.pth>
  python3 seg_visualize.py --data-dir <데이터> --fl-weight <FL.pth> --pid 878676 --n-patients 6
출력: outputs/analysis/seg_viz/<pid>_overlay.png
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.use_cases.liver_segmentation.models.segresnet_morph import build_model
from src.use_cases.liver_segmentation.utils.dataset import LiverSeg9Dataset, seg9_collate, auto_split

SEG = ["Seg1","Seg2","Seg3","Seg4a","Seg4b","Seg5","Seg6","Seg7","Seg8"]
SEG_COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1","#ff9da7","#9c755f"]
CFG = {"num_classes":9,"init_filters":16,"blocks_down":[1,2,2,4],"blocks_up":[1,1,1],
       "image_size":256,"volume_depth":80}


def dice(a, b):
    i = np.logical_and(a, b).sum(); s = a.sum() + b.sum()
    return float(2*i/s) if s > 0 else float("nan")


def mean_dice(gt, pr):
    return np.nanmean([dice(gt[c] > 0, pr[c] > 0) for c in range(9)])


def label_map(seg_stack):
    """(9,H,W) 이진 → (H,W) 라벨(0=배경,1..9). 겹치면 앞 채널 우선."""
    lab = np.zeros(seg_stack.shape[1:], dtype=np.uint8)
    for c in range(seg_stack.shape[0]):
        lab[(seg_stack[c] > 0) & (lab == 0)] = c + 1
    return lab


def norm_ct(sl):
    lo, hi = np.percentile(sl, [1, 99])
    if hi <= lo: hi = lo + 1
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def load_model(weight, dev):
    m = build_model(CFG, dev); m.load_state_dict(torch.load(weight, map_location=dev)); m.eval(); return m


def load_case(data_dir, pid):
    """데이터셋 처리 공간(80,256)에서 CT·GT·입력텐서 반환 — 모두 z 정렬됨."""
    ds = LiverSeg9Dataset(data_dir, [pid], CFG["image_size"], CFG["volume_depth"], "test", CFG["num_classes"])
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=seg9_collate)))
    ct = batch["image"][0, 0].numpy()              # (80,256,256) 처리된 CT
    gt = batch["mask"][0].numpy().astype(np.uint8) # (9,80,256,256) 처리된 GT
    return ct, gt, batch["image"]


def predict(model, img_tensor, dev):
    """→ (9,80,256,256) 이진 예측 (동일 처리공간, z 정렬)."""
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev.type=="cuda"):
        lo, _, _ = model(img_tensor.to(dev))
    return (lo.sigmoid()[0].cpu().numpy() > 0.5).astype(np.uint8)


def render(pid, ct, gt, pr_s, pr_fl, n_slices, out_dir):
    d_fl = mean_dice(gt, pr_fl); d_s = mean_dice(gt, pr_s) if pr_s is not None else None
    stacks = [("GT", gt)]
    if pr_s is not None: stacks.append((f"Single ({d_s:.3f})", pr_s))
    stacks.append((f"FedAvg ({d_fl:.3f})", pr_fl))
    labs = {name: np.stack([label_map(st[:, z]) for z in range(st.shape[1])]) for name, st in stacks}

    zsum = labs["GT"].reshape(labs["GT"].shape[0], -1).astype(bool).sum(1)
    liver_z = np.where(zsum > 0)[0]
    if len(liver_z) == 0: liver_z = np.arange(ct.shape[0])
    picks = liver_z[np.linspace(0, len(liver_z)-1, n_slices).astype(int)]

    ncol = 1 + len(stacks); cmap = ListedColormap(["#00000000"] + SEG_COLORS)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(3*ncol, 3*len(picks)))
    if len(picks) == 1: axes = axes[None, :]
    titles = ["CT"] + [name for name, _ in stacks]
    for ri, z in enumerate(picks):
        base = norm_ct(ct[z])
        for ci in range(ncol):
            ax = axes[ri, ci]; ax.imshow(base, cmap="gray"); ax.axis("off")
            if ci > 0:
                lab = labs[stacks[ci-1][0]][z]
                ax.imshow(np.ma.masked_where(lab == 0, lab), cmap=cmap, vmin=0, vmax=9, alpha=0.5)
            if ri == 0: ax.set_title(titles[ci], fontsize=12)
            if ci == 0: ax.text(-0.1, 0.5, f"z={z}", transform=ax.transAxes, rotation=90, va="center", fontsize=9)
    fig.legend(handles=[Patch(facecolor=c, label=s) for s, c in zip(SEG, SEG_COLORS)],
               loc="lower center", ncol=9, fontsize=8, frameon=False)
    sub = f"patient {pid}" + (f" — FedAvg {d_fl:.3f} vs Single {d_s:.3f} (Δ{d_fl-d_s:+.3f})" if d_s is not None else f" — Dice {d_fl:.3f}")
    fig.suptitle(f"Liver segmentation: {sub}", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{pid}_overlay.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    return out, d_fl, d_s


def main():
    ap = argparse.ArgumentParser(description="FedAvg vs 단일기관 분절 오버레이 시각화")
    ap.add_argument("--data-dir",default="data/combined_80",help="GT 마스크 디렉토리 (환자별 mask.npy)")
    ap.add_argument("--fl-weight", default="weights/model.pth", help="연합학습(FedAvg) 글로벌 가중치")
    ap.add_argument("--single-weight", default="weights/model.pth", help="단일기관 가중치(주면 비교 4열)")
    ap.add_argument("--pid", default=None, help="특정 환자만 (미지정=여러 명 자동)")
    ap.add_argument("--n-patients", type=int, default=6, help="생성할 환자 수")
    ap.add_argument("--select", choices=["top","spread"], default="top",
                    help="top=FL이 가장 크게 이기는 N명(논문용) / spread=순위 전구간 균등(검증용)")
    ap.add_argument("--n-slices", type=int, default=4)
    ap.add_argument("--test-split", default=None, help="test.txt 파일 주면 그 환자만")
    ap.add_argument("--test-only", action="store_true", help="auto_split(seed) held-out test만 시각화(권장)")
    ap.add_argument("--seed", type=int, default=42, help="auto_split 시드 (학습과 동일해야 함)")
    ap.add_argument("--out-dir", default="outputs/analysis/seg_viz")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    if args.test_split and os.path.exists(args.test_split):
        tset = set(open(args.test_split).read().split())
        ids = [p for p in ids if p in tset]
        print(f"test-split 파일 적용: {len(ids)}명")
    elif args.test_only:
        _, _, test_ids = auto_split(ids, 0.8, 0.1, args.seed)   # 학습과 동일 시드
        ids = sorted(test_ids)
        print(f"auto_split(seed={args.seed}) held-out test: {len(ids)}명 → {ids}")
    fl = load_model(args.fl_weight, dev)
    single = load_model(args.single_weight, dev) if args.single_weight else None

    def case_preds(pid):
        ct, gt, tin = load_case(args.data_dir, pid)
        pr_fl = predict(fl, tin, dev)
        pr_s = predict(single, tin, dev) if single is not None else None
        return ct, gt, pr_s, pr_fl

    # ---- 대상 환자 선정 ----
    if args.pid:
        targets = [args.pid]
    else:
        print("전 환자 Dice 산정 중...")
        scored = []
        for p in ids:
            try:
                _, gt, pr_s, pr_fl = case_preds(p)
                df = mean_dice(gt, pr_fl); ds = mean_dice(gt, pr_s) if pr_s is not None else None
            except Exception as e:
                print(f"  {p}: skip ({e})"); continue
            key = (df - ds) if ds is not None else df
            scored.append((p, key)); print(f"  {p}: FL {df:.3f}" + (f" / Single {ds:.3f} / Δ{df-ds:+.3f}" if ds is not None else ""))
        scored.sort(key=lambda x: x[1], reverse=True)   # ΔDice(FL−single) 내림차순
        n = min(args.n_patients, len(scored))
        if args.select == "spread":
            idx = np.unique(np.linspace(0, len(scored)-1, n).astype(int))
            targets = [scored[i][0] for i in idx]; how = "순위 전구간 균등(검증용)"
        else:  # top: FL이 가장 크게 이기는 N명
            targets = [scored[i][0] for i in range(n)]; how = "FL 개선폭 상위(논문용)"
        print(f"\n선택 {len(targets)}명 ({how}): {targets}")

    for pid in targets:
        ct, gt, pr_s, pr_fl = case_preds(pid)
        out, d_fl, d_s = render(pid, ct, gt, pr_s, pr_fl, args.n_slices, args.out_dir)
        print(f"저장: {out}" + (f"  (FL {d_fl:.3f} vs Single {d_s:.3f})" if d_s is not None else f"  (Dice {d_fl:.3f})"))


if __name__ == "__main__":
    main()
