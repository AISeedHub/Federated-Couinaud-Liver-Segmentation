# -*- coding: utf-8 -*-
"""
train_single.py — 분당서울대 단일기관(local-only) 학습 + FL 성능 비교
====================================================================
combined_80(GT)으로 단일기관 모델을 학습하고, 연합학습(FedAvg) 글로벌
모델과 동일 test set에서 성능을 비교한다.

핵심: 데이터 분할·시드·모델 아키텍처·에폭수를 FL과 동일하게 맞춰
      "단일기관 vs 연합학습"을 공정 비교한다.
  - auto_split(train 0.8 / val 0.1 / test 0.1, seed=42)
    → FL의 splits/test.txt 6명과 동일 test set (검증됨)
  - 100 epochs = FL 10라운드 × 10 로컬에폭 (효과적 로컬학습량 일치)

실행:
  # 학습 + 비교 (처음부터)
  python3 train_single.py --fl-weight outputs/global_model_FedAvg_rtx8000.pth

  # 학습 건너뛰고 기존 가중치로 비교만
  python3 train_single.py --eval-only \
      --single-weight outputs/single_center/best_bundang_single.pth \
      --fl-weight outputs/global_model_FedAvg_rtx8000.pth
"""
import os, sys, math, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.use_cases.liver_segmentation.models.segresnet_morph import build_model
from src.use_cases.liver_segmentation.utils.dataset import (
    LiverSeg9Dataset, auto_split, seg9_collate,
)
from src.use_cases.liver_segmentation.utils.metrics import evaluate
from src.use_cases.liver_segmentation.utils.loss import compute_loss

SEG = ["Seg1","Seg2","Seg3","Seg4a","Seg4b","Seg5","Seg6","Seg7","Seg8"]
CFG = {
    "num_classes": 9, "init_filters": 16,
    "blocks_down": [1,2,2,4], "blocks_up": [1,1,1],
    "image_size": 256, "volume_depth": 80,
    "batch_size": 1, "learning_rate": 5.0e-4, "weight_decay": 1.0e-4,
    "train_ratio": 0.8, "val_ratio": 0.1, "seed": 42,   # ★ FL과 동일 시드
    # morph_coeff=0 → FedAvg와 동일한 순수 segmentation loss (공정 비교).
    # FL은 어떤 방법도 실질 morph 미적용(warmup 30 > local_epochs 10)이므로 0으로 맞춤.
    "epochs": 100, "morph_coeff": 0.0, "morph_start_frac": 0.5,
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_loader(data_dir, pids, mode):
    ds = LiverSeg9Dataset(data_dir, pids, CFG["image_size"], CFG["volume_depth"], mode, CFG["num_classes"])
    nw = CFG.get("num_workers", 0)
    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=(mode=="train"),
                      num_workers=nw, persistent_workers=(nw > 0),
                      pin_memory=True, collate_fn=seg9_collate)


def train(data_dir, out_dir, train_ids, val_ids, device):
    os.makedirs(out_dir, exist_ok=True)
    train_loader = make_loader(data_dir, train_ids, "train")
    val_loader   = make_loader(data_dir, val_ids, "val")
    model = build_model(CFG, device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epochs = CFG["epochs"]; morph_start = int(epochs*CFG["morph_start_frac"]); best_val = -1.0
    best_path = os.path.join(out_dir, "best_bundang_single.pth")
    for epoch in range(epochs):
        model.train()
        cur_lr = CFG["learning_rate"] * max(0.5*(1+math.cos(math.pi*epoch/epochs)), 0.3)
        for g in opt.param_groups: g["lr"] = cur_lr
        mc = CFG["morph_coeff"]*((epoch-morph_start)/max(epochs-morph_start,1)) if epoch>=morph_start else 0.0
        total, n = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device); masks = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                seg_logits, _m, vol_ratios = model(images)
                loss, _sl, _ml = compute_loss(seg_logits, vol_ratios, masks, mc)
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item(); n += 1
        if (epoch+1) % 5 == 0 or epoch == epochs-1:
            dv,_,_ = evaluate(model, val_loader, device, CFG["num_classes"])
            val_dice = float(torch.nanmean(dv).item())
            print(f"[Epoch {epoch+1:3d}/{epochs}] loss {total/max(n,1):.4f} | LR {cur_lr:.6f} | val Dice {val_dice:.4f}")
            if val_dice > best_val:
                best_val = val_dice; torch.save(model.state_dict(), best_path)
        else:
            print(f"[Epoch {epoch+1:3d}/{epochs}] loss {total/max(n,1):.4f} | LR {cur_lr:.6f}")
    return best_path, best_val


def eval_model(weight, data_dir, test_ids, device):
    model = build_model(CFG, device)
    model.load_state_dict(torch.load(weight, map_location=device)); model.eval()
    loader = make_loader(data_dir, test_ids, "test")
    dv, hv, vr = evaluate(model, loader, device, CFG["num_classes"])
    return {"dice_per_seg":[float(x) for x in dv.tolist()],
            "dice_mean":float(torch.nanmean(dv).item()),
            "hd95_per_seg":[float(x) for x in hv.tolist()],
            "hd95_mean":float(torch.nanmean(hv).item()),
            "vr_err":float(vr) if not np.isnan(vr) else 0.0}


def main():
    ap = argparse.ArgumentParser(description="단일기관 학습 + FL 성능 비교")
    ap.add_argument("--data-dir", default="data/combined_80")
    ap.add_argument("--out-dir", default="outputs/single_center")
    ap.add_argument("--fl-weight", default="outputs/global_model_FedAvg_rtx8000.pth", help="비교할 FL 글로벌 가중치")
    ap.add_argument("--single-weight", default=None, help="--eval-only 시 사용할 단일기관 가중치")
    ap.add_argument("--eval-only", action="store_true", help="학습 건너뛰고 비교만")
    ap.add_argument("--num-workers", type=int, default=12, help="DataLoader 워커 수 (GPU 놀지 않게)")
    args = ap.parse_args()

    CFG["num_workers"] = args.num_workers
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    # ---- 분할 (FL과 동일 시드) + 검증 ----
    ids = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    train_ids, val_ids, test_ids = auto_split(ids, CFG["train_ratio"], CFG["val_ratio"], CFG["seed"])
    print(f"환자 {len(ids)} → train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)} (seed={CFG['seed']})")
    fixed = os.path.join(REPO, "splits/test.txt")
    if os.path.exists(fixed):
        fl_test = sorted(open(fixed).read().split())
        same = sorted(test_ids) == fl_test
        print(f"FL test.txt와 test set 일치: {'✓ 동일 (공정 비교 성립)' if same else '✗ 불일치!'} ({len(set(test_ids)&set(fl_test))}/{len(fl_test)})")

    # ---- 단일기관 학습 ----
    if args.eval_only:
        single_w = args.single_weight or os.path.join(args.out_dir, "best_single.pth")
        print(f"[eval-only] 단일기관 가중치: {single_w}")
    else:
        single_w, best_val = train(args.data_dir, args.out_dir, train_ids, val_ids, device)
        print(f"학습 완료. best val Dice={best_val:.4f}")

    # ---- 동일 test set에서 양쪽 평가 ----
    print("\n[평가] 단일기관 모델 (test)")
    single = eval_model(single_w, args.data_dir, test_ids, device)
    fl = None
    if args.fl_weight and os.path.exists(args.fl_weight):
        print("[평가] FL 글로벌 모델 (test)")
        fl = eval_model(args.fl_weight, args.data_dir, test_ids, device)

    # ---- 비교 출력/저장 ----
    print("\n" + "="*74)
    print(f"  단일기관(local-only) vs 연합학습(FedAvg) — test set n={len(test_ids)}")
    print("="*74)
    if fl:
        print(f"  {'분절':<8}{'Single':>10}{'FL':>10}{'ΔDice':>10}{'Single HD95':>13}{'FL HD95':>10}")
        for i,s in enumerate(SEG):
            ds=single["dice_per_seg"][i]; df=fl["dice_per_seg"][i]
            print(f"  {s:<8}{ds:>10.4f}{df:>10.4f}{df-ds:>+10.4f}{single['hd95_per_seg'][i]:>13.2f}{fl['hd95_per_seg'][i]:>10.2f}")
        print("  " + "-"*72)
        dd=fl["dice_mean"]-single["dice_mean"]; hh=fl["hd95_mean"]-single["hd95_mean"]
        print(f"  {'평균':<8}{single['dice_mean']:>10.4f}{fl['dice_mean']:>10.4f}{dd:>+10.4f}"
              f"{single['hd95_mean']:>13.2f}{fl['hd95_mean']:>10.2f}")
        print("="*74)
        # Dice: 높을수록 좋음 / HD95: 낮을수록 좋음
        print(f"  → Dice {dd:+.4f} ({dd/single['dice_mean']*100:+.1f}%) — {'연합학습 우세' if dd>0 else '단일기관 우세'}")
        print(f"  → HD95 {hh:+.2f}mm — {'연합학습 개선(경계오차 감소)' if hh<0 else '연합학습 악화(경계오차 증가)'} "
              f"[HD95는 낮을수록 좋음]")
    else:
        print(f"  단일기관 평균 Dice {single['dice_mean']:.4f} / HD95 {single['hd95_mean']:.2f} (FL 가중치 없음)")
    print("="*74)

    out = {"test_ids":sorted(test_ids),"n_test":len(test_ids),"seed":CFG["seed"],
           "single_center":single,"fl_global":fl,
           "delta_dice":(fl["dice_mean"]-single["dice_mean"]) if fl else None,
           "delta_hd95":(fl["hd95_mean"]-single["hd95_mean"]) if fl else None}
    op = os.path.join(args.out_dir, "single_vs_fl_comparison.json")
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(out, open(op,"w"), indent=2, ensure_ascii=False)
    print("저장:", op)


if __name__ == "__main__":
    main()
