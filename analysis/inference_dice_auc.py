# -*- coding: utf-8 -*-
"""
inference_dice_auc.py — 분당서울대 held-out test 6케이스 추론
  · 분절별 Dice (공식 MONAI DiceMetric @256, 빈분절 제외)
  · 분절별 voxel-level AUC (sigmoid 확률 vs GT, sklearn roc_auc_score)
  · Dice/AUC 평균이 0.90(90%) 넘는지 판정

실행:
  python3 inference_dice_auc.py \
    --weight weights/global_model_FedAvg.pth \
    --data-dir data/combined_80 \
    --test-split splits/test.txt
"""
import os, sys, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
from src.use_cases.liver_segmentation.models.segresnet_morph import build_model
from src.use_cases.liver_segmentation.utils.dataset import LiverSeg9Dataset, seg9_collate, auto_split

SEG = ["Seg1","Seg2","Seg3","Seg4a","Seg4b","Seg5","Seg6","Seg7","Seg8"]
CFG = {"num_classes":9,"init_filters":16,"blocks_down":[1,2,2,4],"blocks_up":[1,1,1],
       "image_size":256,"volume_depth":80}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="weights/global_model_FedAvg.pth")
    ap.add_argument("--data-dir", default="data/combined_80")
    ap.add_argument("--test-split", default="splits/test.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresh", type=float, default=0.90, help="합격 기준(90%)")
    ap.add_argument("--auc-subsample", type=int, default=400000,
                    help="AUC 계산 시 배경 복셀 최대 표본(메모리·속도). 0=전체")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}" + (f" | {torch.cuda.get_device_name(0)}" if dev.type=="cuda" else ""))

    ids = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    if args.test_split and os.path.exists(args.test_split):
        test_ids = sorted(x for x in open(args.test_split).read().split() if x in ids)
    else:
        _, _, test_ids = auto_split(ids, 0.8, 0.1, args.seed); test_ids = sorted(test_ids)
    print(f"Test 케이스 {len(test_ids)}명 (익명 Case 1~{len(test_ids)})\n")

    model = build_model(CFG, dev); model.load_state_dict(torch.load(args.weight, map_location=dev)); model.eval()
    ds = LiverSeg9Dataset(args.data_dir, test_ids, 256, 80, "test", 9)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=seg9_collate)
    dm = DiceMetric(include_background=True, reduction="mean")
    rng = np.random.default_rng(args.seed)

    dice_seg = {s: [] for s in SEG}; auc_seg = {s: [] for s in SEG}
    per_pt = {}
    with torch.no_grad():
        for ci,b in enumerate(loader,1):
            pid = b["pids"][0]; img = b["image"].to(dev); masks = b["mask"]  # (1,9,80,256,256)
            with torch.amp.autocast("cuda", enabled=dev.type=="cuda"):
                lo,_,_ = model(img)
            prob = lo.sigmoid()[0].float().cpu().numpy()          # (9,80,256,256) 확률
            pred = (prob > 0.5).astype(np.uint8)
            dds, aas = [], []
            for c, s in enumerate(SEG):
                gt = masks[0, c].numpy().astype(np.uint8)
                if gt.sum() == 0:                                  # 빈 분절 제외(공식과 동일)
                    dds.append(np.nan); aas.append(np.nan); continue
                # Dice (MONAI, @256)
                dm.reset()
                dm(y_pred=torch.from_numpy(pred[c])[None,None], y=torch.from_numpy(gt)[None,None])
                d = float(dm.aggregate().item()); dds.append(d); dice_seg[s].append(d)
                # AUC (voxel-level): 양성 전부 + 배경 표본
                yf = gt.reshape(-1); pf = prob[c].reshape(-1)
                pos = np.flatnonzero(yf == 1); neg = np.flatnonzero(yf == 0)
                if args.auc_subsample and neg.size > args.auc_subsample:
                    neg = rng.choice(neg, args.auc_subsample, replace=False)
                idx = np.concatenate([pos, neg])
                a = float(roc_auc_score(yf[idx], pf[idx])); aas.append(a); auc_seg[s].append(a)
            per_pt[pid] = (np.nanmean(dds), np.nanmean(aas))
            print(f"  Case {ci}:  Dice {per_pt[pid][0]:.4f} | AUC {per_pt[pid][1]:.4f}")

    # ---- 집계 ----
    dvals = np.array([v for s in SEG for v in dice_seg[s]])
    avals = np.array([v for s in SEG for v in auc_seg[s]])
    mDice, mAUC = np.nanmean(dvals), np.nanmean(avals)
    ptD = np.mean([v[0] for v in per_pt.values()]); ptA = np.mean([v[1] for v in per_pt.values()])

    print("\n" + "="*60)
    print(f"{'분절':<8}{'Dice':>10}{'AUC':>10}")
    for s in SEG:
        dd = np.nanmean(dice_seg[s]) if dice_seg[s] else np.nan
        aa = np.nanmean(auc_seg[s]) if auc_seg[s] else np.nan
        print(f"{s:<8}{dd:>10.4f}{aa:>10.4f}")
    print("-"*60)
    print(f"{'분절평균':<8}{mDice:>10.4f}{mAUC:>10.4f}")
    print(f"{'환자평균':<8}{ptD:>10.4f}{ptA:>10.4f}")
    print("="*60)
    t = args.thresh
    print(f"\n[판정] 기준 {t*100:.0f}%")
    print(f"  Dice {mDice*100:.2f}% → {'✅ 90 넘음' if mDice>=t else '❌ 90 미만'}")
    print(f"  AUC  {mAUC*100:.2f}% → {'✅ 90 넘음' if mAUC>=t else '❌ 90 미만'}")


if __name__ == "__main__":
    main()
