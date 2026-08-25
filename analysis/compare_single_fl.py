# -*- coding: utf-8 -*-
"""
compare_single_fl.py — 단일기관 vs 글로벌 연합학습(FedAvg) 부트스트랩 비교
========================================================================
두 가중치(단일기관 / FL 글로벌)를 로드해 동일 데이터에서 환자별 Dice를 뽑고,
환자 단위 paired 부트스트랩으로 Δ(FL − single)의 95% CI와 boot p를 구한다.

  Δ CI가 0을 포함하지 않으면 → 두 모델 성능차가 통계적으로 유의.

실행:
  # test set(held-out 6명) 기준 — 일반화 성능 비교(정직)
  python3 compare_single_fl.py \
      --single-weight outputs/single_center/best_bundang_single.pth \
      --fl-weight outputs/global_model_FedAvg_rtx8000.pth \
      --test-split splits/test.txt

  # 전체 62명 기준 (표본 큼, 단 훈련 포함 → 낙관적)
  python3 compare_single_fl.py --single-weight ... --fl-weight ... --eval-set full
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.use_cases.liver_segmentation.models.segresnet_morph import build_model
from src.use_cases.liver_segmentation.utils.dataset import LiverSeg9Dataset, auto_split, seg9_collate
from src.use_cases.liver_segmentation.utils.metrics import evaluate

SEG = ["Seg1","Seg2","Seg3","Seg4a","Seg4b","Seg5","Seg6","Seg7","Seg8"]
CFG = {"num_classes":9,"init_filters":16,"blocks_down":[1,2,2,4],"blocks_up":[1,1,1],
       "image_size":256,"volume_depth":80,"train_ratio":0.8,"val_ratio":0.1,"seed":42}


def eval_per_patient(weight, data_dir, ids, device):
    """{pid: [dice_seg1..seg9]} — 레포 evaluate()로 환자별 Dice (train_single과 동일 공식 방식)."""
    model = build_model(CFG, device)
    model.load_state_dict(torch.load(weight, map_location=device)); model.eval()
    per = {}
    for pid in ids:
        ds = LiverSeg9Dataset(data_dir, [pid], CFG["image_size"], CFG["volume_depth"], "test", CFG["num_classes"])
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=seg9_collate)
        dv, _, _ = evaluate(model, loader, device, CFG["num_classes"])
        per[pid] = [float(x) for x in dv.tolist()]
    return per


def boot_ci(d, B, rng):
    """1차원 diff 배열 → (관측 mean, 95%CI low, high, boot p)."""
    d = d[np.isfinite(d)]; n = len(d)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2*min((boot <= 0).mean(), (boot >= 0).mean()))
    return float(d.mean()), float(lo), float(hi), float(p), n


def main():
    ap = argparse.ArgumentParser(description="단일기관 vs FL 글로벌 부트스트랩 비교")
    ap.add_argument("--data-dir", default="data/combined_80")
    ap.add_argument("--single-weight", default="outputs/single_center/best_bundang_single.pth")
    ap.add_argument("--fl-weight", default="outputs/global_model_FedAvg_rtx8000.pth")
    ap.add_argument("--test-split", default="splits/test.txt")
    ap.add_argument("--eval-set", choices=["test","full"], default="test",
                    help="test=held-out(정직) / full=전체62명(표본큼·낙관적)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/single_center/bootstrap_single_vs_fl.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))

    # ---- 평가 대상 환자 선택 ----
    ids = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    if args.eval_set == "test":
        if os.path.exists(args.test_split):
            eval_ids = sorted(open(args.test_split).read().split())
        else:
            _, _, eval_ids = auto_split(ids, CFG["train_ratio"], CFG["val_ratio"], CFG["seed"])
        eval_ids = [p for p in eval_ids if p in ids]
    else:
        eval_ids = ids
    print(f"평가셋: {args.eval_set} (n={len(eval_ids)})")
    if args.eval_set == "full":
        print("  ⚠️ 전체 62명 중 다수가 훈련에 포함 → 절대 성능은 낙관적. Δ(paired)는 참고용.")

    # ---- 양쪽 모델 환자별 Dice ----
    print("[1/3] 단일기관 모델 추론"); single_pp = eval_per_patient(args.single_weight, args.data_dir, eval_ids, device)
    print("[2/3] FL 글로벌 모델 추론"); fl_pp = eval_per_patient(args.fl_weight, args.data_dir, eval_ids, device)

    # ---- 부트스트랩 ----
    print(f"[3/3] paired 부트스트랩 (B={args.bootstrap})")
    rng = np.random.default_rng(args.seed)
    s_mean = np.array([np.nanmean(single_pp[p]) for p in eval_ids])
    f_mean = np.array([np.nanmean(fl_pp[p]) for p in eval_ids])
    d_overall = f_mean - s_mean
    ov = boot_ci(d_overall, args.bootstrap, rng)

    perseg = {}
    for si, s in enumerate(SEG):
        d = np.array([fl_pp[p][si] - single_pp[p][si] for p in eval_ids])
        perseg[s] = boot_ci(d, args.bootstrap, rng)

    # ---- 출력 ----
    def star(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
    print("\n" + "="*78)
    print(f"  단일기관 vs 연합학습(FedAvg) — 환자단위 paired 부트스트랩  (n={len(eval_ids)}, {args.eval_set})")
    print("="*78)
    print(f"  {'':<8}{'Single':>9}{'FL':>9}{'ΔDice':>9}{'95% CI':>20}{'boot p':>9}")
    print("  " + "-"*76)
    for s in SEG:
        dl, lo, hi, p, _ = perseg[s]
        smean = np.nanmean([single_pp[pid][SEG.index(s)] for pid in eval_ids])
        fmean = np.nanmean([fl_pp[pid][SEG.index(s)] for pid in eval_ids])
        print(f"  {s:<8}{smean:>9.3f}{fmean:>9.3f}{dl:>+9.3f}  [{lo:+.3f},{hi:+.3f}]{p:>9.3f}{star(p)}")
    print("  " + "-"*76)
    dl, lo, hi, p, n = ov
    print(f"  {'평균':<8}{s_mean.mean():>9.3f}{f_mean.mean():>9.3f}{dl:>+9.3f}  [{lo:+.3f},{hi:+.3f}]{p:>9.3f}{star(p)}")
    print("="*78)
    sig = not (lo <= 0 <= hi)
    print(f"  → 전체 ΔDice = {dl:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], boot p={p:.4f}")
    print(f"    {'유의: FL 우세 (CI가 0 미포함)' if (sig and dl>0) else '유의: 단일 우세' if (sig and dl<0) else '유의차 없음 (CI가 0 포함)'}")
    print("="*78)

    out = {"eval_set":args.eval_set,"n":len(eval_ids),"eval_ids":eval_ids,"B":args.bootstrap,
           "overall":{"single":float(s_mean.mean()),"fl":float(f_mean.mean()),
                      "delta":ov[0],"ci_low":ov[1],"ci_high":ov[2],"boot_p":ov[3]},
           "per_segment":{s:{"delta":perseg[s][0],"ci_low":perseg[s][1],"ci_high":perseg[s][2],"boot_p":perseg[s][3]} for s in SEG}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out,"w"), indent=2, ensure_ascii=False)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
