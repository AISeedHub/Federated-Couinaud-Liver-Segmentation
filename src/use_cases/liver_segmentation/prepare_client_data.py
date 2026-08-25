"""
Validate local data for federated learning.
=============================================

Scans a local data directory to verify that patient folders contain
valid ``image.npy`` and ``mask.npy`` files, and previews the automatic
train/val split that will happen at client startup.

Run this on each client PC BEFORE starting federated learning to ensure
the data is ready.

Usage:
  python prepare_client_data.py --data-dir D:\\data\\liver_ct
  python prepare_client_data.py --data-dir /data/hospital_a/ct
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.use_cases.liver_segmentation.utils.dataset import (
    auto_split,
    discover_patients,
)

NUM_CLASSES = 9


def validate_patient(data_dir: str, pid: str) -> dict:
    """Check a single patient folder for valid data."""
    patient_dir = os.path.join(data_dir, pid)
    img_path = os.path.join(patient_dir, "image.npy")
    mask_path = os.path.join(patient_dir, "mask.npy")

    result = {"pid": pid, "ok": True, "warnings": []}

    try:
        img = np.load(img_path)
        if img.ndim != 3:
            result["warnings"].append(f"image.npy: expected 3D (D,H,W), got {img.ndim}D")
            result["ok"] = False
        result["image_shape"] = img.shape
    except Exception as e:
        result["warnings"].append(f"image.npy: {e}")
        result["ok"] = False
        result["image_shape"] = None

    try:
        mask = np.load(mask_path)
        if mask.ndim != 4:
            result["warnings"].append(f"mask.npy: expected 4D (C,D,H,W), got {mask.ndim}D")
            result["ok"] = False
        elif mask.shape[0] < NUM_CLASSES + 1:
            result["warnings"].append(
                f"mask.npy: expected >= {NUM_CLASSES + 1} channels, got {mask.shape[0]}"
            )
            result["ok"] = False
        result["mask_shape"] = mask.shape

        if mask.ndim == 4:
            seg = mask[1 : NUM_CLASSES + 1]
            active = [c for c in range(NUM_CLASSES) if seg[c].sum() > 0]
            result["active_segments"] = active
    except Exception as e:
        result["warnings"].append(f"mask.npy: {e}")
        result["ok"] = False
        result["mask_shape"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate local data for FedMorph federated learning"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to local CT data directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    print("=" * 60)
    print("FedMorph — Local Data Validation")
    print("=" * 60)
    print(f"  Data dir: {os.path.abspath(args.data_dir)}")
    print()

    pids = discover_patients(args.data_dir)
    if not pids:
        print("ERROR: No valid patient folders found.")
        print("  Each subfolder must contain image.npy and mask.npy.")
        return

    print(f"Found {len(pids)} patients. Validating...")
    print()

    ok_count = 0
    warn_count = 0
    for pid in pids:
        result = validate_patient(args.data_dir, pid)
        if result["ok"]:
            ok_count += 1
        else:
            warn_count += 1
            print(f"  WARNING [{pid}]:")
            for w in result["warnings"]:
                print(f"    - {w}")

    print(f"\nValidation: {ok_count} OK, {warn_count} with issues")

    train_ids, val_ids, test_ids = auto_split(
        pids, args.train_ratio, args.val_ratio, args.seed,
    )

    print(f"\n{'=' * 60}")
    print("Train/Val/Test Split Preview")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val:   {len(val_ids)} patients")
    print(f"  Test:  {len(test_ids)} patients")
    print(
        f"  Ratio: {len(train_ids)/len(pids):.0%} / "
        f"{len(val_ids)/len(pids):.0%} / "
        f"{len(test_ids)/len(pids):.0%}"
    )

    print(f"\n{'=' * 60}")
    print("Ready for Federated Learning")
    print(f"{'=' * 60}")
    print(f"""
  To start this client:

    uv run python src/use_cases/liver_segmentation/main_client.py \\
        --data-dir {os.path.abspath(args.data_dir)} \\
        --server-address <SERVER_IP>:443
""")


if __name__ == "__main__":
    main()
