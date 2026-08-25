"""
FedMorph Liver Segmentation — Federated Client
================================================

Each hospital PC runs one instance of this client.
The client:
  1. Scans its local data directory for CT volumes (image.npy + mask.npy)
  2. Auto-splits into train/val/test sets
  3. Receives global model from the server, trains locally
  4. Returns updated model weights + seg quality metrics to server

Supports running multiple methods sequentially via --methods flag:
  python main_client.py --server-address IP:443 --data-dir ./data --methods FedAvg FedProx FedBN FedMorph

No central data distribution needed — each site only accesses its own data.

Usage:
  python main_client.py --data-dir D:\\data\\liver_ct --server-address 192.168.1.100:443
  python main_client.py --server-address 192.168.1.100:443 --data-dir ./data --methods FedAvg FedMorph
"""

import argparse
import gc
import json
import math
import os
import platform
import random
import sys
import time

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import yaml
from collections import OrderedDict
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.fed_core.fed_client import FedFlowerClient
from src.use_cases.liver_segmentation.models.segresnet_morph import build_model
from src.use_cases.liver_segmentation.utils.dataset import (
    LiverSeg9Dataset,
    auto_split,
    discover_patients,
    seg9_collate,
)
from src.use_cases.liver_segmentation.utils.loss import compute_loss
from src.use_cases.liver_segmentation.utils.metrics import (
    compute_morph_diversity,
    compute_per_segment_dice,
    evaluate,
)

ALL_METHODS = ["FedAvg", "FedProx", "FedBN", "FedMorph"]


def _is_norm(name: str) -> bool:
    return "norm" in name


class LiverSegmentationClient(FedFlowerClient):
    """FedMorph client for 9-segment liver CT segmentation."""

    def __init__(self, client_id: str, config: dict, split=None):
        """
        Args:
            split: optional pre-computed (train_ids, val_ids, test_ids) tuple.
                   If None, auto_split is called internally.
        """
        super().__init__(client_id, config)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Data (load first to detect actual num_classes) ──
        data_dir = config["data_dir"]
        if split is not None:
            train_ids, val_ids, test_ids = split
        else:
            all_pids = discover_patients(data_dir)
            if not all_pids:
                raise RuntimeError(
                    f"No patients found in {data_dir}. "
                    "Each subfolder must contain image.npy and mask.npy."
                )
            train_ids, val_ids, test_ids = auto_split(
                all_pids,
                train_ratio=config.get("train_ratio", 0.70),
                val_ratio=config.get("val_ratio", 0.15),
                seed=config.get("seed", 42),
            )
        nc = config["num_classes"]

        self.train_ds = LiverSeg9Dataset(
            data_dir, train_ids,
            config["image_size"], config["volume_depth"],
            mode="train", num_classes=nc,
        )
        nc = self.train_ds.num_classes
        self.val_ds = LiverSeg9Dataset(
            data_dir, val_ids,
            config["image_size"], config["volume_depth"],
            mode="val", num_classes=nc,
        )
        self.test_ds = LiverSeg9Dataset(
            data_dir, test_ids,
            config["image_size"], config["volume_depth"],
            mode="val", num_classes=nc,
        )

        if nc != config["num_classes"]:
            print(
                f"[Client {client_id}] num_classes adapted: "
                f"{config['num_classes']} → {nc} (based on mask channels)"
            )
            config["num_classes"] = nc

        # ── Model (built after num_classes is confirmed) ──
        self.model = build_model(config, self.device)

        nw = config.get("num_workers", 0)
        pin = torch.cuda.is_available()
        self.train_loader = DataLoader(
            self.train_ds, batch_size=config["batch_size"], shuffle=True,
            num_workers=nw, collate_fn=seg9_collate,
            pin_memory=pin, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=config["batch_size"], shuffle=False,
            num_workers=nw, collate_fn=seg9_collate,
            pin_memory=pin,
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=config["batch_size"], shuffle=False,
            num_workers=nw, collate_fn=seg9_collate,
            pin_memory=pin,
        )

        # ── Mixed precision (CUDA only) ──
        self.scaler = (
            torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
        )

        # ── FL state ──
        self.local_norm_state: OrderedDict | None = None
        self.global_params_for_prox: dict | None = None
        self.current_round = 0
        self.last_local_state: OrderedDict | None = None

        total_patients = len(train_ids) + len(val_ids) + len(test_ids)
        print(f"[Client {client_id}] Device: {self.device}")
        print(f"[Client {client_id}] Data: {data_dir}")
        print(
            f"[Client {client_id}] Patients: {total_patients} total "
            f"(train {len(self.train_ds)}, val {len(self.val_ds)}, "
            f"test {len(self.test_ds)})"
        )
        print(f"[Client {client_id}] Classes: {nc}")

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------
    def get_model_parameters(self) -> list[np.ndarray]:
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict()
        for k, v in zip(keys, parameters):
            state_dict[k] = torch.from_numpy(np.copy(v)).to(self.device)

        method = self.config.get("method", "FedMorph")

        if method == "FedBN" and self.local_norm_state is not None:
            for k, v in self.local_norm_state.items():
                state_dict[k] = v.to(self.device)

        self.model.load_state_dict(state_dict)

        if method == "FedProx":
            self.global_params_for_prox = {
                k: v.clone()
                for k, v in state_dict.items()
                if not k.endswith("num_batches_tracked")
            }

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """Flower fit callback — train locally, return params + FedMorph metadata."""
        self.current_round = config.get("server_round", self.current_round + 1)
        print(
            f"\n[Client {self.client_id}] === Round {self.current_round} ==="
        )

        self.set_model_parameters(parameters)

        epochs = config.get(
            "local_epochs", self.config.get("local_epochs", 10)
        )
        train_metrics = self.train_model(epochs)

        self.last_local_state = OrderedDict(
            (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
        )

        method = self.config.get("method", "FedMorph")
        if method == "FedMorph":
            seg_dices = compute_per_segment_dice(
                self.model, self.val_loader, self.device,
                self.config["num_classes"],
            )
            morph_div = compute_morph_diversity(
                self.model, self.val_loader, self.device,
            )
            train_metrics["seg_dices_json"] = json.dumps(seg_dices.tolist())
            train_metrics["morph_diversity"] = float(morph_div)
            print(
                f"[Client {self.client_id}] "
                f"Seg Dice mean: {seg_dices.mean():.4f}, "
                f"Morph div: {morph_div:.6f}"
            )

        return (
            self.get_model_parameters(),
            self._get_dataset_size(),
            train_metrics,
        )

    def train_model(self, epochs: int) -> dict[str, float]:
        config = self.config
        method = config.get("method", "FedMorph")
        fl_rounds = config.get("fl_rounds", 50)

        lr_scale = 0.5 * (1 + math.cos(math.pi * self.current_round / fl_rounds))
        cur_lr = config["learning_rate"] * max(lr_scale, 0.3)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cur_lr,
            weight_decay=config["weight_decay"],
        )

        fedprox_mu = config.get("fedprox_mu", 0.0) if method == "FedProx" else 0.0

        morph_start = int(fl_rounds * 0.5)
        if method == "FedMorph" and self.current_round >= morph_start:
            alpha = (self.current_round - morph_start) / max(fl_rounds - morph_start, 1)
            mc = 0.005 * alpha
        else:
            mc = 0.0
        local_config = {**config, "morph_coeff": mc}

        total_loss = 0.0
        for epoch in range(epochs):
            global_epoch = self.current_round * epochs + epoch
            loss = self._train_one_epoch(
                optimizer, local_config, fedprox_mu, global_epoch
            )
            total_loss += loss

            if (epoch + 1) % max(1, epochs // 3) == 0:
                print(
                    f"[Client {self.client_id}] "
                    f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, "
                    f"LR: {cur_lr:.6f}"
                )

        if method == "FedBN":
            self.local_norm_state = OrderedDict(
                (k, v.cpu().clone())
                for k, v in self.model.state_dict().items()
                if _is_norm(k)
            )

        avg_loss = total_loss / max(epochs, 1)
        return {"train_loss": float(avg_loss)}

    def _train_one_epoch(self, optimizer, config, fedprox_mu, epoch):
        self.model.train()
        total_loss, n = 0.0, 0

        warmup = config.get("seg_warmup_epochs", 30)
        mc = 0.0 if epoch < warmup else config.get("morph_coeff", 0.0)

        gp = None
        if fedprox_mu > 0 and self.global_params_for_prox is not None:
            gp = self.global_params_for_prox

        use_amp = self.scaler is not None
        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                seg_logits, _morph_feats, vol_ratios = self.model(images)
                loss, _sl, _ml = compute_loss(
                    seg_logits, vol_ratios, masks, mc,
                )

                if fedprox_mu > 0 and gp is not None:
                    prox = sum(
                        ((p - gp[k]) ** 2).sum()
                        for k, p in self.model.named_parameters()
                        if p.requires_grad and k in gp
                    )
                    loss = loss + (fedprox_mu / 2.0) * prox

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def evaluate_model(self) -> tuple[float, float, dict]:
        dv, hv, vr_err = evaluate(
            self.model, self.val_loader, self.device, self.config["num_classes"]
        )

        mean_dice = float(torch.nanmean(dv).item())
        loss_proxy = 1.0 - mean_dice

        metrics = {
            "dice": mean_dice,
            "hd95": float(torch.nanmean(hv).item()),
            "vr_err": float(vr_err) if not np.isnan(vr_err) else 0.0,
            "client_id": str(self.client_id),
        }

        fl_rounds = self.config.get("fl_rounds", 50)
        if self.current_round >= fl_rounds and len(self.test_ds) > 0:
            print(f"[Client {self.client_id}] Last round — running test evaluation...")
            test_results = self.run_final_test()
            metrics["test_results_json"] = json.dumps(test_results)

        print(
            f"[Client {self.client_id}] Eval — "
            f"Dice: {mean_dice:.4f} | "
            f"HD95: {metrics['hd95']:.4f}"
        )
        return loss_proxy, mean_dice, metrics

    def _get_dataset_size(self) -> int:
        return len(self.train_ds)

    def run_final_test(self) -> dict:
        """Run test-set evaluation with both the global and local models."""
        nc = self.config["num_classes"]
        results = {}

        dv, hv, vr_err = evaluate(
            self.model, self.test_loader, self.device, nc,
        )
        results["global"] = {
            "dice": float(torch.nanmean(dv).item()),
            "dice_per_seg": [float(x) for x in dv.tolist()],
            "hd95": float(torch.nanmean(hv).item()),
            "vr_err": float(vr_err) if not np.isnan(vr_err) else 0.0,
        }

        if self.last_local_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.last_local_state.items()}
            )
            dv, hv, vr_err = evaluate(
                self.model, self.test_loader, self.device, nc,
            )
            results["local"] = {
                "dice": float(torch.nanmean(dv).item()),
                "dice_per_seg": [float(x) for x in dv.tolist()],
                "hd95": float(torch.nanmean(hv).item()),
                "vr_err": float(vr_err) if not np.isnan(vr_err) else 0.0,
            }

        return results


def _print_final_report(client_id: int, results: dict, num_classes: int):
    """Print a formatted comparison table of global vs local model on test set."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  FINAL TEST RESULTS — Client {client_id}")
    print(sep)

    header = f"{'Metric':<20} {'Global (Aggregated)':>22} {'Local (Last Train)':>22}"
    print(header)
    print("-" * 72)

    g = results.get("global", {})
    l = results.get("local", {})

    rows = [
        ("Dice (mean)", g.get("dice", float("nan")), l.get("dice", float("nan"))),
        ("HD95 (mean)", g.get("hd95", float("nan")), l.get("hd95", float("nan"))),
        ("VR Error", g.get("vr_err", float("nan")), l.get("vr_err", float("nan"))),
    ]
    for name, gv, lv in rows:
        print(f"  {name:<18} {gv:>22.4f} {lv:>22.4f}")

    print("-" * 72)
    print(f"  {'Per-Segment Dice':<18}")
    g_segs = g.get("dice_per_seg", [])
    l_segs = l.get("dice_per_seg", [])
    for c in range(num_classes):
        gv = g_segs[c] if c < len(g_segs) else float("nan")
        lv = l_segs[c] if c < len(l_segs) else float("nan")
        print(f"    Seg {c + 1:<13} {gv:>22.4f} {lv:>22.4f}")

    print(sep)
    g_dice = g.get("dice", 0)
    l_dice = l.get("dice", 0)
    if g_dice >= l_dice:
        print("  >> Global model wins (federated aggregation is effective)")
    else:
        print("  >> Local model wins (local data distribution advantage)")
    print(sep + "\n")


# ======================================================================
# Entry point
# ======================================================================
def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _connect_with_retry(server_addr, client, max_retries=60, interval=15):
    """Try connecting to the FL server with retries (handles server restart gap).

    Default: 60 retries × 15s = up to 15 minutes wait.
    Windows may need extra time due to TCP TIME_WAIT on port reuse.
    """
    import grpc

    for attempt in range(1, max_retries + 1):
        try:
            fl.client.start_numpy_client(
                server_address=server_addr, client=client,
            )
            return
        except grpc._channel._MultiThreadedRendezvous as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < max_retries:
                print(f"  [Retry {attempt}/{max_retries}] Server not ready, "
                      f"retrying in {interval}s...")
                time.sleep(interval)
            else:
                raise


def _run_one_method_and_save(method, config, client_id, server_addr, out_dir, split):
    """Run one FL method, save models and test results. Returns result dict."""
    method_config = {**config, "method": method}
    client = LiverSegmentationClient(client_id, method_config, split=split)

    print(f"  Connecting to server at {server_addr} ...")
    _connect_with_retry(server_addr, client)

    results = {"method": method}

    # Save models
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"{method}_{client_id}" if len(config.get("_methods", [])) > 1 else client_id

    global_path = os.path.join(out_dir, f"global_model_{suffix}.pth")
    torch.save(client.model.state_dict(), global_path)
    print(f"\n[Client] Global model saved to {global_path}")

    if client.last_local_state is not None:
        local_path = os.path.join(out_dir, f"local_model_{suffix}.pth")
        torch.save(client.last_local_state, local_path)
        print(f"[Client] Local model saved to {local_path}")

    if len(client.test_ds) > 0:
        print("[Client] Running final test evaluation...")
        test_results = client.run_final_test()
        results.update(test_results)
        _print_final_report(client_id, test_results, config["num_classes"])

        result_path = os.path.join(out_dir, f"test_results_{suffix}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"[Client] Test results saved to {result_path}")
    else:
        print("\n[Client] No test data available, skipping final evaluation.")

    del client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _print_multi_method_summary(all_results, client_id):
    """Print a comparison table across all methods."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BENCHMARK SUMMARY — Client [{client_id}]")
    print(sep)

    header = f"  {'Method':<12} {'Dice (G)':>10} {'Dice (L)':>10} {'HD95 (G)':>10} {'HD95 (L)':>10}"
    print(header)
    print("-" * 72)

    for r in all_results:
        method = r.get("method", "?")
        g = r.get("global", {})
        loc = r.get("local", {})
        g_dice = g.get("dice", float("nan"))
        l_dice = loc.get("dice", float("nan"))
        g_hd = g.get("hd95", float("nan"))
        l_hd = loc.get("hd95", float("nan"))
        marker = " *" if method == "FedMorph" else ""
        print(f"  {method + marker:<12} {g_dice:>10.4f} {l_dice:>10.4f} "
              f"{g_hd:>10.2f} {l_hd:>10.2f}")

    print(sep)

    global_dices = [(r["method"], r.get("global", {}).get("dice", 0))
                    for r in all_results]
    if global_dices:
        best = max(global_dices, key=lambda x: x[1])
        print(f"  Best global Dice: {best[0]} ({best[1]:.4f})")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="FedMorph Liver Segmentation — Federated Client"
    )
    parser.add_argument(
        "--client-id", type=str, default=None,
        help="Client ID for logging (default: hostname)",
    )
    parser.add_argument(
        "--server-address", type=str, default=None,
        help="Server address (overrides config)",
    )
    parser.add_argument(
        "--config", type=str,
        default="src/use_cases/liver_segmentation/configs/base.yaml",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Local CT data directory (overrides config)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        choices=ALL_METHODS,
        help="Methods to run sequentially (default: single method from config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    config["data_dir"] = os.environ.get(
        "FEDMORPH_DATA_DIR", args.data_dir or config.get("data_dir", ".")
    )
    server_addr = (
        args.server_address
        or os.environ.get("FEDMORPH_SERVER_ADDRESS")
        or config["server_address"]
    )

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB")
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    client_id = args.client_id or platform.node()

    if args.methods:
        methods = args.methods
    else:
        methods = [config.get("method", "FedMorph")]

    config["_methods"] = methods

    # ── Data split: computed ONCE, shared across all methods ──
    data_dir = config["data_dir"]
    all_pids = discover_patients(data_dir)
    if not all_pids:
        raise RuntimeError(
            f"No patients found in {data_dir}. "
            "Each subfolder must contain image.npy and mask.npy."
        )
    train_ids, val_ids, test_ids = auto_split(
        all_pids,
        train_ratio=config.get("train_ratio", 0.70),
        val_ratio=config.get("val_ratio", 0.15),
        seed=config.get("seed", 42),
    )
    split = (train_ids, val_ids, test_ids)
    total = len(methods)

    print("=" * 60)
    print(f"  FedMorph - Liver Segmentation Client [{client_id}]")
    print("=" * 60)
    print(f"  Methods:  {', '.join(methods)}")
    print(f"  Data:     {data_dir}")
    print(f"  Patients: {len(all_pids)} total "
          f"(train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)})")
    print(f"  Server:   {server_addr}")
    if total > 1:
        print(f"  Split is FIXED across all methods (seed={seed})")
    print("=" * 60)

    out_dir = config.get("output_dir", "outputs")

    all_results = []
    for i, method in enumerate(methods, 1):
        print(f"\n{'#' * 60}")
        print(f"  [{i}/{total}] Method: {method}")
        print(f"{'#' * 60}")

        results = _run_one_method_and_save(
            method, config, client_id, server_addr, out_dir, split,
        )
        all_results.append(results)

        if i < total:
            wait = 5
            print(f"\n  Next method in {wait}s...")
            time.sleep(wait)

    if total > 1:
        _print_multi_method_summary(all_results, client_id)

        summary_path = os.path.join(out_dir, f"benchmark_{client_id}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[Client] Benchmark results saved to {summary_path}")


if __name__ == "__main__":
    main()
