"""
FedMorph Benchmark Client
==========================

Participates in sequential FL experiments for multiple methods.
Pairs with benchmark_server.py — automatically reconnects for each method.

After all methods complete, prints a comparison table of test results.

Usage:
  uv run python src/use_cases/liver_segmentation/benchmark_client.py \
      --server-address 192.168.1.100:443 --data-dir D:\\data\\liver_ct

  uv run python src/use_cases/liver_segmentation/benchmark_client.py \
      --server-address 192.168.1.100:443 --data-dir tests/dummy_data \
      --methods FedAvg FedMorph
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

from src.use_cases.liver_segmentation.main_client import (
    LiverSegmentationClient,
    _print_final_report,
    load_config,
)
from src.use_cases.liver_segmentation.utils.dataset import (
    auto_split,
    discover_patients,
)

ALL_METHODS = ["FedAvg", "FedProx", "FedBN", "FedMorph"]


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


def run_one_method(method, config, client_id, server_addr, out_dir, split):
    """Run a single FL method and return test results."""
    method_config = {**config, "method": method}

    client = LiverSegmentationClient(client_id, method_config, split=split)

    print(f"  Connecting to server at {server_addr} ...")
    _connect_with_retry(server_addr, client)

    results = {"method": method}

    # Save models
    global_path = os.path.join(out_dir, f"global_model_{method}_{client_id}.pth")
    torch.save(client.model.state_dict(), global_path)

    if client.last_local_state is not None:
        local_path = os.path.join(out_dir, f"local_model_{method}_{client_id}.pth")
        torch.save(client.last_local_state, local_path)

    # Test evaluation
    if len(client.test_ds) > 0:
        test_results = client.run_final_test()
        results.update(test_results)
        _print_final_report(client_id, test_results, config["num_classes"])
    else:
        print(f"  No test data, skipping evaluation for {method}")

    del client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def print_benchmark_summary(all_results, client_id):
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
        l = r.get("local", {})
        g_dice = g.get("dice", float("nan"))
        l_dice = l.get("dice", float("nan"))
        g_hd = g.get("hd95", float("nan"))
        l_hd = l.get("hd95", float("nan"))
        marker = " *" if method == "FedMorph" else ""
        print(f"  {method + marker:<12} {g_dice:>10.4f} {l_dice:>10.4f} "
              f"{g_hd:>10.2f} {l_hd:>10.2f}")

    print(sep)

    global_dices = [(r["method"], r.get("global", {}).get("dice", 0))
                    for r in all_results]
    best = max(global_dices, key=lambda x: x[1])
    print(f"  Best global Dice: {best[0]} ({best[1]:.4f})")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description="FedMorph Benchmark Client")
    parser.add_argument("--client-id", type=str, default=None,
                        help="Client ID (default: hostname)")
    parser.add_argument("--server-address", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="src/use_cases/liver_segmentation/configs/base.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Methods to benchmark (default: all 4)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["data_dir"] = os.environ.get(
        "FEDMORPH_DATA_DIR", args.data_dir or config.get("data_dir", ".")
    )
    client_id = args.client_id or platform.node()
    methods = args.methods
    total = len(methods)

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

    # ── Data split: computed ONCE, shared across all methods ──
    data_dir = config["data_dir"]
    all_pids = discover_patients(data_dir)
    if not all_pids:
        raise RuntimeError(f"No patients found in {data_dir}")

    train_ids, val_ids, test_ids = auto_split(
        all_pids,
        train_ratio=config.get("train_ratio", 0.70),
        val_ratio=config.get("val_ratio", 0.15),
        seed=config.get("seed", 42),
    )
    split = (train_ids, val_ids, test_ids)

    print("=" * 60)
    print(f"  FedMorph Benchmark Client [{client_id}]")
    print("=" * 60)
    print(f"  Methods:  {', '.join(methods)}")
    print(f"  Data:     {data_dir}")
    print(f"  Patients: {len(all_pids)} total "
          f"(train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)})")
    print(f"  Server:   {args.server_address}")
    print(f"  Split is FIXED across all methods (seed={config.get('seed', 42)})")
    print("=" * 60)

    out_dir = os.path.join(config.get("output_dir", "outputs"), "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for i, method in enumerate(methods, 1):
        print(f"\n{'#' * 60}")
        print(f"  [{i}/{total}] Method: {method}")
        print(f"{'#' * 60}")

        results = run_one_method(
            method, config, client_id, args.server_address, out_dir, split,
        )
        all_results.append(results)

        if i < total:
            wait = 5
            print(f"\n  Next method in {wait}s...")
            time.sleep(wait)

    # Final summary
    print_benchmark_summary(all_results, client_id)

    # Save all results
    summary_path = os.path.join(out_dir, f"benchmark_{client_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[Benchmark] Results saved to {summary_path}")


if __name__ == "__main__":
    main()
