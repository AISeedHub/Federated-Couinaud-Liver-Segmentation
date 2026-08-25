"""
FedMorph Liver Segmentation — Federated Server
================================================

Starts the Flower gRPC server with FedMorph aggregation strategy.
Waits for ``min_clients`` to connect, then runs ``fl_rounds`` rounds.

After training, saves to server outputs/:
  - Global model weights (.pth) per method
  - Round-by-round metrics log (.json)
  - Client test results collected from last round (.json)

Supports running multiple methods sequentially via --methods flag:
  python main_server.py --methods FedAvg FedProx FedBN FedMorph

Usage:
  python main_server.py
  python main_server.py --config configs/base.yaml
  python main_server.py --methods FedAvg FedMorph
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import flwr as fl
import torch
import yaml
from flwr.common import parameters_to_ndarrays

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.fed_core.fedmorph_strategy import FedMorphStrategy
from src.use_cases.liver_segmentation.models.segresnet_morph import build_model

ALL_METHODS = ["FedAvg", "FedProx", "FedBN", "FedMorph"]


def load_config(
    config_path: str = "src/use_cases/liver_segmentation/configs/base.yaml",
) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_state_keys(config: dict) -> list[str]:
    """Create a temporary model on CPU to get ordered state_dict keys."""
    device = torch.device("cpu")
    model = build_model(config, device)
    keys = list(model.state_dict().keys())
    del model
    return keys


def save_server_outputs(strategy, method, config, model_keys, out_dir):
    """Save global model, round metrics, and client test results."""
    os.makedirs(out_dir, exist_ok=True)

    if strategy.last_parameters is not None:
        ndarrays = parameters_to_ndarrays(strategy.last_parameters)
        model = build_model(config, torch.device("cpu"))
        keys = list(model.state_dict().keys())
        state_dict = OrderedDict()
        for k, arr in zip(keys, ndarrays):
            state_dict[k] = torch.from_numpy(arr)
        model.load_state_dict(state_dict)

        model_path = os.path.join(out_dir, f"global_model_{method}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  [Server] Global model saved: {model_path}")
        del model

    if strategy.round_history:
        history_path = os.path.join(out_dir, f"round_history_{method}.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(strategy.round_history, f, indent=2, ensure_ascii=False)
        print(f"  [Server] Round history saved: {history_path} "
              f"({len(strategy.round_history)} rounds)")

    if strategy.client_test_results:
        test_path = os.path.join(out_dir, f"test_results_{method}.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(
                strategy.client_test_results, f, indent=2, ensure_ascii=False,
            )
        print(f"  [Server] Client test results saved: {test_path} "
              f"({len(strategy.client_test_results)} clients)")
    else:
        print(f"  [Server] No client test results received for {method}")


def run_one_method(method, config, model_keys, server_address, out_dir):
    """Run a single FL method for fl_rounds. Save outputs to server."""
    fl_rounds = config["fl_rounds"]
    min_clients = config["min_clients"]
    local_epochs = config["local_epochs"]

    def fit_config_fn(server_round: int) -> dict:
        lr_scale = 0.5 * (1 + math.cos(math.pi * server_round / fl_rounds))
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "method": method,
            "lr_scale": lr_scale,
        }

    strategy = FedMorphStrategy(
        model_state_keys=model_keys,
        num_classes=config["num_classes"],
        method=method,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=fit_config_fn,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=fl_rounds),
        strategy=strategy,
    )

    save_server_outputs(strategy, method, config, model_keys, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="FedMorph Liver Segmentation — Federated Server"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/use_cases/liver_segmentation/configs/base.yaml",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        choices=ALL_METHODS,
        help="Methods to run sequentially (default: single method from config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    server_address = config.get("server_address", "0.0.0.0:443")
    model_keys = get_model_state_keys(config)

    if args.methods:
        methods = args.methods
    else:
        methods = [config.get("method", "FedMorph")]

    total = len(methods)

    out_dir = os.path.join(config.get("output_dir", "outputs"), "server")

    print("=" * 60)
    print("  FedMorph - Liver Segmentation Server")
    print("=" * 60)
    print(f"  Methods:     {', '.join(methods)}")
    print(f"  Rounds/method: {config['fl_rounds']}")
    print(f"  Min Clients: {config['min_clients']}")
    print(f"  Local Epochs: {config['local_epochs']}")
    print(f"  Model:        SegResNetMorph "
          f"(filters={config['init_filters']})")
    print(f"  Param tensors: {len(model_keys)}")
    print(f"  Address:     {server_address}")
    print(f"  Output:      {out_dir}")
    print("=" * 60)

    for i, method in enumerate(methods, 1):
        print(f"\n{'#' * 60}")
        print(f"  [{i}/{total}] Starting method: {method}")
        print(f"{'#' * 60}")
        print(f"  Waiting for {config['min_clients']} clients to connect...")

        t0 = time.time()
        run_one_method(method, config, model_keys, server_address, out_dir)
        elapsed = time.time() - t0

        print(f"\n  [{i}/{total}] {method} completed in {elapsed:.0f}s")

        if i < total:
            wait = 10
            print(f"  Next method in {wait}s... "
                  f"(clients will auto-reconnect)")
            time.sleep(wait)

    print(f"\n{'=' * 60}")
    if total > 1:
        print("  All methods complete!")
        print(f"  Methods tested: {', '.join(methods)}")
    else:
        print(f"  {methods[0]} complete!")
    print(f"  Server outputs: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
