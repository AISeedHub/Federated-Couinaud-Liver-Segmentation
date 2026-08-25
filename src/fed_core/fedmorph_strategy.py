"""
FedMorph Strategy — Anatomy-Decoupled Aggregation for Flower
=============================================================

Training is identical across all methods (pure segmentation loss).
The only difference is the aggregation strategy:
  FedAvg / FedProx  → data-size weighted average on all params
  FedBN             → FedAvg but GroupNorm stays local
  FedMorph          → per-segment quality-weighted seg_head,
                       FedAvg for everything else
"""

import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

QUALITY_FLOOR = 0.1


def _is_norm(name: str) -> bool:
    return "norm" in name


def _is_seg_head(name: str) -> bool:
    return "conv_final" in name and "norm" not in name


class FedMorphStrategy(FedAvg):
    """Anatomy-Decoupled Aggregation for Flower FL framework.

    Supports multiple FL methods via the ``method`` parameter:
      FedAvg   – standard weighted average
      FedProx  – same aggregation as FedAvg (proximal term is client-side)
      FedBN    – weighted average with local GroupNorm
      FedMorph – seg_head quality-weighted, rest FedAvg

    After training, the following attributes are available:
      last_parameters   – final aggregated Parameters
      round_history     – per-round {round, avg_dice, clients: [...]}
      client_test_results – {client_id: {global: {...}, local: {...}}}
    """

    def __init__(
        self,
        model_state_keys: List[str],
        num_classes: int = 9,
        method: str = "FedMorph",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_state_keys = model_state_keys
        self.num_classes = num_classes
        self.method = method

        self.last_parameters: Optional[Parameters] = None
        self.round_history: List[dict] = []
        self.client_test_results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # aggregate_fit: core aggregation logic
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if self.method in ("FedAvg", "FedProx"):
            params, metrics = super().aggregate_fit(
                server_round, results, failures,
            )
            if params is not None:
                self.last_parameters = params
            return params, metrics

        client_params: List[List[np.ndarray]] = []
        client_weights: List[int] = []
        client_seg_dices: List[np.ndarray] = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_params.append(ndarrays)
            client_weights.append(fit_res.num_examples)

            metrics = fit_res.metrics
            if "seg_dices_json" in metrics:
                client_seg_dices.append(
                    np.array(json.loads(str(metrics["seg_dices_json"])))
                )
            else:
                client_seg_dices.append(
                    np.ones(self.num_classes) / self.num_classes
                )

        n = len(client_params)
        keys = self.model_state_keys
        total_w = sum(client_weights)
        dw = [w / total_w for w in client_weights]

        aggregated: List[np.ndarray] = []
        for idx, key in enumerate(keys):
            tensors = [client_params[i][idx] for i in range(n)]

            if self.method == "FedMorph" and _is_seg_head(key):
                aggregated.append(
                    self._per_segment_weighted(tensors, client_seg_dices, dw)
                )
            elif self.method == "FedBN" and _is_norm(key):
                aggregated.append(tensors[0].copy())
            else:
                merged = sum(
                    dw[i] * tensors[i].astype(np.float64) for i in range(n)
                )
                aggregated.append(merged.astype(tensors[0].dtype))

        parameters = ndarrays_to_parameters(aggregated)
        self.last_parameters = parameters

        agg_metrics: Dict[str, Scalar] = {"server_round": server_round}
        if client_seg_dices:
            avg_dices = np.mean(client_seg_dices, axis=0)
            agg_metrics["avg_dice"] = float(np.mean(avg_dices))

        print(
            f"  [Round {server_round}] Aggregated {n} clients "
            f"(method={self.method})"
        )
        return parameters, agg_metrics

    # ------------------------------------------------------------------
    # aggregate_evaluate: collect per-round metrics + test results
    # ------------------------------------------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        round_data: dict = {"round": server_round, "clients": []}
        weighted_loss = 0.0
        weighted_dice = 0.0
        total_examples = 0

        for _, eval_res in results:
            m = eval_res.metrics
            n_ex = eval_res.num_examples
            total_examples += n_ex
            weighted_loss += eval_res.loss * n_ex
            weighted_dice += m.get("dice", 0.0) * n_ex

            client_info = {
                "num_examples": n_ex,
                "dice": m.get("dice", 0.0),
                "hd95": m.get("hd95", 0.0),
                "vr_err": m.get("vr_err", 0.0),
            }
            if "client_id" in m:
                client_info["client_id"] = str(m["client_id"])

            round_data["clients"].append(client_info)

            if "test_results_json" in m:
                cid = str(m.get("client_id", f"client_{len(self.client_test_results)}"))
                self.client_test_results[cid] = json.loads(
                    str(m["test_results_json"])
                )

        if total_examples == 0:
            return None, {}

        avg_loss = weighted_loss / total_examples
        avg_dice = weighted_dice / total_examples
        round_data["avg_dice"] = avg_dice

        self.round_history.append(round_data)

        print(f"  [Server] Aggregated eval — Dice: {avg_dice:.4f}")
        return avg_loss, {"dice": avg_dice}

    # ------------------------------------------------------------------
    # Per-segment quality × data-size weighted aggregation
    # ------------------------------------------------------------------
    def _per_segment_weighted(self, tensors, seg_dices, dw):
        n = len(tensors)
        result = np.zeros_like(tensors[0], dtype=np.float64)

        if tensors[0].ndim == 0:
            for i in range(n):
                result += dw[i] * tensors[i].astype(np.float64)
            return result.astype(tensors[0].dtype)

        n_seg = min(tensors[0].shape[0], self.num_classes)
        for c in range(n_seg):
            dices = [max(float(seg_dices[i][c]), QUALITY_FLOOR) for i in range(n)]
            quality_w = [dices[i] * dw[i] for i in range(n)]
            total = sum(quality_w)
            w = [1.0 / n] * n if total < 1e-8 else [q / total for q in quality_w]
            for i in range(n):
                result[c] += w[i] * tensors[i][c].astype(np.float64)

        for c in range(n_seg, tensors[0].shape[0]):
            for i in range(n):
                result[c] += dw[i] * tensors[i][c].astype(np.float64)

        return result.astype(tensors[0].dtype)
