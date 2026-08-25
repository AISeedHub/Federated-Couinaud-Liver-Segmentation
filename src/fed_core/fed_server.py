"""Federated Learning Server"""

import flwr as fl
from typing import Dict, Optional, Callable
from flwr.server.strategy.fedavg import FedAvg
# from flwr.common import Metrics


class FedFlowerServer:
    """
    Federated Learning Server
    Coordinates training across multiple clients
    """

    def __init__(
            self,
            num_rounds: int = 3,
            min_clients: int = 2,
            strategy: Optional[fl.server.strategy.Strategy] = None,
            config: Optional[Dict] = None
    ):
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.config = config or {}

        # Default strategy
        if strategy is None:
            strategy = FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=min_clients,
                min_evaluate_clients=min_clients,
                min_available_clients=min_clients,
                evaluate_metrics_aggregation_fn=self._weighted_average,
                on_fit_config_fn=self._fit_config,
            )

        self.strategy = strategy

    def _weighted_average(self, metrics: list) -> Dict:
        """Aggregate metrics using weighted average."""
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}

    def _fit_config(self, server_round: int) -> Dict:
        """Return training configuration for each round."""
        config = {
            "server_round": server_round,
            "local_epochs": self.config.get("local_epochs", 1),
        }
        return config

    def start(self, server_address: str = "0.0.0.0:443"):
        """Start the Federated Learning server."""
        print(f"🌸 Starting FedFlower Server on {server_address}")
        print(f"📊 Rounds: {self.num_rounds} | Min Clients: {self.min_clients}")

        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
        )