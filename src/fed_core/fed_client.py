"""Base Client for Federated Learning"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
import flwr as fl
from collections import OrderedDict


class FedFlowerClient(fl.client.NumPyClient, ABC):
    """
    Base Federated Learning Client

    Task-specific repos should inherit this class and implement:
    - train_model()
    - evaluate_model()
    - get_model_parameters()
    - set_model_parameters()
    """

    def __init__(self, client_id: int, config: Dict):
        self.client_id = client_id
        self.config = config

    @abstractmethod
    def train_model(self, epochs: int) -> Dict[str, float]:
        """Train model locally. Return training metrics."""
        pass

    @abstractmethod
    def evaluate_model(self) -> Tuple[float, float, Dict]:
        """Evaluate model. Return (loss, accuracy, metrics)."""
        pass

    @abstractmethod
    def get_model_parameters(self) -> List[np.ndarray]:
        """Get model parameters as list of numpy arrays."""
        pass

    @abstractmethod
    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from list of numpy arrays."""
        pass

    # Flower interface implementation
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Flower callback: return parameters."""
        return self.get_model_parameters()

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Flower callback: train model."""
        print(f"[Client {self.client_id}] Starting training round...")

        # Set received parameters
        self.set_model_parameters(parameters)

        # Train
        epochs = config.get("local_epochs", 1)
        metrics = self.train_model(epochs)

        # Return updated parameters
        return self.get_model_parameters(), self._get_dataset_size(), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Flower callback: evaluate model."""
        print(f"[Client {self.client_id}] Evaluating...")

        # Set parameters
        self.set_model_parameters(parameters)

        # Evaluate
        loss, accuracy, metrics = self.evaluate_model()
        metrics["accuracy"] = accuracy

        return float(loss), self._get_dataset_size(), metrics

    @abstractmethod
    def _get_dataset_size(self) -> int:
        """Return size of local dataset."""
        pass