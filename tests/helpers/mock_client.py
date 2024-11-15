from __future__ import annotations

import asyncio

import numpy as np

from fl.core.client import Client
from fl.core.protocols import ModelWeights, SecurityProtocol


class MockSecurityProtocol(SecurityProtocol):
    """Mock security protocol for testing."""

    def encrypt(self, data: ModelWeights) -> bytes:
        return b"encrypted"

    def decrypt(self, data: bytes) -> ModelWeights:
        return {
            "layer1": np.zeros((10, 10), dtype=np.float32),
            "layer2": np.zeros((10, 1), dtype=np.float32),
        }


class MockClient(Client):
    """Mock implementation of Client for testing."""

    async def _train_batch(
        self, batch_data: np.ndarray, batch_labels: np.ndarray
    ) -> None:
        """Mock implementation of training batch."""
        await asyncio.sleep(0.01)
