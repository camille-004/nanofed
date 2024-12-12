from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from aiohttp.test_utils import AioHTTPTestCase

from nanofed import HTTPServer
from nanofed.core import ModelVersion


class TestHTTPServer(AioHTTPTestCase):
    async def get_application(self):
        """Create application for testing."""
        # Create a concrete model state dict
        state_dict = {
            "fc.weight": torch.tensor([[1.0, 1.0]]).tolist(),
            "fc.bias": torch.tensor([0.5]).tolist(),
        }

        # Create model mock with a valid state_dict
        class MockModel:
            def state_dict(self):
                return state_dict

        # Create a valid ModelVersion instance
        version = ModelVersion(
            version_id="test_version",
            timestamp=datetime.now(timezone.utc),
            config={"dummy": "config"},
            path=Path("/path/to/mock/model.pt"),  # Mock path
        )

        # Create model manager mock
        model_manager = MagicMock()
        model_manager.current_version = version
        model_manager.model = MockModel()

        # Create a coordinator mock
        coordinator = MagicMock()
        coordinator.model_manager = model_manager

        # Setup server
        server = HTTPServer("localhost", 8080)
        server.set_coordinator(coordinator)

        return server._app

    @pytest.mark.asyncio
    async def test_get_model(self):
        """Test model endpoint."""
        resp = await self.client.get("/model")
        assert resp.status == 200

        data = await resp.json()
        assert data["status"] == "success"
        assert "model_state" in data

        model_state = data["model_state"]
        assert "fc.weight" in model_state
        assert "fc.bias" in model_state
        assert model_state["fc.weight"] == [[1.0, 1.0]]
        assert model_state["fc.bias"] == [0.5]
