from unittest.mock import MagicMock

import pytest
import torch
from aiohttp.test_utils import AioHTTPTestCase

from nanofed import HTTPServer


class TestHTTPServer(AioHTTPTestCase):
    async def get_application(self):
        """Create application for testing."""
        # Setup mock model manager
        manager = MagicMock()
        version = MagicMock()
        version.version_id = "test_version"
        manager.current_version = version
        manager._model = MagicMock()
        manager._model.state_dict.return_value = {
            "fc.weight": torch.tensor([[1.0, 1.0]]),
            "fc.bias": torch.tensor([0.5]),
        }

        # Create server
        server = HTTPServer("localhost", 8080, manager)
        return server._app

    @pytest.mark.asyncio
    async def test_get_model(self):
        """Test model endpoint."""
        resp = await self.client.get("/model")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "success"
        assert "model_state" in data
