# tests/unit/test_http.py

import torch
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from nanofed.communication.http.client import HTTPClient
from tests.unit.helpers import SimpleModel


class TestHTTPClient(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        """Create test application with mock endpoints."""

        async def mock_get_model(request: web.Request) -> web.Response:
            return web.json_response(
                {
                    "status": "success",
                    "message": "Model retrieved",
                    "timestamp": "2024-01-01T00:00:00",
                    "model_params": {
                        "fc.weight": [[1.0, 2.0], [3.0, 4.0]],
                        "fc.bias": [0.1, 0.2],
                    },
                    "round_number": 1,
                    "version_id": "test_v1",
                }
            )

        async def mock_submit_update(request: web.Request) -> web.Response:
            return web.json_response(
                {
                    "status": "success",
                    "message": "Update accepted",
                    "timestamp": "2024-01-01T00:00:00",
                    "update_id": "test_update_1",
                    "accepted": True,
                }
            )

        app = web.Application()
        app.router.add_get("/model", mock_get_model)
        app.router.add_post("/update", mock_submit_update)
        return app

    async def test_fetch_global_model(self):
        client = HTTPClient("", "test_client")
        client._session = self.client

        model_params, round_num = await client.fetch_global_model()

        assert isinstance(model_params, dict)
        assert isinstance(round_num, int)
        assert round_num == 1

        assert "fc.weight" in model_params
        assert "fc.bias" in model_params

        for param in model_params.values():
            assert isinstance(param, torch.Tensor)

    async def test_submit_update(self):
        client = HTTPClient("", "test_client")
        client._session = self.client

        model = SimpleModel()
        metrics = {"loss": 0.5, "accuracy": 0.95}

        success = await client.submit_update(model, metrics)
        assert success is True
