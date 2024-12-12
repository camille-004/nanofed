from pathlib import Path

import pytest

from nanofed import (
    Coordinator,
    CoordinatorConfig,
    HTTPClient,
    HTTPServer,
    ModelManager,
)
from nanofed.core import NanoFedError
from nanofed.models import MNISTModel
from nanofed.server import BaseAggregator


class MockAggregator(BaseAggregator):
    def aggregate(self, updates):
        return updates[0]

    def _compute_weights(self, updates):
        return [1.0] * len(updates)


@pytest.fixture
def test_config(tmp_path: Path):
    """Create test configuration."""
    return CoordinatorConfig(
        num_rounds=1,
        min_clients=1,
        min_completion_rate=1.0,
        round_timeout=10,
        base_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_model_exchange(tmp_path: Path, test_config):
    """Test model exchange between client and server."""
    # Setup server
    model = MNISTModel()
    model_manager = ModelManager(model)
    server = HTTPServer("localhost", 8080)
    aggregator = MockAggregator()

    coordinator = Coordinator(  # noqa
        model_manager=model_manager,
        aggregator=aggregator,
        server=server,
        config=test_config,
    )

    await server.start()

    try:
        async with HTTPClient(server.url, "test_client") as client:
            model_state, round_num = await client.fetch_global_model()
            assert round_num == 0
            assert isinstance(model_state, dict)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_error_handling():
    """Test server error handling scenarios."""
    server = HTTPServer("localhost", 8080)  # Server without coordinator
    await server.start()

    try:
        async with HTTPClient(server.url, "test_client") as client:
            with pytest.raises(NanoFedError):
                await client.fetch_global_model()
    finally:
        await server.stop()
