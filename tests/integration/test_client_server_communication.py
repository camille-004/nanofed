from pathlib import Path

import pytest

from nanofed import HTTPClient, HTTPServer, ModelManager
from nanofed.core import NanoFedError
from nanofed.models import MNISTModel


@pytest.fixture
def test_config(tmp_path: Path):
    """Create test configuration."""
    config = {
        "name": "test_model",
        "version": "1.0",
        "architecture": {"type": "cnn"},
    }
    return config


@pytest.mark.asyncio
async def test_model_exchange(tmp_path: Path, test_config):
    """Test model exchange between client and server."""
    # Create directories
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Setup server
    model = MNISTModel()
    model_manager = ModelManager(model_dir, model)
    model_manager.save_model(test_config)

    server = HTTPServer("localhost", 8080, model_manager)
    await server.start()

    try:
        async with HTTPClient(
            "http://localhost:8080", "test_client"
        ) as client:
            model_state, round_num = await client.fetch_global_model()
            assert round_num == 0
            assert isinstance(model_state, dict)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_error_handling():
    """Test server error handling scenarios."""
    server = HTTPServer("localhost", 8080, None)  # Invalid setup
    await server.start()

    try:
        async with HTTPClient(
            "http://localhost:8080", "test_client"
        ) as client:
            with pytest.raises(NanoFedError):
                await client.fetch_global_model()
    finally:
        await server.stop()
