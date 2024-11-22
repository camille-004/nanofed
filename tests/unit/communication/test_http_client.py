from typing import Any

import pytest
import torch

from nanofed.communication.http import HTTPClient
from nanofed.core.exceptions import NanoFedError


class AsyncContextManagerMock:
    """Async context manager wrapper."""

    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        pass


class MockResponse:
    """Mock HTTP response."""

    def __init__(self, status: int = 200, data: Any = None):
        self.status = status
        self._data = data

    async def json(self):
        return self._data


class MockSession:
    """Mock aiohttp ClientSession."""

    def __init__(self, get_response=None, post_response=None):
        self._get_response = get_response
        self._post_response = post_response

    def get(self, url: str, **kwargs) -> AsyncContextManagerMock:
        return AsyncContextManagerMock(self._get_response)

    def post(self, url: str, **kwargs) -> AsyncContextManagerMock:
        return AsyncContextManagerMock(self._post_response)

    async def close(self):
        pass


@pytest.fixture
def mock_responses():
    """Create mock responses."""
    get_response = MockResponse(
        status=200,
        data={
            "status": "success",
            "message": "OK",
            "model_state": {"fc.weight": [[1.0, 1.0]], "fc.bias": [0.5]},
            "round_number": 1,
            "version_id": "test_version",
        },
    )

    post_response = MockResponse(
        status=200,
        data={
            "status": "success",
            "message": "Update accepted",
            "accepted": True,
        },
    )

    return {"get": get_response, "post": post_response}


@pytest.fixture
def test_client(mock_responses):
    """Create test client with mocked session."""
    session = MockSession(
        get_response=mock_responses["get"],
        post_response=mock_responses["post"],
    )

    client = HTTPClient("http://test", "client1")
    client._session = session
    return client


@pytest.mark.asyncio
async def test_client_fetch_global_model(test_client):
    """Test fetching global model from server."""
    model_state, round_num = await test_client.fetch_global_model()
    assert round_num == 1
    assert "fc.weight" in model_state
    assert "fc.bias" in model_state


@pytest.mark.asyncio
async def test_client_submit_update(test_client):
    """Test submitting model update to server."""

    class MockModel:
        def state_dict(self):
            return {
                "fc.weight": torch.tensor([[1.0, 1.0]]),
                "fc.bias": torch.tensor([0.5]),
            }

    success = await test_client.submit_update(
        MockModel(), {"loss": 0.5, "accuracy": 0.95}
    )
    assert success is True


@pytest.mark.asyncio
async def test_client_fetch_global_model_error(test_client, mock_responses):
    """Test error handling when fetching model."""
    mock_responses["get"].status = 500

    with pytest.raises(NanoFedError) as exc_info:
        await test_client.fetch_global_model()

    assert "Server error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_submit_update_error(test_client, mock_responses):
    """Test error handling when submitting update."""
    mock_responses["post"].status = 500

    class MockModel:
        def state_dict(self):
            return {
                "fc.weight": torch.tensor([[1.0, 1.0]]),
                "fc.bias": torch.tensor([0.5]),
            }

    with pytest.raises(NanoFedError) as exc_info:
        await test_client.submit_update(
            MockModel(), {"loss": 0.5, "accuracy": 0.95}
        )

    assert "Server error" in str(exc_info.value)
