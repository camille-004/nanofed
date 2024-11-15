from __future__ import annotations

from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, status
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from fl.communication.endpoints import (
    ClientCapabilities,
    ClientRegistration,
    get_server,
    router,
    verify_api_key,
)
from fl.config.logging import get_logger
from fl.core.server import ServerState

logger = get_logger(__name__)


async def override_verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    """Override API key verification for testing."""
    if x_api_key != "test-api-key":
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return x_api_key


@pytest.fixture
def mock_server():
    """Fixture for mock server."""
    server = MagicMock()
    server._state = ServerState.READY
    server._active_clients = set()
    server._current_round = None
    server._clients = {}

    # Mock get_status
    server.get_status.return_value = {
        "status": "ready",
        "active_clients": 0,
        "current_round": None,
    }

    # Mock register_client
    async def mock_register(*args, **kwargs):
        return {
            "status": "success",
            "client_id": "test-client",
            "round": 0,
            "server_time": "2024-11-14T00:00:00",
        }

    server.register_client = AsyncMock(side_effect=mock_register)

    return server


@pytest.fixture
def app_with_overrides(mock_server):
    """Fixture for FastAPI test application with dependency overrides."""
    app = FastAPI()
    app.include_router(router)

    async def override_get_server(request: Request | WebSocket):
        return mock_server

    app.dependency_overrides[verify_api_key] = override_verify_api_key
    app.dependency_overrides[get_server] = override_get_server
    app.state.server = mock_server
    return app


@pytest.fixture
def client_registration():
    """Fixture for test client registration."""
    return ClientRegistration(
        client_id="test-client",
        signature="test-signature",
        capabilities=ClientCapabilities(
            version="1.0.0", batch_size=32, local_epochs=1
        ),
    )


@pytest.mark.anyio
async def test_register_client(
    app_with_overrides, client_registration, mock_server
):
    """Test client registration endpoint."""
    transport = ASGITransport(app=app_with_overrides)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/register",
            headers={"X-API-Key": "test-api-key"},
            json=client_registration.model_dump(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["client_id"] == "test-client"
        assert data["status"] == "success"

        mock_server.register_client.assert_called_once_with(
            client_id=client_registration.client_id,
            signature=client_registration.signature,
            capabilities=client_registration.capabilities.model_dump(),
        )


@pytest.mark.anyio
async def test_invalid_api_key(app_with_overrides, client_registration):
    """Test invalid API key handling."""
    transport = ASGITransport(app=app_with_overrides)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/register",
            headers={"X-API-Key": "invalid-key"},
            json=client_registration.model_dump(),
        )
        assert response.status_code == 403


@pytest.mark.anyio
async def test_server_status(app_with_overrides):
    """Test server status endpoint."""
    transport = ASGITransport(app=app_with_overrides)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get(
            "/api/server", headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "active_clients" in data


def test_websocket_connection(app_with_overrides, mock_server):
    """Test WebSocket connection."""
    with patch("fl.communication.endpoints.settings") as mock_settings:
        mock_settings.api_key.get_secret_value.return_value = "test-api-key"

        client = TestClient(app_with_overrides)

        with client.websocket_connect(
            "/api/ws/test-client", headers={"X-API-Key": "test-api-key"}
        ) as websocket:
            data = websocket.receive_json()
            assert data["type"] == "CONNECTION_ACCEPTED"
            assert data["client_id"] == "test-client"


@pytest.mark.anyio
async def test_validation_error(app_with_overrides):
    """Test validation error handling."""
    invalid_registration = {
        "client_id": "",  # Invalid: empty string
        "signature": "test",
        "capabilities": {"version": "invalid"},
    }

    transport = ASGITransport(app=app_with_overrides)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/register",
            headers={"X-API-Key": "test-api-key"},
            json=invalid_registration,
        )
        assert response.status_code == 422
