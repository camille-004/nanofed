import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from websockets.exceptions import WebSocketException

from fl.communication.client import ClientCommunication
from fl.core.exceptions import CommunicationError, SecurityError
from fl.core.protocols import ModelUpdate
from fl.security.encryption import SecurityMiddleware


@pytest.fixture
def model_update():
    weights = {"layer1": [1.0] * 10, "layer2": [2.0] * 5}
    return ModelUpdate(
        client_id="test-client",
        weights=weights,
        round_metrics={"loss": 0.1, "accuracy": 0.9},
        round_number=0,
    )


@pytest.fixture
def security_middleware():
    return SecurityMiddleware(Fernet.generate_key())


@pytest.mark.asyncio
async def test_client_registration(security_middleware):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_post.return_value.__aenter__.return_value = mock_response

        client_comm = ClientCommunication(
            server_url="http://localhost:8000",
            client_id="test-client",
            api_key="test-api-key",
            security=security_middleware,
        )
        await client_comm._register_client()
        mock_post.assert_called_once()


@patch("fl.communication.client.connect", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_websocket_connection(mock_connect, security_middleware):
    with patch("aiohttp.ClientSession.post") as mock_post:
        # Mock the HTTP POST request
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_post.return_value.__aenter__.return_value = mock_response

        # Mock the WebSocket connection
        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            return_value='{"type": "CONNECTION_ACCEPTED"}'
        )
        mock_ws.send = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.closed = False

        mock_connect.return_value = mock_ws

        client_comm = ClientCommunication(
            server_url="http://localhost:8000",
            client_id="test-client",
            api_key="test-api-key",
            security=security_middleware,
        )
        await client_comm.connect()
        assert client_comm.is_connected
        await client_comm.disconnect()
        assert not client_comm.is_connected


@patch("fl.communication.client.connect", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_submit_update(mock_connect, model_update, security_middleware):
    with patch("aiohttp.ClientSession.post") as mock_post:
        # Mock the HTTP POST request
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_post.return_value.__aenter__.return_value = mock_response

        # Mock the WebSocket connection
        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                '{"type": "CONNECTION_ACCEPTED"}',
                '{"status": "success"}',
            ]
        )
        mock_ws.send = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.closed = False

        mock_connect.return_value = mock_ws

        client_comm = ClientCommunication(
            server_url="http://localhost:8000",
            client_id="test-client",
            api_key="test-api-key",
            security=security_middleware,
        )
        await client_comm.connect()
        response = await client_comm.submit_update(model_update)
        assert response["status"] == "success"
        mock_ws.send.assert_called_once()
        await client_comm.disconnect()


@patch("fl.communication.client.connect", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_receive_messages(mock_connect, security_middleware):
    messages = [
        '{"type": "CONNECTION_ACCEPTED"}',
        '{"message_type": "GLOBAL_MODEL", "payload": {"weights": {}, "round": 1}, "sender_id": "server", "round_number": 1}',  # noqa
        '{"message_type": "ROUND_START", "payload": {"round": 1, "deadline": "2024-11-14T10:00:00Z"}, "sender_id": "server", "round_number": 1}',  # noqa
    ]

    with patch("aiohttp.ClientSession.post") as mock_post:
        # Mock the HTTP POST request
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_post.return_value.__aenter__.return_value = mock_response

        # Mock the WebSocket connection
        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            side_effect=messages + [WebSocketException("Test Exception")]
        )
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.closed = False

        mock_connect.return_value = mock_ws

        client_comm = ClientCommunication(
            server_url="http://localhost:8000",
            client_id="test-client",
            api_key="test-api-key",
            security=security_middleware,
        )
        await client_comm.connect()
        received = []
        try:
            async for message in client_comm.receive_messages():
                received.append(message)
        except CommunicationError:
            pass
        assert len(received) == 2
        await client_comm.disconnect()


@patch("fl.communication.client.connect", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_heartbeat(mock_connect, security_middleware):
    with patch("aiohttp.ClientSession.post") as mock_post:
        # Mock the HTTP POST request
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_post.return_value.__aenter__.return_value = mock_response

        # Mock the WebSocket connection
        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            return_value='{"type": "CONNECTION_ACCEPTED"}'
        )
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.closed = False

        mock_connect.return_value = mock_ws

        client_comm = ClientCommunication(
            server_url="http://localhost:8000",
            client_id="test-client",
            api_key="test-api-key",
            security=security_middleware,
        )
        await client_comm.connect()
        await asyncio.sleep(client_comm.HEARTBEAT_INTERVAL + 1)
        mock_ws.ping.assert_called()
        await client_comm.disconnect()


@patch("fl.communication.client.connect", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_handle_timeout(mock_connect, model_update, security_middleware):
    with patch("aiohttp.ClientSession.post") as mock_post:
        # Mock the HTTP POST request
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_post.return_value.__aenter__.return_value = mock_response

        # Mock the WebSocket connection
        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                '{"type": "CONNECTION_ACCEPTED"}',
                asyncio.TimeoutError(),
            ]
        )
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.closed = False

        mock_connect.return_value = mock_ws

        client_comm = ClientCommunication(
            server_url="http://localhost:8000",
            client_id="test-client",
            api_key="test-api-key",
            security=security_middleware,
        )
        await client_comm.connect()
        with pytest.raises(CommunicationError):
            await client_comm.submit_update(model_update)
        await client_comm.disconnect()


@patch("fl.communication.client.connect", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_security_error(mock_connect, model_update, security_middleware):
    with patch("aiohttp.ClientSession.post") as mock_post:
        # Mock the HTTP POST request
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_post.return_value.__aenter__.return_value = mock_response

        # Mock the WebSocket connection
        mock_ws = MagicMock()
        mock_ws.recv = AsyncMock(
            return_value='{"type": "CONNECTION_ACCEPTED"}'
        )
        mock_ws.send = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.closed = False

        mock_connect.return_value = mock_ws

        client_comm = ClientCommunication(
            server_url="http://localhost:8000",
            client_id="test-client",
            api_key="test-api-key",
            security=security_middleware,
        )
        # Mock the encrypt_message method on the security middleware
        with patch.object(
            SecurityMiddleware,
            "encrypt_message",
            side_effect=Exception("Encryption failed"),
        ):
            await client_comm.connect()
            with pytest.raises(SecurityError):
                await client_comm.submit_update(model_update)
            await client_comm.disconnect()
