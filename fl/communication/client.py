import asyncio
import base64
import json
from dataclasses import asdict
from typing import Any, AsyncGenerator, Final
from urllib.parse import urlparse

import aiohttp
from websockets.client import connect
from websockets.exceptions import WebSocketException
from websockets.legacy.client import WebSocketClientProtocol

from fl.communication.endpoints import ClientCapabilities, ClientRegistration
from fl.communication.protocols import Message, MessageType, NumpyJSONEncoder
from fl.config.logging import get_logger
from fl.core.exceptions import (
    CommunicationError,
    SecurityError,
)
from fl.core.protocols import ModelUpdate
from fl.security.encryption import SecurityMiddleware
from fl.utils.common import retry_async, timeout_context


class ClientCommunication:
    """Client-side communication handler."""

    __slots__ = (
        "_server_url",
        "_client_id",
        "_api_key",
        "_security",
        "_ws",
        "_heartbeat_task",
        "_connected",
        "_lock",
        "_session",
        "_message_queue",
        "_max_retries",
        "_retry_delay",
        "_logger",
    )

    HEARTBEAT_INTERVAL: Final[int] = 20
    PING_TIMEOUT: Final[int] = 10
    MAX_QUEUE_SIZE: Final[int] = 1000

    def __init__(
        self,
        server_url: str,
        client_id: str,
        api_key: str,
        security: SecurityMiddleware | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._client_id = client_id
        self._api_key = api_key
        self._security = security
        self._ws: WebSocketClientProtocol | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._connected: bool = False
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue(
            maxsize=self.MAX_QUEUE_SIZE
        )
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Setup logger with context
        self._logger = get_logger(
            "ClientCommunication", context={"client_id": client_id}
        )

    def _get_ws_url(self) -> str:
        """Convert HTTP URL to WebSocket URL."""
        parsed = urlparse(self._server_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{ws_scheme}://{parsed.netloc}/api/ws/{self._client_id}"

    async def _register_client(self) -> None:
        """Register client with server before establishing connection."""
        try:
            if self._security:
                signature_bytes = self._security.create_signature(
                    self._client_id.encode()
                )
                signature = base64.b64encode(signature_bytes).decode("utf-8")
            else:
                signature = ""

            registration = ClientRegistration(
                client_id=self._client_id,
                signature=signature,
                capabilities=ClientCapabilities(version="1.0"),
            )

            async with timeout_context(30):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self._server_url}/api/register",
                        headers={"X-API-Key": self._api_key},
                        json=registration.model_dump(),
                    ) as response:
                        if response.status != 200:
                            raise CommunicationError(
                                f"Registration failed with status "
                                f"{response.status}",
                                details={"response": await response.text()},
                            )
                        return await response.json()
        except asyncio.TimeoutError as e:
            raise CommunicationError(
                "Registration timed out", details={"error": str(e)}
            )
        except Exception as e:
            raise CommunicationError(
                "Registration failed", details={"error": str(e)}
            )

    async def connect(self) -> None:
        async with self._lock:
            if self._connected:
                return

            self._logger.info("Attempting to connect to server")

            async def connect_with_retry() -> None:
                await self._register_client()

                # Connect WebSocket
                ws_url = self._get_ws_url()
                self._ws = await connect(  # type: ignore
                    ws_url,
                    extra_headers={"X-API-Key": self._api_key},
                    ping_interval=self.HEARTBEAT_INTERVAL,
                    ping_timeout=self.PING_TIMEOUT,
                )

                # Wait for connection confirmation
                response = await self._ws.recv()
                data = json.loads(response)
                if data["type"] != "CONNECTION_ACCEPTED":
                    raise CommunicationError(
                        "Connection rejected", details={"response": data}
                    )

            try:
                await retry_async(
                    connect_with_retry,
                    max_attempts=self._max_retries,
                    delay=self._retry_delay,
                    exceptions=(CommunicationError, WebSocketException),
                )

                # Start message handler and heartbeat
                self._start_heartbeat()
                self._connected = True
                self._logger.info("Successfully connected to server")

            except Exception as e:
                self._logger.error(
                    "Failed to connect",
                    extra={"error": str(e), "attempts": self._max_retries},
                )
                raise CommunicationError(
                    f"Failed to connect after {self._max_retries} attempts",
                    details={"error": str(e)},
                )

    def _start_heartbeat(self) -> None:
        """Start background heartbeat task."""

        async def heartbeat() -> None:
            while True:
                try:
                    if self._ws and not self._ws.closed:
                        await self._ws.ping()
                        await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                    else:
                        break
                except Exception as e:
                    self._logger.warning(f"Heartbeat failed: {str(e)}")
                    self._connected = False
                    break

        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(heartbeat())
            self._logger.debug("Started heartbeat task")

    async def submit_update(self, update: ModelUpdate) -> dict:
        """Submit model update to server."""
        if not self._ws or not self._connected:
            raise CommunicationError("Not connected to server")

        try:
            if self._security:
                try:
                    encrypted_weights = self._security.encrypt_message(
                        json.dumps(
                            update.weights, cls=NumpyJSONEncoder
                        ).encode()
                    )
                    update_dict = asdict(update) | {
                        "weights": encrypted_weights
                    }
                except Exception as e:
                    raise SecurityError(
                        "Failed to encrypt weights", details={"error": str(e)}
                    )
            else:
                update_dict = asdict(update)

            message = Message(
                message_type=MessageType.MODEL_UPDATE,
                payload={"update": update_dict},
                sender_id=self._client_id,
                round_number=update.round_number,
                timestamp=None,
            )

            self._logger.debug(
                "Submitting update", extra={"round": update.round_number}
            )

            async with timeout_context(30):
                await self._ws.send(message.to_json())
                response = await self._ws.recv()
                return json.loads(response)

        except WebSocketException as e:
            self._connected = False
            raise CommunicationError(
                "WebSocket error during update", details={"error": str(e)}
            )
        except asyncio.TimeoutError:
            raise CommunicationError("Update submission timed out")
        except SecurityError as e:
            raise e
        except Exception as e:
            raise CommunicationError(
                "Failed to submit update", details={"error": str(e)}
            )

    async def receive_messages(self) -> AsyncGenerator[Message, None]:
        """Recieve and decrypt messages from server."""
        if not self._ws:
            raise CommunicationError("Not connected to server")

        while True:
            try:
                data = await self._ws.recv()
                message = Message.from_json(data)

                # Decrypt payload if needed and security as configured
                if (
                    self._security
                    and message.message_type == MessageType.GLOBAL_MODEL
                    and isinstance(message.payload.get("weights"), bytes)
                ):
                    try:
                        weights = message.payload["weights"]
                        if not isinstance(weights, bytes):
                            raise SecurityError(
                                "Weights must be bytes for decryption"
                            )
                        decrypted_weights = self._security.decrypt_message(
                            weights
                        )
                        message.payload["weights"] = json.loads(
                            decrypted_weights
                        )
                    except Exception as e:
                        raise SecurityError(
                            "Failed to decrypt weights",
                            details={"error": str(e)},
                        )

                self._logger.debug(
                    "Received message",
                    extra={
                        "type": message.message_type.name,
                        "round": message.round_number,
                    },
                )
                yield message
            except asyncio.CancelledError:
                self._logger.warning("Receive messages cancelled")
                break
            except WebSocketException:
                self._connected = False
                self._logger.warning("WebSocket disconnected")
                break
            except Exception as e:
                self._logger.error(f"Error receiving message: {str(e)}")
                raise CommunicationError(
                    "Error receiving message", details={"error": str(e)}
                )

    async def disconnect(self) -> None:
        """Clean disconnect from server."""
        async with self._lock:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None

            if self._ws:
                await self._ws.close()
                self._ws = None

            if self._session:
                await self._session.close()
                self._session = None

            self._connected = False
            self._logger.info("Disconnected from server")

    @property
    def is_connected(self) -> bool:
        return bool(self._connected and self._ws and not self._ws.closed)

    async def __aenter__(self) -> "ClientCommunication":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        await self.disconnect()
