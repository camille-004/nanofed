from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
import torch

from nanofed.communication.http.types import (
    ClientModelUpdateRequest,
    GlobalModelResponse,
    ServerModelUpdateRequest,
)
from nanofed.core.exceptions import NanoFedError
from nanofed.core.interfaces import ModelProtocol
from nanofed.utils.logger import Logger, log_exec


@dataclass(slots=True, frozen=True)
class ClientEndpoints:
    """Client endpoint configuration."""

    get_model: str = "/model"
    submit_update: str = "/update"
    get_status: str = "/status"


class HTTPClient:
    """HTTP client for communication."""

    def __init__(
        self,
        server_url: str,
        client_id: str,
        endpoints: ClientEndpoints | None = None,
        timeout: int = 300,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._client_id = client_id
        self._endpoints = endpoints or ClientEndpoints()
        self._logger = Logger()
        self._timeout = timeout

        # State tracking
        self._current_round: int = 0
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "HTTPClient":
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _get_url(self, endpoint: str) -> str:
        return f"{self._server_url}{endpoint}"

    @log_exec
    async def fetch_global_model(self) -> tuple[dict[str, torch.Tensor], int]:
        """Fetch current global model from server."""
        with self._logger.context("client.http", self._client_id):
            if self._session is None:
                raise NanoFedError("Client session not initialized")

            try:
                url = self._get_url(self._endpoints.get_model)
                async with self._session.get(url) as response:
                    if response.status != 200:
                        raise NanoFedError(
                            f"Server error while fetching model: {response.status}"  # noqa
                        )

                    data: GlobalModelResponse = await response.json()

                    if "status" not in data or data["status"] != "success":
                        raise NanoFedError(
                            f"Error from server: {data.get('message', 'Unknown error')}"  # noqa
                        )

                    if "model_state" not in data or "round_number" not in data:
                        raise NanoFedError(
                            "Invalid server response: missing required fields"
                        )

                    model_state = {
                        key: torch.tensor(value)
                        for key, value in data["model_state"].items()
                    }

                    self._current_round = data["round_number"]
                    return model_state, self._current_round

            except aiohttp.ClientError as e:
                raise NanoFedError(f"HTTP error: {str(e)}")
            except Exception as e:
                raise NanoFedError(f"Failed to fetch global model: {str(e)}")

    @log_exec
    async def submit_update(
        self, model: ModelProtocol, metrics: dict[str, float]
    ) -> bool:
        """Submit model udpate to server."""
        with self._logger.context("client.http", self._client_id):
            if self._session is None:
                raise NanoFedError("Client session not initialized")

            try:
                state_dict = model.state_dict()
                model_state = {
                    key: value.cpu().numpy().tolist()
                    for key, value in state_dict.items()
                }

                update: ClientModelUpdateRequest = {
                    "client_id": self._client_id,
                    "round_number": self._current_round,
                    "model_state": model_state,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                }

                url = self._get_url(self._endpoints.submit_update)
                async with self._session.post(url, json=update) as response:
                    if response.status != 200:
                        raise NanoFedError(f"Server error: {response.status}")

                    data: ServerModelUpdateRequest = await response.json()

                    if data["status"] != "success":
                        raise NanoFedError(
                            f"Error from server: {data['message']}"
                        )

                    return data["accepted"]

            except aiohttp.ClientError as e:
                raise NanoFedError(f"HTTP error: {str(e)}")
            except Exception as e:
                raise NanoFedError(f"Failed to submit update: {str(e)}")

    async def check_server_status(self) -> bool:
        if self._session is None:
            raise NanoFedError("Client session not initialized")

        try:
            url = self._get_url(self._endpoints.get_status)
            async with self._session.get(url) as response:
                if response.status != 200:
                    raise NanoFedError(
                        f"Failed to fetch server status: {response.status}"
                    )

                data = await response.json()
                return data.get("is_training_done", False)

        except aiohttp.ClientError as e:
            raise NanoFedError(f"HTTP error: {str(e)}")
