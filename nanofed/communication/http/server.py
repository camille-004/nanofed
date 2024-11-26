import asyncio
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import torch
from aiohttp import web

from nanofed.communication.http.types import (
    GlobalModelResponse,
    ModelUpdateResponse,
    ServerModelUpdateRequest,
)
from nanofed.server import ModelManager
from nanofed.utils import Logger, get_current_time

TensorLike: TypeAlias = (
    torch.Tensor | list[float] | list[list[float]] | float | int
)
ModelState: TypeAlias = dict[str, list[float] | list[list[float]]]


@dataclass(slots=True, frozen=True)
class ServerEndpoints:
    """Server endpoint configuration."""

    get_model: str = "/model"
    submit_update: str = "/update"
    get_status: str = "/status"


class HTTPServer:
    """HTTP server for FL coordination.

    Manages client connections, model distribution, and update collection using
    HTTP protocol.

    Parameters
    ----------
    host : str
        Server host address.
    port : int
        Server port number.
    model_manager : ModelManager
        Manager for global model versions.
    endpoints : ServerEndpoints, optional
        Custom endpoint configuration.
    max_request_size : int, default=104857600
        Maximum allowed request size in bytes (default: 100MB).

    Attributes
    ----------
    _current_round : int
        Current training round number.
    _updates : dict
        Dictionary of received client updates.
    _is_training_done : bool
        Flag indicating training completion.
    """

    def __init__(
        self,
        host: str,
        port: int,
        model_manager: ModelManager,
        endpoints: ServerEndpoints | None = None,
        max_request_size: int = 100 * 1024 * 1024,  # 100MB default
    ) -> None:
        self._host = host
        self._port = port
        self._model_manager = model_manager
        self._endpoints = endpoints or ServerEndpoints()
        self._logger = Logger()
        self._app = web.Application(client_max_size=max_request_size)
        self._setup_routes()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

        # State tracking
        self._current_round: int = 0
        self._updates: dict[str, ServerModelUpdateRequest] = {}
        self._lock = asyncio.Lock()
        self._is_training_done = False

    def _setup_routes(self) -> None:
        self._logger.debug("Setting up routes for HTTP server.")
        self._app.router.add_get(
            self._endpoints.get_model, self._handle_get_model
        )
        self._app.router.add_post(
            self._endpoints.submit_update, self._handle_submit_update
        )
        self._app.router.add_get(
            self._endpoints.get_status, self._handle_get_status
        )
        self._app.router.add_get("/test", self._handle_test)

    def _convert_tensor(
        self, value: TensorLike
    ) -> list[float] | list[list[float]]:
        if isinstance(value, torch.Tensor):
            arr = value.cpu().detach().numpy()
            return cast(list[float] | list[list[float]], arr.tolist())
        elif isinstance(value, (list, np.ndarray)):
            return value
        elif isinstance(value, (int, float)):
            return [float(value)]

    async def _handle_test(self, request: web.Request) -> web.Response:
        return web.Response(text="Server is running")

    async def _handle_get_model(self, request: web.Request) -> web.Response:
        """Handle request for global model."""
        with self._logger.context("server.http", "get_model"):
            self._logger.debug("Processing /model request.")
            try:
                if self._is_training_done:
                    self._logger.info(
                        "Training complete. Sending termination signal."
                    )
                    return web.json_response(
                        {
                            "status": "terminated",
                            "message": "Training is complete",
                            "timestamp": get_current_time().isoformat(),
                            "model_state": None,
                            "round_number": -1,
                        }
                    )

                version = self._model_manager.current_version
                if version is None:
                    version = self._model_manager.load_model()

                self._logger.debug(
                    f"Serving model version {version.version_id}"
                )

                state_dict = self._model_manager._model.state_dict()
                model_state = {
                    key: self._convert_tensor(value)
                    for key, value in state_dict.items()
                }

                response: GlobalModelResponse = {
                    "status": "success",
                    "message": "Global model retrieved",
                    "timestamp": get_current_time().isoformat(),
                    "model_state": model_state,
                    "round_number": self._current_round,
                    "version_id": version.version_id,
                }
                self._logger.debug(
                    f"Model response prepared for version {version.version_id}"
                )
                return web.json_response(response)

            except Exception as e:
                self._logger.error(f"Error serving model: {str(e)}")
                return web.json_response(
                    {
                        "status": "error",
                        "message": str(e),
                        "timestamp": get_current_time().isoformat(),
                    },
                    status=500,
                )

    async def _handle_submit_update(
        self, request: web.Request
    ) -> web.Response:
        """Handle model update submission from client."""
        with self._logger.context("server.http", "submit_update"):
            self._logger.debug("Processing /update request.")
            try:
                data: dict[str, Any] = await request.json()

                required_keys = {
                    "client_id",
                    "round_number",
                    "model_state",
                    "metrics",
                    "timestamp",
                }
                if not required_keys.issubset(data.keys()):
                    missing_keys = required_keys - data.keys()
                    return web.json_response(
                        {
                            "status": "error",
                            "message": f"Missing keys: {', '.join(missing_keys)}",  # noqa
                            "timestamp": get_current_time().isoformat(),
                        },
                        status=400,
                    )

                update: ServerModelUpdateRequest = {
                    "client_id": data["client_id"],
                    "round_number": data["round_number"],
                    "model_state": data["model_state"],
                    "metrics": data["metrics"],
                    "timestamp": data["timestamp"],
                    "status": data.get("status", "success"),
                    "message": data.get("mesage", ""),
                    "accepted": data.get("accepted", True),
                }

                async with self._lock:
                    if update["round_number"] != self._current_round:
                        self._logger.warning(
                            f"Udpate round mismatch: expected {self._current_round}, "  # noqa
                            f"got {update['round_number']} from client {update['client_id']}"  # noqa
                        )
                        return web.json_response(
                            {
                                "status": "error",
                                "message": "Invalid round number",
                                "timestamp": get_current_time().isoformat(),
                            },
                            status=400,
                        )

                    client_id = update["client_id"]
                    self._updates[client_id] = update
                    self._logger.info(
                        f"Accepted update from client {client_id} for round in {self._current_round}"  # noqa
                    )

                    response: ModelUpdateResponse = {
                        "status": "success",
                        "message": "Updated accepted",
                        "timestamp": get_current_time().isoformat(),
                        "update_id": f"update_{client_id}_{self._current_round}",  # noqa
                        "accepted": True,
                    }
                    return web.json_response(response)

            except Exception as e:
                self._logger.error(f"Error handling update: {str(e)}")
                return web.json_response(
                    {
                        "status": "error",
                        "message": str(e),
                        "timestamp": get_current_time().isoformat(),
                    },
                    status=500,
                )

    async def _handle_get_status(self, request: web.Request) -> web.Response:
        self._logger.info("Processing /status request.")
        return web.json_response(
            {
                "status": "success",
                "message": "Server is running",
                "timestamp": get_current_time().isoformat(),
                "current_round": self._current_round,
                "num_updates": len(self._updates),
                "is_training_done": self._is_training_done,
            }
        )

    async def stop_training(self) -> None:
        self._is_training_done = True
        self._logger.info(
            "Training completed. Broadcasting termination signal to clients."
        )

    async def start(self) -> None:
        """Start HTTP server."""
        self._logger.info("Starting HTTP server...")
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner,
            self._host,
            self._port,
            reuse_address=True,
            reuse_port=True,
        )
        await self._site.start()
        self._logger.info(f"HTTP server started on {self._host}:{self._port}")

    async def stop(self) -> None:
        """Stop HTTP server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._logger.info("Server stopped")
