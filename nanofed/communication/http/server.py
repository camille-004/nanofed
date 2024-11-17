import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aiohttp import web

from nanofed.communication.http.types import (
    GlobalModelResponse,
    ModelUpdateResponse,
    ServerModelUpdateRequest,
)
from nanofed.server.model_manager.manager import ModelManager
from nanofed.utils.logger import Logger


@dataclass(slots=True, frozen=True)
class ServerEndpoints:
    """Server endpoint configuration."""

    get_model: str = "/model"
    submit_update: str = "/update"
    get_status: str = "/status"


class HTTPServer:
    """HTTP server for communication."""

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

    async def _handle_test(self, request: web.Request) -> web.Response:
        return web.Response(text="Server is running")

    async def _handle_get_model(self, request: web.Request) -> web.Response:
        """Handle request for global model."""
        with self._logger.context("server.http"):
            try:
                version = self._model_manager.current_version
                if version is None:
                    version = self._model_manager.load_model()

                self._logger.debug(
                    f"Serving model version {version.version_id}"
                )

                # Convert model parameters to list for JSON serialization
                state_dict = self._model_manager._model.state_dict()
                model_state = {
                    key: value.cpu().numpy().tolist()
                    for key, value in state_dict.items()
                }

                response: GlobalModelResponse = {
                    "status": "success",
                    "message": "Global model retrieved",
                    "timestamp": datetime.now().isoformat(),
                    "model_state": model_state,
                    "round_number": self._current_round,
                    "version_id": version.version_id,
                }
                return web.json_response(response)

            except Exception as e:
                self._logger.error(f"Error serving model: {str(e)}")
                return web.json_response(
                    {
                        "status": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat(),
                    },
                    status=500,
                )

    async def _handle_submit_update(
        self, request: web.Request
    ) -> web.Response:
        """Handle model update submission from client."""
        with self._logger.context("server.http"):
            try:
                data: dict[str, Any] = await request.json()

                # Validate required fields
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
                            "timestamp": datetime.now().isoformat(),
                        },
                        status=400,
                    )

                update: ServerModelUpdateRequest = {
                    "client_id": data["client_id"],
                    "round_number": data["round_number"],
                    "model_state": data["model_state"],
                    "metrics": data["metrics"],
                    "timestamp": data["timestamp"],
                    "status": data["status"],
                    "message": data["message"],
                    "accepted": data["accepted"],
                }

                async with self._lock:
                    if update["round_number"] != self._current_round:
                        return web.json_response(
                            {
                                "status": "error",
                                "message": "Invalid round number",
                                "timestamp": datetime.now().isoformat(),
                            },
                            status=400,
                        )

                    client_id = update["client_id"]
                    self._updates[client_id] = update

                    response: ModelUpdateResponse = {
                        "status": "success",
                        "message": "Updated accepted",
                        "timestamp": datetime.now().isoformat(),
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
                        "timestamp": datetime.now().isoformat(),
                    },
                    status=500,
                )

    async def _handle_get_status(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "success",
                "message": "Server is running",
                "timestamp": datetime.now().isoformat(),
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

    async def stop(self) -> None:
        """Stop HTTP server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._logger.info("Server stopped")
