from datetime import datetime
from typing import Annotated, Any, Final

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from fl.communication.protocols import (
    Message,
    MessageType,
    MessageValidationError,
)
from fl.config.logging import get_logger
from fl.config.settings import get_settings
from fl.core.exceptions import ServerError, ValidationError
from fl.core.protocols import ModelUpdate
from fl.utils.common import timed

settings = get_settings()
router = APIRouter(tags=["Communication"])
logger = get_logger("Endpoints")

# Constants
MAX_CLIENTS: Final[int] = 1000
MAX_RETRIES: Final[int] = 3
RETRY_DELAY: Final[float] = 1.0
API_KEY_HEADER: Final[str] = "X-API-Key"


class ServerStatus(BaseModel):
    """Server status response model."""

    status: str
    active_clients: int
    current_round: int | None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainingResponse(BaseModel):
    """Training response model."""

    status: str
    timestamp: str


class ClientCapabilities(BaseModel):
    """Client capabilities model."""

    version: str = Field(..., pattern=r"^\d+\.\d+(\.\d+)?$")
    batch_size: int = Field(default=32, gt=0, lt=1024)
    local_epochs: int = Field(default=1, gt=0, lt=100)
    encryption_type: str | None = Field(default=None)
    device_info: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "version": "1.0.0",
                "batch_size": 32,
                "local_epochs": 1,
                "encryption_type": "basic",
                "device_info": {
                    "cpu_count": 4,
                    "memory_gb": 8,
                    "gpu_available": False,
                },
            }
        },
    )


class ClientRegistration(BaseModel):
    """Client registration request model."""

    client_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique identifier for the client",
    )
    signature: str = Field(..., min_length=1)
    capabilities: ClientCapabilities


class RegistrationResponse(BaseModel):
    """Client registration response model."""

    status: str = Field(default="success")
    client_id: str
    round: int = Field(default=0, ge=0)
    server_time: datetime = Field(default_factory=datetime.now)
    message: str | None = None
    token: str | None = Field(
        default=None, description="Authentication token for future requests"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str
    error_code: int = Field(default=status.HTTP_400_BAD_REQUEST)
    timestamp: datetime = Field(default_factory=datetime.now)
    trace_id: str | None = None


class WebSocketRequest:
    def __init__(self, websocket: WebSocket) -> None:
        self.scope = websocket.scope
        self.app = websocket.app


async def get_server(request: Request) -> Any:
    """Get server instance from app state."""
    try:
        return request.app.state.server
    except AttributeError:
        logger.error("Server not initalized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized",
        )


async def verify_api_key_header(api_key: str | None) -> None:
    """Helper function to verify the API key."""
    if not api_key or api_key != settings.api_key.get_secret_value():
        raise ValidationError("Invalid API Key")


async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str | None:
    try:
        await verify_api_key_header(x_api_key)
    except ValidationError as e:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e)
        )
    return x_api_key


async def send_connection_accepted(
    websocket: WebSocket, client_id: str, connection_id: str
) -> None:
    logger.info(
        "WebSocket connection accepted",
        extra={"client_id": client_id, "connection_id": connection_id},
    )
    await websocket.send_json(
        {
            "type": "CONNECTION_ACCEPTED",
            "client_id": client_id,
            "connection_id": connection_id,
            "message": "Connected successfully",
            "server_time": datetime.now().isoformat(),
        }
    )


async def handle_message(
    websocket: WebSocket,
    client_id: str,
    connection_id: str,
    message: Message,
    server: Any,
) -> None:
    if message.message_type == MessageType.MODEL_UPDATE:
        await handle_model_update(websocket, client_id, message, server)
    elif message.message_type == MessageType.HEARTBEAT:
        await handle_heartbeat(websocket, client_id, connection_id)
    else:
        await handle_unknown_message_type(websocket, client_id, message)


async def handle_model_update(
    websocket: WebSocket, client_id: str, message: Message, server: Any
) -> None:
    update_dict = message.payload.get("update")
    if not update_dict or not isinstance(update_dict, dict):
        logger.warning(
            "Misisng update data",
            extra={"client_id": client_id},
        )
        await send_error(
            websocket,
            "Invalid update format: 'update' field is missing",
        )
        return
    try:
        update = ModelUpdate(**update_dict)
        response, duration = await timed(server.submit_update)(
            client_id=client_id, model_update=update
        )

        logger.info(
            "Processed model update",
            extra={
                "client_id": client_id,
                "round": update.round_number,
                "duration": duration,
            },
        )

        await websocket.send_json(
            {
                "type": "UPDATE_RESPONSE",
                "status": "success",
                "payload": response,
                "processing_time": duration,
            }
        )
    except ValidationError as e:
        logger.warning(
            "Invalid update format",
            extra={"client_id": client_id, "error": str(e)},
        )
        await send_error(websocket, f"Invalid update format {str(e)}")


async def handle_heartbeat(
    websocket: WebSocket, client_id: str, connection_id: str
) -> None:
    await websocket.send_json(
        {
            "type": "HEARTBEAT_RESPONSE",
            "status": "success",
            "payload": {
                "server_time": datetime.now().isoformat(),
                "status": "alive",
                "connection_id": connection_id,
            },
        }
    )


async def handle_unknown_message_type(
    websocket: WebSocket, client_id: str, message: Message
) -> None:
    logger.warning(
        "Unknown message type",
        extra={
            "client_id": client_id,
            "message_type": message.message_type,
        },
    )
    await send_error(
        websocket, f"Unknown message type: {message.message_type}"
    )


async def handle_message_validation_error(
    websocket: WebSocket, client_id: str, error: Exception
) -> None:
    logger.warning(
        "Message validation error",
        extra={"client_id": client_id, "error": str(error)},
    )
    await send_error(websocket, f"MEssage vaidation error: {str(error)}")


async def handle_general_error(
    websocket: WebSocket, client_id: str, error: Exception
) -> None:
    logger.error(
        "WebSocket error", extra={"client_id": client_id, "error": str(error)}
    )
    await send_error(websocket, f"Server error: {str(error)}")


async def send_error(websocket: WebSocket, error_message: str) -> None:
    await websocket.send_json(
        {
            "type": "ERROR",
            "status": "error",
            "payload": {"error": error_message},
        }
    )


async def log_request(request: Request, call_next) -> None:
    """Middleware to log requests."""
    start_time = datetime.now()
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration": duration,
            },
        )
        return response
    except Exception as e:
        logger.error(
            "Request failed",
            extra={
                "method": request.method,
                "url": str(request.url),
                "Error": str(e),
            },
        )
        raise


@router.post(
    "/api/register",
    response_model=RegistrationResponse,
    responses={
        403: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    description="Register a new client with the server",
)
async def register_client(
    request: Request,
    client_info: ClientRegistration,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
) -> RegistrationResponse:
    """Register a new client with the server."""
    try:
        server = await get_server(request)
        if len(server._clients) >= MAX_CLIENTS:
            raise ValidationError("Maximum number of clients reached")

        response = await server.register_client(
            client_id=client_info.client_id,
            signature=client_info.signature,
            capabilities=client_info.capabilities.model_dump(),
        )

        logger.info(
            "Client registered successfully",
            extra={
                "client_id": client_info.client_id,
                "capabilities": client_info.capabilities.model_dump(),
            },
        )

        background_tasks.add_task(
            cleanup_inactive_clients, server, settings.timeout
        )

        registration_response = RegistrationResponse(
            status="success",
            client_id=response["client_id"],
            round=response["round"],
            server_time=datetime.now(),
        )

        return registration_response

    except ValidationError as e:
        logger.warning(
            "Client registration validation failed",
            extra={"client_id": client_info.client_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except ServerError as e:
        logger.error(
            "Server error during registration",
            extra={"client_id": client_info.client_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error during registration",
            extra={"client_id": client_info.client_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.websocket("/api/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
) -> None:
    """Websocket endpoint for client-server communication."""
    api_key = websocket.headers.get(API_KEY_HEADER)
    try:
        await verify_api_key_header(api_key)
    except ValidationError:
        logger.warning(
            "Invalid API key attempt", extra={"client_id": client_id}
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    connection_id = f"{client_id}_{datetime.now().timestamp()}"
    logger.info("WebSocket connection attempt", extra={"client_id": client_id})

    server = websocket.app.state.server
    if server is None:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    await websocket.accept()
    await send_connection_accepted(websocket, client_id, connection_id)

    while True:
        try:
            data = await websocket.receive_text()
            message = Message.from_json(data)
            await handle_message(
                websocket, client_id, connection_id, message, server
            )
        except MessageValidationError as e:
            await handle_message_validation_error(websocket, client_id, e)
        except WebSocketDisconnect:
            logger.info(
                "WebSocket disconnected",
                extra={"client_id": client_id, "connection_id": connection_id},
            )
            break
        except Exception as e:
            await handle_general_error(websocket, client_id, e)
            break


@router.get(
    "/api/server",
    response_model=ServerStatus,
    responses={503: {"model": ErrorResponse}},
)
async def get_server_info(request: Request) -> ServerStatus:
    """Get server status information."""
    try:
        server = await get_server(request)
        status_info = server.get_status()
        logger.info("Server status requested", extra=status_info)
        return ServerStatus(**status_info)
    except Exception as e:
        logger.error(f"Error getting server status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get server status",
        )


@router.post(
    "/api/start_training",
    response_model=TrainingResponse,
    responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def start_server_training(request: Request) -> TrainingResponse:
    """Start the federated learning training process."""
    try:
        server = await get_server(request)
        await server.start_training()
        logger.info("Training started successfully")
        return TrainingResponse(
            status="training_started", timestamp=datetime.now().isoformat()
        )
    except ValidationError as e:
        logger.warning(f"Training start validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except ServerError as e:
        logger.error(f"Training start server error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def cleanup_inactive_clients(server: Any, timeout: int) -> None:
    """Background task to cleanup inactive clients."""
    try:
        current_time = datetime.now().timestamp()
        inactive_threshold = current_time - timeout

        inactive_count = 0
        for client_id, info in list(server._clients.items()):
            if info.last_seen < inactive_threshold:
                info.is_active = False
                server._active_clients.discard(client_id)
                inactive_count += 1

        if inactive_count > 0:
            logger.info(
                "Cleaned up inactive clients", extra={"count": inactive_count}
            )

    except Exception as e:
        logger.error(f"Error in cleanup task: {str(e)}")
