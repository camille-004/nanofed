import base64
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, ClassVar, Final, TypeAlias, TypeGuard

import numpy as np

JSONValue: TypeAlias = (
    str | int | float | bool | None | dict[str, Any] | list[Any]
)
PayloadType: TypeAlias = dict[str, JSONValue]


class MessageType(Enum):
    REGISTER = "REGISTER"  # Client registration
    MODEL_UPDATE = "MODEL_UPDATE"  # Client sending model update
    GLOBAL_MODEL = "GLOBAL_MODEL"  # Server sending global model
    ROUND_START = "ROUND_START"  # Server announcing round start
    ROUND_END = "ROUND_END"  # Server announcing round end
    ERROR = "ERROR"  # Error message
    HEARTBEAT = "HEARTBEAT"  # Connection keepalive
    METRICS = "METRICS"  # Training metrics


class MessageValidationError(Exception):
    """Raised when message validation fails."""

    pass


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and bytes."""

    NUMPY_TYPE_PREFIX: Final[str] = "ndarray"
    BYTES_TYPE_PREFIX: Final[str] = "bytes"

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__type__": self.NUMPY_TYPE_PREFIX,
                "data": base64.b64encode(obj.tobytes()).decode("utf-8"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, bytes):
            return {
                "__type__": self.BYTES_TYPE_PREFIX,
                "data": base64.b64encode(obj).decode("utf-8"),
            }
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class NumpyJSONDecoder(json.JSONDecoder):
    """Custom JSON decoder for numpy arrays and bytes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(object_hook=self._object_hook, *args, **kwargs)

    def _object_hook(self, obj: dict) -> dict[str, Any] | np.ndarray | bytes:
        obj_type = obj.get("__type__")
        if not obj_type:
            return obj

        if obj_type == NumpyJSONEncoder.NUMPY_TYPE_PREFIX:
            try:
                data = base64.b64decode(obj["data"])
                return np.frombuffer(
                    data, dtype=np.dtype(obj["dtype"])
                ).reshape(obj["shape"])
            except (KeyError, ValueError, TypeError) as e:
                raise MessageValidationError(
                    f"Invalid numpy array format: {e}"
                )

        if obj_type == NumpyJSONEncoder.BYTES_TYPE_PREFIX:
            try:
                return base64.b64decode(obj["data"])
            except (KeyError, ValueError, TypeError) as e:
                raise MessageValidationError(f"Invalid bytes format: {e}")

        return obj


@dataclass(frozen=True, slots=True)
class Message:
    """Immutable message exchanged between client and server."""

    message_type: MessageType
    payload: PayloadType
    sender_id: str
    round_number: int | None = None
    timestamp: float | None = field(
        default_factory=lambda: datetime.now().timestamp()
    )

    MAX_PAYLOAD_SIZE: ClassVar[int] = 1024 * 1024 * 10  # 10MB
    REQUIRED_PAYLOAD_FIELDS: ClassVar[dict[MessageType, set[str]]] = {
        MessageType.REGISTER: {"client_id", "signature"},
        MessageType.MODEL_UPDATE: {"update"},
        MessageType.GLOBAL_MODEL: {"weights", "round"},
        MessageType.ROUND_START: {"round", "deadline"},
        MessageType.ROUND_END: {"round", "summary"},
        MessageType.ERROR: {"error", "code"},
        MessageType.METRICS: {"metrics"},
    }

    def __post_init__(self) -> None:
        self._validate_message()

    def _validate_message(self) -> None:
        if not isinstance(self.message_type, MessageType):
            raise MessageValidationError(
                f"Invalid message type: {self.message_type}"
            )

        if not isinstance(self.payload, dict):
            raise MessageValidationError(
                f"Payload must be a dictionary, got {type(self.payload)}"
            )

        # Check required fields for message type
        if self.message_type in self.REQUIRED_PAYLOAD_FIELDS:
            missing = self.REQUIRED_PAYLOAD_FIELDS[self.message_type] - set(
                self.payload.keys()
            )
            if missing:
                raise MessageValidationError(
                    f"Missing required payload fields: {missing}"
                )

        # Validate payload size
        payload_size = len(json.dumps(self.payload, cls=NumpyJSONEncoder))
        if payload_size > self.MAX_PAYLOAD_SIZE:
            raise MessageValidationError(
                f"Payload size ({payload_size} bytes) exceeds maximum "
                f"({self.MAX_PAYLOAD_SIZE} bytes)"
            )

    def to_json(self) -> str:
        """Convert message to JSON string."""
        try:
            return json.dumps(asdict(self), cls=NumpyJSONEncoder)
        except (TypeError, ValueError) as e:
            raise MessageValidationError(f"Failed to serialize message: {e}")

    @classmethod
    def from_json(cls, json_str) -> "Message":
        """Create message from JSON string."""
        try:
            data = json.loads(json_str, cls=NumpyJSONDecoder)
            data["message_type"] = MessageType(data["message_type"])
            return cls(**data)
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            raise MessageValidationError(f"Failed to deserialize message: {e}")

    def is_valid_round(self) -> bool:
        """Check if round number is valid for message type."""
        needs_round = self.message_type in {
            MessageType.MODEL_UPDATE,
            MessageType.GLOBAL_MODEL,
            MessageType.ROUND_START,
            MessageType.ROUND_END,
        }
        return not needs_round or (
            self.round_number is not None and self.round_number >= 0
        )


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols."""

    MAX_RETRY_ATTEMPTS: ClassVar[int] = 3
    RETRY_DELAY: ClassVar[float] = 1.0

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def send(self, message: Message) -> None:
        pass

    @abstractmethod
    async def receive(self) -> AsyncGenerator[Message, None]:
        pass

    def validate_numpy_array(self, arr: Any) -> TypeGuard[np.ndarray]:
        return (
            isinstance(arr, np.ndarray)
            and not np.any(np.isnan(arr))
            and not np.any(np.isinf(arr))
        )


def decode_weights(weights_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    """Decode weights from serialized format back to numpy arrays."""
    try:
        decoded = {}
        for key, value in weights_dict.items():
            if isinstance(value, dict) and value.get("__type__") == "ndarray":
                data = base64.b64decode(value["data"])
                dtype = np.dtype(value["dtype"])
                shape = tuple(value["shape"])
                arr = np.frombuffer(data, dtype=dtype).reshape(shape)
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    raise ValueError(
                        f"Invalid values in array {key}: contains NaN or Inf"
                    )
                decoded[key] = arr
            else:
                decoded[key] = value
        return decoded
    except (KeyError, ValueError, TypeError) as e:
        raise MessageValidationError(f"Failed to decode weights: {e}")
