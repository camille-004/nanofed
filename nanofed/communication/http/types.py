from typing import Literal, TypedDict

from nanofed.privacy.accountant import PrivacySpent


class BaseResponse(TypedDict):
    """Base response structure."""

    status: Literal["success", "error"]
    message: str
    timestamp: str


class ClientModelUpdateRequest(TypedDict):
    """Model update request structure."""

    client_id: str
    round_number: int
    model_state: dict[str, list[float] | list[list[float]]]
    metrics: dict[str, float]
    timestamp: str


class ServerModelUpdateRequest(TypedDict, total=False):
    """Model update request structure (processed by the server)."""

    client_id: str
    round_number: int
    model_state: dict[str, list[float] | list[list[float]]]
    metrics: dict[str, float]
    timestamp: str
    status: Literal["success", "error"]
    message: str
    accepted: bool
    privacy_spent: PrivacySpent


class ModelUpdateResponse(BaseResponse):
    """Response for model update submission."""

    update_id: str
    accepted: bool


class GlobalModelResponse(BaseResponse):
    """Response containing global model info."""

    model_state: dict[str, list[float] | list[list[float]]]
    round_number: int
    version_id: str
