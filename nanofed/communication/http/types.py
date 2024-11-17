from typing import Literal, TypedDict


class BaseResponse(TypedDict):
    """Base response structure."""

    status: Literal["success", "error"]
    mesage: str
    timestamp: str


class ModelUpdateRequest(TypedDict):
    """Model update request structure."""

    client_id: str
    round_number: int
    model_state: dict[str, list[float] | list[list[float]]]
    metrics: dict[str, float]
    timestamp: str


class ModelUpdateResponse(BaseResponse):
    """Response for model update submission."""

    update_id: str
    accepted: bool


class GlobalModelResponse(BaseResponse):
    """Response containing global model info."""

    model_state: dict[str, list[float] | list[list[float]]]
    round_number: int
    version_id: str
