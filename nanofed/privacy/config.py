from enum import Enum, auto

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NOISE_MULTIPLIER,
    MAX_DELTA,
    MAX_EPSILON,
    MIN_DELTA,
    MIN_EPSILON,
)


class NoiseType(Enum):
    """Type of noise distributions."""

    GAUSSIAN = auto()
    LAPLACIAN = auto()


class PrivacyConfig(BaseModel):
    """Privacy mechanism configuration.

    Parameters
    ----------
    epsilon : float
        Privacy parameter epsilon (ε)
    delta : float
        Privacy parameter delta (δ)
    max_gradient_norm : float
        Maximum L2 norm for gradient clipping
    noise_multiplier : float
        Scale of noise addition
    noise_type : NoiseType
        Type of noise distribution to use
    """

    epsilon: float = Field(
        default=DEFAULT_EPSILON,
        description="Privacy parameter epsilon (ε)",
        ge=MIN_EPSILON,
        le=MAX_EPSILON,
    )
    delta: float = Field(
        default=DEFAULT_DELTA,
        description="Privacy parameter delta (δ)",
        ge=MIN_DELTA,
        le=MAX_DELTA,
    )
    max_gradient_norm: float = Field(
        default=DEFAULT_MAX_GRAD_NORM,
        description="Maximum L2 norm for gradient clipping",
        gt=0,
    )
    noise_multiplier: float = Field(
        default=DEFAULT_NOISE_MULTIPLIER,
        description="Scale of noise addition",
        gt=0,
    )
    noise_type: NoiseType = Field(
        default=NoiseType.GAUSSIAN, description="Type of noise distribution"
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("epsilon")
    @classmethod
    def validate_epsilon(cls, v: float) -> float:
        if v < MIN_EPSILON or v > MAX_EPSILON:
            raise ValueError(
                f"epsilon must be between {MIN_EPSILON} and {MAX_EPSILON}"
            )
        return v

    @field_validator("delta")
    @classmethod
    def validate_delta(cls, v: float) -> float:
        if v < MIN_DELTA or v > MAX_DELTA:
            raise ValueError(
                f"delta must be between {MIN_DELTA} and {MAX_DELTA}"
            )
        return v
