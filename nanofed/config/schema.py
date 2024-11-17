from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ServerConfig(BaseModel):
    """Server configuration schema."""

    host: str = Field(default="localhost")
    port: int = Field(default=8080, ge=1024, le=65535)
    max_clients: int = Field(default=100, ge=1)
    timeout: int = Field(default=30, ge=0)
    model_save_dir: Path = Field(default=Path("models"))

    @field_validator("model_save_dir")
    def validate_model_save_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class ClientConfig(BaseModel):
    """Client configuration schema."""

    server_url: str
    client_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    batch_size: int = Field(default=32, ge=1)
    local_epochs: int = Field(default=1, ge=1)
    device: Literal["cpu", "cuda"] = Field(default="cpu")

    @field_validator("server_url")
    def validate_server_url(cls, v: str) -> str:
        if not v.startswith("http://") or v.startswith("https://"):
            raise ValueError("server_url must start with http:// or https://")
        return v
