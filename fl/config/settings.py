from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecurityConfig(TypedDict):
    encryption_type: Literal[
        "homomorphic", "differential_privacy", "secure_aggregation"
    ]
    key_size: int
    noise_scale: float


class Settings(BaseSettings):
    api_key: SecretStr = Field(
        default=SecretStr("dev-key-000"), description="API key"
    )
    max_clients: int = Field(
        default=100, gt=0, description="Maximum number of clients allowed"
    )
    security: SecurityConfig = Field(
        default_factory=lambda: {
            "encryption_type": "homomorphic",
            "key_size": 2048,
            "noise_scale": 0.1,
        },
        description="Security settings",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for storing data",
    )
    model_dir: Path = Field(
        default=Path("./models"),
        description="Directory for storing models",
    )
    server_url: str = Field(
        default="http://localhost:8000",
        description="URL of the server",
    )
    timeout: int = Field(
        default=30,
        gt=0,
        description="Timeout for network operations",
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Default batch_size for training",
    )
    local_epochs: int = Field(
        default=1,
        gt=0,
        description="Default nubmer of local epochs",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FL_",
        protected_namespaces=("settings_",),
        case_sensitive=False,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_path(cls, values: dict) -> dict:
        data_dir = values.get("data_dir", Path("./data"))
        model_dir = values.get("model_dir", Path("./models"))

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        return values


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def load_yaml_settings(path: Path) -> dict:
    import yaml

    with path.open("r") as f:
        return yaml.safe_load(f)
