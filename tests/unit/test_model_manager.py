import shutil

import pytest

from nanofed.core.types import ModelConfig
from nanofed.server.model_manager.manager import ModelManager
from tests.unit.helpers import SimpleModel


@pytest.fixture
def test_dir(tmp_path):
    model_dir = tmp_path / "test_models"
    yield model_dir
    if model_dir.exists():
        shutil.rmtree(model_dir)


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def manager(test_dir, model):
    return ModelManager(test_dir, model)


def test_save_and_load_model(manager):
    config = ModelConfig(
        name="test_model", version="1.0", architecture={"type": "simple"}
    )

    version = manager.save_model(config)
    assert version.version_id is not None
    assert version.path.exists()

    loaded_version = manager.load_model(version.version_id)
    assert loaded_version.version_id == version.version_id
    assert loaded_version.config == config


def test_list_versions(manager):
    config = ModelConfig(
        name="test_model", version="1.0", architecture={"type": "simple"}
    )

    v1 = manager.save_model(config)
    v2 = manager.save_model(config)

    versions = manager.list_versions()
    assert len(versions) == 2
    assert versions[0].version_id == v1.version_id
    assert versions[1].version_id == v2.version_id
