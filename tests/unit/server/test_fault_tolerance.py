import pytest
import torch

from nanofed.server import (
    CheckpointMetadata,
    FaultTolerantCoordinator,
    FileStateStore,
    RoundState,
    SimpleRecoveryStrategy,
)
from nanofed.utils import get_current_time


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def mock_state():
    return {
        "model_state": {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        },
        "optimizer_state": {
            "step": 100,
            "lr": 0.01,
        },
    }


def mock_update_to_serializable(tensor: torch.Tensor) -> list:
    """Convert a tensor to a list for JSON serialization."""
    return tensor.tolist()


@pytest.fixture
def mock_client_updates():
    updates = {}
    for i in range(3):
        updates[f"client_{i}"] = {
            "client_id": f"client_{i}",
            "round_number": 1,
            "model_state": {
                "layer1.weight": mock_update_to_serializable(
                    torch.randn(10, 5)
                ),
                "layer1.bias": mock_update_to_serializable(torch.randn(10)),
            },
            "metrics": {"accuracy": 0.9},
        }
    return updates


class TestFileStateStore:
    """Test file-based state persistence."""

    def test_save_load_checkpoint(
        self, temp_checkpoint_dir, mock_state, mock_client_updates
    ):
        """Test saving and loading checkpoints."""
        store = FileStateStore(temp_checkpoint_dir)

        metadata = CheckpointMetadata(
            round_id=1,
            timestamp=get_current_time(),
            num_clients=3,
            client_updates=mock_client_updates,
            global_model_version="v1",
            state=RoundState.COMPLETED,
        )

        store.save_checkpoint(metadata, mock_state)

        loaded = store.load_checkpoint(1)
        assert loaded is not None
        loaded_metadata, loaded_state = loaded

        assert loaded_metadata.round_id == metadata.round_id
        assert loaded_metadata.num_clients == metadata.num_clients
        assert (
            loaded_metadata.global_model_version
            == metadata.global_model_version
        )
        assert loaded_metadata.state == metadata.state

        assert set(loaded_state.keys()) == set(mock_state.keys())
        assert "model_state" in loaded_state
        assert "optimizer_state" in loaded_state

    def test_list_checkpoints(self, temp_checkpoint_dir, mock_state):
        """Test listing checkpoints."""
        store = FileStateStore(temp_checkpoint_dir)

        rounds = [1, 2, 3]
        for round_id in rounds:
            metadata = CheckpointMetadata(
                round_id=round_id,
                timestamp=get_current_time(),
                num_clients=3,
                client_updates={},
                global_model_version=f"v{round_id}",
                state=RoundState.COMPLETED,
            )
            store.save_checkpoint(metadata, mock_state)

        checkpoints = store.list_checkpoints()
        assert len(checkpoints) == len(rounds)
        assert [cp.round_id for cp in checkpoints] == rounds


class TestSimpleRecoveryStrategy:
    """Test recovery strategy."""

    def test_should_recover(self):
        """Test recovery determination."""
        strategy = SimpleRecoveryStrategy()

        assert strategy.should_recover(TimeoutError())
        assert strategy.should_recover(ConnectionError())
        assert strategy.should_recover(RuntimeError())

        assert not strategy.should_recover(ValueError())
        assert not strategy.should_recover(KeyError())

    def test_get_recovery_point(self):
        """Test recovery point selection."""
        strategy = SimpleRecoveryStrategy()

        checkpoints = [
            CheckpointMetadata(
                round_id=i,
                timestamp=get_current_time(),
                num_clients=3,
                client_updates={},
                global_model_version=f"v{i}",
                state=state,
            )
            for i, state in enumerate(
                [
                    RoundState.COMPLETED,
                    RoundState.FAILED,
                    RoundState.COMPLETED,
                    RoundState.IN_PROGRESS,
                ]
            )
        ]

        recovery_point = strategy.get_recovery_point(checkpoints)
        assert recovery_point is not None
        assert recovery_point.round_id == 2
        assert recovery_point.state == RoundState.COMPLETED


class TestFaultTolerantCoordinator:
    """Test fault-tolerant coordinator."""

    def test_checkpoint_restore(
        self, temp_checkpoint_dir, mock_state, mock_client_updates
    ):
        """Test checkpointing and restoration."""
        coordinator = FaultTolerantCoordinator(temp_checkpoint_dir)

        coordinator.checkpoint_round(
            round_id=1,
            client_updates=mock_client_updates,
            model_version="v1",
            state=mock_state,
            round_state=RoundState.COMPLETED,
        )

        restored = coordinator.restore_round(1)
        assert restored is not None
        metadata, state = restored

        assert metadata.round_id == 1
        assert metadata.global_model_version == "v1"
        assert metadata.state == RoundState.COMPLETED
        assert len(metadata.client_updates) == len(mock_client_updates)

    def test_failure_handling(
        self, temp_checkpoint_dir, mock_state, mock_client_updates
    ):
        """Test failure handling and recovery."""
        coordinator = FaultTolerantCoordinator(temp_checkpoint_dir)

        for i in range(3):
            coordinator.checkpoint_round(
                round_id=i,
                client_updates=mock_client_updates,
                model_version=f"v{i}",
                state=mock_state,
                round_state=RoundState.COMPLETED
                if i < 2
                else RoundState.FAILED,
            )

        recovery = coordinator.handle_failure(
            TimeoutError("Connection timeout"), current_round=3
        )

        assert recovery is not None
        metadata, state = recovery
        assert metadata.round_id == 1
        assert metadata.state == RoundState.COMPLETED

        recovery = coordinator.handle_failure(
            ValueError("Invalid input"), current_round=3
        )
        assert recovery is None
