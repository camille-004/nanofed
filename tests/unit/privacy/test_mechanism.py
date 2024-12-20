from unittest.mock import MagicMock, Mock

import pytest
import torch

from nanofed.privacy.accountant import PrivacySpent
from nanofed.privacy.config import PrivacyConfig
from nanofed.privacy.mechanisms import (
    CentralPrivacyMechanism,
    LocalPrivacyMechanism,
    PrivacyMechanismFactory,
    PrivacyType,
    UpdateMetadata,
)
from nanofed.privacy.noise import GaussianNoiseGenerator


@pytest.fixture
def privacy_config() -> PrivacyConfig:
    """Create privacy configuration for testing."""
    return PrivacyConfig(
        epsilon=2.5,
        delta=1e-5,
        noise_multiplier=1.1,
        max_gradient_norm=1.0,
    )


@pytest.fixture
def high_max_gradient_config() -> PrivacyConfig:
    """Create privacy configuration with high max_gradient_norm."""
    return PrivacyConfig(
        epsilon=2.5,
        delta=1e-5,
        noise_multiplier=1.1,
        max_gradient_norm=100.0,
    )


@pytest.fixture
def model_parameters() -> dict[str, torch.Tensor]:
    """Create dummy model parameters."""
    return {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
        "layer2.bias": torch.randn(5),
    }


class MockAccountant:
    """Mock privacy accountant that accumulates epsilon_spent."""

    def __init__(self, config: PrivacyConfig):
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.config = config
        self.add_noise_event = MagicMock(
            side_effect=self._add_noise_event_impl
        )

    def _add_noise_event_impl(self, sigma: float, samples: int):
        self.epsilon_spent += sigma * samples

    def get_privacy_spent(self) -> PrivacySpent:
        return PrivacySpent(
            epsilon_spent=self.epsilon_spent, delta_spent=self.delta_spent
        )

    def validate_budget(self) -> bool:
        return self.epsilon_spent > self.config.epsilon


@pytest.fixture
def mock_accountant_fixture(privacy_config: PrivacyConfig) -> MockAccountant:
    """Create a mock privacy accountant that accumulates epsilon spent."""
    return MockAccountant(privacy_config)


@pytest.fixture
def mock_noise_generator() -> Mock:
    generator = Mock(spec=GaussianNoiseGenerator)

    def generate_side_effect(*args, **kwargs):
        if args:
            shape = args[0]
        else:
            shape = kwargs.get("shape")

        scale = kwargs.get("scale", 1.0)

        return torch.ones(shape) * scale

    generator.generate.side_effect = generate_side_effect
    return generator


class TestBasePrivacyMechanism:
    """Test base privacy mechanism functionality."""

    def test_clip_update(
        self,
        privacy_config: PrivacyConfig,
        model_parameters: dict[str, torch.Tensor],
    ):
        """Test parameter clipping."""
        mechanism = CentralPrivacyMechanism(privacy_config)
        clipped, metadata = mechanism._clip_update(
            model_parameters, max_norm=1.0
        )

        # Check clipped norm
        total_norm = torch.norm(
            torch.cat([p.flatten() for p in clipped.values()])
        )
        assert float(total_norm) <= 1.0 + 1e-6

        assert isinstance(metadata, UpdateMetadata)
        assert metadata.total_norm > metadata.clipped_norm
        assert metadata.num_parameters == sum(
            p.numel() for p in model_parameters.values()
        )

    def test_privacy_spent_tracking(
        self,
        privacy_config: PrivacyConfig,
        model_parameters: dict[str, torch.Tensor],
        mock_accountant_fixture: MockAccountant,
    ):
        """Test privacy budget tracking."""
        mechanism = CentralPrivacyMechanism(
            privacy_config, accountant=mock_accountant_fixture
        )

        for _ in range(3):
            mechanism.add_noise(model_parameters, batch_size=32)

        assert mock_accountant_fixture.add_noise_event.call_count == 3


class TestCentralPrivacyMechanism:
    """Test central DP mechanism."""

    def test_noise_addition(
        self,
        high_max_gradient_config: PrivacyConfig,
        model_parameters: dict[str, torch.Tensor],
        mock_noise_generator: Mock,
    ):
        """Test noise addition in central DP."""
        mechanism = CentralPrivacyMechanism(
            high_max_gradient_config, noise_generator=mock_noise_generator
        )
        batch_size = 32
        noised = mechanism.add_noise(model_parameters, batch_size)

        # Verify noise generator called correctly
        assert mock_noise_generator.generate.call_count == len(
            model_parameters
        )
        for call in mock_noise_generator.generate.call_args_list:
            _, kwargs = call
            assert "scale" in kwargs
            expected_scale = pytest.approx(
                high_max_gradient_config.noise_multiplier
                * high_max_gradient_config.max_gradient_norm
                / batch_size
            )
            assert kwargs["scale"] == expected_scale

        for key in model_parameters:
            expected_noise = torch.ones(model_parameters[key].shape) * (
                high_max_gradient_config.noise_multiplier
                * high_max_gradient_config.max_gradient_norm
                / batch_size
            )
            assert torch.allclose(
                noised[key] - model_parameters[key], expected_noise
            ), f"Noise not added correctly for {key}"

    def test_privacy_type(self, privacy_config: PrivacyConfig):
        """Test privacy type property."""
        mechanism = CentralPrivacyMechanism(privacy_config)
        assert mechanism.privacy_type == PrivacyType.CENTRAL


class TestLocalPrivacyMechanism:
    """Test local DP mechanism."""

    def test_noise_addition(
        self,
        high_max_gradient_config: PrivacyConfig,
        model_parameters: dict[str, torch.Tensor],
        mock_noise_generator: Mock,
    ):
        """Test noise addition in local DP."""
        mechanism = LocalPrivacyMechanism(
            high_max_gradient_config, noise_generator=mock_noise_generator
        )

        # Local DP should ignore batch size
        batch_size = 32
        noised = mechanism.add_noise(model_parameters, batch_size)

        # Verify noise generator called correctly
        assert mock_noise_generator.generate.call_count == len(
            model_parameters
        )
        for call in mock_noise_generator.generate.call_args_list:
            _, kwargs = call
            assert "scale" in kwargs
            expected_scale = pytest.approx(
                high_max_gradient_config.noise_multiplier
                * high_max_gradient_config.max_gradient_norm
            )
            assert kwargs["scale"] == expected_scale

        for key in model_parameters:
            expected_noise = torch.ones(model_parameters[key].shape) * (
                high_max_gradient_config.noise_multiplier
                * high_max_gradient_config.max_gradient_norm
            )
            assert torch.allclose(
                noised[key] - model_parameters[key], expected_noise
            ), f"Noise not added correctly for {key}"

    def test_privacy_type(self, privacy_config: PrivacyConfig):
        """Test privacy type property."""
        mechanism = LocalPrivacyMechanism(privacy_config)
        assert mechanism.privacy_type == PrivacyType.LOCAL


class TestPrivacyMechanismFactory:
    """Test privacy mechanism factory."""

    def test_central_creation(self, privacy_config: PrivacyConfig):
        """Test central mechanism creation."""
        mechanism = PrivacyMechanismFactory.create(
            PrivacyType.CENTRAL, privacy_config
        )
        assert isinstance(mechanism, CentralPrivacyMechanism)

    def test_local_creation(self, privacy_config: PrivacyConfig):
        """Test local mechanism creation."""
        mechanism = PrivacyMechanismFactory.create(
            PrivacyType.LOCAL, privacy_config
        )
        assert isinstance(mechanism, LocalPrivacyMechanism)

    def test_invalid_type(self, privacy_config: PrivacyConfig):
        """Test invalid privacy type handling."""
        with pytest.raises(ValueError, match="Unknown privacy type: invalid"):
            PrivacyMechanismFactory.create("invalid", privacy_config)


@pytest.mark.parametrize("batch_size", [1, 32, 128])
def test_noise_scaling(
    privacy_config: PrivacyConfig,
    model_parameters: dict[str, torch.Tensor],
    mock_noise_generator: Mock,
    batch_size: int,
):
    """Test noise scaling with different batch sizes."""
    mechanism = CentralPrivacyMechanism(
        privacy_config, noise_generator=mock_noise_generator
    )
    noised = mechanism.add_noise(model_parameters, batch_size)

    for key in model_parameters:
        diff = torch.norm(noised[key] - model_parameters[key])
        assert diff > 0  # Should add some noise
        if batch_size > 1:
            # More batches should mean less noise per parameter
            assert diff < torch.norm(model_parameters[key])


@pytest.mark.parametrize(
    "mechanism_type", [PrivacyType.CENTRAL, PrivacyType.LOCAL]
)
def test_privacy_budget_validation(
    privacy_config: PrivacyConfig,
    model_parameters: dict[str, torch.Tensor],
    mechanism_type: PrivacyType,
    mock_accountant_fixture: MockAccountant,
):
    """Test privacy budget validation."""
    mechanism = PrivacyMechanismFactory.create(
        mechanism_type, privacy_config, accountant=mock_accountant_fixture
    )

    count = 0
    max_iter = 1000
    while (
        mechanism.get_privacy_spent().epsilon_spent <= privacy_config.epsilon
    ):
        mechanism.add_noise(model_parameters, batch_size=32)
        count += 1
        if count > max_iter:
            pytest.fail("Privacy budget not properly tracked")

    # Assert that the privacy budget has been exceeded
    assert mechanism.validate_budget(), "Privacy budget should be exceeded."

    # Additionally, verify that epsilon_spent is indeed greater than epsilon
    spent = mechanism.get_privacy_spent()
    assert spent.epsilon_spent > privacy_config.epsilon
