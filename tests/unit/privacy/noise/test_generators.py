import pytest
import torch

from nanofed.privacy.noise import (
    GaussianNoiseGenerator,
    LaplacianNoiseGenerator,
)


@pytest.fixture
def gaussian_generator():
    """Fixture for gaussian noise generator."""
    return GaussianNoiseGenerator(seed=42)


@pytest.fixture
def laplacian_generator():
    """Fixture for laplacian noise generator."""
    return LaplacianNoiseGenerator(seed=42)


class TestGaussianNoise:
    """Test suite for Gaussian noise generator."""

    def test_shape(self, gaussian_generator):
        """Test output tensor shape."""
        shape = (100, 20)
        noise = gaussian_generator.generate(shape, scale=1.0)
        assert noise.shape == shape

    def test_scale(self, gaussian_generator):
        """Test noise scale."""
        shape = (1000,)
        scale = 2.0
        noise = gaussian_generator.generate(shape, scale=scale)
        assert abs(noise.std().item() - scale) < 0.1

    def test_reproducibility(self, gaussian_generator):
        """Test noise reproducibility"""
        shape = (100, 20)
        noise1 = gaussian_generator.generate(shape, scale=1.0)

        gen2 = GaussianNoiseGenerator(seed=42)
        noise2 = gen2.generate(shape, scale=1.0)

        assert torch.allclose(noise1, noise2)


class TestLaplacianNoise:
    """Test suite for Laplacian noise generator."""

    def test_shape(self, laplacian_generator):
        """Test output tensor shape."""
        shape = (100, 20)
        noise = laplacian_generator.generate(shape, scale=1.0)
        assert noise.shape == shape

    def test_scale(self, laplacian_generator):
        """Test noise scale."""
        shape = (10000,)
        scale = 2.0
        noise = laplacian_generator.generate(shape, scale=scale)
        # For Laplace distribution, mean absolute deviation = scale
        assert abs(noise.abs().mean().item() - scale) < 0.1


def test_invalid_inputs():
    """Test invalid input handling."""
    generator = GaussianNoiseGenerator()

    with pytest.raises(ValueError):
        generator.generate((-1, 10), scale=1.0)

    with pytest.raises(ValueError):
        generator.generate((10, 10), scale=-1.0)

    with pytest.raises(ValueError):
        generator.generate((10.5, 10), scale=1.0)
