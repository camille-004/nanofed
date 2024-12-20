from nanofed.privacy.accountant import PrivacySpent
from nanofed.privacy.config import PrivacyConfig


def test_privacy_spent_validation():
    """Test privacy spent validation."""
    config = PrivacyConfig(epsilon=1.0, delta=1e-5)

    # Test within budget
    spent = PrivacySpent(epsilon_spent=0.5, delta_spent=1e-6)
    assert spent.validate(config)

    # Test epsilon exceeded
    spent = PrivacySpent(epsilon_spent=1.5, delta_spent=1e-6)
    assert not spent.validate(config)

    # Test delta exceeded
    spent = PrivacySpent(epsilon_spent=0.5, delta_spent=1e-4)
    assert not spent.validate(config)
