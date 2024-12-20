class PrivacyError(Exception):
    """Base class for privacy-related errors."""

    pass


class PrivacyBudgetExceededError(PrivacyError):
    """Raised when privacy budget is exceeded."""

    pass


class PrivacyConfigurationError(PrivacyError):
    """Raised for invalid privacy configurations."""

    pass


class NoiseGenerationError(PrivacyError):
    """Raised when noise generation fails."""

    pass
