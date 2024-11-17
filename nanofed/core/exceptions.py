class NanoFedError(Exception):
    """Base exception class."""

    pass


class AggregationError(NanoFedError):
    """Raised when model aggregation fails."""

    pass


class ModelManagerError(NanoFedError):
    """Raised when model management operations fail."""

    pass
