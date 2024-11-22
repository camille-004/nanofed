# CHANGELOG.md
# Changelog

## [0.1.0] - 2024-11-21

### Added
- Basic client-server architecture with HTTP communication
- Simple global model management with versioning
- FedAvg implementation for model aggregation
- Local training support with PyTorch integration
- Synchronous training coordination
- Basic error handling and logging
- CLI for running server and clients
- MNIST example implementation
- Comprehensive test suite
- Type checking with MyPy
- Code quality checks with Ruff

### Known Limitations
- Synchronous training only (single round must complete before next begins)
- Basic error handling and recovery
- Limited model management features
- No built-in privacy or security features
