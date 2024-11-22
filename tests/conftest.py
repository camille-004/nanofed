import asyncio

import pytest


@pytest.fixture
def event_loop():
    """Create event loop for testing."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
