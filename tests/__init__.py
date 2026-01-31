"""
Test Suite Initialization

Aegis Agent Platform test configuration.
"""

import pytest


@pytest.fixture
def anyio_backend():
    """Use asyncio for async tests."""
    return "asyncio"
