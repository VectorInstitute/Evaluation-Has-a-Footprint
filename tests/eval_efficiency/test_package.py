"""Smoke tests for the public package."""

from importlib.metadata import metadata

import pytest

import eval_efficiency


def test_package_imports() -> None:
    """The public package is importable."""
    assert eval_efficiency.__doc__


@pytest.mark.integration_test()
def test_distribution_metadata() -> None:
    """Installed distribution metadata matches the release skeleton."""
    assert metadata("eval-efficiency")["Name"] == "eval-efficiency"
