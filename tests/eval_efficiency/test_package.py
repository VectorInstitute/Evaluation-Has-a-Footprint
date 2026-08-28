"""Smoke tests for the public package."""

from importlib.metadata import metadata

import pytest

import evaluation_has_a_footprint


def test_package_imports() -> None:
    """The public package is importable."""
    assert evaluation_has_a_footprint.__doc__


@pytest.mark.integration_test()
def test_distribution_metadata() -> None:
    """Installed distribution metadata matches the release skeleton."""
    assert metadata("evaluation-has-a-footprint")["Name"] == "evaluation-has-a-footprint"
