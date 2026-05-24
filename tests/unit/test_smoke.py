"""Smoke test -- confirms the package imports cleanly."""

import neurotrack


def test_package_version_string() -> None:
    assert isinstance(neurotrack.__version__, str)
    assert neurotrack.__version__.count(".") >= 1
