"""Tests for target_mode config support (price | return | log_return).

target_mode is an additive, non-destructive flag. Default MUST be "price" so
every existing config keeps working unchanged.
"""

import pytest

from src.utils.config import resolve_target_mode


def test_default_is_price_when_absent():
    """A config without target_mode resolves to 'price' (backward compatible)."""
    cfg = {"data": {"source": "lunarcrush", "coin": "btc"}}
    assert resolve_target_mode(cfg) == "price"


def test_explicit_price():
    cfg = {"data": {"target_mode": "price"}}
    assert resolve_target_mode(cfg) == "price"


def test_explicit_return():
    cfg = {"data": {"target_mode": "return"}}
    assert resolve_target_mode(cfg) == "return"


def test_explicit_log_return():
    cfg = {"data": {"target_mode": "log_return"}}
    assert resolve_target_mode(cfg) == "log_return"


def test_invalid_target_mode_raises():
    cfg = {"data": {"target_mode": "log_returns"}}  # typo: trailing 's'
    with pytest.raises(ValueError, match="target_mode"):
        resolve_target_mode(cfg)


def test_top_level_target_mode_also_supported():
    """Allow target_mode at top level too, not only under data."""
    cfg = {"target_mode": "return", "data": {}}
    assert resolve_target_mode(cfg) == "return"
