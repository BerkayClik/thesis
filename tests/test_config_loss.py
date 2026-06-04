"""Config resolution for loss selection (mse | directional_mse).

resolve_loss(config) -> (loss_type, lambda_dir, k)
Default is mse so all existing configs/runs are unchanged. Directional loss is
only valid in return-mode (sign of a return is a direction; sign of a price is not).
"""

import pytest

from src.utils.config import resolve_loss


def test_default_is_mse():
    cfg = {"training": {}, "data": {}}
    loss_type, lam, k = resolve_loss(cfg)
    assert loss_type == "mse"
    assert lam == 0.0


def test_explicit_mse():
    cfg = {"training": {"loss_type": "mse"}, "data": {}}
    assert resolve_loss(cfg)[0] == "mse"


def test_directional_in_return_mode():
    cfg = {
        "training": {"loss_type": "directional_mse", "lambda_dir": 2.0, "directional_k": 500.0},
        "data": {"target_mode": "return"},
    }
    loss_type, lam, k = resolve_loss(cfg)
    assert loss_type == "directional_mse"
    assert lam == 2.0
    assert k == 500.0


def test_directional_default_lambda_and_k():
    cfg = {
        "training": {"loss_type": "directional_mse"},
        "data": {"target_mode": "log_return"},
    }
    loss_type, lam, k = resolve_loss(cfg)
    assert loss_type == "directional_mse"
    assert lam > 0.0
    assert k > 0.0


def test_directional_rejected_in_price_mode():
    cfg = {
        "training": {"loss_type": "directional_mse"},
        "data": {"target_mode": "price"},
    }
    with pytest.raises(ValueError, match="directional"):
        resolve_loss(cfg)


def test_unknown_loss_type_raises():
    cfg = {"training": {"loss_type": "huber"}, "data": {"target_mode": "return"}}
    with pytest.raises(ValueError, match="loss_type"):
        resolve_loss(cfg)
