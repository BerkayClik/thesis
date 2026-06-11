"""Real LSTM + RevIN ablation models (Oracle O3 parity with quaternion models).

These wrappers must follow the exact normalization protocol of the quaternion
models: per-window RevIN on input, denorm_scalar on the scalar output in
price-mode ONLY (return-mode outputs are returns and must not be price-scale
denormalized). They also must fix the static Z-score OOD failure: predictions
should live on each window's own price scale, not the training-set scale.
"""

import torch

from src.models.real_lstm_revin import RealLSTMRevIN, RealLSTMAttentionRevIN


def _spy_denorm(model):
    """Wrap norm_layer.denorm_scalar to record whether it was invoked."""
    calls = {"n": 0}
    original = model.norm_layer.denorm_scalar

    def wrapper(x, feature_idx):
        calls["n"] += 1
        return original(x, feature_idx)

    model.norm_layer.denorm_scalar = wrapper
    return calls


def test_real_revin_denorm_applied_in_price_mode():
    torch.manual_seed(0)
    model = RealLSTMRevIN(input_size=4, hidden_size=8, target_col=3).eval()
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        out = model(x)
    assert calls["n"] == 1, "denorm_scalar MUST run in price-mode"
    assert out.shape == (2, 1)


def test_real_revin_denorm_skipped_in_return_mode():
    torch.manual_seed(0)
    model = RealLSTMRevIN(input_size=4, hidden_size=8, target_col=3,
                          target_mode="return").eval()
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        model(x)
    assert calls["n"] == 0, "denorm_scalar must NOT run in return-mode"


def test_real_attention_revin_denorm_applied_in_price_mode():
    torch.manual_seed(0)
    model = RealLSTMAttentionRevIN(input_size=4, hidden_size=8, target_col=3).eval()
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        model(x)
    assert calls["n"] == 1, "denorm_scalar MUST run in price-mode"


def test_real_attention_revin_denorm_skipped_in_return_mode():
    torch.manual_seed(0)
    model = RealLSTMAttentionRevIN(input_size=4, hidden_size=8, target_col=3,
                                   target_mode="return").eval()
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        model(x)
    assert calls["n"] == 0, "denorm_scalar must NOT run in return-mode"


def test_price_mode_predictions_track_window_scale():
    """The point of the ablation: with RevIN, predictions follow each window's
    own price level — a window around 100 and the same shape around 10000 must
    give predictions near their respective levels (static Z-score cannot)."""
    torch.manual_seed(0)
    model = RealLSTMRevIN(input_size=4, hidden_size=8, target_col=3).eval()
    base = torch.randn(1, 6, 4) * 0.5
    low = base + 100.0
    high = base + 10_000.0
    with torch.no_grad():
        out_low = model(low)
        out_high = model(high)
    assert 50.0 < out_low.item() < 200.0, f"low-scale pred off-window: {out_low.item()}"
    assert 9_000.0 < out_high.item() < 11_000.0, f"high-scale pred off-window: {out_high.item()}"


def test_dish_ts_norm_type_supported():
    torch.manual_seed(0)
    model = RealLSTMRevIN(input_size=4, hidden_size=8, target_col=3,
                          norm_type="dish_ts", seq_len=6).eval()
    x = torch.randn(2, 6, 4) + 100.0
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()
