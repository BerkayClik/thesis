"""Oracle O3: in return-mode the model output is a RETURN, so the RevIN/Dish-TS
price-scale denormalization MUST be bypassed. Inverse-applying price mean/std to
a return value silently corrupts every downstream metric.

These tests prove:
  - price-mode  -> denorm_scalar IS called (output differs from pre-denorm scalar)
  - return-mode -> denorm_scalar is NOT called (output equals pre-denorm scalar)
"""

import torch

from src.models.qnn_attention_model import QNNAttentionModel, QuaternionLSTMNoAttention
from src.models.hierarchical_qlstm import HierarchicalQLSTM


def _spy_denorm(model):
    """Wrap norm_layer.denorm_scalar to record whether it was invoked."""
    calls = {"n": 0}
    original = model.norm_layer.denorm_scalar

    def wrapper(x, feature_idx):
        calls["n"] += 1
        return original(x, feature_idx)

    model.norm_layer.denorm_scalar = wrapper
    return calls


def test_qnn_attention_denorm_skipped_in_return_mode():
    torch.manual_seed(0)
    model = QNNAttentionModel(hidden_size=8, num_features=4, target_col=3,
                              target_mode="return").eval()
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        model(x)
    assert calls["n"] == 0, "denorm_scalar must NOT run in return-mode"


def test_qnn_attention_denorm_applied_in_price_mode():
    torch.manual_seed(0)
    model = QNNAttentionModel(hidden_size=8, num_features=4, target_col=3).eval()  # default price
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        model(x)
    assert calls["n"] == 1, "denorm_scalar MUST run in price-mode"


def test_qnn_no_attention_denorm_skipped_in_return_mode():
    torch.manual_seed(0)
    model = QuaternionLSTMNoAttention(hidden_size=8, num_features=4, target_col=3,
                                      target_mode="log_return").eval()
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        model(x)
    assert calls["n"] == 0


def test_hierarchical_denorm_skipped_in_return_mode():
    torch.manual_seed(0)
    model = HierarchicalQLSTM(hidden_size=8, num_features=16, target_col=3,
                              seq_len=6, target_mode="return").eval()
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        model(x)
    assert calls["n"] == 0


def test_hierarchical_denorm_applied_in_price_mode():
    torch.manual_seed(0)
    model = HierarchicalQLSTM(hidden_size=8, num_features=16, target_col=3,
                              seq_len=6).eval()  # default price
    calls = _spy_denorm(model)
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        model(x)
    assert calls["n"] == 1
