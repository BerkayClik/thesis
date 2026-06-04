"""Oracle Design B eval: in return-mode evaluate_model must reconstruct prices
from raw prev_close (pred_price = prev_raw * (1 + pred_return)) and MUST NOT run
the z-score denormalize() on a return prediction.

Price-mode behavior must stay byte-identical (regression lock).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

import experiments.run_experiments as R


class _ConstReturnModel(torch.nn.Module):
    """Emits a constant scalar (a 'return') for every sample."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        b = x.shape[0]
        return torch.full((b, 1), self.value)


def _loader(x, y):
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_return_mode_reconstructs_price_from_raw_prev():
    # 3 samples: each x window ends at a known raw close (prev_raw).
    # prev_raw values: 100, 200, 400  -> pred_return 0.05 -> pred_price 105,210,420
    prev_raw = torch.tensor([100.0, 200.0, 400.0])
    # window length 2, single feature; last row of each window is prev_raw
    x = torch.stack([
        torch.tensor([[90.0], [100.0]]),
        torch.tensor([[190.0], [200.0]]),
        torch.tensor([[390.0], [400.0]]),
    ])
    y_return = torch.tensor([0.05, 0.05, 0.05])  # true returns (unused for pred recon)
    model = _ConstReturnModel(0.05)
    norm_stats = {"mean": [0.0], "std": [1.0], "return_std": 0.01}

    out = R.evaluate_model(
        model, _loader(x, y_return), torch.device("cpu"), norm_stats,
        needs_denorm=False, target_col=0,
        target_mode="return", prev_closes=prev_raw.numpy(),
    )
    preds = out["predictions"]
    # pred_price = prev_raw * (1 + 0.05)
    assert abs(preds[0] - 105.0) < 1e-4
    assert abs(preds[1] - 210.0) < 1e-4
    assert abs(preds[2] - 420.0) < 1e-4
    # prev_closes echoed in price scale
    assert abs(out["prev_closes"][0] - 100.0) < 1e-4


def test_return_mode_does_not_denormalize_prediction():
    # If denormalize were (wrongly) applied with std=1000, pred would blow up.
    prev_raw = torch.tensor([100.0])
    x = torch.tensor([[[90.0], [100.0]]])
    y_return = torch.tensor([0.05])
    model = _ConstReturnModel(0.05)
    # std deliberately huge: proves denormalize() is NOT applied to the return.
    norm_stats = {"mean": [0.0], "std": [1000.0], "return_std": 0.01}

    out = R.evaluate_model(
        model, _loader(x, y_return), torch.device("cpu"), norm_stats,
        needs_denorm=True, target_col=0,  # even with needs_denorm True
        target_mode="return", prev_closes=prev_raw.numpy(),
    )
    # pred_price must be 105, NOT 100*(1 + 0.05*1000)=5100
    assert abs(out["predictions"][0] - 105.0) < 1e-4


def test_price_mode_unchanged_regression():
    # Price-mode path must behave exactly as before (no target_mode kwarg given).
    x = torch.tensor([[[90.0], [100.0]], [[190.0], [200.0]]])
    y_price = torch.tensor([110.0, 210.0])

    class _ConstPrice(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return torch.full((b, 1), 0.0)  # normalized 0 -> denorm to mean

    norm_stats = {"mean": [50.0], "std": [10.0], "return_std": 0.01}
    out = R.evaluate_model(
        _ConstPrice(), _loader(x, y_price), torch.device("cpu"), norm_stats,
        needs_denorm=True, target_col=0,
    )
    # pred 0 denormalized -> 0*10 + 50 = 50 for both samples
    assert abs(out["predictions"][0] - 50.0) < 1e-4
    assert abs(out["predictions"][1] - 50.0) < 1e-4
