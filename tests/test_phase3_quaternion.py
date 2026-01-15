"""
Phase 3 Tests: Quaternion Operations

Tests fundamental quaternion operations for correctness:
- Hamilton product
- Quaternion conjugate
- Quaternion norm
- QuaternionLinear layer
- Gradient flow
"""

import pytest
import torch
import torch.nn as nn

from src.models.quaternion_ops import (
    hamilton_product,
    quaternion_conjugate,
    quaternion_norm,
    QuaternionLinear,
)


class TestHamiltonProduct:
    """Tests for Hamilton product."""

    def test_shape_preserved(self):
        """Hamilton product preserves input shape."""
        p = torch.randn(10, 4)
        q = torch.randn(10, 4)
        result = hamilton_product(p, q)
        assert result.shape == (10, 4)

    def test_batch_shape_preserved(self):
        """Hamilton product works with batch dimensions."""
        p = torch.randn(5, 10, 4)
        q = torch.randn(5, 10, 4)
        result = hamilton_product(p, q)
        assert result.shape == (5, 10, 4)

    def test_identity_multiplication(self):
        """q * 1 = q where 1 = (1, 0, 0, 0)."""
        q = torch.randn(10, 4)
        identity = torch.zeros(10, 4)
        identity[:, 0] = 1.0  # (1, 0, 0, 0)

        result = hamilton_product(q, identity)
        assert torch.allclose(result, q, atol=1e-6)

    def test_identity_multiplication_left(self):
        """1 * q = q where 1 = (1, 0, 0, 0)."""
        q = torch.randn(10, 4)
        identity = torch.zeros(10, 4)
        identity[:, 0] = 1.0

        result = hamilton_product(identity, q)
        assert torch.allclose(result, q, atol=1e-6)

    def test_known_product_i_j_equals_k(self):
        """i * j = k in quaternion algebra."""
        # i = (0, 1, 0, 0), j = (0, 0, 1, 0), k = (0, 0, 0, 1)
        i = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        j = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        k = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

        result = hamilton_product(i, j)
        assert torch.allclose(result, k, atol=1e-6)

    def test_known_product_j_i_equals_neg_k(self):
        """j * i = -k in quaternion algebra (non-commutative)."""
        i = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        j = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        neg_k = torch.tensor([[0.0, 0.0, 0.0, -1.0]])

        result = hamilton_product(j, i)
        assert torch.allclose(result, neg_k, atol=1e-6)

    def test_known_product_i_squared_equals_neg_one(self):
        """i^2 = -1 in quaternion algebra."""
        i = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        neg_one = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])

        result = hamilton_product(i, i)
        assert torch.allclose(result, neg_one, atol=1e-6)

    def test_known_product_j_squared_equals_neg_one(self):
        """j^2 = -1 in quaternion algebra."""
        j = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        neg_one = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])

        result = hamilton_product(j, j)
        assert torch.allclose(result, neg_one, atol=1e-6)

    def test_known_product_k_squared_equals_neg_one(self):
        """k^2 = -1 in quaternion algebra."""
        k = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        neg_one = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])

        result = hamilton_product(k, k)
        assert torch.allclose(result, neg_one, atol=1e-6)

    def test_associativity(self):
        """(p * q) * r = p * (q * r)."""
        torch.manual_seed(42)
        p = torch.randn(5, 4)
        q = torch.randn(5, 4)
        r = torch.randn(5, 4)

        left = hamilton_product(hamilton_product(p, q), r)
        right = hamilton_product(p, hamilton_product(q, r))

        assert torch.allclose(left, right, atol=1e-5)

    def test_non_commutativity(self):
        """Hamilton product is NOT commutative: p * q != q * p in general."""
        torch.manual_seed(42)
        p = torch.randn(5, 4)
        q = torch.randn(5, 4)

        pq = hamilton_product(p, q)
        qp = hamilton_product(q, p)

        # They should NOT be equal in general
        assert not torch.allclose(pq, qp, atol=1e-6)

    def test_distributivity_left(self):
        """p * (q + r) = p * q + p * r."""
        torch.manual_seed(42)
        p = torch.randn(5, 4)
        q = torch.randn(5, 4)
        r = torch.randn(5, 4)

        left = hamilton_product(p, q + r)
        right = hamilton_product(p, q) + hamilton_product(p, r)

        assert torch.allclose(left, right, atol=1e-5)

    def test_distributivity_right(self):
        """(p + q) * r = p * r + q * r."""
        torch.manual_seed(42)
        p = torch.randn(5, 4)
        q = torch.randn(5, 4)
        r = torch.randn(5, 4)

        left = hamilton_product(p + q, r)
        right = hamilton_product(p, r) + hamilton_product(q, r)

        assert torch.allclose(left, right, atol=1e-5)


class TestQuaternionConjugate:
    """Tests for quaternion conjugate."""

    def test_shape_preserved(self):
        """Conjugate preserves shape."""
        q = torch.randn(10, 4)
        result = quaternion_conjugate(q)
        assert result.shape == (10, 4)

    def test_conjugate_correctness(self):
        """Conjugate of (a, b, c, d) is (a, -b, -c, -d)."""
        q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        expected = torch.tensor([[1.0, -2.0, -3.0, -4.0]])

        result = quaternion_conjugate(q)
        assert torch.allclose(result, expected)

    def test_double_conjugate_is_identity(self):
        """Conjugate of conjugate is original: conj(conj(q)) = q."""
        q = torch.randn(10, 4)
        result = quaternion_conjugate(quaternion_conjugate(q))
        assert torch.allclose(result, q, atol=1e-6)

    def test_conjugate_of_product(self):
        """conj(p * q) = conj(q) * conj(p)."""
        torch.manual_seed(42)
        p = torch.randn(5, 4)
        q = torch.randn(5, 4)

        left = quaternion_conjugate(hamilton_product(p, q))
        right = hamilton_product(quaternion_conjugate(q), quaternion_conjugate(p))

        assert torch.allclose(left, right, atol=1e-5)

    def test_q_times_conjugate_is_norm_squared(self):
        """q * conj(q) = ||q||^2 as real quaternion."""
        torch.manual_seed(42)
        q = torch.randn(5, 4)

        result = hamilton_product(q, quaternion_conjugate(q))
        norm_sq = quaternion_norm(q) ** 2

        # Result should be (||q||^2, 0, 0, 0)
        assert torch.allclose(result[:, 0], norm_sq, atol=1e-5)
        assert torch.allclose(result[:, 1:], torch.zeros(5, 3), atol=1e-5)


class TestQuaternionNorm:
    """Tests for quaternion norm."""

    def test_shape_correct(self):
        """Norm reduces last dimension."""
        q = torch.randn(10, 4)
        result = quaternion_norm(q)
        assert result.shape == (10,)

    def test_batch_shape(self):
        """Norm works with batch dimensions."""
        q = torch.randn(5, 10, 4)
        result = quaternion_norm(q)
        assert result.shape == (5, 10)

    def test_norm_correctness(self):
        """Norm of (3, 4, 0, 0) is 5."""
        q = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
        result = quaternion_norm(q)
        assert torch.isclose(result[0], torch.tensor(5.0))

    def test_norm_non_negative(self):
        """Norm is always non-negative."""
        q = torch.randn(100, 4)
        result = quaternion_norm(q)
        assert (result >= 0).all()

    def test_norm_zero_for_zero_quaternion(self):
        """Norm of zero quaternion is zero."""
        q = torch.zeros(1, 4)
        result = quaternion_norm(q)
        assert torch.isclose(result[0], torch.tensor(0.0))

    def test_norm_multiplicative(self):
        """||p * q|| = ||p|| * ||q||."""
        torch.manual_seed(42)
        p = torch.randn(10, 4)
        q = torch.randn(10, 4)

        pq = hamilton_product(p, q)
        norm_pq = quaternion_norm(pq)
        norm_p_times_norm_q = quaternion_norm(p) * quaternion_norm(q)

        assert torch.allclose(norm_pq, norm_p_times_norm_q, atol=1e-5)


class TestQuaternionLinear:
    """Tests for QuaternionLinear layer."""

    def test_output_shape(self):
        """Output has correct shape."""
        layer = QuaternionLinear(in_features=4, out_features=8)
        x = torch.randn(16, 4, 4)  # (batch, in_features, 4)
        output = layer(x)
        assert output.shape == (16, 8, 4)

    def test_batch_independence(self):
        """Each batch element processed independently."""
        layer = QuaternionLinear(in_features=4, out_features=8)
        x = torch.randn(16, 4, 4)

        # Process all at once
        output_batch = layer(x)

        # Process individually
        for i in range(16):
            output_single = layer(x[i : i + 1])
            assert torch.allclose(output_batch[i], output_single[0], atol=1e-6)

    def test_parameter_count(self):
        """Correct number of parameters."""
        layer = QuaternionLinear(in_features=4, out_features=8)
        # Weight: (8, 4, 4) = 128 params
        # Bias: (8, 4) = 32 params
        # Total: 160
        total_params = sum(p.numel() for p in layer.parameters())
        assert total_params == 160

    def test_zero_input_gives_bias(self):
        """Zero input should give output equal to bias."""
        layer = QuaternionLinear(in_features=4, out_features=8)
        x = torch.zeros(1, 4, 4)
        output = layer(x)
        assert torch.allclose(output[0], layer.bias, atol=1e-6)


class TestGradientFlow:
    """Tests for gradient flow through quaternion operations."""

    def test_hamilton_product_gradient(self):
        """Gradients flow through Hamilton product."""
        p = torch.randn(5, 4, requires_grad=True)
        q = torch.randn(5, 4, requires_grad=True)

        result = hamilton_product(p, q)
        loss = result.sum()
        loss.backward()

        assert p.grad is not None
        assert q.grad is not None
        assert not torch.isnan(p.grad).any()
        assert not torch.isnan(q.grad).any()

    def test_conjugate_gradient(self):
        """Gradients flow through conjugate."""
        q = torch.randn(5, 4, requires_grad=True)

        result = quaternion_conjugate(q)
        loss = result.sum()
        loss.backward()

        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_norm_gradient(self):
        """Gradients flow through norm."""
        q = torch.randn(5, 4, requires_grad=True)

        result = quaternion_norm(q)
        loss = result.sum()
        loss.backward()

        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_quaternion_linear_gradient(self):
        """Gradients flow through QuaternionLinear."""
        layer = QuaternionLinear(in_features=4, out_features=8)
        x = torch.randn(16, 4, 4, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check parameter gradients
        for param in layer.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_chained_operations_gradient(self):
        """Gradients flow through chain of operations."""
        layer = QuaternionLinear(in_features=4, out_features=8)
        x = torch.randn(16, 4, 4, requires_grad=True)

        # Chain: linear -> hamilton product with self -> conjugate -> norm
        output = layer(x)
        product = hamilton_product(output, output)
        conj = quaternion_conjugate(product)
        norm = quaternion_norm(conj)

        loss = norm.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_no_nan_in_backward_pass(self):
        """No NaN values during backward pass with random inputs."""
        torch.manual_seed(42)

        for _ in range(10):
            layer = QuaternionLinear(in_features=8, out_features=16)
            x = torch.randn(32, 8, 4, requires_grad=True)

            output = layer(x)
            loss = output.pow(2).sum()
            loss.backward()

            assert not torch.isnan(x.grad).any()
            for param in layer.parameters():
                assert not torch.isnan(param.grad).any()


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_small_values(self):
        """Operations handle small values."""
        p = torch.ones(5, 4) * 1e-8
        q = torch.ones(5, 4) * 1e-8

        result = hamilton_product(p, q)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_large_values(self):
        """Operations handle large values."""
        p = torch.ones(5, 4) * 1e6
        q = torch.ones(5, 4) * 1e6

        result = hamilton_product(p, q)
        assert not torch.isnan(result).any()

    def test_mixed_scale_values(self):
        """Operations handle mixed scale values."""
        p = torch.randn(5, 4) * 1e-4
        q = torch.randn(5, 4) * 1e4

        result = hamilton_product(p, q)
        assert not torch.isnan(result).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
