"""Tests for pinns.core.

Each test checks a value the mathematics fixes independently of the
implementation: an autodiff derivative against the analytic one, an error
metric against a hand-computed number, an initialisation against the variance
the thesis derives. Nothing here asserts "it ran".
"""
import math

import pytest
import torch

from pinns.core import (
    DTYPE,
    FNN,
    abs_linf,
    flat_grad,
    grad,
    imbalance,
    rel_l2,
    set_seed,
    train_adam,
    uniform_1d,
)


# ---------------------------------------------------------------- grad
def test_grad_matches_the_analytic_first_derivative():
    x = torch.linspace(0.1, 1.0, 40, dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
    y = torch.sin(3 * x) * torch.exp(-x)
    got = grad(y, x)
    want = 3 * torch.cos(3 * x) * torch.exp(-x) - torch.sin(3 * x) * torch.exp(-x)
    assert torch.allclose(got, want, atol=1e-12)


def test_grad_matches_the_analytic_second_derivative():
    # u = sin(2 pi x) is the thesis's test function; u'' = -(2 pi)^2 u exactly,
    # which is the identity every PINN residual in Chapter 5 is built on.
    x = torch.linspace(0.0, 1.0, 64, dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
    u = torch.sin(2 * math.pi * x)
    assert torch.allclose(grad(u, x, order=2), -(2 * math.pi) ** 2 * u, atol=1e-10)


def test_grad_keeps_the_graph_so_a_third_derivative_exists():
    x = torch.linspace(0.2, 1.2, 16, dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
    y = x ** 4
    assert torch.allclose(grad(y, x, order=3), 24 * x, atol=1e-9)


# ---------------------------------------------------------------- metrics
def test_rel_l2_is_the_ratio_of_norms():
    pred = torch.tensor([[3.0], [4.0]], dtype=DTYPE)
    exact = torch.tensor([[0.0], [0.0]], dtype=DTYPE) + torch.tensor([[3.0], [0.0]], dtype=DTYPE)
    # ||pred - exact|| = 4, ||exact|| = 3
    assert rel_l2(pred, exact) == pytest.approx(4.0 / 3.0, rel=1e-12)


def test_rel_l2_is_zero_on_an_exact_match():
    v = torch.tensor([[1.5], [-2.0], [0.25]], dtype=DTYPE)
    assert rel_l2(v, v) == pytest.approx(0.0, abs=1e-15)


def test_abs_linf_is_the_largest_pointwise_gap():
    pred = torch.tensor([[1.0], [2.0], [3.0]], dtype=DTYPE)
    exact = torch.tensor([[1.1], [2.5], [2.9]], dtype=DTYPE)
    assert abs_linf(pred, exact) == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------- grid
def test_uniform_1d_includes_both_endpoints():
    x = uniform_1d(5, 0.0, 1.0)
    assert x.shape == (5, 1)
    assert x[0].item() == pytest.approx(0.0)
    assert x[-1].item() == pytest.approx(1.0)
    assert torch.allclose(x.diff(dim=0), torch.full((4, 1), 0.25, dtype=DTYPE))


def test_uniform_1d_reproduces_the_mean_square_quoted_for_experiment_1():
    # core.py's docstring claims: with n = 256 on [0,1] and f = 4 pi^2 sin(2 pi x),
    # the mean of f^2 over this grid is 776.229, matching J_r reported in TN1.
    x = uniform_1d(256, 0.0, 1.0)
    f = 4 * math.pi ** 2 * torch.sin(2 * math.pi * x)
    assert (f ** 2).mean().item() == pytest.approx(776.229, rel=1e-5)


# ---------------------------------------------------------------- network
def test_fnn_shapes_and_parameter_count():
    net = FNN([1, 8, 8, 1], seed=0)
    # weights + biases for 1->8, 8->8, 8->1
    assert net.n_params() == (1 * 8 + 8) + (8 * 8 + 8) + (8 * 1 + 1)
    assert net(uniform_1d(13)).shape == (13, 1)


def test_fnn_biases_start_at_zero_as_assumption_A2_requires():
    net = FNN([1, 16, 16, 1], seed=3)
    for lin in net.lins:
        assert torch.count_nonzero(lin.bias) == 0


def test_fnn_output_layer_is_linear_not_squashed():
    # A tanh on the output would bound it to (-1, 1); the thesis's convention is
    # a linear output layer, so a large input must be able to leave that range.
    net = FNN([1, 8, 1], seed=1)
    with torch.no_grad():
        for lin in net.lins:
            lin.weight.mul_(25.0)
    out = net(torch.tensor([[5.0]], dtype=DTYPE))
    assert out.abs().item() > 1.0


def test_same_seed_gives_identical_networks_and_different_seeds_do_not():
    a = FNN([1, 8, 1], seed=11)
    b = FNN([1, 8, 1], seed=11)
    c = FNN([1, 8, 1], seed=12)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters(), strict=True))


def test_fourier_features_change_the_input_width_not_the_output():
    net = FNN([1, 8, 1], fourier_m=6, seed=2)
    assert net.lins[0].in_features == 12      # cos and sin of 6 frequencies
    assert net(uniform_1d(7)).shape == (7, 1)


def test_relu_network_has_zero_second_derivative_which_is_why_the_thesis_rejects_it():
    # The stated reason tanh is used: sigma'' != 0 is necessary for a second-order
    # operator. A ReLU net is piecewise linear, so u_xx vanishes wherever it exists.
    net = FNN([1, 16, 16, 1], act="relu", seed=5)
    x = uniform_1d(64, 0.05, 0.95).requires_grad_(True)
    assert torch.allclose(grad(net(x), x, order=2), torch.zeros_like(x), atol=1e-12)


def test_tanh_network_has_a_nonzero_second_derivative():
    net = FNN([1, 16, 16, 1], act="tanh", seed=5)
    x = uniform_1d(64, 0.05, 0.95).requires_grad_(True)
    assert grad(net(x), x, order=2).abs().max().item() > 1e-6


# ---------------------------------------------------------------- gradients
def test_flat_grad_returns_one_entry_per_parameter():
    net = FNN([1, 4, 1], seed=7)
    x = uniform_1d(10).requires_grad_(True)
    g = flat_grad((net(x) ** 2).mean(), list(net.parameters()))
    assert g.numel() == net.n_params()


def test_imbalance_is_the_ratio_the_definition_states():
    net = FNN([1, 4, 1], seed=9)
    x = uniform_1d(10).requires_grad_(True)
    params = list(net.parameters())
    lr, li = (net(x) ** 2).mean(), (net(x) ** 2).sum()
    gr, gi = flat_grad(lr, params), flat_grad(li, params)
    assert imbalance(lr, li, params) == pytest.approx(
        gr.abs().max().item() / gi.abs().mean().item(), rel=1e-12)


def test_imbalance_is_infinite_when_the_second_loss_has_no_gradient():
    net = FNN([1, 4, 1], seed=9)
    x = uniform_1d(10).requires_grad_(True)
    params = list(net.parameters())
    const = torch.zeros((), dtype=DTYPE, requires_grad=True) * 0
    assert math.isinf(imbalance((net(x) ** 2).mean(), const + 0 * net(x).sum(), params))


# ---------------------------------------------------------------- training
def test_adam_reduces_a_loss_it_can_actually_reduce():
    set_seed(0)
    net = FNN([1, 16, 16, 1], seed=0)
    x = uniform_1d(64)
    target = torch.sin(2 * math.pi * x)

    def loss_fn():
        L = ((net(x) - target) ** 2).mean()
        return L, {"fit": L.item()}

    before = loss_fn()[0].item()
    train_adam(net, loss_fn, iters=200, lr=1e-2)
    assert loss_fn()[0].item() < 0.5 * before


def test_set_seed_makes_a_training_run_reproducible():
    def run():
        set_seed(4)
        net = FNN([1, 8, 1], seed=4)
        x = uniform_1d(32)
        t = torch.cos(math.pi * x)

        def loss_fn():
            L = ((net(x) - t) ** 2).mean()
            return L, {}

        train_adam(net, loss_fn, iters=25, lr=1e-2)
        return loss_fn()[0].item()

    assert run() == pytest.approx(run(), rel=1e-12)
