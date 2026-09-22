"""Thanh phan dung chung cho moi thi nghiem PINN cua luan van.

Moi lua chon thiet ke o day deu tuong ung voi mot ket qua trong Chuong 2-3:

  - tanh          : Hq. 2.x (sigma'' != 0 la dieu kien can cho toan tu bac hai)
  - Xavier        : eq:xavier, Var(w) = 2/(n_in + n_out)
  - bias = 0      : gia thiet (A2) cua phan tich lan truyen phuong sai
  - lop ra tuyen tinh : quy uoc eq:fnn-output
  - float64       : de phan biet sai so xap xi voi sai so lam tron
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def set_seed(seed: int) -> None:
    """Co dinh moi nguon ngau nhien de tai lap duoc."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def grad(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Dao ham cap `order` cua y theo x, giu lai do thi de vi phan tiep.

    Hien thuc truc tiep cua hai he thuc truy hoi eq:G-truy-hoi va
    eq:S-truy-hoi; co `create_graph=True` la dieu kien de lay tiep dao ham
    theo tham so (xem Muc 3.2.6).
    """
    for _ in range(order):
        y = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return y


class FNN(nn.Module):
    """Mang truyen thang, tuy chon nhung dac trung Fourier.

    layers: vi du [1, 32, 32, 32, 1] cho kien truc bai toan 1D cua luan van.
    """

    def __init__(self, layers, act="tanh", fourier_m=0, fourier_s=1.0, seed=None):
        super().__init__()
        if seed is not None:
            set_seed(seed)
        self.fourier_m = fourier_m
        if fourier_m > 0:
            B = torch.randn(fourier_m, layers[0]) * fourier_s
            self.register_buffer("B", B)          # co dinh, khong hoc
            layers = [2 * fourier_m] + list(layers[1:])
        self.lins = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        for lin in self.lins:                     # khoi tao Xavier, eq:xavier
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.act = {"tanh": torch.tanh, "relu": torch.relu}[act]

    def forward(self, x):
        if self.fourier_m > 0:
            p = 2 * math.pi * x @ self.B.T
            x = torch.cat([torch.cos(p), torch.sin(p)], dim=1)
        for lin in self.lins[:-1]:
            x = self.act(lin(x))
        return self.lins[-1](x)                   # lop ra tuyen tinh

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def uniform_1d(n: int, a: float = 0.0, b: float = 1.0) -> torch.Tensor:
    """n diem phoi tri deu tren [a,b], KE CA hai diem bien.

    Day dung la quy uoc cua luan van: voi n = 256 tren (0,1) va
    f = 4 pi^2 sin(2 pi x), trung binh f^2 tren luoi nay bang 776,229 --
    khop voi gia tri J_r bao cao o TN1.
    """
    return torch.linspace(a, b, n, dtype=DTYPE).reshape(-1, 1)


# ----------------------------------------------------------------------
# Chi tieu sai so (Dinh nghia 5.2)
# ----------------------------------------------------------------------
def rel_l2(pred: torch.Tensor, exact: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - exact) / torch.linalg.norm(exact)).item()


def abs_linf(pred: torch.Tensor, exact: torch.Tensor) -> float:
    return (pred - exact).abs().max().item()


# ----------------------------------------------------------------------
# Chi so mat can bang gradient (Dinh nghia 5.1)
# ----------------------------------------------------------------------
def flat_grad(loss: torch.Tensor, params) -> torch.Tensor:
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).reshape(-1)
        for g, p in zip(gs, params, strict=True)
    ])


def imbalance(loss_r: torch.Tensor, loss_i: torch.Tensor, params) -> float:
    """rho_i = max|grad J_r| / mean|grad J_i|."""
    gr = flat_grad(loss_r, params)
    gi = flat_grad(loss_i, params)
    denom = gi.abs().mean().item()
    return float("inf") if denom == 0.0 else gr.abs().max().item() / denom


# ----------------------------------------------------------------------
# Huan luyen hai giai doan Adam -> L-BFGS (Thuat toan 3.3)
# ----------------------------------------------------------------------
def train_adam(net, loss_fn, iters, lr=1e-3, log_every=0, log=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for it in range(iters + 1):
        loss, parts = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log is not None and log_every and it % log_every == 0:
            log.append((it, loss.item(), {k: v for k, v in parts.items()}))
    return net


def train_lbfgs(net, loss_fn, max_iter=500, history_size=50):
    opt = torch.optim.LBFGS(
        net.parameters(),
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",   # Bo de 3.x: dieu kien Wolfe giu H_k xac dinh duong
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
    )

    def closure():
        opt.zero_grad()
        loss, _ = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    return net
