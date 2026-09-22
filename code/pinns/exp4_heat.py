"""TN4 --- Phuong trinh khuech tan voi nghiem giai tich.

  du/dt = nu d2u/dx2 tren (0,1)x(0,1],  nu = 0,1
  u(x,0) = sin(pi x),  u(0,t) = u(1,t) = 0
  nghiem dung: u*(x,t) = sin(pi x) exp(-nu pi^2 t)

Doi chung "lanh manh": nu khong nho nen hang so on dinh o thang O(1), va
nghiem chi chua mot mode Fourier duy nhat nen khong co van de thien kien pho.
So sanh trong so co dinh voi thuat toan u trong so (alg:annealing).
"""
from __future__ import annotations

import json
import math
import os

import torch

from .core import (
    DTYPE,
    FNN,
    abs_linf,
    grad,
    imbalance,
    rel_l2,
    set_seed,
    train_lbfgs,
)

PI = math.pi
NU = 0.1
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def u_exact(X):
    return torch.sin(PI * X[:, 0:1]) * torch.exp(-NU * PI**2 * X[:, 1:2])


def sample(seed, Nr=2000, Nb=300, N0=200):
    g = torch.Generator().manual_seed(seed)
    Xr = torch.rand(Nr, 2, generator=g, dtype=DTYPE)
    tb = torch.rand(Nb, 1, generator=g, dtype=DTYPE)
    Xb = torch.cat([torch.cat([torch.zeros_like(tb), tb], 1),
                    torch.cat([torch.ones_like(tb), tb], 1)], 0)
    x0 = torch.rand(N0, 1, generator=g, dtype=DTYPE)
    X0 = torch.cat([x0, torch.zeros_like(x0)], 1)
    return Xr, Xb, X0


def build_loss(net, Xr, Xb, X0, lam):
    u0 = torch.sin(PI * X0[:, 0:1])

    def parts():
        X = Xr.clone().requires_grad_(True)
        u = net(X)
        g1 = grad(u, X)
        ut, ux = g1[:, 1:2], g1[:, 0:1]
        uxx = grad(ux, X)[:, 0:1]
        Jr = ((ut - NU * uxx) ** 2).mean()
        Jb = (net(Xb) ** 2).mean()
        J0 = ((net(X0) - u0) ** 2).mean()
        return Jr, Jb, J0

    def loss_fn():
        Jr, Jb, J0 = parts()
        J = Jr + lam["b"] * Jb + lam["0"] * J0
        return J, {"Jr": Jr.item(), "Jb": Jb.item(), "J0": J0.item()}

    return parts, loss_fn


def run(anneal: bool, seed=0, adam_iters=8000, lbfgs_iters=500,
        alpha=0.1, N_up=100):
    set_seed(seed)
    net = FNN([2, 32, 32, 32, 32, 1], act="tanh")
    Xr, Xb, X0 = sample(seed)
    lam = {"b": 1.0, "0": 1.0}
    parts, loss_fn = build_loss(net, Xr, Xb, X0, lam)
    params = list(net.parameters())

    gx = torch.linspace(0, 1, 101, dtype=DTYPE)
    XX = torch.stack(torch.meshgrid(gx, gx, indexing="ij"), -1).reshape(-1, 2)
    ue = u_exact(XX)

    rho_hist, max_rho = [], 0.0
    for it in range(adam_iters + 1):
        if it % N_up == 0:
            Jr, Jb, _ = parts()
            r = imbalance(Jr, Jb, params)
            rho_hist.append((it, r))
            max_rho = max(max_rho, r)
            if anneal:
                lam["b"] = (1 - alpha) * lam["b"] + alpha * r
                Jr2, _, J02 = parts()
                lam["0"] = (1 - alpha) * lam["0"] + alpha * imbalance(Jr2, J02, params)
        loss, _ = loss_fn()
        for p in params:
            p.grad = None
        loss.backward()
        if it == 0:
            opt = torch.optim.Adam(params, lr=1e-3)
        opt.step()

    with torch.no_grad():
        e_adam = rel_l2(net(XX), ue)
    train_lbfgs(net, loss_fn, max_iter=lbfgs_iters)
    with torch.no_grad():
        pred = net(XX)
        e_l, e_i = rel_l2(pred, ue), abs_linf(pred, ue)
    return {"u_trong_so": anneal, "eps_L2_sau_Adam": e_adam,
            "eps_L2_cuoi": e_l, "eps_Linf_cuoi": e_i,
            "lam_b_cuoi": lam["b"], "max_rho_b": max_rho,
            "rho_dau": rho_hist[0][1], "rho_cuoi": rho_hist[-1][1],
            "rho_hist": rho_hist}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = [run(False), run(True)]
    print(f"{'chien luoc':<24}{'sau Adam':>12}{'cuoi':>12}{'eps_Linf':>12}"
          f"{'lam_b cuoi':>13}{'max rho_b':>12}{'rho dau':>10}{'rho cuoi':>10}")
    for r in res:
        ten = "U trong so" if r["u_trong_so"] else "Trong so co dinh"
        print(f"{ten:<24}{r['eps_L2_sau_Adam']:12.4e}{r['eps_L2_cuoi']:12.4e}"
              f"{r['eps_Linf_cuoi']:12.4e}{r['lam_b_cuoi']:13.4g}"
              f"{r['max_rho_b']:12.1f}{r['rho_dau']:10.1f}{r['rho_cuoi']:10.1f}")
    a, b = res
    print(f"\nL-BFGS cai thien {a['eps_L2_sau_Adam']/a['eps_L2_cuoi']:.1f} lan")
    print(f"U trong so lam xau {b['eps_L2_cuoi']/a['eps_L2_cuoi']:.1f} lan")
    print(f"rho_b (trong so co dinh): {a['rho_dau']:.1f} -> {a['rho_cuoi']:.1f}"
          f"  ({'GIAM' if a['rho_cuoi'] < a['rho_dau'] else 'TANG'})")
    json.dump(res, open(os.path.join(OUT, "exp4_heat.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
