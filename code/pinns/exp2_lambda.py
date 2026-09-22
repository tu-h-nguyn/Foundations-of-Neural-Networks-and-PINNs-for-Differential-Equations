"""TN2 --- Cai gia cua rang buoc mem (Menh de md:phat-bien-phat).

Quet lam_b qua sau bac do lon tren bai toan Poisson cua TN1, do sai so bien
va hoi quy log--log de uoc luong so mu, doi chieu voi du bao lam_b^{-1}.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from .core import DTYPE, FNN, abs_linf, grad, rel_l2, set_seed, train_adam, uniform_1d

PI = math.pi
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def run(lam_b, seed=0, iters=4000, n_col=256):
    set_seed(seed)
    net = FNN([1, 32, 32, 32, 1], act="tanh")
    xr = uniform_1d(n_col)
    xb = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
    xe = uniform_1d(1001)
    f = lambda x: 4 * PI**2 * torch.sin(2 * PI * x)

    def loss_fn():
        x = xr.clone().requires_grad_(True)
        Jr = ((-grad(net(x), x, 2) - f(x)) ** 2).mean()
        Jb = (net(xb) ** 2).mean()
        return Jr + lam_b * Jb, {"Jr": Jr.item(), "Jb": Jb.item()}

    train_adam(net, loss_fn, iters)
    _, parts = loss_fn()
    with torch.no_grad():
        pred, exact = net(xe), torch.sin(2 * PI * xe)
        sai_so_bien = net(xb).abs().max().item()
    return {"lam_b": lam_b, "sai_so_bien": sai_so_bien, "Jr": parts["Jr"],
            "eps_L2": rel_l2(pred, exact), "eps_Linf": abs_linf(pred, exact)}


def main():
    os.makedirs(OUT, exist_ok=True)
    lams = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    rows = [run(l) for l in lams]
    # hoi quy log--log tren dai [0,1 ; 100] dung nhu luan van
    sub = [w for w in rows if 0.1 <= w["lam_b"] <= 100.0]
    A = np.log10([w["lam_b"] for w in sub])
    B = np.log10([w["sai_so_bien"] for w in sub])
    slope, intercept = np.polyfit(A, B, 1)
    print(f"{'lam_b':>9}{'sai so bien':>14}{'eps_L2':>13}{'eps_Linf':>13}{'Jr':>13}")
    for w in rows:
        print(f"{w['lam_b']:9g}{w['sai_so_bien']:14.4e}{w['eps_L2']:13.4e}"
              f"{w['eps_Linf']:13.4e}{w['Jr']:13.4e}")
    best = min(rows, key=lambda w: w["eps_L2"])
    print(f"\nso mu do duoc tren [0,1; 100] = {slope:.4f}   (du bao: -1)")
    print(f"lech so mu = {abs(slope + 1) * 100:.1f}%")
    print(f"eps_L2 nho nhat tai lam_b = {best['lam_b']:g}  -> lam_b toi uu la HUU HAN")
    json.dump({"rows": rows, "so_mu": slope},
              open(os.path.join(OUT, "exp2_lambda.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
