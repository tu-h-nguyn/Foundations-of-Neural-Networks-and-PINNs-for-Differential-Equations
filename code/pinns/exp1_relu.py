"""TN1 --- Ham kich hoat va su suy bien cua ReLU (Hq. hq:pinn-hong).

-u'' = f tren (0,1), u(0)=u(1)=0, u* = sin(2 pi x), f = 4 pi^2 sin(2 pi x).
Khang dinh la mot DANG THUC CHINH XAC (grad = 0), nen kiem chung toi chu so may.
"""
from __future__ import annotations

import json
import math
import os

import torch

from .core import DTYPE, FNN, abs_linf, flat_grad, grad, rel_l2, set_seed, train_adam, uniform_1d

PI = math.pi
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def run(act: str, seed=0, iters=4000, n_col=256, lam_b=100.0):
    set_seed(seed)
    net = FNN([1, 32, 32, 32, 1], act=act)
    xr = uniform_1d(n_col)          # quy uoc luan van: ke ca diem bien
    xb = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
    xe = torch.linspace(0, 1, 1001, dtype=DTYPE).reshape(-1, 1)
    f = lambda x: 4 * PI**2 * torch.sin(2 * PI * x)

    def parts():
        x = xr.clone().requires_grad_(True)
        Jr = ((-grad(net(x), x, 2) - f(x)) ** 2).mean()
        Jb = (net(xb) ** 2).mean()
        return Jr, Jb

    # do dao ham bac hai va gradient TAI KHOI TAO
    x0 = xe.clone().requires_grad_(True)
    uxx0 = grad(net(x0), x0, 2)
    max_uxx = uxx0.abs().max().item()
    Jr0, _ = parts()
    g0 = flat_grad(Jr0, list(net.parameters()))
    gmax0, gl2_0 = g0.abs().max().item(), g0.norm().item()

    loss_fn = lambda: ((lambda Jr, Jb: (Jr + lam_b * Jb,
                                        {"Jr": Jr.item(), "Jb": Jb.item()}))(*parts()))
    train_adam(net, loss_fn, iters)

    Jr1, _ = parts()
    with torch.no_grad():
        pred, exact = net(xe), torch.sin(2 * PI * xe)
    return {
        "act": act,
        "max_uxx_khoi_tao": max_uxx,
        "grad_Jr_inf_khoi_tao": gmax0,
        "grad_Jr_l2_khoi_tao": gl2_0,
        "Jr_cuoi": Jr1.item(),
        "eps_L2": rel_l2(pred, exact),
        "eps_Linf": abs_linf(pred, exact),
        "so_tham_so": net.n_params(),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    # gia tri tham chieu lien tuc: trung binh cua f^2 tren (0,1) = 8 pi^4
    res = {"f2_lien_tuc": 8 * PI**4, "cau_hinh": [run("tanh"), run("relu")]}
    xe = torch.linspace(0, 1, 1001, dtype=DTYPE)
    res["f2_tren_luoi_danh_gia"] = ((4 * PI**2 * torch.sin(2 * PI * xe)) ** 2).mean().item()
    print(f"trung binh f^2 (lien tuc)        = {res['f2_lien_tuc']:.3f}")
    print(f"trung binh f^2 (luoi 1001 diem)  = {res['f2_tren_luoi_danh_gia']:.3f}\n")
    hdr = f"{'act':<6}{'max|u_xx| kt':>15}{'||gJr||_inf kt':>16}{'Jr cuoi':>13}{'eps_L2':>12}{'eps_Linf':>12}"
    print(hdr)
    for c in res["cau_hinh"]:
        print(f"{c['act']:<6}{c['max_uxx_khoi_tao']:15.6e}{c['grad_Jr_inf_khoi_tao']:16.6e}"
              f"{c['Jr_cuoi']:13.4e}{c['eps_L2']:12.4e}{c['eps_Linf']:12.4e}")
    t, r = res["cau_hinh"]
    print(f"\nchenh lech do chinh xac tanh/ReLU = {r['eps_L2']/t['eps_L2']:.4g} lan")
    print(f"so tham so = {t['so_tham_so']}")
    with open(os.path.join(OUT, "exp1_relu.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
