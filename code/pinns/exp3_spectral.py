"""TN3 --- Thien kien pho va su dao chieu do toan tu vi phan.

Nghiem che tao da tan so, bien do BANG NHAU (dieu kien thiet ke then chot):
    u*(x) = sum_{k in {1,4,8}} sin(2 pi k x),  f(x) = sum 4 pi^2 k^2 sin(2 pi k x).

Ba che do tren CUNG muc tieu, CUNG kien truc, CUNG hat giong:
  (a) hoi quy truc tiep      -> P == 1
  (b) mat mat phan du        -> P(k) = 4 pi^2 k^2
  (c) phan du + dac trung Fourier (M = 32, s = 6)

Theo doi he so sai so theo mode  c_k(n) = 2 * int_0^1 (u_theta - u*) sin(2 pi k x) dx
va t10(k) = vong lap dau tien |c_k| giam con 10% gia tri ban dau.

LUU Y khi doc: t10 la thoi diem CHAM NGUONG LAN DAU va KHONG don dieu; can cu
dung de so sanh thu tu uu tien la |c_k| tai diem dung, khong phai t10.
"""
from __future__ import annotations

import json
import math
import os

import torch

from .core import DTYPE, FNN, grad, rel_l2, set_seed, train_adam, uniform_1d

PI = math.pi
KS = (1, 4, 8)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def u_exact(x):
    return sum(torch.sin(2 * PI * k * x) for k in KS)


def f_rhs(x):
    return sum(4 * PI**2 * k**2 * torch.sin(2 * PI * k * x) for k in KS)


def mode_coeffs(model, xe):
    """c_k = 2 * int (u_theta - u*) sin(2 pi k x) dx, quy tac hinh thang."""
    with torch.no_grad():
        e = (model(xe) - u_exact(xe)).squeeze()
        xs = xe.squeeze()
        return {k: abs(2 * torch.trapz(e * torch.sin(2 * PI * k * xs), xs).item())
                for k in KS}


def run(mode: str, seed=0, iters=8000, n_col=256, lam_b=100.0, every=100):
    set_seed(seed)
    fm, fs = (32, 6.0) if mode == "fourier" else (0, 1.0)
    net = FNN([1, 64, 64, 64, 1], act="tanh", fourier_m=fm, fourier_s=fs)
    xr, xb, xe = uniform_1d(n_col), torch.tensor([[0.0], [1.0]], dtype=DTYPE), uniform_1d(2001)

    def loss_fn():
        if mode == "regression":
            J = ((net(xr) - u_exact(xr)) ** 2).mean()
            return J, {"J": J.item()}
        x = xr.clone().requires_grad_(True)
        Jr = ((-grad(net(x), x, 2) - f_rhs(x)) ** 2).mean()
        Jb = (net(xb) ** 2).mean()
        return Jr + lam_b * Jb, {"Jr": Jr.item(), "Jb": Jb.item()}

    c0 = mode_coeffs(net, xe)
    t10 = {k: None for k in KS}
    hist = [(0, dict(c0))]
    for it in range(0, iters, every):
        train_adam(net, loss_fn, every)
        c = mode_coeffs(net, xe)
        hist.append((it + every, dict(c)))
        for k in KS:
            if t10[k] is None and c[k] <= 0.1 * c0[k]:
                t10[k] = it + every
    return {"mode": mode, "c_dau": c0, "c_cuoi": hist[-1][1], "t10": t10,
            "eps_L2": rel_l2(net(xe).detach(), u_exact(xe)), "lich_su": hist}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = [run(m) for m in ("regression", "residual", "fourier")]
    ten = {"regression": "(a) Hoi quy", "residual": "(b) Phan du",
           "fourier": "(c) Phan du+Fourier"}
    print(f"{'che do':<22}{'t10(1)':>9}{'t10(4)':>9}{'t10(8)':>9}"
          f"{'|c1|':>11}{'|c4|':>11}{'|c8|':>11}{'eps_L2':>12}")
    for r in res:
        t = [("---" if r["t10"][k] is None else str(r["t10"][k])) for k in KS]
        c = [r["c_cuoi"][k] for k in KS]
        print(f"{ten[r['mode']]:<22}{t[0]:>9}{t[1]:>9}{t[2]:>9}"
              f"{c[0]:11.4f}{c[1]:11.4f}{c[2]:11.4f}{r['eps_L2']:12.4e}")
    b = next(r for r in res if r["mode"] == "residual")
    print(f"\nche do (b): |c1|/|c8| = {b['c_cuoi'][1]/b['c_cuoi'][8]:.1f}"
          f"  -> sai so don ve mode TAN SO THAP nhat")
    f_ = next(r for r in res if r["mode"] == "fourier")
    print(f"nhung Fourier cai thien {b['eps_L2']/f_['eps_L2']:.0f} lan")
    json.dump(res, open(os.path.join(OUT, "exp3_spectral.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
