"""TN5 --- Phuong trinh Burgers do nhot nho.

  du/dt + u du/dx - nu d2u/dx2 = 0 tren (-1,1)x(0,1],  nu = 0,01/pi
  u(x,0) = -sin(pi x),  u(-1,t) = u(1,t) = 0

Bai toan ma CA BA tang cua Bang tab:ba-nguyen-nhan cung bat loi.

Nghiem tham chieu: sai phan huu han bac hai + Runge--Kutta bac bon tuong minh,
dang bao toan (u^2/2)_x sai phan trung tam, khuech tan trung tam ba diem.

Cau hinh L-BFGS (max_iter = 800, history_size = 50, Wolfe manh) lay dung theo
Doan ma 5.3 cua luan van. Rieng TN4 dung 500 vong theo dung mo ta trong Muc 5.4.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from .core import DTYPE, FNN, abs_linf, grad, imbalance, rel_l2, set_seed, train_lbfgs

PI = math.pi
NU = 0.01 / PI
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


# ----------------------------------------------------------------------
# Nghiem tham chieu bang sai phan huu han
# ----------------------------------------------------------------------
def reference(Nx=2047, dt=5e-5, T=1.0, n_save=201):
    """Tra ve (x_full, t_save, U) voi U[i] la nghiem tai t_save[i]."""
    x = np.linspace(-1.0, 1.0, Nx + 2)
    dx = x[1] - x[0]
    u = -np.sin(PI * x)
    u[0] = u[-1] = 0.0

    def rhs(u):
        flux = 0.5 * u**2
        du = np.zeros_like(u)
        du[1:-1] = (-(flux[2:] - flux[:-2]) / (2 * dx)
                    + NU * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2)
        return du

    nsteps = int(round(T / dt))
    save_every = max(1, nsteps // (n_save - 1))
    ts, Us = [0.0], [u.copy()]
    for n in range(1, nsteps + 1):
        k1 = rhs(u); k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2); k4 = rhs(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        u[0] = u[-1] = 0.0
        if n % save_every == 0:
            ts.append(n * dt); Us.append(u.copy())
    return x, np.array(ts), np.array(Us), dx


def steepness(x, ts, U):
    """max |u_x| va thoi diem dat cuc dai."""
    dx = x[1] - x[0]
    g = np.abs(np.gradient(U, dx, axis=1)).max(axis=1)
    i = int(np.argmax(g))
    return float(g[i]), float(ts[i])


# ----------------------------------------------------------------------
# PINN
# ----------------------------------------------------------------------
def sample(seed, Nr=2500, Nb=300, N0=300):
    g = torch.Generator().manual_seed(seed)
    Xr = torch.cat([torch.rand(Nr, 1, generator=g, dtype=DTYPE) * 2 - 1,
                    torch.rand(Nr, 1, generator=g, dtype=DTYPE)], 1)
    tb = torch.rand(Nb, 1, generator=g, dtype=DTYPE)
    Xb = torch.cat([torch.cat([-torch.ones_like(tb), tb], 1),
                    torch.cat([torch.ones_like(tb), tb], 1)], 0)
    x0 = torch.rand(N0, 1, generator=g, dtype=DTYPE) * 2 - 1
    X0 = torch.cat([x0, torch.zeros_like(x0)], 1)
    return Xr, Xb, X0


def run(anneal_alpha, seed=0, adam_iters=6000, lbfgs_iters=800, N_up=100,
        ref=None):
    set_seed(seed)
    net = FNN([2, 32, 32, 32, 32, 1], act="tanh")
    Xr, Xb, X0 = sample(seed)
    u0 = -torch.sin(PI * X0[:, 0:1])
    lam = {"b": 1.0, "0": 1.0}
    params = list(net.parameters())

    def parts():
        X = Xr.clone().requires_grad_(True)
        u = net(X)
        g1 = grad(u, X)
        ut, ux = g1[:, 1:2], g1[:, 0:1]
        uxx = grad(ux, X)[:, 0:1]
        Jr = ((ut + u * ux - NU * uxx) ** 2).mean()
        Jb = (net(Xb) ** 2).mean()
        J0 = ((net(X0) - u0) ** 2).mean()
        return Jr, Jb, J0

    def loss_fn():
        Jr, Jb, J0 = parts()
        return Jr + lam["b"] * Jb + lam["0"] * J0, {"Jr": Jr.item()}

    xref, tref, Uref, _ = ref
    Xe = torch.tensor(np.stack(np.meshgrid(xref, tref, indexing="ij"), -1)
                      .reshape(-1, 2), dtype=DTYPE)
    ue = torch.tensor(Uref.T.reshape(-1, 1), dtype=DTYPE)

    opt = torch.optim.Adam(params, lr=1e-3)
    rho_hist, max_rho, wnorm = [], 0.0, []
    for it in range(adam_iters + 1):
        if it % N_up == 0:
            Jr, Jb, _ = parts()
            r = imbalance(Jr, Jb, params)
            rho_hist.append((it, r)); max_rho = max(max_rho, r)
            wnorm.append((it, net.lins[0].weight.norm().item()))
            if anneal_alpha is not None:
                a = anneal_alpha
                lam["b"] = (1 - a) * lam["b"] + a * r
                Jr2, _, J02 = parts()
                lam["0"] = (1 - a) * lam["0"] + a * imbalance(Jr2, J02, params)
        loss, _ = loss_fn()
        for p in params:
            p.grad = None
        loss.backward(); opt.step()

    with torch.no_grad():
        e_adam = rel_l2(net(Xe), ue)
    train_lbfgs(net, loss_fn, max_iter=lbfgs_iters)
    with torch.no_grad():
        pred = net(Xe)
        e_l, e_i = rel_l2(pred, ue), abs_linf(pred, ue)
    return {"alpha_u": anneal_alpha, "eps_L2_Adam": e_adam, "eps_L2_cuoi": e_l,
            "eps_Linf": e_i, "lam_b_cuoi": lam["b"], "max_rho_b": max_rho,
            "rho_dau": rho_hist[0][1], "rho_cuoi": rho_hist[-1][1],
            "W1_dau": wnorm[0][1], "W1_cuoi": wnorm[-1][1]}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Dang tinh nghiem tham chieu (Nx = 2047) ...")
    ref = reference(Nx=2047)
    x, ts, U, dx = ref
    gmax, tmax = steepness(x, ts, U)
    Pe = float(np.abs(U).max() * dx / NU)
    print(f"  max|u_x| = {gmax:.1f} tai t = {tmax:.3f}")
    print(f"  so Peclet luoi = {Pe:.3f}  ({'on dinh' if Pe < 2 else 'CANH BAO'})")
    print(f"  thoi diem hinh thanh lop trong (ly thuyet khong nhot) t_b = 1/pi = {1/PI:.3f}")

    print("Kiem chung hoi tu luoi voi Nx = 1023 ...")
    x2, ts2, U2, _ = reference(Nx=1023)
    Ui = np.stack([np.interp(x2, x, U[i]) for i in range(len(ts))])
    d = float(np.linalg.norm(Ui - U2) / np.linalg.norm(U2))
    print(f"  sai lech L2 tuong doi giua hai luoi = {d:.4e}\n")

    res = [run(None, ref=ref), run(0.9, ref=ref), run(0.1, ref=ref)]
    print(f"{'chien luoc':<26}{'eps_L2 Adam':>13}{'eps_L2 cuoi':>13}"
          f"{'eps_Linf':>12}{'lam_b cuoi':>13}{'max rho_b':>12}")
    for r in res:
        ten = "Trong so co dinh" if r["alpha_u"] is None else f"U trong so, a={r['alpha_u']}"
        print(f"{ten:<26}{r['eps_L2_Adam']:13.4e}{r['eps_L2_cuoi']:13.4e}"
              f"{r['eps_Linf']:12.4e}{r['lam_b_cuoi']:13.4g}{r['max_rho_b']:12.4g}")
    a = res[0]
    print(f"\nrho_b (trong so co dinh): {a['rho_dau']:.1f} -> {a['rho_cuoi']:.1f}"
          f"  ({'TANG' if a['rho_cuoi'] > a['rho_dau'] else 'giam'})")
    print(f"||W1||_F: {a['W1_dau']:.3f} -> {a['W1_cuoi']:.3f}"
          f"  (binh phuong tang {(a['W1_cuoi']/a['W1_dau'])**2:.2f} lan)")
    print(f"eps_Linf/eps_L2 = {a['eps_Linf']/a['eps_L2_cuoi']:.1f}"
          f"  -> sai so tap trung vao dai hep quanh lop trong")
    json.dump({"tham_chieu": {"max_ux": gmax, "t_max": tmax, "Pe": Pe,
                              "hoi_tu_luoi": d}, "ket_qua": res},
              open(os.path.join(OUT, "exp5_burgers.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
