"""TN10 --- PINN tang cuong cho Burgers.

Muc Han che cua bao cao tu neu hai ly do khien sai so Burgers o TN5 xa
muc 1e-3 -- 1e-4 cua cong trinh goc: ngan sach huan luyen han che, va chua
thu lay mau thich nghi RAR ("ky thuat co kha nang cai thien ro nhat va dang
duoc thu truoc tien"). Thi nghiem nay lam dung hai viec ay, cung luc:

  TN5 (goc)                      TN10 (tang cuong)
  mang [2, 32x4, 1]         ->   [2, 20x8, 1]  (sau hon, theo Raissi 2019)
  Nr = 2 500                ->   10 000, cong them diem RAR
  6 000 Adam + 800 L-BFGS   ->   3 000 Adam + toi da 12 000 L-BFGS
  diem phoi tri co dinh     ->   RAR: sau moi vong L-BFGS dau, them 500
                                 diem noi |phan du| lon nhat

Sai so do DUNG dinh nghia cua TN5 (L2 tuong doi tren luoi tham chieu
Nx = 2047), nen hai so so sanh truc tiep duoc.

Trong luc huan luyen, ghi anh chup u_theta(x,t) tren mot luoi thua de dung
animation (xem pinns/animate.py). Anh chup la float16 va KHONG dua vao git.

Chay mot hat giong:   OMP_NUM_THREADS=1 python -m pinns.exp10_burgers_manh 0
Tong ket:             python -m pinns.exp10_burgers_manh tong_ket
"""
from __future__ import annotations
import json, os, sys, time

import numpy as np
import torch

from .core import DTYPE, FNN, abs_linf, grad, rel_l2, set_seed
from . import exp5_burgers as B

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = [0, 1, 2, 3, 4]
LAYERS = [2] + [20] * 8 + [1]
NR = 10_000
ADAM_ITERS = 3_000
LBFGS_VONG = 6                # so vong L-BFGS
LBFGS_MOI_VONG = 2_000        # buoc toi da moi vong -> tong toi da 12 000
RAR_VONG = 3                  # them diem RAR sau 3 vong dau
RAR_THEM = 500
RAR_UNG_VIEN = 20_000
CHUP_MOI = 100                # anh chup moi 100 lan danh gia ham


def _luoi_danh_gia(ref):
    x, ts, U, _ = ref
    Xe = torch.tensor(np.stack(np.meshgrid(x, ts, indexing="ij"), -1)
                      .reshape(-1, 2), dtype=DTYPE)
    ue = torch.tensor(U.T.reshape(-1, 1), dtype=DTYPE)
    return Xe, ue


def _luoi_chup(ref):
    """Luoi thua cho animation: moi 8 diem x, moi 2 moc t."""
    x, ts, _, _ = ref
    xs, tt = x[::8], ts[::2]
    X = torch.tensor(np.stack(np.meshgrid(xs, tt, indexing="ij"), -1)
                     .reshape(-1, 2), dtype=DTYPE)
    return X, xs, tt


def run(seed, ref=None):
    ref = ref if ref is not None else B.reference(Nx=2047)
    Xe, ue = _luoi_danh_gia(ref)
    Xc, xs, tt = _luoi_chup(ref)

    set_seed(seed)
    net = FNN(LAYERS, act="tanh")
    Xr, Xb, X0 = B.sample(seed, Nr=NR)
    u0 = -torch.sin(B.PI * X0[:, 0:1])
    g = torch.Generator().manual_seed(10_000 + seed)   # cho RAR, tach khoi mau

    def residual(X):
        X = X.clone().requires_grad_(True)
        u = net(X)
        g1 = grad(u, X)
        uxx = grad(g1[:, 0:1], X)[:, 0:1]
        return g1[:, 1:2] + u * g1[:, 0:1] - B.NU * uxx

    def loss_fn():
        Jr = (residual(Xr) ** 2).mean()
        Jb = (net(Xb) ** 2).mean()
        J0 = ((net(X0) - u0) ** 2).mean()
        return Jr + Jb + J0

    dem = {"n": 0}
    anh, duong = [], []           # anh chup, (buoc, sai so)

    def chup(giai_doan):
        with torch.no_grad():
            e = rel_l2(net(Xe), ue)
            anh.append(net(Xc).reshape(len(xs), len(tt)).numpy().astype(np.float16))
        duong.append({"buoc": dem["n"], "giai_doan": giai_doan, "eps_L2": e})

    t0 = time.time()
    chup("khoi_tao")

    # --- giai doan 1: Adam khoi dong --------------------------------------
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(1, ADAM_ITERS + 1):
        J = loss_fn()
        opt.zero_grad(); J.backward(); opt.step()
        dem["n"] += 1
        if it % CHUP_MOI == 0:
            chup("adam")
    with torch.no_grad():
        e_adam = rel_l2(net(Xe), ue)
    print(f"[TN10 s{seed}] sau Adam: {e_adam:.3e}  ({time.time()-t0:.0f}s)",
          flush=True)

    # --- giai doan 2: L-BFGS theo vong, xen RAR ---------------------------
    them_rar = 0
    for vong in range(LBFGS_VONG):
        lb = torch.optim.LBFGS(net.parameters(), max_iter=LBFGS_MOI_VONG,
                               history_size=50, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-14, tolerance_change=1e-16)

        def closure():
            lb.zero_grad()
            J = loss_fn()
            J.backward()
            dem["n"] += 1
            if dem["n"] % CHUP_MOI == 0:
                chup("lbfgs")
            return J

        lb.step(closure)
        with torch.no_grad():
            e = rel_l2(net(Xe), ue)
        print(f"[TN10 s{seed}] vong {vong+1}/{LBFGS_VONG}: {e:.3e}  "
              f"Nr={len(Xr)}  ({time.time()-t0:.0f}s)", flush=True)

        if vong < RAR_VONG:       # RAR: them diem noi phan du lon nhat
            ung = torch.cat([torch.rand(RAR_UNG_VIEN, 1, generator=g, dtype=DTYPE) * 2 - 1,
                             torch.rand(RAR_UNG_VIEN, 1, generator=g, dtype=DTYPE)], 1)
            r = residual(ung).detach().abs().squeeze()
            chon = torch.topk(r, RAR_THEM).indices
            Xr = torch.cat([Xr, ung[chon]], 0)
            them_rar += RAR_THEM

    chup("cuoi")
    with torch.no_grad():
        pred = net(Xe)
        e_l, e_i = rel_l2(pred, ue), abs_linf(pred, ue)
    giay = time.time() - t0

    ket = {"seed": seed, "layers": LAYERS, "Nr_dau": NR, "Nr_cuoi": len(Xr),
           "rar_them": them_rar, "eps_L2_Adam": e_adam, "eps_L2_cuoi": e_l,
           "eps_Linf": e_i, "so_danh_gia": dem["n"], "giay": giay,
           "duong_sai_so": duong}
    os.makedirs(OUT, exist_ok=True)
    json.dump(ket, open(os.path.join(OUT, f"exp10_seed{seed}.json"), "w"), indent=1)
    # Trong so cuoi (float64, ~3 000 tham so): de ve sai so dung do chinh xac
    # kep -- anh chup float16 chi dung cho hinh u, khong dung cho sai so.
    torch.save(net.state_dict(), os.path.join(OUT, f"exp10_trongso_seed{seed}.pt"))
    np.savez_compressed(os.path.join(OUT, f"exp10_anh_seed{seed}.npz"),
                        anh=np.stack(anh), xs=xs, tt=tt,
                        buoc=np.array([d["buoc"] for d in duong]),
                        eps=np.array([d["eps_L2"] for d in duong]))
    print(f"[TN10 s{seed}] XONG: eps_L2 = {e_l:.4e}, Linf = {e_i:.3e}, "
          f"{dem['n']} lan danh gia, {giay/60:.1f} phut", flush=True)
    return ket


def fd_cung_muc(eps_muc, ref_min=None):
    """Luoi sai phan huu han tho nhat dat sai so <= eps_muc, va thoi gian cua no.

    Voi moi Nx, chon buoc thoi gian LON NHAT trong {5e-3, 1e-3, 5e-4, 2,5e-4,
    1e-4} ma van nho hon mot nua chan on dinh cua RK4 tuong minh cho chinh
    luoi ay (doi luu dx/max|u|, khuech tan dx^2/(2 nu)) -- de khong lap lai
    loi o TN9, noi dt chon cho luoi min lam chi phi FD bi thoi phong 100 lan.
    Thoi gian la trung vi cua 3 lan chay.
    """
    import statistics as st
    ref_min = ref_min if ref_min is not None else B.reference(Nx=2047)
    xm, tm, Um, _ = ref_min
    rows = []
    for Nx in (63, 127, 255, 511, 1023):
        dx = 2.0 / (Nx + 1)
        chan = min(dx / 1.0, dx * dx / (2 * B.NU))
        dt = next(d for d in (5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4) if d <= chan / 2)
        ts_ = []
        for _ in range(3):
            t0 = time.perf_counter()
            x, ts, U, _ = B.reference(Nx=Nx, dt=dt)
            ts_.append(time.perf_counter() - t0)
        Ui = np.stack([np.interp(xm, x, U[i]) for i in range(len(ts))])
        e = float(np.linalg.norm(Ui - Um) / np.linalg.norm(Um))
        rows.append({"Nx": Nx, "dt": dt, "giay": st.median(ts_), "eps_L2": e,
                     "dat": e <= eps_muc})
        print(f"  FD Nx={Nx:5d} dt={dt:.1e}: {st.median(ts_):.4f} s, eps={e:.3e}"
              f"  {'<- dat' if e <= eps_muc else ''}", flush=True)
        if e <= eps_muc:
            break
    return rows


def tong_ket():
    import statistics as st
    rs = []
    for s in SEEDS:
        f = os.path.join(OUT, f"exp10_seed{s}.json")
        if os.path.exists(f):
            rs.append(json.load(open(f)))
    if not rs:
        print("chua co ket qua"); return
    e = sorted(r["eps_L2_cuoi"] for r in rs)
    li = sorted(r["eps_Linf"] for r in rs)
    ph = sorted(r["giay"] / 60 for r in rs)
    tn5 = json.load(open(os.path.join(OUT, "exp8_burgers.json")))["co_dinh"]
    e5 = sorted(r["eps_L2_cuoi"] for r in tn5)
    print(f"TN10 tren {len(rs)} hat giong:")
    for r in rs:
        print(f"  hat giong {r['seed']}: eps_L2 {r['eps_L2_cuoi']:.4e}  "
              f"Linf {r['eps_Linf']:.3e}  ({r['giay']/60:.1f} phut)")
    print(f"  eps_L2 : trung vi {st.median(e):.4e}  [{e[0]:.4e}; {e[-1]:.4e}]")
    print(f"  Linf   : trung vi {st.median(li):.3e}  [{li[0]:.3e}; {li[-1]:.3e}]")
    print(f"  thoi gian: trung vi {st.median(ph):.1f} phut")
    print(f"\nTN5 goc (5 hat giong): trung vi {st.median(e5):.4e}  "
          f"[{e5[0]:.4e}; {e5[-1]:.4e}]")
    print(f"=> cai thien trung vi: {st.median(e5)/st.median(e):.0f} lan")
    print(f"\nChi phi sai phan huu han o cung muc sai so {st.median(e):.2e}:")
    fd = fd_cung_muc(st.median(e))
    dat = [r for r in fd if r["dat"]]
    if dat:
        print(f"  => PINN {st.median(ph):.1f} phut so voi FD {dat[0]['giay']:.3f} s"
              f"  ({st.median(ph)*60/dat[0]['giay']:.0f} lan)")
    json.dump({"fd": fd, "fd_dat": dat[0] if dat else None,
               "n": len(rs), "eps_trung_vi": st.median(e), "eps_min": e[0],
               "eps_max": e[-1], "linf_trung_vi": st.median(li),
               "phut_trung_vi": st.median(ph),
               "tn5_trung_vi": st.median(e5),
               "cai_thien_lan": st.median(e5) / st.median(e)},
              open(os.path.join(OUT, "exp10_tomtat.json"), "w"), indent=2)


def main():
    """Cho run_all: chay tuan tu cac hat giong chua co ket qua, roi tong ket.

    Moi hat giong mat khoang nua gio tren mot loi; chay song song thi goi
    tung hat giong rieng (xem dau tep)."""
    torch.set_num_threads(1)
    for s in SEEDS:
        if not os.path.exists(os.path.join(OUT, f"exp10_seed{s}.json")):
            run(s)
    tong_ket()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "tong_ket"
    if arg == "tong_ket":
        tong_ket()
    else:
        torch.set_num_threads(1)
        run(int(arg))
