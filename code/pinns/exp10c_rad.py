"""TN10c --- RAR dang LAY MAU LAI TOAN BO (RAD, Wu va cs. 2023).

TN10b cho thay RAR dang THEM DIEM doi sai so o lop soc lay sai so o vung tron:
sai so lon nhat giam, nhung sai so dien hinh ngoai dai soc tang, va hai tac
dong gan nhu triet tieu nhau trong eps_L2. Nhan xet nx:rar-cuc-bo cua bao cao
du doan mot bien the giu mat do diem tren vung tron co the tranh duoc su danh
doi ay. Thi nghiem nay kiem du doan do.

RAD: sau moi dot L-BFGS 1--5, BO toan bo Nr = 10 000 diem phan du va lay lai
10 000 diem tu 100 000 ung vien deu, voi xac suat

        p(x)  ~  |r(x)|^k / mean(|r|^k)  +  c .

Hai nhanh, re tu CUNG trang thai sau dot L-BFGS 1 nhu TN10b:
  rad_c1   k = 1, c = 1  -- mac dinh cua Wu va cs.; hang c giu mot nua "khoi
           luong" xac suat phan bo deu, tuc giu mat do diem tren vung tron
  rad_c0   k = 1, c = 0  -- ti le thuan |r|, KHONG giu mat do vung tron
So diem luon la 10 000, bang nhanh `khong_them` cua TN10b -> so sanh cap voi
nhanh ay (va voi `rar`) tren cung hat giong.

GIA THUYET VA TIEU CHI, VIET TRUOC KHI CHAY:
  H1  rad_c1 tot hon `khong_them` ve eps_L2 o >= 4/5 hat giong.
  H2  rad_c1 KHONG lam te sai so dien hinh ngoai dai soc: trung vi cua no
      <= trung vi cua `khong_them` (1,02e-4).
  H3  rad_c0 (khong giu vung tron) te hon rad_c1 ve sai so ngoai dai o
      >= 4/5 hat giong -- tuc chinh hang c la thu tranh duoc su danh doi.
Bat ky gia thuyet nao khong dat se duoc bao cao la khong dat.

Chay mot hat giong:  OMP_NUM_THREADS=1 python -m pinns.exp10c_rad 0
Tong ket:            python -m pinns.exp10c_rad tong_ket
"""
from __future__ import annotations
import copy, json, math, os, sys, time

import torch

from .core import DTYPE, FNN, grad, set_seed
from . import exp5_burgers as B
from .exp10_burgers_manh import (ADAM_ITERS, LAYERS, LBFGS_MOI_VONG, LBFGS_VONG,
                                 NR, OUT, SEEDS, _luoi_danh_gia)
from .exp10b_rar_tachbien import DAI_SOC, _chi_so_dai, _do

NHANH = {"rad_c1": (1.0, 1.0), "rad_c0": (1.0, 0.0)}      # ten: (k, c)
UNG_VIEN = 100_000
KHUC = 20_000                   # tinh phan du theo khuc de gioi han bo nho


def _trong_dai(X):
    return ((X[:, 0].abs() < DAI_SOC) & (X[:, 1] > 1 / math.pi)).double().mean().item()


def run(seed, ref=None):
    ref = ref if ref is not None else B.reference(Nx=2047)
    Xe, ue = _luoi_danh_gia(ref)
    dai = _chi_so_dai(Xe)

    # ---------------- phan chung: y het TN10 / TN10b ----------------------
    set_seed(seed)
    net = FNN(LAYERS, act="tanh")
    Xr, Xb, X0 = B.sample(seed, Nr=NR)
    u0 = -torch.sin(B.PI * X0[:, 0:1])

    def residual(net, X):
        X = X.clone().requires_grad_(True)
        u = net(X)
        g1 = grad(u, X)
        uxx = grad(g1[:, 0:1], X)[:, 0:1]
        return g1[:, 1:2] + u * g1[:, 0:1] - B.NU * uxx

    def loss_fn(net, Xr):
        return ((residual(net, Xr) ** 2).mean() + (net(Xb) ** 2).mean()
                + ((net(X0) - u0) ** 2).mean())

    dem = {"n": 0}

    def mot_dot(net, Xr):
        lb = torch.optim.LBFGS(net.parameters(), max_iter=LBFGS_MOI_VONG,
                               history_size=50, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-14, tolerance_change=1e-16)

        def closure():
            lb.zero_grad()
            J = loss_fn(net, Xr)
            J.backward()
            dem["n"] += 1
            return J
        lb.step(closure)

    t0 = time.time()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(ADAM_ITERS):
        J = loss_fn(net, Xr)
        opt.zero_grad(); J.backward(); opt.step()
        dem["n"] += 1
    mot_dot(net, Xr)
    chung = _do(net, Xe, ue, dai)
    n_chung = dem["n"]
    print(f"[TN10c s{seed}] phan chung xong: dot 1 {chung['eps_L2']:.3e}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    trang_thai = copy.deepcopy(net.state_dict())

    # ---------------- cac nhanh RAD ---------------------------------------
    ket = {"seed": seed, "sau_dot_1": chung, "nhanh": {}}
    for ten, (k, c) in NHANH.items():
        t1 = time.time(); dem["n"] = n_chung
        nb = FNN(LAYERS, act="tanh")
        nb.load_state_dict(trang_thai)
        X = Xr.clone()
        g = torch.Generator().manual_seed(30_000 + seed)
        theo_dot = [dict(dot=1, trong_dai_diem=_trong_dai(X), **chung)]
        for vong in range(LBFGS_VONG):
            if vong > 0:
                mot_dot(nb, X)
                m = _do(nb, Xe, ue, dai)
                theo_dot.append(dict(dot=vong + 1, trong_dai_diem=_trong_dai(X), **m))
                print(f"[TN10c s{seed}] {ten} dot {vong+1}/{LBFGS_VONG}: "
                      f"{m['eps_L2']:.3e}  diem trong dai {_trong_dai(X):.1%}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if vong < LBFGS_VONG - 1:          # lay mau lai truoc dot 2..6
                ung = torch.cat([
                    torch.rand(UNG_VIEN, 1, generator=g, dtype=DTYPE) * 2 - 1,
                    torch.rand(UNG_VIEN, 1, generator=g, dtype=DTYPE)], 1)
                r = torch.cat([residual(nb, ung[i:i + KHUC]).detach().abs().squeeze()
                               for i in range(0, UNG_VIEN, KHUC)])
                rk = r ** k
                p = rk / rk.mean() + c
                chon = torch.multinomial(p / p.sum(), NR, replacement=False,
                                         generator=g)
                X = ung[chon]
        cuoi = _do(nb, Xe, ue, dai)
        ket["nhanh"][ten] = {"k": k, "c": c, "cuoi": cuoi, "theo_dot": theo_dot,
                             "diem_trong_dai_cuoi": _trong_dai(X),
                             "so_danh_gia": dem["n"], "giay_nhanh": time.time() - t1}
        torch.save(nb.state_dict(), os.path.join(OUT, f"exp10c_{ten}_seed{seed}.pt"))
        print(f"[TN10c s{seed}] {ten} XONG: eps_L2 {cuoi['eps_L2']:.4e}, "
              f"Linf {cuoi['eps_Linf']:.3e}, ngoai dai (trung vi) "
              f"{cuoi['trung_vi_ngoai_dai']:.2e}", flush=True)

    json.dump(ket, open(os.path.join(OUT, f"exp10c_seed{seed}.json"), "w"), indent=1)
    return ket


def tong_ket():
    import statistics as st
    rs, rb = [], {}
    for s in SEEDS:
        f = os.path.join(OUT, f"exp10c_seed{s}.json")
        if os.path.exists(f):
            rs.append(json.load(open(f)))
            rb[s] = json.load(open(os.path.join(OUT, f"exp10b_seed{s}.json")))
    if not rs:
        print("chua co ket qua"); return
    # phan chung phai trung TN10b tung bit
    khop = [r["sau_dot_1"]["eps_L2"] == rb[r["seed"]]["sau_dot_1"]["eps_L2"] for r in rs]
    print(f"TN10c tren {len(rs)} hat giong; phan chung khop TN10b: {khop}")

    def lay(ten, r):
        return (r["nhanh"][ten]["cuoi"] if ten in r["nhanh"]
                else rb[r["seed"]]["nhanh"][ten]["cuoi"])

    tom = {"n": len(rs), "khop_TN10b": khop, "nhanh": {}}
    tat_ca = ["khong_them", "rar", "rad_c0", "rad_c1"]
    for ten in tat_ca:
        c = [lay(ten, r) for r in rs]
        tv = {q: st.median(x[q] for x in c) for q in c[0]}
        tom["nhanh"][ten] = {"trung_vi": tv,
                             "tung_hat": {r["seed"]: lay(ten, r) for r in rs}}
        print(f"  {ten:10s}: eps_L2 {tv['eps_L2']:.3e} [{min(x['eps_L2'] for x in c):.3e};"
              f" {max(x['eps_L2'] for x in c):.3e}]  Linf {tv['eps_Linf']:.3e}  "
              f"ngoai dai {tv['trung_vi_ngoai_dai']:.2e}  "
              f"phan trong dai {tv['phan_sai_so_trong_dai']:.1%}")
    for ten in NHANH:
        d = [r["nhanh"][ten]["diem_trong_dai_cuoi"] for r in rs]
        print(f"  {ten}: diem phoi tri trong dai (lan lay mau cuoi) "
              + " ".join(f"{v:.1%}" for v in d))
        tom["nhanh"][ten]["diem_trong_dai"] = d

    def thang(a, b, q):          # so hat giong a < b ve dai luong q
        return sum(lay(a, r)[q] < lay(b, r)[q] for r in rs)

    for a, b in [("rad_c1", "khong_them"), ("rad_c1", "rar"), ("rad_c0", "khong_them"),
                 ("rad_c1", "rad_c0")]:
        tom[f"{a}<{b}"] = {q: thang(a, b, q) for q in
                           ("eps_L2", "eps_Linf", "trung_vi_ngoai_dai")}
        print(f"  {a} tot hon {b}: " + ", ".join(
            f"{q} {v}/{len(rs)}" for q, v in tom[f'{a}<{b}'].items()))

    kt = tom["nhanh"]["khong_them"]["trung_vi"]["trung_vi_ngoai_dai"]
    h1 = tom["rad_c1<khong_them"]["eps_L2"] >= 4
    h2 = tom["nhanh"]["rad_c1"]["trung_vi"]["trung_vi_ngoai_dai"] <= kt
    h3 = tom["rad_c1<rad_c0"]["trung_vi_ngoai_dai"] >= 4
    tom["gia_thuyet"] = {"H1": h1, "H2": h2, "H3": h3}
    print(f"  H1 (rad_c1 thang khong_them eps_L2 >= 4/5): {'DAT' if h1 else 'KHONG DAT'}")
    print(f"  H2 (rad_c1 khong te vung tron):            {'DAT' if h2 else 'KHONG DAT'}")
    print(f"  H3 (hang c giu vung tron, >= 4/5):         {'DAT' if h3 else 'KHONG DAT'}")
    json.dump(tom, open(os.path.join(OUT, "exp10c_tomtat.json"), "w"), indent=1)


def main():
    torch.set_num_threads(1)
    for s in SEEDS:
        if not os.path.exists(os.path.join(OUT, f"exp10c_seed{s}.json")):
            run(s)
    tong_ket()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "tong_ket"
    if arg == "tong_ket":
        tong_ket()
    else:
        torch.set_num_threads(1)
        run(int(arg))
