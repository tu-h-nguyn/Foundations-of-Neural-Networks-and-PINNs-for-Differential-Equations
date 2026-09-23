"""TN10b --- tach bien: vai tro rieng cua RAR trong TN10.

TN10 doi bon thu cung luc (mang sau hon, Nr lon hon, L-BFGS dai hon, RAR),
nen khong quy cong duoc cho RAR. Thi nghiem nay giu nguyen MOI THU cua TN10
va chi doi cach them diem phoi tri.

Ba nhanh re ra tu CUNG mot trang thai: sau Adam va dot L-BFGS thu nhat --
dung thoi diem ma RAR bat dau can thiep. Phan chung ay chi chay mot lan.

  rar          dung nhu TN10: sau dot 1--3, them 500 diem co |r| lon nhat
               trong 20 000 ung vien  (phai khop TN10 tung bit -- kiem chung)
  ngau_nhien   sau dot 1--3, them 500 diem RAI DEU ngau nhien
               -> cung SO diem nhu RAR, chi khac VI TRI: tach "thich nghi"
                  khoi "nhieu diem hon"
  khong_them   Nr giu 10 000 den het

Ca ba nhanh chay cung 5 dot L-BFGS con lai, cung tham so.

Ngoai eps_L2, ghi them sai so trong dai soc |x| < 0,02, t > 1/pi (noi RAR
dat diem) va ngoai dai, vi cau hoi co y nghia la RAR giup O DAU.

Chay mot hat giong:  OMP_NUM_THREADS=1 python -m pinns.exp10b_rar_tachbien 0
Tong ket:            python -m pinns.exp10b_rar_tachbien tong_ket
"""
from __future__ import annotations
import copy, json, math, os, sys, time

import numpy as np
import torch

from .core import DTYPE, FNN, abs_linf, grad, rel_l2, set_seed
from . import exp5_burgers as B
from .exp10_burgers_manh import (ADAM_ITERS, LAYERS, LBFGS_MOI_VONG, LBFGS_VONG,
                                 NR, OUT, RAR_THEM, RAR_UNG_VIEN, RAR_VONG,
                                 SEEDS, _luoi_danh_gia)

NHANH = ["rar", "ngau_nhien", "khong_them"]
DAI_SOC = 0.02


def _chi_so_dai(Xe):
    x, t = Xe[:, 0], Xe[:, 1]
    return (x.abs() < DAI_SOC) & (t > 1 / math.pi)


def _do(net, Xe, ue, dai):
    with torch.no_grad():
        p = net(Xe)
    e2 = ((p - ue) ** 2).squeeze()
    return {"eps_L2": rel_l2(p, ue), "eps_Linf": abs_linf(p, ue),
            "ti_le_dai": dai.double().mean().item(),
            "phan_sai_so_trong_dai": (e2[dai].sum() / e2.sum()).item(),
            "max_trong_dai": e2[dai].max().sqrt().item(),
            "max_ngoai_dai": e2[~dai].max().sqrt().item(),
            "trung_vi_ngoai_dai": e2[~dai].sqrt().median().item()}


def run(seed, ref=None):
    ref = ref if ref is not None else B.reference(Nx=2047)
    Xe, ue = _luoi_danh_gia(ref)
    dai = _chi_so_dai(Xe)

    # ---------------- phan chung: y het TN10 toi het dot L-BFGS thu nhat ----
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
    e_adam = _do(net, Xe, ue, dai)["eps_L2"]
    mot_dot(net, Xr)
    chung = _do(net, Xe, ue, dai)
    giay_chung, n_chung = time.time() - t0, dem["n"]
    print(f"[TN10b s{seed}] phan chung xong: sau Adam {e_adam:.3e}, "
          f"dot 1 {chung['eps_L2']:.3e}  ({giay_chung:.0f}s)", flush=True)
    trang_thai = copy.deepcopy(net.state_dict())

    # ---------------- ba nhanh --------------------------------------------
    ket = {"seed": seed, "eps_L2_Adam": e_adam, "sau_dot_1": chung,
           "giay_phan_chung": giay_chung, "danh_gia_phan_chung": n_chung,
           "nhanh": {}}
    for ten in NHANH:
        t1 = time.time(); dem["n"] = n_chung
        nb = FNN(LAYERS, act="tanh")
        nb.load_state_dict(trang_thai)
        X = Xr.clone()
        g = torch.Generator().manual_seed(10_000 + seed)    # nhu TN10
        g_deu = torch.Generator().manual_seed(20_000 + seed)
        theo_dot = [dict(dot=1, Nr=len(X), **chung)]
        for vong in range(LBFGS_VONG):
            if vong > 0:
                mot_dot(nb, X)
                m = _do(nb, Xe, ue, dai)
                theo_dot.append(dict(dot=vong + 1, Nr=len(X), **m))
                print(f"[TN10b s{seed}] {ten:10s} dot {vong+1}/{LBFGS_VONG}: "
                      f"{m['eps_L2']:.3e}  Nr={len(X)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if vong < RAR_VONG:
                if ten == "rar":
                    ung = torch.cat([
                        torch.rand(RAR_UNG_VIEN, 1, generator=g, dtype=DTYPE) * 2 - 1,
                        torch.rand(RAR_UNG_VIEN, 1, generator=g, dtype=DTYPE)], 1)
                    r = residual(nb, ung).detach().abs().squeeze()
                    X = torch.cat([X, ung[torch.topk(r, RAR_THEM).indices]], 0)
                elif ten == "ngau_nhien":
                    X = torch.cat([X, torch.cat([
                        torch.rand(RAR_THEM, 1, generator=g_deu, dtype=DTYPE) * 2 - 1,
                        torch.rand(RAR_THEM, 1, generator=g_deu, dtype=DTYPE)], 1)], 0)
        cuoi = _do(nb, Xe, ue, dai)
        # ti le diem phoi tri them vao nam trong dai soc
        them = X[NR:]
        trong = ((them[:, 0].abs() < DAI_SOC) & (them[:, 1] > 1 / math.pi)).double()
        ket["nhanh"][ten] = {
            "cuoi": cuoi, "theo_dot": theo_dot, "Nr_cuoi": len(X),
            "diem_them_trong_dai": trong.mean().item() if len(them) else None,
            "so_danh_gia": dem["n"], "giay_nhanh": time.time() - t1}
        torch.save(nb.state_dict(),
                   os.path.join(OUT, f"exp10b_{ten}_seed{seed}.pt"))
        print(f"[TN10b s{seed}] {ten:10s} XONG: eps_L2 {cuoi['eps_L2']:.4e}, "
              f"Linf {cuoi['eps_Linf']:.3e}, trong dai "
              f"{cuoi['phan_sai_so_trong_dai']:.1%}", flush=True)

    # kiem chung: nhanh rar phai trung TN10 tung bit
    f10 = os.path.join(OUT, f"exp10_seed{seed}.json")
    if os.path.exists(f10):
        e10 = json.load(open(f10))["eps_L2_cuoi"]
        e_rar = ket["nhanh"]["rar"]["cuoi"]["eps_L2"]
        ket["khop_TN10"] = (e10 == e_rar)
        print(f"[TN10b s{seed}] nhanh rar {'KHOP' if e10 == e_rar else 'LECH'} "
              f"TN10: {e_rar!r} / {e10!r}", flush=True)

    json.dump(ket, open(os.path.join(OUT, f"exp10b_seed{seed}.json"), "w"), indent=1)
    return ket


def tong_ket():
    import statistics as st
    rs = [json.load(open(os.path.join(OUT, f"exp10b_seed{s}.json")))
          for s in SEEDS if os.path.exists(os.path.join(OUT, f"exp10b_seed{s}.json"))]
    if not rs:
        print("chua co ket qua"); return
    tom = {"n": len(rs), "khop_TN10": [r.get("khop_TN10") for r in rs]}
    print(f"TN10b tren {len(rs)} hat giong; nhanh rar khop TN10: {tom['khop_TN10']}")
    e1 = [r["sau_dot_1"]["eps_L2"] for r in rs]
    print(f"  diem re nhanh (sau dot 1): trung vi {st.median(e1):.3e}")
    for ten in NHANH:
        c = [r["nhanh"][ten]["cuoi"] for r in rs]
        d = {k: sorted(q[k] for q in c) for k in c[0]}
        tv = {k: st.median(v) for k, v in d.items()}
        dt = [r["nhanh"][ten]["diem_them_trong_dai"] for r in rs]
        tom[ten] = {"trung_vi": tv, "min": {k: v[0] for k, v in d.items()},
                    "max": {k: v[-1] for k, v in d.items()},
                    "tung_hat": {r["seed"]: r["nhanh"][ten]["cuoi"]["eps_L2"] for r in rs},
                    "diem_them_trong_dai": dt}
        print(f"  {ten:10s}: eps_L2 {tv['eps_L2']:.3e} [{d['eps_L2'][0]:.3e}; "
              f"{d['eps_L2'][-1]:.3e}]  Linf {tv['eps_Linf']:.3e}  "
              f"max trong dai {tv['max_trong_dai']:.2e}  ngoai {tv['max_ngoai_dai']:.2e}  "
              f"phan trong dai {tv['phan_sai_so_trong_dai']:.1%}")
    # so sanh cap, cung hat giong
    for a, b in [("rar", "khong_them"), ("rar", "ngau_nhien"), ("ngau_nhien", "khong_them")]:
        ts = [r["nhanh"][b]["cuoi"]["eps_L2"] / r["nhanh"][a]["cuoi"]["eps_L2"] for r in rs]
        tl = [r["nhanh"][b]["cuoi"]["eps_Linf"] / r["nhanh"][a]["cuoi"]["eps_Linf"] for r in rs]
        tom[f"{b}/{a}"] = {"eps_L2": ts, "eps_Linf": tl}
        print(f"  eps({b})/eps({a}) theo hat giong: L2 "
              + " ".join(f"{v:.2f}" for v in ts) + "  | Linf "
              + " ".join(f"{v:.2f}" for v in tl))
    json.dump(tom, open(os.path.join(OUT, "exp10b_tomtat.json"), "w"), indent=1)


def main():
    torch.set_num_threads(1)
    for s in SEEDS:
        if not os.path.exists(os.path.join(OUT, f"exp10b_seed{s}.json")):
            run(s)
    tong_ket()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "tong_ket"
    if arg == "tong_ket":
        tong_ket()
    else:
        torch.set_num_threads(1)
        run(int(arg))
