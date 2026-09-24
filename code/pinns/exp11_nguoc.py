"""TN11 --- Bai toan nguoc: nhan dang z1, z2 cua phuong trinh Burgers.

    u_t + z1 u u_x - z2 u_xx = 0,   z1 = 1,  z2 = 0,01/pi   (eq:burgers-param)

Muoi thi nghiem truoc deu la bai toan THUAN, tuc nam ngoai ca ba dong luc ma bao
cao dung de bien minh cho PINNs (Muc subsec:gioi-han-co-dien). Thi nghiem nay
dat PINN vao dung dong luc thu nhat va thu hai: bai toan nguoc, voi dieu kien
dau va dieu kien bien KHONG duoc cho.

Du lieu (dung thiet lap cua Raissi va cs. 2019, doi chieu voi ma nguon goc
appendix/continuous_time_identification (Burgers)/Burgers.py):
  N_u = 2000 diem (x, t) rut ngau nhien, khong lap, tu luoi cua nghiem tham chieu
  (2049 x 201); nhieu y = u + sigma * std(u) * N(0, 1), sigma in {0; 0,01}.
  Cung mot tap diem cho ca hai muc nhieu cua mot hat giong.

Ba doi thu, CUNG du lieu:
  A  PINN        mang [2, 20x8, 1], z1 khoi tao 0, log z2 khoi tao -6 (nhu ma goc);
                 phan du tinh tai chinh 2000 diem du lieu (nhu ma goc); KHONG dung
                 dieu kien dau / bien. Adam 2 000 vong roi L-BFGS toi da 5 x 2 000.
  C  co dien, CUNG thong tin: sai phan huu han + RK4 (luoc do cua nghiem tham
                 chieu) voi dieu kien dau va hai dieu kien bien la AN SO (spline bac
                 ba 33 nut / 11 nut), gradient bang phuong phap lien hop roi rac viet
                 tay (pinns/lienhop_burgers.py), toi uu L-BFGS-B. Luoi Nx = 1023
                 (chinh), Nx = 511 (phu). z2 bi chan tren 0,01 vi on dinh cua luoc do
                 tuong minh -- mot rang buoc PINN khong can.
  B  co dien, DU thong tin: nhu C nhung dieu kien dau / bien cho dung; chi con
                 (z1, z2) la an. De doi chieu, khong co gia thuyet.

GIA THUYET VA TIEU CHI --- VIET VA COMMIT TRUOC KHI CHAY PINN VA NHANH C:
  H1  PINN, du lieu sach: sai so tuong doi cua z2 < 1% o 5/5 hat giong.
      (Raissi va cs. bao cao 0,12%.)
  H2  PINN, nhieu 1%: sai so tuong doi cua z2 < 5% o 5/5 hat giong.
      (Raissi va cs. bao cao 0,84%.)
  H3  Vi du vd:burgers-inverse cua bao cao viet "sai so cua z2 LUON lon hon sai
      so cua z1 KHOANG MOT BAC". Kiem: err(z2) > err(z1) o >= 9/10 lan chay PINN,
      VA trung vi ti so err(z2)/err(z1) nam trong [3; 30].
  H4  Muc subsec:thuan-nguoc va Nhan xet nx:doc-tn9 (iii) viet rang voi bai toan
      nguoc, phuong phap co dien bat loi ("giai lap bai toan thuan nhieu lan",
      "ca mot bo may lien hop"). Kiem, voi CUNG thong tin (A so voi C, Nx = 1023):
      H4a  trung vi sai so z2 cua PINN nho hon cua C, o CA HAI muc nhieu;
      H4b  trung vi thoi gian cua PINN ngan hon cua C, o CA HAI muc nhieu.
Moi gia thuyet khong dat se duoc bao cao la khong dat.

Chay:  OMP_NUM_THREADS=1 python -m pinns.exp11_nguoc pinn <hat_giong>
       OMP_NUM_THREADS=1 python -m pinns.exp11_nguoc codien <hat_giong>
       python -m pinns.exp11_nguoc tong_ket
"""
from __future__ import annotations
import json, math, os, statistics as st, sys, time

import numpy as np
import torch
from scipy.optimize import minimize

from .core import DTYPE, FNN, grad, set_seed
from . import exp5_burgers as B
from .lienhop_burgers import BaiToanNguoc

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = [0, 1, 2, 3, 4]
NHIEU = [0.0, 0.01]
N_U = 2000
Z1, Z2 = 1.0, 0.01 / math.pi
LAYERS = [2] + [20] * 8 + [1]
ADAM, LBFGS_VONG, LBFGS_MOI_VONG = 2_000, 5, 2_000
CHUP_MOI = 100
LUOI_CO_DIEN = [(1023, 2.5e-4), (511, 5e-4)]          # (Nx, dt); 1023 la chinh
MAX_DANH_GIA_CO_DIEN = 3_000


def du_lieu(seed, sigma, ref):
    """N_u diem tu luoi tham chieu; cung tap diem cho moi muc nhieu."""
    x, ts, U, _ = ref
    rng = np.random.default_rng(40_000 + seed)
    idx = rng.choice(U.size, N_U, replace=False)
    li, xi = np.unravel_index(idx, U.shape)
    u = U[li, xi]
    y = u + sigma * u.std() * np.random.default_rng(50_000 + seed).standard_normal(N_U)
    return x[xi], ts[li], y


def _sai_so(z1, z2):
    return abs(z1 - Z1) / Z1, abs(z2 - Z2) / Z2


def _ten(sigma):
    return "sach" if sigma == 0 else f"nhieu{int(round(sigma * 100))}"


# ============================================================ A. PINN
def chay_pinn(seed, sigma, ref):
    x, ts, U, _ = ref
    xd, td, yd = du_lieu(seed, sigma, ref)
    X = torch.tensor(np.stack([xd, td], 1), dtype=DTYPE)
    Y = torch.tensor(yd[:, None], dtype=DTYPE)
    set_seed(seed)
    net = FNN(LAYERS, act="tanh")
    l1 = torch.nn.Parameter(torch.tensor(0.0, dtype=DTYPE))     # nhu ma goc
    s2 = torch.nn.Parameter(torch.tensor(-6.0, dtype=DTYPE))
    tham_so = list(net.parameters()) + [l1, s2]

    def loss_fn():
        Xg = X.clone().requires_grad_(True)
        u = net(Xg)
        g1 = grad(u, Xg)
        uxx = grad(g1[:, 0:1], Xg)[:, 0:1]
        f = g1[:, 1:2] + l1 * u * g1[:, 0:1] - torch.exp(s2) * uxx
        return ((u - Y) ** 2).mean() + (f ** 2).mean()

    # luoi danh gia truong nghiem va luoi thua cho anh chup
    Xe = torch.tensor(np.stack(np.meshgrid(x, ts, indexing="ij"), -1).reshape(-1, 2), dtype=DTYPE)
    ue = torch.tensor(U.T.reshape(-1, 1), dtype=DTYPE)
    xs, tt = x[::8], ts[::2]
    Xc = torch.tensor(np.stack(np.meshgrid(xs, tt, indexing="ij"), -1).reshape(-1, 2), dtype=DTYPE)

    dem = {"n": 0}
    duong, anh = [], []

    def ghi():
        with torch.no_grad():
            e1, e2 = _sai_so(l1.item(), math.exp(s2.item()))
            duong.append({"buoc": dem["n"], "z1": l1.item(), "z2": math.exp(s2.item()),
                          "sai_z1": e1, "sai_z2": e2})
            if dem["n"] % CHUP_MOI == 0:
                anh.append(net(Xc).reshape(len(xs), len(tt)).numpy().astype(np.float16))

    t0 = time.time()
    ghi()
    opt = torch.optim.Adam(tham_so, lr=1e-3)
    for _ in range(ADAM):
        J = loss_fn(); opt.zero_grad(); J.backward(); opt.step()
        dem["n"] += 1
        if dem["n"] % 50 == 0:
            ghi()
    for vong in range(LBFGS_VONG):
        lb = torch.optim.LBFGS(tham_so, max_iter=LBFGS_MOI_VONG, history_size=50,
                               line_search_fn="strong_wolfe",
                               tolerance_grad=1e-14, tolerance_change=1e-16)

        def closure():
            lb.zero_grad(); J = loss_fn(); J.backward()
            dem["n"] += 1
            if dem["n"] % 50 == 0:
                ghi()
            return J
        lb.step(closure)
        e1, e2 = _sai_so(l1.item(), math.exp(s2.item()))
        print(f"[TN11 PINN s{seed} {_ten(sigma)}] dot {vong+1}: z1 {l1.item():.5f} "
              f"z2 {math.exp(s2.item()):.5e} (sai {e2:.2%})  ({time.time()-t0:.0f}s)", flush=True)
    giay = time.time() - t0
    with torch.no_grad():
        eps_truong = float(torch.linalg.norm(net(Xe) - ue) / torch.linalg.norm(ue))
    J_cuoi = float(loss_fn().detach())
    z1, z2 = l1.item(), math.exp(s2.item())
    e1, e2 = _sai_so(z1, z2)
    ket = {"doi_thu": "pinn", "seed": seed, "sigma": sigma, "z1": z1, "z2": z2,
           "sai_z1": e1, "sai_z2": e2, "eps_truong": eps_truong, "J": J_cuoi,
           "so_danh_gia": dem["n"], "giay": giay, "duong": duong}
    ten = f"exp11_pinn_{_ten(sigma)}_seed{seed}"
    json.dump(ket, open(os.path.join(OUT, ten + ".json"), "w"), indent=1)
    torch.save({"net": net.state_dict(), "l1": l1.detach(), "s2": s2.detach()},
               os.path.join(OUT, ten + ".pt"))
    np.savez_compressed(os.path.join(OUT, f"exp11_anh_{_ten(sigma)}_seed{seed}.npz"),
                        anh=np.stack(anh), xs=xs, tt=tt,
                        buoc=np.array([d["buoc"] for d in duong if d["buoc"] % CHUP_MOI == 0]))
    print(f"[TN11 PINN s{seed} {_ten(sigma)}] XONG: z1 {z1:.5f} ({e1:.3%}), z2 {z2:.5e} "
          f"({e2:.3%}), truong {eps_truong:.2e}, {giay/60:.1f} phut", flush=True)
    return ket


# ============================================================ B, C. co dien
def chay_co_dien(seed, sigma, ref, Nx, dt, biet):
    x, ts, U, _ = ref
    xd, td, yd = du_lieu(seed, sigma, ref)
    bt = BaiToanNguoc(Nx, dt, xd, td, yd, biet_dau_bien=biet)
    p0 = np.zeros(bt.n_an); p0[1] = -6.0                # cung khoi tao voi PINN
    dem = {"n": 0}
    duong = []
    t0 = time.time()

    def f(p):
        dem["n"] += 1
        L, g = bt.muc_tieu(p)
        if dem["n"] % 10 == 0 or dem["n"] == 1:
            e1, e2 = _sai_so(p[0], math.exp(p[1]))
            duong.append({"buoc": dem["n"], "z1": float(p[0]), "z2": math.exp(p[1]),
                          "sai_z1": e1, "sai_z2": e2, "L": L})
        return L, g

    bien = [(None, None), (None, math.log(0.01))] + [(None, None)] * (bt.n_an - 2)
    r = minimize(f, p0, jac=True, method="L-BFGS-B", bounds=bien,
                 options={"maxiter": MAX_DANH_GIA_CO_DIEN, "maxfun": MAX_DANH_GIA_CO_DIEN,
                          "ftol": 1e-15, "gtol": 1e-12})
    giay = time.time() - t0
    z1, z2 = float(r.x[0]), math.exp(r.x[1])
    e1, e2 = _sai_so(z1, z2)
    # sai so truong nghiem: nghiem sai phan (luoi Nx) noi suy len luoi tham chieu
    muc, _ = bt.giai(r.x)
    Ui = np.stack([np.interp(x, bt.xg, muc[i]) for i in range(len(ts))])
    eps_truong = float(np.linalg.norm(Ui - U) / np.linalg.norm(U))
    ket = {"doi_thu": "B" if biet else "C", "Nx": Nx, "dt": dt, "seed": seed,
           "sigma": sigma, "z1": z1, "z2": z2, "sai_z1": e1, "sai_z2": e2,
           "eps_truong": eps_truong, "so_danh_gia": dem["n"], "giay": giay,
           "L": float(r.fun), "thong_diep": str(r.message),
           "so_lan_no_nghiem": getattr(bt, "so_lan_no", 0), "chan_z2_cham": bool(
               abs(r.x[1] - math.log(0.01)) < 1e-9), "duong": duong}
    print(f"[TN11 {ket['doi_thu']} Nx={Nx} s{seed} {_ten(sigma)}] z1 {z1:.5f} ({e1:.3%}), "
          f"z2 {z2:.5e} ({e2:.3%}), truong {eps_truong:.2e}, {dem['n']} lan, "
          f"{giay:.0f}s, {r.message}", flush=True)
    return ket


def chay_tat_ca_co_dien(seed, ref):
    kq = []
    for biet in (True, False):
        for Nx, dt in LUOI_CO_DIEN:
            for sigma in NHIEU:
                kq.append(chay_co_dien(seed, sigma, ref, Nx, dt, biet))
                json.dump(kq, open(os.path.join(OUT, f"exp11_codien_seed{seed}.json"), "w"),
                          indent=1)
    return kq


# ============================================================ tong ket
def tong_ket():
    A, CB = [], []
    for s in SEEDS:
        for sigma in NHIEU:
            f = os.path.join(OUT, f"exp11_pinn_{_ten(sigma)}_seed{s}.json")
            if os.path.exists(f):
                A.append(json.load(open(f)))
        f = os.path.join(OUT, f"exp11_codien_seed{s}.json")
        if os.path.exists(f):
            CB += json.load(open(f))
    if not A:
        print("chua co ket qua"); return
    tom = {"nhom": {}}

    def nhom(ds, ten):
        if not ds:
            return
        d = {k: st.median(q[k] for q in ds) for k in ("sai_z1", "sai_z2", "eps_truong", "giay")}
        d.update({"min_sai_z2": min(q["sai_z2"] for q in ds),
                  "max_sai_z2": max(q["sai_z2"] for q in ds), "n": len(ds),
                  "so_danh_gia": st.median(q["so_danh_gia"] for q in ds)})
        tom["nhom"][ten] = d
        print(f"  {ten:22s} n={len(ds)}  sai z1 {d['sai_z1']:.3%}  sai z2 {d['sai_z2']:.3%} "
              f"[{d['min_sai_z2']:.3%}; {d['max_sai_z2']:.3%}]  truong {d['eps_truong']:.2e}  "
              f"{d['giay']:.0f}s  ({d['so_danh_gia']:.0f} lan)")

    for sigma in NHIEU:
        print(f"--- {_ten(sigma)}")
        nhom([q for q in A if q["sigma"] == sigma], f"A_pinn_{_ten(sigma)}")
        for Nx, _ in LUOI_CO_DIEN:
            for dt_ in ("C", "B"):
                nhom([q for q in CB if q["sigma"] == sigma and q["Nx"] == Nx
                      and q["doi_thu"] == dt_], f"{dt_}_{Nx}_{_ten(sigma)}")
    # gia thuyet
    s0 = [q for q in A if q["sigma"] == 0.0]
    s1 = [q for q in A if q["sigma"] == 0.01]
    h1 = len(s0) == 5 and all(q["sai_z2"] < 0.01 for q in s0)
    h2 = len(s1) == 5 and all(q["sai_z2"] < 0.05 for q in s1)
    ti = [q["sai_z2"] / q["sai_z1"] for q in A]
    h3 = sum(t > 1 for t in ti) >= 9 and 3 <= st.median(ti) <= 30
    g = tom["nhom"]
    h4a = all(g[f"A_pinn_{_ten(s)}"]["sai_z2"] < g[f"C_1023_{_ten(s)}"]["sai_z2"] for s in NHIEU) \
        if all(f"C_1023_{_ten(s)}" in g for s in NHIEU) else None
    h4b = all(g[f"A_pinn_{_ten(s)}"]["giay"] < g[f"C_1023_{_ten(s)}"]["giay"] for s in NHIEU) \
        if all(f"C_1023_{_ten(s)}" in g for s in NHIEU) else None
    tom["ti_so_z2_z1"] = ti
    tom["gia_thuyet"] = {"H1": h1, "H2": h2, "H3": h3, "H4a": h4a, "H4b": h4b}
    print(f"  ti so err(z2)/err(z1): {sum(t > 1 for t in ti)}/{len(ti)} lan > 1, "
          f"trung vi {st.median(ti):.1f}")
    for k, v in tom["gia_thuyet"].items():
        print(f"  {k}: {'DAT' if v else ('KHONG DAT' if v is not None else 'chua du so lieu')}")
    json.dump(tom, open(os.path.join(OUT, "exp11_tomtat.json"), "w"), indent=1)


def main():
    torch.set_num_threads(1)
    ref = B.reference(Nx=2047)
    for s in SEEDS:
        for sigma in NHIEU:
            if not os.path.exists(os.path.join(OUT, f"exp11_pinn_{_ten(sigma)}_seed{s}.json")):
                chay_pinn(s, sigma, ref)
        if not os.path.exists(os.path.join(OUT, f"exp11_codien_seed{s}.json")):
            chay_tat_ca_co_dien(s, ref)
    tong_ket()


if __name__ == "__main__":
    torch.set_num_threads(1)
    viec = sys.argv[1] if len(sys.argv) > 1 else "tong_ket"
    if viec == "tong_ket":
        tong_ket()
    else:
        s = int(sys.argv[2])
        ref = B.reference(Nx=2047)
        if viec == "pinn":
            for sigma in NHIEU:
                chay_pinn(s, sigma, ref)
        elif viec == "codien":
            chay_tat_ca_co_dien(s, ref)
