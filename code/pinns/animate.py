"""Animation cho ket qua thuc nghiem.

Moi animation xuat ra HAI dang tu cung mot chuoi khung hinh:
  - GIF  -> Images/anim/<ten>.gif   (README tren GitHub hien thi truc tiep)
  - PDF nhieu trang -> slides/anim/<ten>.pdf  (goi `animate` cua LaTeX dung
    moi trang lam mot khung hinh; phat duoc trong Adobe Reader, Okular, Foxit)

Bon animation, moi cai ke mot ket qua cua bao cao:
  relu      TN1  - ReLU dung yen tuyet doi, tanh hoi tu
  pho       TN3  - thien kien pho bi dao chieu boi toan tu vi phan
  burgers   TN10 - PINN tang cuong hoc phuong trinh Burgers
  vatly     TN10 - nghiem theo thoi gian: song sin dung thanh soc

Chay:  python -m pinns.animate [relu|pho|burgers|vatly|tat_ca]
(burgers va vatly can ket qua cua exp10_burgers_manh truoc.)

Mau lay tu bang mau da kiem dinh mu mau (xem README):
PINN = xanh #2a78d6, tham chieu = xam trung tinh, ba mode tan so = ba o dau
cua bang phan loai; ban do nhiet u(x,t) dung thang phan ky xanh <-> do voi
diem giua xam, vi u doi dau quanh 0.
"""
from __future__ import annotations
import json, math, os, statistics as st, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from .core import DTYPE, FNN, grad, set_seed, uniform_1d

GOC = os.path.join(os.path.dirname(__file__), "..", "..")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
GIF_DIR = os.path.join(GOC, "Images", "anim")
PDF_DIR = os.path.join(GOC, "slides", "anim")

# ---------------------------------------------------------------- mau & kieu
MAT = "#fcfcfb"          # nen bieu do
CHU = "#0b0b0b"          # chu chinh
CHU2 = "#52514e"         # chu phu
LUOI = "#e4e3de"         # luoi mo
PINN = "#2a78d6"         # slot 1 - xanh
THAM = "#8a8984"         # tham chieu - xam trung tinh
CAM = "#eb6834"          # slot 2 - cam
NGOC = "#1baf7a"         # slot 3 - xanh ngoc
MODE = [PINN, CAM, NGOC]
PHAN_KY = LinearSegmentedColormap.from_list(
    "xanh_do", ["#0d366b", "#256abf", "#86b6ef", "#f0efec",
                "#f1a3a2", "#e34948", "#8a1f1e"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.facecolor": MAT, "figure.facecolor": MAT, "savefig.facecolor": MAT,
    "axes.edgecolor": LUOI, "axes.labelcolor": CHU2, "axes.titlecolor": CHU,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "xtick.color": CHU2, "ytick.color": CHU2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": LUOI, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})


class GhiKhung:
    """Gom khung hinh vao GIF va PDF nhieu trang cung luc."""

    def __init__(self, ten, fig, pdf_moi=1):
        os.makedirs(GIF_DIR, exist_ok=True); os.makedirs(PDF_DIR, exist_ok=True)
        self.ten, self.fig, self.pdf_moi = ten, fig, pdf_moi
        self.khung, self.dem = [], 0
        self.pdf = PdfPages(os.path.join(PDF_DIR, ten + ".pdf"))

    def ghi(self):
        self.fig.canvas.draw()
        a = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        self.khung.append(Image.fromarray(a.copy()))
        if self.dem % self.pdf_moi == 0:
            self.pdf.savefig(self.fig)
        self.dem += 1

    def dong(self, ms=90, giu_cuoi=24):
        self.pdf.savefig(self.fig)           # khung cuoi luon co trong PDF
        self.pdf.close()
        ks = self.khung + [self.khung[-1]] * giu_cuoi
        # Mot bang mau chung cho moi khung (lay tu khung dau, giua, cuoi): pixel
        # khong doi giua hai khung giu nguyen chi so mau, nen GIF nen tot hon
        # han va khong nhap nhay dai mau nhu khi moi khung tu luong tu hoa.
        mau = [ks[0], ks[len(ks) // 2], ks[-1]]
        ghep = Image.new("RGB", (mau[0].width, mau[0].height * 3))
        for i, k in enumerate(mau):
            ghep.paste(k, (0, i * k.height))
        bang = ghep.quantize(colors=224, method=Image.Quantize.MEDIANCUT)
        ks = [k.quantize(palette=bang, dither=Image.Dither.NONE) for k in ks]
        duong = os.path.join(GIF_DIR, self.ten + ".gif")
        ks[0].save(duong, save_all=True, append_images=ks[1:], duration=ms,
                   loop=0, optimize=True, disposal=2)
        plt.close(self.fig)
        print(f"  -> {duong}  ({len(ks)} khung, "
              f"{os.path.getsize(duong)/1e6:.1f} MB)")


def _nhan_buoc(ax, txt):
    return ax.text(0.99, 0.97, txt, transform=ax.transAxes, ha="right",
                   va="top", fontsize=9, color=CHU2)


# ============================================================ 1. ReLU vs tanh
def _chay_poisson(act, iters=4000, moi=50, seed=0):
    set_seed(seed)
    net = FNN([1, 32, 32, 32, 1], act=act)
    xr, xb, xe = uniform_1d(256), torch.tensor([[0.0], [1.0]], dtype=DTYPE), uniform_1d(401)
    f = lambda x: 4 * math.pi ** 2 * torch.sin(2 * math.pi * x)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ra = []
    for it in range(iters + 1):
        x = xr.clone().requires_grad_(True)
        Jr = ((-grad(net(x), x, 2) - f(x)) ** 2).mean()
        J = Jr + 100.0 * (net(xb) ** 2).mean()
        if it % moi == 0:
            with torch.no_grad():
                ra.append((it, Jr.item(), net(xe).squeeze().numpy().copy()))
        opt.zero_grad(); J.backward(); opt.step()
    return xe.squeeze().numpy(), ra


def anim_relu():
    print("[anim] relu: huan luyen hai mang ...")
    x, tanh_ = _chay_poisson("tanh")
    _, relu_ = _chay_poisson("relu")
    ustar = np.sin(2 * np.pi * x)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.14, wspace=0.28)
    fig.suptitle("TN1 · Với ReLU, phần dư của phương trình bậc hai có gradient "
                 "bằng đúng 0", x=0.07, ha="left", fontsize=12, fontweight="bold",
                 color=CHU)
    fig.text(0.07, 0.868, r"$-u'' = 4\pi^2\sin 2\pi x$ trên $(0,1)$ · cùng kiến trúc, "
             "cùng hạt giống, cùng 4 000 vòng Adam", fontsize=9, color=CHU2)

    a1.plot(x, ustar, color=THAM, lw=3.2, label="nghiệm đúng")
    l_t, = a1.plot([], [], color=PINN, label="tanh")
    l_r, = a1.plot([], [], color=CAM, label="ReLU")
    a1.set_xlim(0, 1); a1.set_ylim(-1.6, 1.6); a1.set_xlabel("x")
    a1.set_title("Nghiệm học được", loc="left")
    a1.legend(loc="lower left", ncol=3)
    nb = _nhan_buoc(a1, "")

    its = [r[0] for r in tanh_]
    a2.set_yscale("log"); a2.set_xlim(0, its[-1]); a2.set_ylim(1e-3, 3e3)
    a2.set_xlabel("vòng lặp Adam"); a2.set_title("Hàm mục tiêu phần dư $J_r$", loc="left")
    c_t, = a2.plot([], [], color=PINN, label="tanh")
    c_r, = a2.plot([], [], color=CAM, label="ReLU")
    a2.legend(loc="lower left")
    ghichu = a2.text(0.97, 0.52, "", transform=a2.transAxes, ha="right",
                     fontsize=9, color=CHU)

    g = GhiKhung("tn1_relu_vs_tanh", fig, pdf_moi=2)
    for i in range(len(its)):
        l_t.set_data(x, tanh_[i][2]); l_r.set_data(x, relu_[i][2])
        c_t.set_data(its[:i + 1], [r[1] for r in tanh_[:i + 1]])
        c_r.set_data(its[:i + 1], [r[1] for r in relu_[:i + 1]])
        nb.set_text(f"vòng {its[i]:,}".replace(",", " "))
        ghichu.set_text(f"ReLU: $J_r$ = {relu_[i][1]:.2f} — đứng yên\n"
                        f"tanh: $J_r$ = {tanh_[i][1]:.2e}")
        g.ghi()
    g.dong()


# ======================================================= 2. thien kien pho
KS = (1, 4, 8)


def _chay_pho(che_do, iters=8000, moi=100, seed=0):
    set_seed(seed)
    fm, fs = (32, 6.0) if che_do == "fourier" else (0, 1.0)
    net = FNN([1, 64, 64, 64, 1], act="tanh", fourier_m=fm, fourier_s=fs)
    xr, xb, xe = uniform_1d(256), torch.tensor([[0.0], [1.0]], dtype=DTYPE), uniform_1d(801)
    ue = lambda x: sum(torch.sin(2 * math.pi * k * x) for k in KS)
    fr = lambda x: sum(4 * math.pi ** 2 * k ** 2 * torch.sin(2 * math.pi * k * x) for k in KS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xs = xe.squeeze()

    def chup():
        with torch.no_grad():
            p = net(xe); e = (p - ue(xe)).squeeze()
            c = [abs(2 * torch.trapz(e * torch.sin(2 * math.pi * k * xs), xs).item())
                 for k in KS]
        return p.squeeze().numpy().copy(), c

    ra = [(0,) + chup()]
    for it in range(1, iters + 1):
        if che_do == "regression":
            J = ((net(xr) - ue(xr)) ** 2).mean()
        else:
            x = xr.clone().requires_grad_(True)
            J = ((-grad(net(x), x, 2) - fr(x)) ** 2).mean() + 100.0 * (net(xb) ** 2).mean()
        opt.zero_grad(); J.backward(); opt.step()
        if it % moi == 0:
            ra.append((it,) + chup())
    return xs.numpy(), ue(xe).squeeze().numpy(), ra


def anim_pho():
    ten = {"regression": ("Hồi quy", "học tần số thấp trước"),
           "residual": ("Phần dư", "đảo chiều: bỏ rơi tần số thấp"),
           "fourier": ("Phần dư + Fourier", "dải tần bị dịch chuyển")}
    kq = {}
    for m in ten:
        print(f"[anim] pho: {m} ...", flush=True)
        kq[m] = _chay_pho(m)
    x, ust, _ = kq["regression"]

    fig, ax = plt.subplots(2, 3, figsize=(10.6, 5.6),
                           gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.06, right=0.985, top=0.785, bottom=0.08,
                        hspace=0.46, wspace=0.22)
    fig.suptitle("TN3 · Cùng nghiệm đích, ba hàm mục tiêu — thứ tự học các tần số",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color=CHU)
    fig.text(0.06, 0.918, r"$u^\star(x)=\sin 2\pi x+\sin 8\pi x+\sin 16\pi x$ · "
             r"cột dưới: sai số theo mode $|c_k|$ (thang log)",
             fontsize=9, color=CHU2)

    dong, cot, nhan = {}, {}, {}
    for j, m in enumerate(ten):
        a = ax[0, j]
        a.plot(x, ust, color=THAM, lw=2.6, label="nghiệm đích")
        dong[m], = a.plot([], [], color=PINN, lw=1.6, label="mạng")
        a.set_xlim(0, 1); a.set_ylim(-3.6, 3.6)
        a.set_title(f"{ten[m][0]}\n", loc="left")
        a.text(0, 1.035, ten[m][1], transform=a.transAxes, fontsize=8.8,
               color=CHU2, va="bottom")
        if j == 0:
            a.legend(loc="lower left", ncol=2)
        b = ax[1, j]
        cot[m] = b.bar(range(3), [1, 1, 1], color=MODE, width=0.62,
                       edgecolor=MAT, linewidth=2)
        b.set_yscale("log"); b.set_ylim(1e-4, 20)
        b.set_xticks(range(3), [f"k = {k}" for k in KS])
        b.grid(axis="x", visible=False)
        nhan[m] = [b.text(i, 1, "", ha="center", va="bottom", fontsize=8,
                          color=CHU) for i in range(3)]
    nb = fig.text(0.985, 0.918, "", ha="right", fontsize=9, color=CHU2)

    n = len(kq["regression"][2])
    g = GhiKhung("tn3_thien_kien_pho", fig, pdf_moi=2)
    for i in range(n):
        for m in ten:
            it, pred, c = kq[m][2][i]
            dong[m].set_data(x, pred)
            for r, v, t in zip(cot[m], c, nhan[m]):
                r.set_height(max(v, 1.1e-4))
                t.set_y(max(v, 1.1e-4) * 1.25); t.set_text(f"{v:.1e}")
        nb.set_text(f"vòng {it:,}".replace(",", " "))
        g.ghi()
    g.dong()


# ============================================== 3 & 4. Burgers tu TN10
def _chon_hat_giong_trung_vi():
    rs = []
    for s in range(5):
        f = os.path.join(RES, f"exp10_seed{s}.json")
        if os.path.exists(f):
            rs.append(json.load(open(f)))
    if not rs:
        raise SystemExit("chua co ket qua exp10 -- chay pinns.exp10_burgers_manh truoc")
    rs.sort(key=lambda r: r["eps_L2_cuoi"])
    r = rs[len(rs) // 2]           # hat giong TRUNG VI, khong phai hat tot nhat
    return r, rs


def _du_doan_cuoi(seed, x, ts):
    """u_theta cuoi cua mot hat giong tren luoi (ts, x), float64 day du.

    Anh chup float16 chi du cho ban do mau u; buoc luong tu cua float16 gan
    |u| ~ 1 la 4,9e-4 -- cung bac voi chinh sai so can ve -- nen moi hinh SAI
    SO phai tinh lai tu trong so da luu."""
    from .exp10_burgers_manh import LAYERS
    f = os.path.join(RES, f"exp10_trongso_seed{seed}.pt")
    if not os.path.exists(f):
        raise SystemExit(f"thieu {f} -- chay lai pinns.exp10_burgers_manh {seed}")
    net = FNN(LAYERS, act="tanh")
    net.load_state_dict(torch.load(f))
    X = torch.tensor(np.stack(np.meshgrid(x, ts, indexing="ij"), -1).reshape(-1, 2),
                     dtype=DTYPE)
    with torch.no_grad():
        return net(X).reshape(len(x), len(ts)).numpy().T


def anim_burgers():
    from . import exp5_burgers as B
    r, rs = _chon_hat_giong_trung_vi()
    s = r["seed"]
    d = np.load(os.path.join(RES, f"exp10_anh_seed{s}.npz"))
    anh, xs, tt, buoc, eps = (d["anh"].astype(np.float32), d["xs"], d["tt"],
                              d["buoc"], d["eps"])
    print(f"[anim] burgers: hat giong trung vi s{s}, {len(anh)} anh chup")
    x, ts, U, _ = B.reference(Nx=2047)
    adam_het = 3000

    fig = plt.figure(figsize=(10.8, 5.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.6, 1, 1], height_ratios=[1, 1, 1],
                          left=0.075, right=0.985, top=0.84, bottom=0.09,
                          hspace=0.75, wspace=0.34)
    fig.suptitle("TN10 · PINN học phương trình Burgers, "
                 r"$u_t + u\,u_x = (0{,}01/\pi)\,u_{xx}$",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color=CHU)
    fig.text(0.06, 0.905, f"hạt giống trung vị trong 5 (s = {s}) · mạng 8×20 · "
             "10 000 điểm + RAR · 3 000 Adam rồi L-BFGS", fontsize=9, color=CHU2)

    ah = fig.add_subplot(gs[:, 0])
    im = ah.imshow(anh[0], origin="lower", aspect="auto", cmap=PHAN_KY,
                   vmin=-1, vmax=1, extent=[tt[0], tt[-1], xs[0], xs[-1]],
                   interpolation="bilinear")
    ah.grid(False); ah.set_xlabel("t"); ah.set_ylabel("x")
    ah.set_title(r"$u_\theta(x,t)$", loc="left")
    cb = fig.colorbar(im, ax=ah, fraction=0.05, pad=0.02, ticks=[-1, 0, 1])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)
    for tl in (0.25, 0.5, 0.75):
        ah.axvline(tl, color=CHU, lw=0.8, ls=(0, (2, 3)), alpha=0.55)

    lat = []
    for i, tl in enumerate((0.25, 0.5, 0.75)):
        a = fig.add_subplot(gs[i, 1])
        k = int(np.argmin(abs(ts - tl)))
        a.plot(x, U[k], color=THAM, lw=3.0, label="tham chiếu")
        kk = int(np.argmin(abs(tt - tl)))
        l, = a.plot(xs, anh[0][:, kk], color=PINN, lw=1.6, label="PINN")
        a.set_xlim(-1, 1); a.set_ylim(-1.2, 1.2)
        a.set_title(f"t = {tl}", loc="left", fontsize=9.5)
        if i == 0:
            a.legend(loc="upper right", fontsize=7.5)
        if i < 2:
            a.tick_params(labelbottom=False)
        lat.append((l, kk))

    ae = fig.add_subplot(gs[:, 2])
    ae.set_yscale("log"); ae.set_xlim(0, buoc[-1] * 1.02)
    ae.set_ylim(min(eps.min() * 0.5, 1e-3), 2)
    ae.axvline(adam_het, color=CHU2, lw=0.8, ls=(0, (3, 3)))
    ae.text(adam_het * 0.5, 1.2, "Adam", ha="center", fontsize=8, color=CHU2)
    ae.text(adam_het + (buoc[-1] - adam_het) * 0.5, 1.2, "L-BFGS", ha="center",
            fontsize=8, color=CHU2)
    tn5 = json.load(open(os.path.join(RES, "exp10_tomtat.json")))["tn5_trung_vi"]
    ae.axhline(tn5, color=CAM, lw=1.2, ls=(0, (4, 2)))
    ae.text(buoc[-1], tn5 * 1.25, f"TN5 gốc (trung vị): {tn5:.2f}", ha="right",
            fontsize=8, color=CHU)
    ae.axhline(6.7e-4, color=THAM, lw=1.0, ls=(0, (1, 2)))
    ae.text(buoc[-1], 6.7e-4 * 0.62, "Raissi 2019: 6,7e-4", ha="right",
            va="top", fontsize=8, color=CHU2)
    ce, = ae.plot([], [], color=PINN)
    ch, = ae.plot([], [], "o", color=PINN, ms=6, mec=MAT, mew=1.5)
    ae.set_xlabel("số lần đánh giá hàm mục tiêu")
    ae.set_title(r"sai số $\varepsilon_{L^2}$", loc="left")
    tieu = fig.text(0.985, 0.905, "", ha="right", fontsize=9.5, color=CHU,
                    fontweight="bold")

    chon = list(range(len(anh)))
    if len(chon) > 120:                       # giu GIF nhe: toi da ~120 khung
        buoc_nhay = len(chon) / 120
        chon = sorted({int(i * buoc_nhay) for i in range(120)} | {len(anh) - 1})
    g = GhiKhung("tn10_burgers_huan_luyen", fig, pdf_moi=3)
    for i in chon:
        im.set_data(anh[i])
        for l, kk in lat:
            l.set_ydata(anh[i][:, kk])
        ce.set_data(buoc[:i + 1], eps[:i + 1]); ch.set_data([buoc[i]], [eps[i]])
        tieu.set_text(f"bước {buoc[i]:,} · ε = {eps[i]:.2e}".replace(",", " "))
        g.ghi()
    g.dong(ms=80, giu_cuoi=30)


def anim_vatly():
    from . import exp5_burgers as B
    r, _ = _chon_hat_giong_trung_vi()
    s = r["seed"]
    x, ts, U, _ = B.reference(Nx=2047)
    P = _du_doan_cuoi(s, x, ts)                  # (len(ts), len(x)), float64
    print(f"[anim] vatly: hat giong trung vi s{s}, "
          f"eps_L2 tinh lai = {np.linalg.norm(P - U) / np.linalg.norm(U):.4e}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1]})
    fig.subplots_adjust(left=0.10, right=0.975, top=0.84, bottom=0.10, hspace=0.16)
    fig.suptitle("TN10 · Sóng sin dựng dần thành sốc tại x = 0",
                 x=0.10, ha="left", fontsize=12, fontweight="bold", color=CHU)
    fig.text(0.10, 0.905, f"PINN sau huấn luyện (hạt giống trung vị s = {s}) so với "
             "nghiệm tham chiếu sai phân hữu hạn Nx = 2 047", fontsize=9, color=CHU2)
    lr, = a1.plot([], [], color=THAM, lw=3.4, label="tham chiếu")
    lp, = a1.plot([], [], color=PINN, lw=1.7, label="PINN")
    a1.set_ylim(-1.25, 1.25); a1.set_ylabel("u(x, t)")
    a1.legend(loc="upper right")
    nb = a1.text(0.02, 0.92, "", transform=a1.transAxes, fontsize=10.5,
                 color=CHU, fontweight="bold", va="top")
    le, = a2.plot([], [], color=PINN, lw=1.4)
    a2.set_yscale("log"); a2.set_ylim(1e-7, 1); a2.set_xlim(-1, 1)
    a2.set_ylabel("|sai số|"); a2.set_xlabel("x")

    g = GhiKhung("tn10_burgers_theo_thoi_gian", fig, pdf_moi=2)
    for k in range(0, len(ts), 2):
        lr.set_data(x, U[k]); lp.set_data(x, P[k])
        le.set_data(x, np.maximum(abs(P[k] - U[k]), 1.1e-7))
        nb.set_text(f"t = {ts[k]:.2f}")
        g.ghi()
    g.dong(ms=70, giu_cuoi=28)



# ================================== hinh TINH cho bao cao (khong phai animation)
def hinh_tn10():
    """Hinh 5.x cua bao cao: duong hoi tu cua CA NAM hat giong + nghiem cuoi
    cua hat giong trung vi. Xuat PDF vector vao Images/chap_5/."""
    from . import exp5_burgers as B
    r, rs = _chon_hat_giong_trung_vi()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.15, wspace=0.26)
    tn5 = json.load(open(os.path.join(RES, "exp10_tomtat.json")))["tn5_trung_vi"]
    for q in sorted(rs, key=lambda q: q["seed"]):
        b = [d["buoc"] for d in q["duong_sai_so"]]
        e = [d["eps_L2"] for d in q["duong_sai_so"]]
        trung = q["seed"] == r["seed"]
        a1.plot(b, e, color=PINN if trung else THAM, lw=2.0 if trung else 1.0,
                zorder=3 if trung else 2,
                label=f"hạt giống trung vị (s = {q['seed']})" if trung else None)
    a1.plot([], [], color=THAM, lw=1.0, label="bốn hạt giống còn lại")
    a1.axhline(tn5, color=CAM, lw=1.2, ls=(0, (4, 2)), label=f"TN5 gốc, trung vị {tn5:.2f}")
    a1.axvline(3000, color=CHU2, lw=0.8, ls=(0, (3, 3)))
    a1.text(1500, 1.4, "Adam", ha="center", fontsize=8.5, color=CHU2)
    a1.text(3400, 1.4, "L-BFGS  →", ha="left", fontsize=8.5, color=CHU2)
    a1.set_yscale("log"); a1.set_ylim(3e-4, 3); a1.set_xlim(0, None)
    a1.set_xlabel("số lần đánh giá hàm mục tiêu")
    a1.set_ylabel(r"$\varepsilon_{L^2}$")
    a1.set_title("(a) Hội tụ trên năm hạt giống", loc="left")
    a1.legend(loc="upper right", fontsize=8)

    x, ts, U, _ = B.reference(Nx=2047)
    P = _du_doan_cuoi(r["seed"], x, ts)
    E = np.log10(np.maximum(abs(P - U), 1e-7))
    DON_SAC = LinearSegmentedColormap.from_list(
        "xanh_don", [MAT, "#cfe0f5", "#86b6ef", "#2a78d6", "#0d366b"])
    im = a2.imshow(E.T, origin="lower", aspect="auto", cmap=DON_SAC, vmin=-5,
                   vmax=-2, extent=[ts[0], ts[-1], x[0], x[-1]],
                   interpolation="nearest")
    a2.grid(False); a2.set_xlabel("t"); a2.set_ylabel("x")
    a2.set_title(rf"(b) $|u_\theta - u|$, hạt giống s = {r['seed']}", loc="left")
    cb = fig.colorbar(im, ax=a2, fraction=0.05, pad=0.03,
                      ticks=[-5, -4, -3, -2])
    cb.ax.set_yticklabels([rf"$10^{{{k}}}$" for k in range(-5, -1)])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)
    ra = os.path.join(GOC, "Images", "chap_5", "fig57_tn10.pdf")
    fig.savefig(ra); plt.close(fig)
    print("  ->", ra)

def hinh_tn10b():
    """Hinh 5.8: tach bien RAR. Moi hat giong mot duong noi ba nhanh, de doc
    so sanh CAP (cung hat giong, cung trang thai re nhanh) chu khong chi trung vi."""
    rs = [json.load(open(os.path.join(RES, f"exp10b_seed{s}.json")))
          for s in range(5)]
    nhanh = ["khong_them", "ngau_nhien", "rar"]
    nhan = ["không thêm", "ngẫu nhiên\n(cùng số điểm)", "RAR"]
    mau = [THAM, CAM, PINN]
    muc = [("eps_L2", r"(a) $\varepsilon_{L^2}$", True),
           ("eps_Linf", r"(b) $\max\,|u_\theta - u|$", True),
           ("phan_sai_so_trong_dai", "(c) % bình phương sai số\ntrong dải sốc", False)]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.7))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.76, bottom=0.18, wspace=0.42)
    for ax, (k, tieu, log) in zip(axs, muc):
        for r in rs:
            v = [r["nhanh"][n]["cuoi"][k] * (1 if log else 100) for n in nhanh]
            ax.plot(range(3), v, color=LUOI, lw=1.2, zorder=1)
            for i in range(3):
                ax.plot(i, v[i], "o", color=mau[i], ms=7, mec=MAT, mew=1.5, zorder=3)
        tv = [st.median(r["nhanh"][n]["cuoi"][k] for r in rs) * (1 if log else 100)
              for n in nhanh]
        for i in range(3):
            ax.plot([i - 0.22, i + 0.22], [tv[i]] * 2, color=CHU, lw=2, zorder=4)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 100); ax.set_ylabel("%")
        ax.set_xticks(range(3)); ax.set_xticklabels(nhan, fontsize=8.5)
        ax.set_xlim(-0.5, 2.5); ax.grid(axis="x", visible=False)
        ax.set_title(tieu, loc="left")
    axs[2].axhline(100 * rs[0]["sau_dot_1"]["ti_le_dai"], color=CHU2, lw=0.8,
                   ls=(0, (3, 3)))
    axs[2].text(2.45, 100 * rs[0]["sau_dot_1"]["ti_le_dai"] + 2.5,
                "diện tích dải: 1,4%", ha="right", fontsize=7.5, color=CHU2)
    fig.text(0.075, 0.93, "Mỗi đường nối là một hạt giống; vạch đen là trung vị. "
             "Ba nhánh rẽ ra từ cùng trạng thái sau đợt L-BFGS thứ nhất.",
             fontsize=8.5, color=CHU2)
    ra = os.path.join(GOC, "Images", "chap_5", "fig58_tn10b.pdf")
    fig.savefig(ra); plt.close(fig)
    print("  ->", ra)


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cai = sys.argv[1] if len(sys.argv) > 1 else "tat_ca"
    bang = {"relu": anim_relu, "pho": anim_pho, "burgers": anim_burgers,
            "vatly": anim_vatly, "hinh": hinh_tn10, "tachbien": hinh_tn10b}
    for k in (bang if cai == "tat_ca" else [cai]):
        bang[k]()
