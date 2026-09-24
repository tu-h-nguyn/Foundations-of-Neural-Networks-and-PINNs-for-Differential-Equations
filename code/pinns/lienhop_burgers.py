"""Bo giai lien hop roi rac cho bai toan nguoc Burgers --- doi thu CO DIEN cua TN11.

Phuong trinh co tham so (eq:burgers-param):
    u_t + z1 * (u^2/2)_x - z2 * u_xx = 0   tren (-1, 1) x (0, 1].

Luoc do THUAN y het nghiem tham chieu cua TN5: sai phan trung tam bac hai dang bao
toan, Runge--Kutta bac bon tuong minh. Khac biet duy nhat: moi thu chua biet deu la
AN SO toi uu hoa:
    z1, s2 (z2 = exp(s2)),
    dieu kien dau u(x, 0)  = spline bac ba qua K_ic nut deu tren [-1, 1],
    dieu kien bien u(-1,t), u(1,t) = spline bac ba qua K_bc nut deu tren [0, 1].
Khi `biet_dau_bien=True`, dieu kien dau -sin(pi x) va bien 0 duoc cho dung, chi con
(z1, s2) la an.

Gradient cua sai so khop du lieu theo MOI an so tinh bang phuong phap LIEN HOP ROI
RAC viet tay: mot lan quet xuoi luu u_n, mot lan quet nguoc truyen bien lien hop
qua tung buoc RK4. Chi phi mot gradient ~ 3 lan mot lan giai thuan, doc lap voi so
an so --- day chinh la "bo may lien hop" ma Muc subsec:thuan-nguoc noi phuong phap
co dien can den. `kiem_gradient()` doi chieu voi vi phan tu dong cua PyTorch.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.interpolate import CubicSpline


def _co_so_spline(nut, diem):
    """Ma tran S (len(diem) x len(nut)): gia tri spline bac ba qua cac nut = S @ c."""
    I = np.eye(len(nut))
    return np.stack([CubicSpline(nut, I[k])(diem) for k in range(len(nut))], 1)


class NoNghiem(Exception):
    """Luoc do tuong minh mat on dinh tai bo tham so dang thu."""


class BaiToanNguoc:
    def __init__(self, Nx, dt, x_du_lieu, t_du_lieu, y_du_lieu, K_ic=33, K_bc=11,
                 biet_dau_bien=False, T=1.0, dt_luu=0.005):
        self.Nx, self.dt = Nx, dt
        self.xg = np.linspace(-1.0, 1.0, Nx + 2)
        self.dx = self.xg[1] - self.xg[0]
        self.nst = int(round(T / dt))
        self.moi = int(round(dt_luu / dt))                 # luu moi `moi` buoc
        assert abs(self.moi * dt - dt_luu) < 1e-12, "dt phai chia het dt_luu"
        self.biet = biet_dau_bien
        self.K_ic, self.K_bc = K_ic, K_bc
        if not biet_dau_bien:
            self.S_ic = _co_so_spline(np.linspace(-1, 1, K_ic), self.xg)[1:-1]
            n = np.arange(self.nst) * dt
            tt = np.concatenate([n, n + dt / 2, n + dt])   # thoi diem cac tang RK4
            self.S_bc = _co_so_spline(np.linspace(0, 1, K_bc), tt)
            self.S_bc0 = _co_so_spline(np.linspace(0, 1, K_bc), np.array([0.0]))[0]
        # du lieu -> (muc luu, o luoi, trong so noi suy tuyen tinh)
        lv = np.rint(np.asarray(t_du_lieu) / dt_luu).astype(int)
        assert np.allclose(lv * dt_luu, t_du_lieu, atol=1e-9), "t du lieu phai tren muc luu"
        s = (np.asarray(x_du_lieu) + 1.0) / self.dx
        j = np.clip(np.floor(s).astype(int), 0, Nx)
        self.lv, self.j, self.w = lv, j, s - j
        self.y = np.asarray(y_du_lieu, dtype=float)
        self.n_an = 2 + (0 if biet_dau_bien else K_ic + 2 * K_bc)

    # --------------------------------------------------------------- tach an so
    def _tach(self, p):
        z1, s2 = p[0], p[1]
        if self.biet:
            u0 = -np.sin(np.pi * self.xg[1:-1])
            gL = np.zeros(3 * self.nst); gR = np.zeros(3 * self.nst)
            g0 = (0.0, 0.0)
        else:
            c = p[2:2 + self.K_ic]
            cL = p[2 + self.K_ic:2 + self.K_ic + self.K_bc]
            cR = p[2 + self.K_ic + self.K_bc:]
            u0 = self.S_ic @ c
            gL, gR = self.S_bc @ cL, self.S_bc @ cR
            g0 = (self.S_bc0 @ cL, self.S_bc0 @ cR)
        return z1, math.exp(s2), u0, gL, gR, g0

    def _F(self, u, a, b, z1, z2):
        f = np.concatenate(([a], u, [b]))
        q = 0.5 * f * f
        return (-z1 * (q[2:] - q[:-2]) / (2 * self.dx)
                + z2 * (f[2:] - 2 * f[1:-1] + f[:-2]) / self.dx ** 2)

    def _F_vjp(self, u, a, b, z1, z2, gb):
        """(g_u, g_a, g_b, g_z1, g_z2) = J^T gb tai trang thai (u, a, b)."""
        dx = self.dx
        f = np.concatenate(([a], u, [b]))
        G = np.zeros(self.Nx + 4); G[2:-2] = gb           # G[k+1] <-> chi so day du k
        Gm, G0, Gp = G[:-2], G[1:-1], G[2:]               # G_{j-1}, G_j, G_{j+1}
        gf = (-z1 * f / (2 * dx) * (Gm - Gp) + z2 / dx ** 2 * (Gm + Gp - 2 * G0))
        q = 0.5 * f * f
        g_z1 = float(gb @ (-(q[2:] - q[:-2]) / (2 * dx)))
        g_z2 = float(gb @ ((f[2:] - 2 * f[1:-1] + f[:-2]) / dx ** 2))
        return gf[1:-1], gf[0], gf[-1], g_z1, g_z2

    # --------------------------------------------------------------- giai thuan
    def giai(self, p, luu_moi_buoc=False):
        z1, z2, u, gL, gR, g0 = self._tach(p)
        h, N = self.dt, self.nst
        muc = [np.concatenate(([g0[0]], u, [g0[1]]))]
        tat_ca = [u] if luu_moi_buoc else None
        for n in range(N):
            k1 = self._F(u, gL[n], gR[n], z1, z2)
            k2 = self._F(u + 0.5 * h * k1, gL[N + n], gR[N + n], z1, z2)
            k3 = self._F(u + 0.5 * h * k2, gL[N + n], gR[N + n], z1, z2)
            k4 = self._F(u + h * k3, gL[2 * N + n], gR[2 * N + n], z1, z2)
            u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if luu_moi_buoc:
                tat_ca.append(u)
            if n % 50 == 0 and not (np.all(np.isfinite(u)) and np.abs(u).max() < 1e3):
                raise NoNghiem
            if (n + 1) % self.moi == 0:
                muc.append(np.concatenate(([gL[2 * N + n]], u, [gR[2 * N + n]])))
        return np.array(muc), tat_ca

    def du_doan(self, muc):
        M = muc[self.lv]
        r = np.arange(len(self.lv))
        return (1 - self.w) * M[r, self.j] + self.w * M[r, self.j + 1]

    # --------------------------------------------------------------- muc tieu + gradient
    def muc_tieu(self, p):
        """Tra ve (L, dL/dp) voi L = trung binh binh phuong sai lech du lieu."""
        z1, z2, u0, gL, gR, g0 = self._tach(p)
        try:
            with np.errstate(over="raise", invalid="raise"):
                muc, us = self.giai(p, luu_moi_buoc=True)
        except (NoNghiem, FloatingPointError):
            # Tham so dang thu lam no nghiem: tra ve gia tri lon de tim kiem
            # theo tia lui buoc. PINN khong co co che nay vi khong co buoc thoi gian.
            self.so_lan_no = getattr(self, "so_lan_no", 0) + 1
            return 1e10, np.zeros(self.n_an)
        e = self.du_doan(muc) - self.y
        L = float(np.mean(e * e))
        # dL/d(trang thai day du tai tung muc luu)
        D = np.zeros_like(muc)
        c = 2.0 / len(e) * e
        np.add.at(D, (self.lv, self.j), c * (1 - self.w))
        np.add.at(D, (self.lv, self.j + 1), c * self.w)

        h, N = self.dt, self.nst
        gA = np.zeros(3 * N); gB = np.zeros(3 * N)
        g_z1 = g_z2 = 0.0
        lam = np.zeros(self.Nx)
        for n in range(N - 1, -1, -1):
            if (n + 1) % self.moi == 0:                  # u_{n+1} la mot muc luu
                d = D[(n + 1) // self.moi]
                lam = lam + d[1:-1]
                gA[2 * N + n] += d[0]; gB[2 * N + n] += d[-1]
            un = us[n]
            a0, b0 = gL[n], gR[n]; ah, bh = gL[N + n], gR[N + n]; a1, b1 = gL[2 * N + n], gR[2 * N + n]
            k1 = self._F(un, a0, b0, z1, z2)
            v2 = un + 0.5 * h * k1; k2 = self._F(v2, ah, bh, z1, z2)
            v3 = un + 0.5 * h * k2; k3 = self._F(v3, ah, bh, z1, z2)
            v4 = un + h * k3
            gk1 = h / 6 * lam; gk2 = h / 3 * lam; gk3 = h / 3 * lam; gk4 = h / 6 * lam
            gu = lam.copy()
            gv, ga, gb_, a_, b_ = self._F_vjp(v4, a1, b1, z1, z2, gk4)
            gu += gv; gk3 = gk3 + h * gv; gA[2 * N + n] += ga; gB[2 * N + n] += gb_; g_z1 += a_; g_z2 += b_
            gv, ga, gb_, a_, b_ = self._F_vjp(v3, ah, bh, z1, z2, gk3)
            gu += gv; gk2 = gk2 + 0.5 * h * gv; gA[N + n] += ga; gB[N + n] += gb_; g_z1 += a_; g_z2 += b_
            gv, ga, gb_, a_, b_ = self._F_vjp(v2, ah, bh, z1, z2, gk2)
            gu += gv; gk1 = gk1 + 0.5 * h * gv; gA[N + n] += ga; gB[N + n] += gb_; g_z1 += a_; g_z2 += b_
            gv, ga, gb_, a_, b_ = self._F_vjp(un, a0, b0, z1, z2, gk1)
            gu += gv; gA[n] += ga; gB[n] += gb_; g_z1 += a_; g_z2 += b_
            lam = gu
        d0 = D[0]
        lam = lam + d0[1:-1]
        g = np.zeros(self.n_an)
        g[0] = g_z1
        g[1] = g_z2 * z2                                  # z2 = exp(s2)
        if not self.biet:
            g[2:2 + self.K_ic] = self.S_ic.T @ lam
            g[2 + self.K_ic:2 + self.K_ic + self.K_bc] = self.S_bc.T @ gA + self.S_bc0 * d0[0]
            g[2 + self.K_ic + self.K_bc:] = self.S_bc.T @ gB + self.S_bc0 * d0[-1]
        return L, g


def kiem_gradient(Nx=63, dt=2.5e-3, seed=0):
    """Doi chieu gradient lien hop voi vi phan tu dong cua PyTorch (float64)."""
    import torch
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1, 1, 300); ts = rng.integers(0, 201, 300) * 0.005
    ys = rng.normal(size=300) * 0.5
    for biet in (True, False):
        bt = BaiToanNguoc(Nx, dt, xs, ts, ys, K_ic=9, K_bc=5, biet_dau_bien=biet)
        p = np.concatenate(([0.9, math.log(0.004)],
                            rng.normal(size=bt.n_an - 2) * 0.3))
        L, g = bt.muc_tieu(p)
        # ban autograd, cung luoc do
        P = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        z1, z2 = P[0], torch.exp(P[1])
        if biet:
            u = torch.tensor(-np.sin(np.pi * bt.xg[1:-1]))
            gL = torch.zeros(3 * bt.nst, dtype=torch.float64); gR = gL.clone()
            g0 = (torch.tensor(0.0, dtype=torch.float64),) * 2
        else:
            K, Kb = bt.K_ic, bt.K_bc
            u = torch.tensor(bt.S_ic) @ P[2:2 + K]
            gL = torch.tensor(bt.S_bc) @ P[2 + K:2 + K + Kb]
            gR = torch.tensor(bt.S_bc) @ P[2 + K + Kb:]
            g0 = (torch.tensor(bt.S_bc0) @ P[2 + K:2 + K + Kb],
                  torch.tensor(bt.S_bc0) @ P[2 + K + Kb:])

        def F(u, a, b):
            f = torch.cat([a.reshape(1), u, b.reshape(1)]); q = 0.5 * f * f
            return (-z1 * (q[2:] - q[:-2]) / (2 * bt.dx)
                    + z2 * (f[2:] - 2 * f[1:-1] + f[:-2]) / bt.dx ** 2)
        N, h = bt.nst, bt.dt
        muc = [torch.cat([g0[0].reshape(1), u, g0[1].reshape(1)])]
        for n in range(N):
            k1 = F(u, gL[n], gR[n]); k2 = F(u + 0.5 * h * k1, gL[N + n], gR[N + n])
            k3 = F(u + 0.5 * h * k2, gL[N + n], gR[N + n]); k4 = F(u + h * k3, gL[2 * N + n], gR[2 * N + n])
            u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if (n + 1) % bt.moi == 0:
                muc.append(torch.cat([gL[2 * N + n].reshape(1), u, gR[2 * N + n].reshape(1)]))
        M = torch.stack(muc)[torch.tensor(bt.lv)]
        r = torch.arange(len(bt.lv)); j = torch.tensor(bt.j); w = torch.tensor(bt.w)
        pred = (1 - w) * M[r, j] + w * M[r, j + 1]
        Lt = ((pred - torch.tensor(ys)) ** 2).mean()
        Lt.backward()
        gt = P.grad.numpy()
        loi = np.abs(g - gt).max() / np.abs(gt).max()
        print(f"  biet_dau_bien={biet}: L {L:.12e} / {Lt.item():.12e}, "
              f"sai lech gradient tuong doi {loi:.2e} ({bt.n_an} an so)")
        assert abs(L - Lt.item()) < 1e-12 * max(1, abs(L)) and loi < 1e-9
    print("  gradient lien hop KHOP vi phan tu dong")


if __name__ == "__main__":
    kiem_gradient()
