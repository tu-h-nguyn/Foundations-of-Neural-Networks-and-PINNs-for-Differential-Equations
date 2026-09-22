"""Kiem chung lai moi con so duoc cong bo, tu cac tep results/*.json.

Vi sao khong so sanh tung byte. Chay lai mot thi nghiem tren may khac se KHONG
cho lai dung tung chu so: so luong luong BLAS quyet dinh thu tu cong don trong
phep nhan ma tran, nen sai khac co epsilon may o vong dau duoc quy dao toi uu
hoa khuech dai (xem phan "Tai lap" trong code/README.md). Mot cong kiem tra
`git diff --exit-code` tren results/ vi the se do kim va bao dong gia.

Thay vao do, kiem tra ba lop khang dinh KHONG phu thuoc thu tu cong don:

  A. Dang thuc dong kin   -- dung den chu so may (8 pi^4, 1/pi^2, ...).
  B. Nhat quan noi bo     -- moi ty so / do doc hoi quy duoc tinh lai tu chinh
                             cac thanh phan nam trong tep.
  C. Bat dang thuc / dau  -- cac khang dinh dinh tinh ma luan van rut ra, voi
                             bien do dung bang do on dinh da do duoc.

Chay:  python verify_results.py              (ma thoat 1 neu co muc nao hong)
       python verify_results.py --chay-lai-tn1  (them: chay lai TN1 tu dau)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

PI = math.pi
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

_fails: list[str] = []
_n = 0


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def check(label, ok, detail=""):
    global _n
    _n += 1
    print(f"  {'ok  ' if ok else 'HONG'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fails.append(label)


def close(label, got, want, rtol, detail=""):
    ok = abs(got - want) <= rtol * max(abs(want), 1e-300)
    check(label, ok, detail or f"duoc {got:.10g}, can {want:.10g}, rtol {rtol:g}")


# --------------------------------------------------------------------------
# A. Dang thuc dong kin
# --------------------------------------------------------------------------
def muc_A():
    print("\nA. Dang thuc dong kin (den chu so may)")
    e1 = _load("exp1_relu.json")

    # trung binh cua f^2 tren (0,1) voi f = 4 pi^2 sin(2 pi x):
    #   (4 pi^2)^2 * <sin^2> = 16 pi^4 * 1/2 = 8 pi^4
    close("TN1  trung binh f^2 lien tuc = 8 pi^4",
          e1["f2_lien_tuc"], 8 * PI**4, 1e-14)

    # cung dai luong do tren luoi deu 1001 diem KE CA hai bien. Luoi dong bo
    # voi chu ky cua sin^2 nen trung binh roi rac lech khoi 1/2 mot luong biet
    # truoc: <sin^2(2 pi x)> tren {0, h, ..., 1} voi h = 1/1000.
    n = 1001
    xs = np.linspace(0.0, 1.0, n)
    want = float(((4 * PI**2 * np.sin(2 * PI * xs)) ** 2).mean())
    close("TN1  trung binh f^2 tren luoi 1001 diem",
          e1["f2_tren_luoi_danh_gia"], want, 1e-12)

    # ReLU: sigma'' = 0 hau khap noi, nen u_xx = 0 DUNG BANG 0, va do do
    # grad J_r = 0 dung bang 0. Day la dang thuc, khong phai xap xi.
    relu = next(c for c in e1["cau_hinh"] if c["act"] == "relu")
    check("TN1  ReLU: max|u_xx| tai khoi tao = 0 chinh xac",
          relu["max_uxx_khoi_tao"] == 0.0, f"duoc {relu['max_uxx_khoi_tao']!r}")
    check("TN1  ReLU: ||grad J_r||_inf = 0 chinh xac",
          relu["grad_Jr_inf_khoi_tao"] == 0.0, f"duoc {relu['grad_Jr_inf_khoi_tao']!r}")
    check("TN1  ReLU: ||grad J_r||_2 = 0 chinh xac",
          relu["grad_Jr_l2_khoi_tao"] == 0.0, f"duoc {relu['grad_Jr_l2_khoi_tao']!r}")

    # Gradient bang 0 => Adam khong doi duoc tham so nao => J_r giu nguyen gia
    # tri khoi tao. Voi mang le (u == 0) thi J_r = trung binh f^2 tren luoi
    # PHOI TRI 256 diem -- chinh la 776,229 ma README giai thich.
    m = 256
    xr = np.linspace(0.0, 1.0, m)
    jr0 = float(((4 * PI**2 * np.sin(2 * PI * xr)) ** 2).mean())
    close("TN1  ReLU: J_r cuoi = trung binh f^2 tren luoi phoi tri (mang chet)",
          relu["Jr_cuoi"], jr0, 1e-12, f"duoc {relu['Jr_cuoi']:.6f}, can {jr0:.6f}")

    # Hai mang cung kien truc (1,32,32,32,1) => cung so tham so. Dem tay:
    #   (1*32+32) + (32*32+32) * 2 + (32*1+1)
    want_p = (1 * 32 + 32) + 2 * (32 * 32 + 32) + (32 * 1 + 1)
    for c in e1["cau_hinh"]:
        check(f"TN1  so tham so ({c['act']}) = {want_p}",
              c["so_tham_so"] == want_p, f"duoc {c['so_tham_so']}")

    # Hang so Poincare 1/pi^2 ma TN6 doi chieu.
    close("TN6  hang so chan on dinh 1/pi^2", 1.0 / PI**2, 0.10132118364233778, 1e-14)


# --------------------------------------------------------------------------
# B. Nhat quan noi bo
# --------------------------------------------------------------------------
def muc_B():
    print("\nB. Nhat quan noi bo (tinh lai tu chinh cac thanh phan trong tep)")

    # TN2: so mu la do doc hoi quy log-log cua sai so bien theo lam_b, lay
    # tren dai [0,1 ; 100] dung nhu luan van. Khop lai tu 'rows'.
    e2 = _load("exp2_lambda.json")
    sub = [w for w in e2["rows"] if 0.1 <= w["lam_b"] <= 100.0]
    check("TN2  dai hoi quy gom dung 4 diem", len(sub) == 4, f"duoc {len(sub)}")
    slope, _ = np.polyfit(np.log10([w["lam_b"] for w in sub]),
                          np.log10([w["sai_so_bien"] for w in sub]), 1)
    close("TN2  so mu khop lai tu cac hang", e2["so_mu"], float(slope), 1e-10)

    # TN6: moi ty so phai bang thuong cua hai chuan nam ngay canh no.
    e6 = _load("exp6_stability.json")
    bad = []
    for key, rows in e6.items():
        for w in rows:
            for num, name in ((w["e"], "ty_so_e"), (w["ep"], "ty_so_ep")):
                if not math.isclose(w[name], num / w["r"], rel_tol=1e-12):
                    bad.append(f"{key}/{w['moc']}/{name}")
    check("TN6  moi ty so = ||e||/||r|| tinh lai khop (36 gia tri)", not bad,
          ", ".join(bad[:3]) if bad else "36/36")

    # TN1 va TN2 chia se cung mot lan chay: hang lam_b = 100 cua TN2 chinh la
    # cau hinh tanh cua TN1. Cac con so phai TRUNG KHOP TUNG BIT.
    e1 = _load("exp1_relu.json")
    tanh = next(c for c in e1["cau_hinh"] if c["act"] == "tanh")
    r100 = next(w for w in e2["rows"] if w["lam_b"] == 100.0)
    for k in ("Jr", "eps_L2", "eps_Linf"):
        k1 = {"Jr": "Jr_cuoi"}.get(k, k)
        check(f"TN1/TN2 dung mot lan chay: {k} trung khop tung bit",
              tanh[k1] == r100[k], f"{tanh[k1]!r} vs {r100[k]!r}")


# --------------------------------------------------------------------------
# C. Bat dang thuc va dau
# --------------------------------------------------------------------------
def muc_C():
    print("\nC. Bat dang thuc / dau (cac khang dinh luan van rut ra)")

    # TN6 --- khang dinh trung tam. Menh de chan on dinh noi rang NEU dieu kien
    # bien duoc thoa CHINH XAC thi ||e|| <= (1/pi^2)||r|| va ||e'|| <= (1/pi)||r||.
    e6 = _load("exp6_stability.json")
    cung = e6["rang_buoc_cung"]
    check("TN6  rang buoc cung: sai so bien = 0 chinh xac tai moi moc",
          all(w["sai_so_bien"] == 0.0 for w in cung))
    viol_e = [w["moc"] for w in cung if w["ty_so_e"] > 1 / PI**2]
    viol_ep = [w["moc"] for w in cung if w["ty_so_ep"] > 1 / PI]
    check("TN6  rang buoc cung: ||e||/||r|| <= 1/pi^2 tai moi moc", not viol_e,
          f"bien do lon nhat {max(w['ty_so_e'] for w in cung):.6f} <= {1/PI**2:.6f}")
    check("TN6  rang buoc cung: ||e'||/||r|| <= 1/pi tai moi moc", not viol_ep,
          f"bien do lon nhat {max(w['ty_so_ep'] for w in cung):.6f} <= {1/PI:.6f}")

    # ... va gia thiet la CAN THIET: bo no di thi chan bi vi pham.
    mem = e6["rang_buoc_mem_0.01"]
    check("TN6  rang buoc mem yeu (lam_b=0,01): gia thiet bien bi vi pham",
          any(w["sai_so_bien"] > 1e-3 for w in mem),
          f"sai so bien lon nhat {max(w['sai_so_bien'] for w in mem):.4g}")
    check("TN6  rang buoc mem yeu: chan 1/pi^2 that su bi vuot",
          any(w["ty_so_e"] > 1 / PI**2 for w in mem),
          f"ty so lon nhat {max(w['ty_so_e'] for w in mem):.4g}")

    # TN1 --- ReLU hong hoan toan: sai so tuong doi ~ 1 (mang du doan gan nhu 0).
    e1 = _load("exp1_relu.json")
    tanh = next(c for c in e1["cau_hinh"] if c["act"] == "tanh")
    relu = next(c for c in e1["cau_hinh"] if c["act"] == "relu")
    check("TN1  ReLU: eps_L2 ~ 1 (khong hoc duoc gi)", 0.9 < relu["eps_L2"] < 1.1,
          f"{relu['eps_L2']:.4f}")
    check("TN1  tanh: eps_L2 < 1e-3", tanh["eps_L2"] < 1e-3, f"{tanh['eps_L2']:.3e}")
    check("TN1  tanh chinh xac hon ReLU it nhat 1e3 lan",
          relu["eps_L2"] / tanh["eps_L2"] > 1e3,
          f"{relu['eps_L2'] / tanh['eps_L2']:.4g} lan")

    # TN2 --- so mu do duoc phai gan du bao lam_b^{-1}, va lam_b toi uu HUU HAN.
    e2 = _load("exp2_lambda.json")
    check("TN2  so mu do duoc gan -1 (du bao lam_b^-1)", abs(e2["so_mu"] + 1) < 0.15,
          f"{e2['so_mu']:.4f}, lech {abs(e2['so_mu'] + 1) * 100:.1f}%")
    rows = e2["rows"]
    best = min(rows, key=lambda w: w["eps_L2"])
    check("TN2  lam_b toi uu la huu han (khong phai dau hay cuoi dai quet)",
          best["lam_b"] not in (rows[0]["lam_b"], rows[-1]["lam_b"]),
          f"toi uu tai lam_b = {best['lam_b']:g}")
    check("TN2  sai so bien giam don dieu theo lam_b",
          all(a["sai_so_bien"] > b["sai_so_bien"] for a, b in zip(rows, rows[1:], strict=False)))

    # TN3 --- thien kien pho va su DAO CHIEU do toan tu bac hai gay ra.
    e3 = _load("exp3_spectral.json")
    reg = next(r for r in e3 if r["mode"] == "regression")
    check("TN3  hoi quy: tan so thap hoi tu truoc (t10 tang theo k)",
          reg["t10"]["1"] < reg["t10"]["4"] < reg["t10"]["8"],
          f"t10 = {reg['t10']['1']}, {reg['t10']['4']}, {reg['t10']['8']}")
    check("TN3  hoi quy: moi bien do deu giam manh so voi khoi tao",
          all(reg["c_cuoi"][k] < 0.05 * reg["c_dau"][k] for k in ("1", "4", "8")))

    # TN5 --- bai toan Burgers do nhot nho la bai toan KHO: do doc lon, va
    # can can gradient bi lech nang. Day la ket luan am cua luan van.
    e5 = _load("exp5_burgers.json")
    ref = e5["tham_chieu"]
    check("TN5  nghiem tham chieu that su doc (max|u_x| > 100)",
          ref["max_ux"] > 100, f"max|u_x| = {ref['max_ux']:.1f}")
    check("TN5  bo giai tham chieu hoi tu luoi (sai khac < 1e-3)",
          ref["hoi_tu_luoi"] < 1e-3, f"{ref['hoi_tu_luoi']:.3e}")
    check("TN5  can can gradient xau di trong huan luyen (rho_cuoi > rho_dau)",
          all(w["rho_cuoi"] > w["rho_dau"] for w in e5["ket_qua"]))
    check("TN5  khong cau hinh nao dat duoc eps_L2 < 5e-2 (ket luan am)",
          all(w["eps_L2_cuoi"] >= 5e-2 for w in e5["ket_qua"]),
          f"tot nhat {min(w['eps_L2_cuoi'] for w in e5['ket_qua']):.4f}")

    # TN4 --- phuong trinh khuech tan: L-BFGS cai thien them sau Adam.
    e4 = _load("exp4_heat.json")
    check("TN4  L-BFGS cai thien them sau Adam o moi cau hinh",
          all(w["eps_L2_cuoi"] < w["eps_L2_sau_Adam"] for w in e4))
    check("TN4  dat eps_L2 < 1e-3 o it nhat mot cau hinh",
          any(w["eps_L2_cuoi"] < 1e-3 for w in e4),
          f"tot nhat {min(w['eps_L2_cuoi'] for w in e4):.3e}")


# --------------------------------------------------------------------------
# D. Chay lai TN1 tu dau (tuy chon, ~50 giay)
# --------------------------------------------------------------------------
# Phan nhom duoi day da tung SAI, va CI bat duoc.
#
# Lan dau no duoc hieu chinh bang cach chay lai TN1 voi OMP_NUM_THREADS = 1, 2,
# 4 TREN CUNG MOT MAY, roi ket luan rang moi dai luong khong di qua quy dao toi
# uu hoa deu tai lap tung bit. Chay tren runner cua GitHub -- CPU khac -- ba
# dai luong trong nhom do lech ngay: tanh.max_uxx_khoi_tao,
# tanh.grad_Jr_inf_khoi_tao va relu.eps_L2.
#
# Sai lam la do CHI thay doi mot yeu to roi cho rang do la yeu to duy nhat. So
# luong luong khong phai nguon duy nhat lam doi thu tu cong don: CPU khac chon
# nhan BLAS khac (AVX2 / AVX-512), va thu tu cong don lai doi lan nua.
#
# Nhom duy nhat that su bat bien tren MOI may la nhung dai luong khong he duoc
# tinh bang phep cong don dau ca:
#   * dai luong cua mang ReLU bang KHONG dung nghia den -- sigma'' = 0 hau khap
#     noi, nen u_xx va grad J_r la so 0 cau truc, khong phai tong cua cac so
#     hang trai dau;
#   * so tham so la mot phep dem nguyen.
# Moi thu con lai deu la so thuc dau phay dong, va deu co the lech.
BAT_BIEN_CAU_TRUC = [
    ("relu", "max_uxx_khoi_tao"),
    ("relu", "grad_Jr_inf_khoi_tao"),
    ("relu", "grad_Jr_l2_khoi_tao"),
    ("tanh", "so_tham_so"),
    ("relu", "so_tham_so"),
]
# Dai luong chi di qua MOT luot truyen xuoi/nguoc, khong tich luy qua vong lap:
# sai khac giua cac may chi o muc ulp.
MOT_LUOT = [
    ("tanh", "max_uxx_khoi_tao"),
    ("tanh", "grad_Jr_inf_khoi_tao"),
    ("tanh", "grad_Jr_l2_khoi_tao"),
    ("relu", "Jr_cuoi"),
    ("relu", "eps_L2"),
    ("relu", "eps_Linf"),
]
DUNG_SAI_MOT_LUOT = 1e-9
# Lech tuong doi lon nhat do duoc tren 1/2/4 luong: 3,4% (eps_L2 tai 2 luong).
# Nguong 5% = khop den 2 chu so co nghia, con du bien nhung khong vo nghia.
DUNG_SAI_HUAN_LUYEN = 5e-2


def muc_D():
    """Chay lai TN1 tu dau va doi chieu voi so da cong bo.

    Day moi la kiem tra tai lap THAT SU. No duoc hieu chinh theo dung muc on
    dinh da do duoc, nen khong do kim theo ca hai huong: doi hoi trung khop
    tung bit o moi dai luong se bao dong gia tren may co so luong BLAS khac,
    con chi doi hoi "cung bac do lon" thi se bo lot loi that.
    """
    print("\nD. Chay lai TN1 tu dau va doi chieu (~50 giay)")
    from pinns import exp1_relu

    goc = {c["act"]: c for c in _load("exp1_relu.json")["cau_hinh"]}
    moi = {a: exp1_relu.run(a) for a in ("tanh", "relu")}

    lech = [f"{a}.{k}" for a, k in BAT_BIEN_CAU_TRUC if moi[a][k] != goc[a][k]]
    check(f"TN1  chay lai: {len(BAT_BIEN_CAU_TRUC)} dai luong bat bien CAU TRUC "
          "trung khop tung bit", not lech,
          ", ".join(lech) if lech else f"{len(BAT_BIEN_CAU_TRUC)}/{len(BAT_BIEN_CAU_TRUC)}")

    # In ra do lech thuc te de lan sau con hieu chinh duoc bang so do duoc,
    # thay vi bang phong doan.
    worst, worst_name = 0.0, ""
    for a, k in MOT_LUOT:
        got, want = moi[a][k], goc[a][k]
        rel = abs(got - want) / abs(want) if want else abs(got - want)
        if rel > worst:
            worst, worst_name = rel, f"{a}.{k}"
    check(f"TN1  chay lai: {len(MOT_LUOT)} dai luong mot luot khop den "
          f"{DUNG_SAI_MOT_LUOT:g}", worst < DUNG_SAI_MOT_LUOT,
          f"lech lon nhat {worst:.3e} tai {worst_name}"
          if worst else "trung khop tung bit")

    for k in ("Jr_cuoi", "eps_L2", "eps_Linf"):
        got, want = moi["tanh"][k], goc["tanh"][k]
        rel = abs(got - want) / abs(want)
        check(f"TN1  chay lai: tanh.{k} khop den 2 chu so co nghia",
              rel < DUNG_SAI_HUAN_LUYEN, f"lech tuong doi {rel * 100:.2f}%")


def main():
    print(f"Kiem chung {len(os.listdir(RESULTS))} tep ket qua trong {RESULTS}")
    muc_A()
    muc_B()
    muc_C()
    if "--chay-lai-tn1" in sys.argv:
        muc_D()
    print(f"\n{_n - len(_fails)}/{_n} muc dat.")
    if _fails:
        print("\nHONG:")
        for f in _fails:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
