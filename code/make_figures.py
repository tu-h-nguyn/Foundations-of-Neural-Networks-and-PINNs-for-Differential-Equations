"""Ve cac hinh dung trong README, DUNG TU results/*.json da cong bo.

Khong huan luyen lai gi ca: moi hinh chi doc lai cac tep ket qua, nen hinh
trong README va so trong luan van khong the roi nhau.

Chay:  python make_figures.py      -> ../docs/figures/*.svg
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PI = math.pi
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.normpath(os.path.join(HERE, "..", "docs", "figures"))

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"  viet {os.path.relpath(path, os.path.join(HERE, '..'))}")


def fig_stability_bound():
    """TN6 --- chan ||e|| <= (1/pi^2)||r|| va gia thiet lam no dung."""
    e6 = _load("exp6_stability.json")
    nhan = {
        "rang_buoc_cung": ("rang buoc cung: u = x(1-x)N(x)\n(gia thiet thoa CHINH XAC)",
                           "#1b6ca8", "o", "-"),
        "rang_buoc_mem_100": ("rang buoc mem, $\\lambda_b=100$\n(gia thiet thoa gan dung)",
                              "#2e8b57", "s", "--"),
        "rang_buoc_mem_0.01": ("rang buoc mem, $\\lambda_b=0{,}01$\n(gia thiet BI VI PHAM)",
                               "#c0392b", "^", "-."),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    moc = [w["moc"] for w in e6["rang_buoc_cung"]]
    xs = range(len(moc))
    for key, (lab, col, mk, ls) in nhan.items():
        ax.plot(xs, [w["ty_so_e"] for w in e6[key]], marker=mk, color=col,
                linestyle=ls, label=lab, lw=1.6, ms=5)

    ax.axhline(1 / PI**2, color="black", lw=1.4)
    ax.annotate("chan ly thuyet  $1/\\pi^2 \\approx 0{,}1013$",
                xy=(0.02, 1 / PI**2), xycoords=("axes fraction", "data"),
                xytext=(0, 6), textcoords="offset points", fontsize=8.5)
    ax.axhspan(1 / PI**2, 1e2, color="#c0392b", alpha=0.055, lw=0)
    # Dat o goc tren ben TRAI: goc phai la noi duong mau do di qua.
    ax.annotate("vung VI PHAM chan", xy=(0.02, 18.0),
                xycoords=("axes fraction", "data"), ha="left", fontsize=8.5,
                color="#c0392b")

    ax.set_yscale("log")
    ax.set_ylim(3e-4, 4e1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(moc)
    ax.set_xlabel("moc trong qua trinh huan luyen")
    ax.set_ylabel(r"$\|e\|_{L^2}\ /\ \|r\|_{L^2}$")
    ax.set_title("Chan on dinh dung -- va chi dung khi gia thiet bien duoc thoa",
                 fontsize=10.5, pad=10)
    ax.legend(fontsize=7.8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    _save(fig, "stability_bound.svg")


def fig_spectral_reversal():
    """TN3 --- toan tu bac hai dao chieu thu tu uu tien theo tan so.

    Do bang |c_k| CON LAI tai diem dung, chuan hoa theo gia tri ban dau --
    dung chi tieu ma chinh docstring cua TN3 chi dinh, chu khong phai t10
    (t10 khong don dieu nen doc sai thu tu).
    """
    e3 = _load("exp3_spectral.json")
    ten = {"regression": "(a) hoi quy truc tiep\n$P(k)=1$",
           "residual": "(b) mat mat phan du\n$P(k)=4\\pi^2k^2$",
           "fourier": "(c) phan du + dac trung Fourier\n$P(k)=4\\pi^2k^2$, $M=32$"}
    mau = {"1": "#1b6ca8", "4": "#e08a1e", "8": "#c0392b"}
    ks = ("1", "4", "8")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w = 0.26
    for j, k in enumerate(ks):
        vals = [r["c_cuoi"][k] / r["c_dau"][k] for r in
                (next(q for q in e3 if q["mode"] == m) for m in ten)]
        pos = [i + (j - 1) * w for i in range(len(ten))]
        bars = ax.bar(pos, vals, w, color=mau[k], label=f"$k={k}$")
        for p, v in zip(bars, vals, strict=True):
            ax.annotate(f"{v:.2g}".replace(".", ","),
                        (p.get_x() + p.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=7.2, xytext=(0, 2),
                        textcoords="offset points")

    ax.set_yscale("log")
    ax.set_ylim(2e-5, 3e1)
    ax.axhline(1.0, color="black", lw=1.0, ls=":")
    ax.annotate("khong hoc duoc gi", xy=(0.995, 1.0),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=7.6)
    ax.set_xticks(range(len(ten)))
    ax.set_xticklabels(list(ten.values()), fontsize=8.5)
    ax.set_ylabel("phan sai so CON LAI o mode $k$\n" r"$|c_k^{\rm cuoi}| / |c_k^{\rm dau}|$")
    ax.set_title("Thien kien pho bi toan tu vi phan dao chieu\n"
                 "(thap hon = mode do duoc hoc tot hon)", fontsize=10.5, pad=10)
    ax.legend(fontsize=8, ncol=3, loc="upper left", frameon=False)
    _save(fig, "spectral_reversal.svg")


def main():
    print(f"Ve hinh tu {RESULTS}")
    fig_stability_bound()
    fig_spectral_reversal()


if __name__ == "__main__":
    main()
