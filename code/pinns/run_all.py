"""Chay lan luot moi thi nghiem va ghi ket qua ra results/."""
import importlib
import time

MODULES = [
    ("TN1  ham kich hoat / ReLU",        "pinns.exp1_relu"),
    ("TN2  cai gia rang buoc mem",       "pinns.exp2_lambda"),
    ("TN3  thien kien pho",              "pinns.exp3_spectral"),
    ("TN4  phuong trinh khuech tan",     "pinns.exp4_heat"),
    ("TN5  Burgers do nhot nho",         "pinns.exp5_burgers"),
    ("TN6  kiem chung chan on dinh",     "pinns.exp6_stability"),
    ("TN7  quet nu, ba hat giong",       "pinns.exp7_nu_sweep"),
    ("TN8  do phan tan giua hat giong",  "pinns.exp8_seeds"),
    ("TN9  chi phi tinh toan thuc do",   "pinns.exp9_chiphi"),
    ("TN10 Burgers, cau hinh manh",      "pinns.exp10_burgers_manh"),
    ("TN10b tach bien RAR",              "pinns.exp10b_rar_tachbien"),
    ("TN10c RAD, lay mau lai toan bo",   "pinns.exp10c_rad"),
]


def main():
    for ten, mod in MODULES:
        print("\n" + "=" * 78)
        print(ten)
        print("=" * 78)
        t0 = time.time()
        importlib.import_module(mod).main()
        print(f"[{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
