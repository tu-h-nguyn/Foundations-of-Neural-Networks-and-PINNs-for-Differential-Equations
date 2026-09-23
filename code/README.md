# Mã nguồn thực nghiệm

Cài đặt PINNs bằng PyTorch thuần, không dùng thư viện PINNs đóng gói sẵn, dùng
cho toàn bộ Chương 5 của báo cáo.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
python -m pinns.exp1_relu        # TN1: hàm kích hoạt, sự suy biến của ReLU
python -m pinns.exp2_lambda      # TN2: cái giá của ràng buộc mềm
python -m pinns.exp3_spectral    # TN3: thiên kiến phổ, đảo chiều do toán tử
python -m pinns.exp4_heat        # TN4: phương trình khuếch tán
python -m pinns.exp5_burgers     # TN5: Burgers độ nhớt nhỏ
python -m pinns.exp6_stability   # TN6: kiểm chứng chặn ổn định
python -m pinns.exp7_nu_sweep    # TN7: quét ν, kiểm chứng chặn suy biến 1/ν
python -m pinns.exp8_seeds       # TN8: lặp TN2/TN3/TN4/TN5 trên năm hạt giống
python -m pinns.exp9_chiphi      # TN9: chi phí tính toán, PINN so với sai phân
python -m pinns.run_all          # chạy tất cả
```

**TN10** (Burgers, cấu hình của công trình gốc + RAR) mất khoảng 40 phút một hạt
giống trên một lõi, nên chạy từng hạt giống song song rồi tổng kết:

```bash
export OMP_NUM_THREADS=1
for s in 0 1 2 3 4; do python -m pinns.exp10_burgers_manh $s & done; wait
python -m pinns.exp10_burgers_manh tong_ket   # gộp 5 hạt giống + đo chi phí sai phân
python -m pinns.animate tat_ca                # GIF (Images/anim), PDF khung (slides/anim), hình 5.7
```

**TN10b** tách biến vai trò của RAR: cùng cấu hình TN10, rẽ ba nhánh (RAR / thêm
điểm ngẫu nhiên cùng số lượng / không thêm) từ cùng trạng thái sau đợt L-BFGS
thứ nhất. Nhánh RAR phải khớp TN10 từng bit — script tự kiểm và ghi `khop_TN10`.

```bash
for s in 0 1 2 3 4; do python -m pinns.exp10b_rar_tachbien $s & done; wait
python -m pinns.exp10b_rar_tachbien tong_ket
```

Mỗi hạt giống ghi `results/exp10_seed<s>.json` (số liệu), `exp10_trongso_seed<s>.pt`
(trọng số cuối, `float64`) và `exp10_anh_seed<s>.npz` (ảnh chụp `float16` cho
hoạt hình — không đưa vào git). Ảnh chụp `float16` chỉ dùng để vẽ `u`; mọi hình
**sai số** tính lại từ trọng số, vì bước lượng tử của `float16` gần `|u| ≈ 1` là
`4,9e-4`, cùng bậc với chính sai số cần vẽ.

`exp8_seeds` ghi kết quả ra `results/` **sau mỗi hạt giống** và đọc lại phần đã
có khi chạy lại, nên một lần chạy bị ngắt không làm mất gì. Nó nhận thêm tham số
`heat`, `burgers`, `pho` (TN3) hoặc `lambda` (TN2) để chạy riêng một phần.

Kết quả số ghi ra `results/*.json`.

## Quy ước bám theo báo cáo

| Thành phần | Thiết lập | Truy về |
|---|---|---|
| Hàm kích hoạt | `tanh` | điều kiện $\sigma''\not\equiv0$ cho toán tử bậc hai |
| Khởi tạo | Xavier chuẩn tắc, $\mathrm{Var}(w)=2/(n_{in}+n_{out})$ | `eq:xavier` |
| Độ chệch | $0$ | giả thiết (A2) |
| Lớp ra | tuyến tính | `eq:fnn-output` |
| Độ chính xác | `float64` | phân biệt sai số xấp xỉ với sai số làm tròn |
| Tối ưu | Adam $10^{-3}$ → L-BFGS (Wolfe mạnh) | Thuật toán hai giai đoạn |
| Điểm phối trí 1D | `linspace(a, b, n)`, **kể cả hai điểm biên** | xem ghi chú dưới |
| Lưới đánh giá | khác tập huấn luyện | Định nghĩa chỉ tiêu sai số |

**Ghi chú về lưới điểm phối trí.** Lưới 1D bao gồm cả hai điểm biên. Với
$n=256$ trên $(0,1)$ và $f=4\pi^2\sin(2\pi x)$, trung bình $f^2$ trên lưới này
bằng $776{,}229$ — đúng giá trị $J_r$ mà TN1 báo cáo cho mạng ReLU. Quy ước này
được xác định bằng cách đối chiếu ngược với số liệu trong báo cáo.

## Tái lập

Mọi thí nghiệm dùng một hạt giống cố định (`seed = 0`) và `float64`.

**Nhưng hạt giống cố định là chưa đủ để tái lập từng chữ số.** Kết quả còn phụ
thuộc **số luồng BLAS**. Đo trực tiếp trên bài toán Poisson, mạng `(1,32,32,32,1)`:

| | `OMP_NUM_THREADS=1` | `=2` | `=4` |
|---|---|---|---|
| `‖W⁽¹⁾‖` khởi tạo | `1.5813022424358212` | giống hệt | giống hệt |
| `J` tại vòng 0 | `777.60307687791374` | giống hệt | giống hệt |
| `J` tại vòng 100 | `28.153617819651501` | `...505` | `...505` |
| `J` tại vòng 1499 | `0.023262825` | `0.023287981` | `0.023260501` |

Khởi tạo và vòng lặp đầu tiên giống nhau từng bit; sai khác chỉ xuất hiện khi
tích luỹ qua nhiều bước, vì số luồng đổi **thứ tự cộng dồn** trong phép nhân ma
trận. Sai khác cỡ epsilon máy ban đầu được quỹ đạo tối ưu hoá khuếch đại: sau
$1\,500$ vòng đã lệch ở chữ số có nghĩa thứ tư.

**Trên bài toán khó thì mức khuếch đại lớn hơn hẳn.** Chạy `exp5_burgers` với
hạt giống `0`, cùng một mã, cùng `float64`, cùng $6\,000$ vòng Adam $+\ 800$ vòng
L-BFGS — chỉ khác biến môi trường:

| | `OMP_NUM_THREADS=1` | không đặt (nhiều luồng) | hệ số |
|---|---|---|---|
| `eps_L2` cuối | `1.382437e-01` | `8.185651e-02` | **1.69×** |
| `max rho_b` | `602.9` | `2101.1` | **3.48×** |
| `rho_b` cuối | `419.8` | `705.5` | 1.68× |
| `‖W⁽¹⁾‖_F` cuối | `3.608211` | `3.969849` | 1.10× |

Hai lần chạy khởi tạo **giống hệt nhau** (`rho_dau = 10.538083367986...` ở cả
hai), nên toàn bộ chênh lệch sinh ra trong quá trình huấn luyện. Nói cách khác:
trên bài toán Burgers độ nhớt nhỏ, **số luồng BLAS một mình nó đã đổi sai số cuối
gần hai lần và chỉ số `max rho_b` gấp ba lần rưỡi.** Con số `max rho_b` là đại
lượng mà Chương 5 dùng để chẩn đoán bệnh lý gradient, nên nó phải được đọc theo
bậc độ lớn chứ không theo giá trị.

Mọi số trong `results/*.json` của kho này đều được sinh với `OMP_NUM_THREADS=1`.

Hệ quả thực hành:

- Muốn tái lập từng chữ số, phải cố định **cả** hạt giống **lẫn** số luồng
  (`OMP_NUM_THREADS`), và dùng cùng phiên bản PyTorch.
- Với các thí nghiệm dùng lấy mẫu ngẫu nhiên (TN3, TN4, TN5), chỉ nên kỳ vọng
  tái lập được **kết luận định tính**, không phải con số.
- Các thí nghiệm dùng lưới tất định (TN1, TN2, TN6) và bộ giải tham chiếu sai
  phân hữu hạn của TN5 thì ổn định hơn hẳn: chúng tái lập tới ba đến bốn chữ số
  có nghĩa so với số liệu trong báo cáo.

Xem thêm phần Hạn chế của báo cáo về việc một hạt giống duy nhất là chưa đủ để
rút kết luận thống kê.
