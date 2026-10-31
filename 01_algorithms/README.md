# 01_algorithms - Thuật toán lượng tử cơ bản

Mạch chính của cả folder: **phase kickback**. Bài `00` xây nó từ đầu, ba bài sau
chỉ là các cách khai thác nó. Học theo thứ tự.

| File | Nội dung | Ý chính |
|---|---|---|
| `00_phase_kickback_va_deutsch.ipynb` | Oracle, phase kickback, Deutsch, Deutsch-Jozsa | 1 lần gọi thay vì $2^{n-1}+1$ |
| `01_bernstein_vazirani.ipynb` | Lấy chuỗi bí mật $n$ bit trong 1 lần gọi | interference gom amplitude về đúng $\lvert s\rangle$ |
| `02_grover.ipynb` | Phase oracle + diffuser, số vòng tối ưu, nhiều nghiệm | $O(\sqrt N)$, lợi thế **bình phương** |
| `03_teleportation_va_superdense.ipynb` | No-cloning, teleportation, superdense coding | mid-circuit measurement + `if_test` |

Cần học trước: cả folder [`00_fundamental/`](../00_fundamental/README.md).

## Chạy

Giống folder `00_fundamental`: mở trong VS Code, chọn kernel **`Python 3.12 (Shared Venv)`**
(`~/.venv`), chạy từng ô từ trên xuống.

## Ba điều dễ hiểu sai, bài nào cũng nhắc lại

1. **Chỉ đo qubit dữ liệu.** Dùng `measure_all()` thì ancilla (đang ở $\lvert -\rangle$) cũng bị đo
   và cho bit ngẫu nhiên, làm chuỗi counts có bit rác.
2. **Lợi thế ở đây là số lần gọi oracle**, không phải tổng số cổng. Oracle bên trong vẫn tốn
   $O(n)$ cổng - xem mục 5 của bài `01`.
3. **Grover không phải càng lặp càng tốt.** Quá $k \approx \frac{\pi}{4}\sqrt{N/M}$ thì xác suất tụt xuống.

## Học gì tiếp

QFT và phase estimation (cửa ngõ vào Shor), variational circuit / VQE (hướng dùng nhiều nhất trên máy
thực tế), và mô phỏng nhiễu bằng `qiskit_aer.noise`.
