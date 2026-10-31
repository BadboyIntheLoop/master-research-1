# 00_fundamental - Nền tảng: qubit, cổng lượng tử, Qiskit

Học theo đúng thứ tự dưới đây. Kiến thức nối tiếp nhau, nhưng mỗi notebook vẫn chạy độc lập được.

| File | Nội dung |
|---|---|
| `00_qubit_va_statevector.ipynb` | Qubit, amplitude, superposition, Bloch sphere, đo và collapse, thứ tự bit little-endian |
| `01_cong_1_qubit.ipynb` | X, Y, Z, H, S, T, P, RX/RY/RZ; global phase; thứ tự nhân ma trận ngược với thứ tự vẽ mạch |
| `02_cong_nhieu_qubit.ipynb` | Tích tensor, CX, CZ, SWAP, Toffoli, trạng thái Bell/GHZ, đo entanglement |
| `03_qiskit_workflow.ipynb` | Quy trình 5 bước: xây - xem - đo - chạy - transpile; bảng API cũ vs API 2.x |

## Chạy notebook

Notebook đã cấu hình sẵn kernel **`Python 3.12 (Shared Venv)`** - trỏ tới `~/.venv`
(đã có qiskit 2.5, qiskit-aer, matplotlib, pylatexenc). Mở file `.ipynb` trong VS Code,
chọn đúng kernel đó ở góc trên bên phải, rồi chạy từng ô từ trên xuống.

Kiểm tra nhanh môi trường:

```bash
~/.venv/bin/python -c "import qiskit, qiskit_aer; print(qiskit.__version__, qiskit_aer.__version__)"
```

## Ghi chú

- `~/.venv` không có `sympy`, nên các lệnh render LaTeX của Qiskit
  (`Statevector.draw("latex")`, `array_to_latex`) sẽ báo lỗi. Notebook tránh dùng chúng -
  dùng hàm `ket()` tự viết thay thế. Nếu muốn dùng: `~/.venv/bin/pip install sympy`.
- `~/.venv` cũng không có JupyterLab (chỉ có `ipykernel`), nên hãy mở notebook bằng VS Code.
- Notebook `03` có ghi file `03_mach.png` ra thư mục đang làm việc khi chạy.
