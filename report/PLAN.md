# Kế hoạch báo cáo "Chuyên đề nghiên cứu 1"

## 1. Tên đề tài (đề xuất)

**Tìm hiểu điện toán lượng tử cơ bản và bài toán biên dịch mạch lượng tử**
*(Fundamentals of Quantum Computing and Quantum Circuit Compilation)*

Phương án thay thế, nếu muốn nhấn mạnh hướng nghiên cứu tiếp theo:

- *Từ qubit đến transpiler: nền tảng điện toán lượng tử và biên dịch mạch lượng tử như một bài toán tối ưu*
- *Biên dịch mạch lượng tử (quantum circuit compilation) và hướng tiếp cận bằng reinforcement learning*

Thông tin trang bìa:

- Học phần: Chuyên đề nghiên cứu 1
- Học viên: Nguyễn Tất Chung
- Giảng viên hướng dẫn: Phạm Văn Cảnh, Nguyễn Văn Duy
  (thêm học hàm/học vị, ví dụ "TS.", vào `report.tex` nếu cần)

## 2. Mạch logic của báo cáo

Báo cáo đi theo đúng thứ tự của tutorial `master-research-1`, mỗi chương là nền cho chương sau:

```
ket / qubit  ->  cổng lượng tử  ->  thuật toán  ->  compilation  ->  compilation như bài toán MDP
 (00_fundamental)                 (01_algorithms)    (02_compilation)
```

Ý chính xuyên suốt: **thuật toán được viết trên mạch lý tưởng, nhưng chạy trên phần cứng có ràng
buộc** (basis gates, connectivity, noise). Compiler là cầu nối, và chất lượng của nó quyết định trực
tiếp xác suất ra kết quả đúng. Vì vậy compilation là một bài toán tối ưu đáng nghiên cứu.

## 3. Dàn ý chi tiết (khớp với `report.tex`)

| Chương | Nội dung | Nguồn trong repo |
|---|---|---|
| Tóm tắt | Mục tiêu, phạm vi, kết quả chính (vài con số) | toàn bộ |
| 1. Mở đầu | Bối cảnh NISQ, vì sao cần compiler, mục tiêu, phạm vi, công cụ (Qiskit 2.x), cấu trúc báo cáo | `README.md`, `02_compilation/00` |
| 2. Cơ sở điện toán lượng tử | qubit, ký hiệu Dirac (ket, bra, tích trong), superposition, quy tắc Born, Bloch sphere, đo và shot, relative vs global phase, nhiều qubit và tích tensor, little-endian | `00_fundamental/00` |
| 3. Cổng lượng tử | unitary, Pauli, H, S/T/P, R_X/R_Y/R_Z, U(θ,φ,λ), thứ tự nhân ma trận; CX, CZ, SWAP, Toffoli; Bell, GHZ, đo entanglement; tập cổng universal; quy trình Qiskit 5 bước | `00_fundamental/01–03` |
| 4. Thuật toán lượng tử | oracle, phase kickback, Deutsch, Deutsch–Jozsa, Bernstein–Vazirani, Grover (hình học, P(k), số vòng tối ưu), no-cloning, teleportation, superdense coding | `01_algorithms/00–03` |
| 5. Biên dịch mạch lượng tử | ba ràng buộc phần cứng, 6 stage của pass manager, DAGCircuit và front layer, hàm cost; layout; routing và SABRE; synthesis (Euler, KAK, Clifford+T, T-count); tự viết pass | `02_compilation/00–04` |
| 6. Compilation như bài toán MDP | phát biểu MDP cho routing, encode state, action masking, ba kiểu reward, cổng kiểm tra, hai framing khác | `02_compilation/05`, `routing_env.py` |
| 7. Thực nghiệm và thảo luận | phương pháp đo (`bench.py`), bảng số liệu routing/topology/seed, synthesis, thang mốc cho RL, năm "bẫy im lặng" | `02_compilation/README.md`, `bench.py` |
| 8. Kết luận và hướng phát triển | tóm tắt đóng góp, hạn chế, kế hoạch cho Chuyên đề nghiên cứu 2 (DQN/PPO trên `RoutingEnv`) | `02_compilation/README.md` |
| Phụ lục A | Bảng ký hiệu | |
| Phụ lục B | Ánh xạ chương ↔ notebook, cách tái lập kết quả | `README.md` |
| Tài liệu tham khảo | Nielsen & Chuang, các paper gốc của từng thuật toán, SABRE, KAK, Solovay–Kitaev, gridsynth, các paper RL cho compilation | |

## 4. Quy ước trình bày

- Viết bằng tiếng Việt; **giữ nguyên tiếng Anh** cho thuật ngữ chuyên ngành: qubit, ket, bra,
  superposition, entanglement, amplitude, unitary, oracle, phase kickback, transpiler, pass,
  layout, routing, synthesis, coupling map, front layer, reward, policy...
  Lần đầu xuất hiện có thể kèm giải nghĩa tiếng Việt trong ngoặc.
- Tên hàm/lớp Qiskit đặt trong `\texttt{}` (ví dụ `SabreSwap`, `DAGCircuit`).
- Mạch lượng tử vẽ bằng `quantikz`, sơ đồ bằng TikZ, đồ thị bằng `pgfplots` — không cần file ảnh
  ngoài (trừ logo trường), nên sửa nội dung không phải vẽ lại hình.
- Định dạng giống `cv-report`: A4, 11pt, Latin Modern qua `fontspec`, LuaLaTeX, trang bìa có khung
  theo mẫu Trường Đại học Phenikaa, cùng bảng màu `ProbRed / OkGreen / AccentBlue / ...`.
- Mọi con số trong chương 7 lấy từ notebook/README của repo (qiskit 2.5.2), tái lập được.

## 5. Các bước thực hiện

1. [x] Đọc toàn bộ tutorial, trích ý chính và số liệu của từng notebook.
2. [x] Chốt tên đề tài và dàn ý (mục 1 và 3 ở trên).
3. [x] Dựng `report.tex` theo template `cv-report` (preamble, trang bìa, tóm tắt, mục lục).
4. [x] Viết chương 1–4 (nền tảng và thuật toán), kèm mạch `quantikz`.
5. [x] Viết chương 5–6 (compilation và MDP), kèm sơ đồ pipeline.
6. [x] Viết chương 7–8, phụ lục, tài liệu tham khảo.
7. [x] Biên dịch bằng LuaLaTeX, sửa lỗi, kiểm tra bố cục.
8. [ ] **Việc của học viên:** đọc lại, bổ sung học vị giảng viên, chỉnh tên đề tài nếu muốn, thêm
   nhận xét/đánh giá cá nhân, và chạy lại notebook nếu muốn chèn hình chụp kết quả.

## 6. Biên dịch

```bash
cd report
latexmk -pdflua -interaction=nonstopmode -outdir=build report.tex
```

Trên Overleaf: chọn compiler **LuaLaTeX**.
