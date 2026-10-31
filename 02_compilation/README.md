# 02_compilation - Transpiler như một bài toán tối ưu

Folder này coi compiler lượng tử là **bài toán tối ưu** chứ không phải công cụ dùng xong là xong.
Mục tiêu: đủ nền để dựng được một environment reinforcement learning **đúng semantics** và một
**baseline trung thực** - hai thứ mà thiếu cái nào thì mọi kết quả RL sau đó cũng vô giá trị.

Folder này **không** train agent nào. Đó là việc của `03_`.

| File | Nội dung |
|---|---|
| `00_transpiler_pipeline.ipynb` | Ba ràng buộc phần cứng, 6 stage của pass manager, `DAGCircuit`, **front layer**, hàm cost |
| `01_layout.ipynb` | Coupling map, interaction graph, 4 layout pass, `TranspileLayout`, noise-aware layout |
| `02_routing.ipynb` | Vòng lặp routing, tự viết `mini_router`, 3 heuristic của SABRE, độ nhạy seed, topology |
| `03_synthesis_va_tcount.ipynb` | Euler, **KAK** (0/1/2/3 `cx`), Clifford+T, Solovay-Kitaev vs gridsynth, T-count |
| `04_custom_pass_va_benchmark.ipynb` | `AnalysisPass` / `TransformationPass`, thay stage, dựng benchmark harness |
| `05_compilation_as_mdp.ipynb` | Phát biểu MDP, `gymnasium.Env` thật, action masking, 3 kiểu reward, các mốc so sánh |

Hai module bàn giao cho `03_`:

| Module | Dùng để |
|---|---|
| `bench.py` | đo đạc: `chay_benchmark`, `nhieu_router`, `so_voi_baseline`, `bang_tom_tat` |
| `routing_env.py` | `RoutingEnv` (gymnasium), `policy_ngau_nhien`, `policy_greedy`, `kiem_tra_ket_qua` |

## Chạy notebook

Kernel **`Python 3.12 (Shared Venv)`** (`~/.venv`), mở bằng VS Code như hai folder trước. Folder này
cần thêm `gymnasium` (notebook `05`) - đã cài sẵn:

```bash
~/.venv/bin/python -c "import gymnasium, torch; print(gymnasium.__version__, torch.__version__)"
# 1.3.0 2.14.1+cpu
```

`torch` đã cài bản **CPU-only** (`--index-url https://download.pytorch.org/whl/cpu`), chưa dùng ở
folder này - nó là chuẩn bị cho DQN/PPO ở `03_`.

Notebook `04` và `05` `import bench` / `import routing_env`, nên phải chạy **từ trong** folder này
(VS Code tự làm đúng; nếu chạy bằng script thì nhớ `sys.path`).

## Năm cái bẫy tìm được khi viết folder này

Cả năm đều **im lặng** - không có exception nào, chỉ có số liệu sai. Ghi lại vì chúng đều đánh trúng
người tự dựng pipeline routing:

1. **`SabreLayout` tự route bên trong.** Nối nó với `ApplyLayout` là apply layout **hai lần** và ra
   mạch không hợp lệ. Dấu hiệu: pass nào ghi `property_set["final_layout"]` thì đã route rồi, dùng
   một mình. Muốn tự route thì dùng `TrivialLayout` / `DenseLayout` / `VF2Layout`.
   → `01_layout.ipynb`
2. **`Operator(da_compile).equiv(Operator(goc))` trả `False` dù mạch đúng**, vì layout đã hoán vị
   qubit. Dùng `Operator.from_circuit(da_compile)`. → `01_layout.ipynb`
3. **Chiều của hoán vị.** Khi tự route và tự theo dõi $\pi$, phải bù bằng **nghịch đảo** của $\pi$
   (`inv[vật_lý] = logic`). Dùng chính $\pi$ cho `False` ngay khi hoán vị không phải một phép đổi
   chỗ đơn lẻ - và một phép đổi chỗ đơn lẻ là đúng trường hợp test nhỏ hay rơi vào, nên bug này sống
   sót qua test. → `02_routing.ipynb`
4. **`unitary_synthesis_method` chỉ tác dụng lên `UnitaryGate`.** Đưa vào `rz` thường thì
   `BasisTranslator` xử lý trước, plugin không bao giờ chạy, `epsilon` bị bỏ qua im lặng - T-count y
   hệt 127 cho mọi `epsilon` từ `1e-1` tới `1e-6`. Cách tự bảo vệ: luôn **đo sai số thực**.
   → `03_synthesis_va_tcount.ipynb`
5. **Tên plugin.** Ross-Selinger đăng ký là `"gridsynth"`, không phải `"rossselinger"` (dù class tên
   `RossSelingerSynthesis`). Và `unitary_synthesis_method="sk"` **panic ở tầng Rust**, `try/except`
   không đỡ được - dùng pass `SolovayKitaev`. → `03_synthesis_va_tcount.ipynb`

## Các con số làm mốc

Đo trong `~/.venv` với qiskit 2.5.2, tái lập được bằng notebook tương ứng. Số routing là trung
bình trên **5 mạch × 5 seed router = 25 lần đo** mỗi topology; `±` là độ lệch chuẩn, và nó lớn vì
trộn cả "mạch này khó hơn mạch kia" - dùng `bench.nhieu_router` để tách riêng phần nhiễu do seed.

| Đo gì | Kết quả |
|---|---|
| Routing mạch 9 qubit (~95 `cx`), `SabreSwap(decay)`, layout tầm thường | line **75.2 ± 32.8** swap / ring **42.8 ± 15.8** / grid(3,3) **24.5 ± 11.4** / full **0** |
| Dao động chỉ do seed của `SabreSwap(decay)` | **~9%** - cải thiện nhỏ hơn thế là nhiễu |
| `SabreLayout` so với `SabreSwap` + layout tầm thường | tốt hơn **16%** - layout và routing không tách rời |
| Layout hoàn hảo (`VF2Layout`) vs `TrivialLayout` | **0** vs **3** swap trên cùng mạch 4 qubit |
| KAK | unitary 2-qubit ngẫu nhiên cần **đúng 3** `cx`; SWAP cũng 3 |
| Solovay-Kitaev ở sai số 2.5e-3 | **478** cổng T |
| gridsynth ở sai số 9.7e-4 (tốt hơn) | **66** cổng T - hơn **7 lần** |
| `OptimizeCliffordT` | giảm **34%** T-count, sai số không đổi |
| `LookaheadSwap` | chậm gấp **~200×** `SabreSwap` ở 7 qubit mà **cùng** số swap - đã loại khỏi `bench.bo_router` |

Thang mốc cho RL (20 mạch 5 qubit trên `line(5)`, ghép cặp từng mạch, `05_compilation_as_mdp.ipynb`):

| Cách | swap trung bình | kém `SabreSwap(decay)` |
|---|---|---|
| `SabreSwap(decay)` | 8.25 | 0% |
| `SabreSwap(basic)` | 8.45 | 2.3% |
| `GreedyPolicy` (trong env) | 10.45 | **24.9%** |
| `BasicSwap` | 13.90 | 64.6% |
| `RandomPolicy` (trong env) | 27.95 | 266% |

Mục tiêu đầu tiên của `03_` không phải thắng `SabreSwap` mà là **vượt `GreedyPolicy`** - tức là
thu hẹp khoảng 24.9%.

## Ba điều dễ hiểu sai về RL cho biên dịch

1. **Hợp lệ không đồng nghĩa với đúng.** `CheckMap` chỉ kiểm tra cổng 2-qubit nằm trên edge thật.
   Một agent vẫn có thể sinh mạch hợp lệ mà sai unitary. Phải kiểm tra **cả hai**, mọi episode -
   `routing_env.kiem_tra_ket_qua` làm đúng việc đó.
2. **Baseline yếu làm kết quả vô nghĩa.** `mini_router` ở notebook `02` cài **đúng** công thức
   heuristic `basic` trong paper SABRE mà vẫn tốn 10 swap so với 8 của `SabreSwap` - cài đúng công
   thức chưa bằng implementation thật. So agent với `mini_router` thì agent "thắng" dễ và con số đó
   không nói gì.
3. **Ít SWAP không đồng nghĩa với mạch tốt hơn.** `grid(3,3)` ít swap hơn `ring(9)` nhưng depth cao
   hơn. Ba kiểu reward trong `RoutingEnv` (`swap` / `depth` / `fidelity`) **không cùng chiều**;
   `fidelity` dùng error rate thật của từng edge nên gần mục tiêu cuối cùng nhất.

## Đọc thêm

Baseline và các hướng RL đang được nghiên cứu, theo đúng ba bài toán con mà folder này đặt mốc cho:

**Routing**

- Gushu Li, Yufei Ding, Yuan Xie, *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices*,
  ASPLOS '19, 1001-1014. [doi:10.1145/3297858.3304023](https://doi.org/10.1145/3297858.3304023) -
  đây là SABRE, baseline trong `SabreSwap`/`SabreLayout`. Đọc mục về *reverse traversal* và
  *decay effect*.
- M. G. Pozzi, S. J. Herbert, A. Sengupta, R. D. Mullins, *Using Reinforcement Learning to Perform
  Qubit Routing in Quantum Compilers*, ACM Transactions on Quantum Computing 3(2), 10:1-10:25
  (2022) - deep Q-learning, mục tiêu là tối thiểu hoá **depth** thêm vào chứ không phải số SWAP.

**Synthesis và T-count**

- Neil J. Ross, Peter Selinger, *Optimal ancilla-free Clifford+T approximation of z-rotations*,
  Quantum Information and Computation (2016) - thuật toán sau `"gridsynth"`. T-count điển hình
  $4\log_2(1/\epsilon) + O(\log\log(1/\epsilon))$, tối ưu tới sai khác $O(\log\log(1/\epsilon))$.
- Lorenzo Moro, Matteo G. A. Paris, Marcello Restelli, Enrico Prati, *Quantum compiling by deep
  reinforcement learning*, [Communications Physics 4, 178 (2021)](https://www.nature.com/articles/s42005-021-00684-3) -
  RL xấp xỉ unitary 1 qubit bằng gate set rời rạc.
- F. J. R. Ruiz et al., *Quantum circuit optimization with AlphaTensor*,
  [Nature Machine Intelligence (2025)](https://www.nature.com/articles/s42256-025-01001-1) -
  AlphaTensor-Quantum: MCTS + policy network, khai thác liên hệ giữa T-count và phân tích tensor.
  Giảm 37% T-count trên benchmark chuẩn, 47% trên mạch cho elliptic curve cryptography.

**Tối ưu mạch bằng rewrite**

- Thomas Fösel, Murphy Yuezhen Niu, Florian Marquardt, Li Li, *Quantum circuit optimization with
  deep reinforcement learning*, [arXiv:2103.07585](https://arxiv.org/abs/2103.07585).
- Jordi Riu, Jan Nogué, Gerard Vilaplana, Artur Garcia-Saez, Marta P. Estarellas, *Reinforcement
  Learning Based Quantum Circuit Optimization via ZX-Calculus*,
  [Quantum 9, 1758 (2025)](https://quantum-journal.org/papers/q-2025-05-28-1758/) - PPO + graph
  neural network trên ZX-diagram; train trên mạch 5 qubit, tổng quát hoá tới 80 qubit.
