"""Qubit routing dưới dạng một môi trường reinforcement learning.

Môi trường tuân thủ API của ``gymnasium`` (đã qua ``gymnasium.utils.env_checker.check_env``),
nên gắn trực tiếp được vào DQN/PPO ở folder ``03_``.

Phát biểu MDP
-------------

* **state** - front layer của DAG còn lại (những cổng 2-qubit *chạy được ngay* nếu chúng kề nhau),
  cộng mapping :math:`\\pi` hiện tại. Xem :meth:`RoutingEnv._quan_sat` để biết cách encode.
* **action** - chọn một edge của coupling map để chèn SWAP. ``action_space`` là
  ``Discrete(số edge vô hướng)``.
* **transition** - chèn SWAP đó, cập nhật :math:`\\pi`, rồi **thực thi hết** các cổng đã hợp lệ
  (phần này là bắt buộc (deterministic), không phải quyết định).
* **reward** - xem :data:`KIEU_REWARD`.
* **terminated** - DAG rỗng. **truncated** - vượt ``so_buoc_toi_da``.

Ba chỗ dễ làm sai, đã xử lý sẵn
-------------------------------

1. **Action masking.** Không phải edge nào cũng có nghĩa ở mỗi bước. ``info["action_mask"]``
   đánh dấu các edge có liên quan tới qubit đang bị kẹt; SWAP ngoài tập đó chỉ làm dài mạch.
   Môi trường **vẫn nhận** action bị mask (RL library nào cũng cần ``step`` không raise), nhưng
   tính nó là một bước tốn reward.
2. **Hợp lệ không đồng nghĩa với đúng.** :func:`kiem_tra_ket_qua` kiểm tra **cả hai**: mạch khớp
   coupling map (``CheckMap``) và cùng unitary với mạch gốc (bù hoán vị bằng **nghịch đảo** của
   :math:`\\pi`).
3. **Reward "âm số SWAP" không phải mục tiêu thật.** Ít SWAP không tự động nghĩa là depth thấp hay
   fidelity cao - xem :data:`KIEU_REWARD` và notebook ``05``.

Khoá dict trả về (``"hop_le"``, ``"dung"``, ``"swap"``, ``"buoc"``, ...) để **ASCII không dấu** có
chủ ý - chúng là API, truy cập bằng ``kq["dung"]``, nên thuộc nhóm identifier chứ không phải prose.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CheckMap

#: Các kiểu reward. Mỗi bước = một SWAP, nên reward luôn âm; agent tối đa hoá tức là
#: tối thiểu hoá cost.
#:
#: * ``"swap"`` - ``-1`` mỗi SWAP. Đơn giản, nhưng bỏ qua depth và noise.
#: * ``"depth"`` - ``-(depth tăng thêm)``. Thưởng cho SWAP song song được với nhau.
#: * ``"fidelity"`` - ``log(1 - err)**3`` của edge bị SWAP (1 SWAP = 3 cổng 2-qubit).
#:   Gần mục tiêu thật nhất: nó biết edge nào tốt edge nào tệ. Cần ``error_rate``.
KIEU_REWARD = ("swap", "depth", "fidelity")


class RoutingEnv(gym.Env):
    """Môi trường routing cho một coupling map cố định và một bộ mạch.

    Parameters
    ----------
    coupling_map:
        Topology của thiết bị. Số qubit của mạch phải bằng ``coupling_map.size()``.
    sinh_mach:
        Callable ``(rng) -> QuantumCircuit``. Được gọi ở mỗi ``reset()``, nên một agent học
        trên *phân phối* mạch chứ không học thuộc một mạch.
    kieu_reward:
        Một trong :data:`KIEU_REWARD`.
    error_rate:
        ``{(a, b): error}`` cho ``kieu_reward="fidelity"``; lấy từ
        ``backend.target["cx"]``.
    so_buoc_toi_da:
        Cắt episode. Mặc định ``8 * số cổng 2-qubit`` của mạch.
    """

    metadata = {"render_modes": []}

    def __init__(self, coupling_map: CouplingMap, sinh_mach, kieu_reward: str = "swap",
                 error_rate: dict | None = None, so_buoc_toi_da: int | None = None):
        super().__init__()
        if kieu_reward not in KIEU_REWARD:
            raise ValueError(f"kieu_reward phải là một trong {KIEU_REWARD}")
        if kieu_reward == "fidelity" and not error_rate:
            raise ValueError("kieu_reward='fidelity' cần error_rate")

        self.cm = coupling_map
        self.n = coupling_map.size()
        self.D = np.asarray(coupling_map.distance_matrix)
        self.edges = sorted(set(tuple(sorted(e)) for e in coupling_map.get_edges()))
        self.sinh_mach = sinh_mach
        self.kieu_reward = kieu_reward
        self.error_rate = error_rate or {}
        self.so_buoc_toi_da = so_buoc_toi_da

        self.action_space = spaces.Discrete(len(self.edges))
        # quan sát: [front layer trên qubit vật lý | lớp kế tiếp | tiến độ]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 * self.n * self.n + 1,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self.mach_goc: QuantumCircuit | None = None

    # ------------------------------------------------------------------ nội bộ

    def _cap_front(self):
        """Các cặp qubit **logic** cần kề nhau ở front layer."""
        return [tuple(self._dag.find_bit(q).index for q in nd.qargs)
                for nd in self._dag.front_layer() if len(nd.qargs) == 2]

    def _quan_sat(self) -> np.ndarray:
        """Encode state thành vector.

        Hai ma trận ``n x n`` trên **qubit vật lý** (không phải logic - nhờ vậy quan sát bất
        biến với cách đánh số qubit logic), cộng một số đo tiến độ:

        * ``front[a][b] = 1`` nếu có cổng ở front layer cần hai qubit vật lý ``a``, ``b`` kề nhau
        * ``ke_tiep`` - tương tự cho lớp sau front layer (phần "lookahead")
        * tiến độ - tỉ lệ cổng đã thực thi
        """
        front = np.zeros((self.n, self.n), dtype=np.float32)
        for a, b in self._cap_front():
            front[self.pi[a], self.pi[b]] = front[self.pi[b], self.pi[a]] = 1.0

        # "extended set" kiểu SABRE: các cổng 2-qubit ngay sau front layer. Lấy bằng
        # successors thay vì copy DAG - copy ở mỗi observation thì quá đắt.
        ke_tiep = np.zeros((self.n, self.n), dtype=np.float32)
        for nd in self._dag.front_layer():
            for s in self._dag.successors(nd):
                if getattr(s, "qargs", None) is None or len(s.qargs) != 2:
                    continue
                a, b = (self._dag.find_bit(q).index for q in s.qargs)
                ke_tiep[self.pi[a], self.pi[b]] = ke_tiep[self.pi[b], self.pi[a]] = 1.0

        tien_do = np.float32(1.0 - self._dag.size() / max(self._size_goc, 1))
        return np.concatenate([front.ravel(), ke_tiep.ravel(), [tien_do]]).astype(np.float32)

    def _thuc_thi_duoc(self) -> int:
        """Thực thi mọi cổng ở front layer mà mapping hiện tại làm cho hợp lệ.

        Lặp tới khi không thực thi thêm được - vì thực thi một cổng có thể mở ra cổng mới.
        Trả về số cổng đã thực thi.
        """
        da_lam = 0
        while True:
            tien_do = False
            for node in list(self._dag.front_layer()):
                qs = [self._dag.find_bit(q).index for q in node.qargs]
                if len(qs) == 1 or self.D[self.pi[qs[0]]][self.pi[qs[1]]] == 1:
                    self.mach_ra.append(node.op, [self.pi[q] for q in qs])
                    self._dag.remove_op_node(node)
                    tien_do = True
                    da_lam += 1
            if not tien_do:
                return da_lam

    def action_mask(self) -> np.ndarray:
        """``True`` cho các edge có liên quan tới qubit đang bị kẹt.

        SWAP ngoài tập này không thể giúp cổng nào ở front layer, nên với một policy học được
        thì mask này thu nhỏ không gian tìm kiếm rất nhiều.
        """
        cap = self._cap_front()
        if not cap:
            return np.ones(len(self.edges), dtype=bool)
        ket = {self.pi[q] for c in cap for q in c}
        return np.array([e[0] in ket or e[1] in ket for e in self.edges], dtype=bool)

    def _info(self) -> dict:
        return {
            "action_mask": self.action_mask(),
            "so_swap": self.so_swap,
            "con_lai": self._dag.size(),
            "pi": list(self.pi),
        }

    # ------------------------------------------------------------- gymnasium API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.mach_goc = self.sinh_mach(self._rng)
        if self.mach_goc.num_qubits != self.n:
            raise ValueError(
                f"mạch có {self.mach_goc.num_qubits} qubit, coupling map có {self.n}")

        self._dag = circuit_to_dag(self.mach_goc)
        self._size_goc = self._dag.size()
        self.pi = list(range(self.n))          # layout ban đầu: tầm thường
        self.mach_ra = QuantumCircuit(self.n)
        self.mach_ra.global_phase = self.mach_goc.global_phase
        self.so_swap = 0
        self.so_buoc = 0
        self._depth_truoc = 0
        self._gioi_han = self.so_buoc_toi_da or max(
            8 * len(self.mach_goc.get_instructions("cx")), 50)

        self._thuc_thi_duoc()                  # cổng nào hợp lệ sẵn thì làm luôn
        # đặt mốc depth SAU bước trên: những cổng đó không do agent quyết định, tính vào
        # reward bước đầu là phạt oan
        self._depth_truoc = self.mach_ra.depth()
        return self._quan_sat(), self._info()

    def step(self, action: int):
        action = int(action)
        a, b = self.edges[action]

        la, lb = self.pi.index(a), self.pi.index(b)
        self.pi[la], self.pi[lb] = self.pi[lb], self.pi[la]
        self.mach_ra.append(SwapGate(), [a, b])
        self.so_swap += 1
        self.so_buoc += 1

        self._thuc_thi_duoc()

        if self.kieu_reward == "swap":
            reward = -1.0
        elif self.kieu_reward == "depth":
            d = self.mach_ra.depth()
            reward = -float(d - self._depth_truoc)
            self._depth_truoc = d
        else:                                   # fidelity
            err = self.error_rate.get((a, b), self.error_rate.get((b, a), 1e-3))
            reward = 3.0 * float(np.log1p(-err))

        terminated = self._dag.size() == 0
        truncated = (not terminated) and self.so_buoc >= self._gioi_han
        return self._quan_sat(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------ tiện ích

    def ket_qua(self) -> tuple[QuantumCircuit, list[int]]:
        """Mạch đã route và mapping cuối. Chỉ có nghĩa khi episode đã ``terminated``."""
        return self.mach_ra, list(self.pi)


# --------------------------------------------------------------------- kiểm tra


def kiem_tra_ket_qua(mach_goc: QuantumCircuit, mach_ra: QuantumCircuit,
                     pi: list[int], cm: CouplingMap, kiem_unitary: bool = True) -> dict:
    """Kiểm tra output của một episode: **hợp lệ** và **đúng**.

    ``kiem_unitary`` tốn :math:`O(4^n)` nên chỉ bật với mạch nhỏ (≤ 6 qubit).
    Hoán vị được bù bằng **nghịch đảo** của ``pi`` - dùng chính ``pi`` sẽ cho ``False``
    ngay khi hoán vị không phải một phép đổi chỗ đơn lẻ.
    """
    pm = PassManager([CheckMap(cm)])
    pm.run(mach_ra)
    hop_le = bool(pm.property_set["is_swap_mapped"])

    dung = None
    if kiem_unitary:
        inv = [pi.index(i) for i in range(len(pi))]
        dung = bool(Operator(mach_ra).equiv(Operator(mach_goc).apply_permutation(inv)))
    return {"hop_le": hop_le, "dung": dung,
            "swap": mach_ra.count_ops().get("swap", 0), "depth": mach_ra.depth()}


# --------------------------------------------------------------------- policy mốc


def policy_ngau_nhien(obs, info, env, rng) -> int:
    """Chọn ngẫu nhiên trong các action được mask. Mốc **dưới**."""
    hop_le = np.flatnonzero(info["action_mask"])
    return int(rng.choice(hop_le))


def policy_greedy(obs, info, env, rng) -> int:
    """Chọn SWAP giảm nhiều nhất tổng khoảng cách của front layer.

    Đây là heuristic ``basic`` của SABRE, cài lại trong ngôn ngữ của môi trường. Nó là mốc
    **giữa**: hơn random rất nhiều, nhưng vẫn kém ``SabreSwap(decay)``.
    """
    cap = env._cap_front()
    if not cap:
        return int(rng.choice(np.flatnonzero(info["action_mask"])))

    def H(p):
        return sum(env.D[p[a]][p[b]] for a, b in cap)

    tot, H_tot = None, None
    for idx in np.flatnonzero(info["action_mask"]):
        a, b = env.edges[idx]
        p = list(env.pi)
        la, lb = p.index(a), p.index(b)
        p[la], p[lb] = p[lb], p[la]
        h = H(p)
        if H_tot is None or h < H_tot:
            tot, H_tot = int(idx), h
    return tot


def chay_episode(env: RoutingEnv, policy, seed: int, rng=None) -> dict:
    """Chạy một episode bằng ``policy``, trả về metric kèm kết quả kiểm tra."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    tong_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy(obs, info, env, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        tong_reward += reward
    mach_ra, pi = env.ket_qua()
    kq = {"reward": tong_reward, "swap": env.so_swap, "buoc": env.so_buoc,
          "xong": terminated, "truncated": truncated}
    if terminated:
        kq.update(kiem_tra_ket_qua(env.mach_goc, mach_ra, pi, env.cm,
                                   kiem_unitary=env.n <= 6))
    return kq
