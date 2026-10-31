"""Benchmark harness cho các bài toán biên dịch lượng tử.

Dùng ở notebook `04` và `05`, và ở folder `03_` khi cần so RL agent với baseline.

Bốn nguyên tắc, mỗi cái chặn một kiểu số liệu rác:

1. **Luôn kiểm tra hợp lệ.** Mỗi lần đo đều chạy ``CheckMap``. Kết quả không hợp lệ làm
   ``chay_benchmark`` raise ngay, thay vì lặng lẽ vào bảng.
2. **Tách seed của mạch khỏi seed của router.** Nếu dùng chung một seed thì độ lệch chuẩn
   trộn lẫn "mạch này khó hơn mạch kia" với "router dao động" - hai thứ cần đọc riêng.
   ``nhieu_router()`` đo cái thứ hai, ``bang_tom_tat()`` đọc cái thứ nhất.
3. **Cố định layout khi so router.** Layout tốt giảm ~16% số SWAP, nhiều hơn khác biệt giữa
   các router - không cố định thì đang so sai bài toán.
4. **Đếm SWAP trước stage translation.** Sau translation mỗi SWAP đã thành 3 ``cx`` và
   ``count_ops()["swap"]`` về 0.

Khoá của các dict trả về (``"swap"``, ``"depth"``, ``"hop_le"``, ``"giay"``, ...) để **ASCII không
dấu** có chủ ý: chúng là API, được truy cập bằng ``kq["hop_le"]`` và dùng làm tên cột DataFrame,
nên thuộc nhóm identifier chứ không phải prose.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasicSwap,
    CheckMap,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    LookaheadSwap,
    SabreSwap,
    TrivialLayout,
)

BASIS = ["cx", "rz", "sx", "x"]


def mach_ngau_nhien(n_qubit: int, do_sau: int, seed: int) -> QuantumCircuit:
    """Mạch ngẫu nhiên đã dịch về BASIS, chưa route. Tái lập được theo seed."""
    qc = random_circuit(n_qubit, do_sau, max_operands=2, seed=seed)
    return transpile(qc, basis_gates=BASIS, optimization_level=0)


def bo_topology(n_qubit: int) -> dict[str, CouplingMap]:
    """Các coupling map cùng số qubit, xếp theo bậc tăng dần."""
    bo = {f"line({n_qubit})": CouplingMap.from_line(n_qubit)}
    canh = int(np.sqrt(n_qubit))
    if canh * canh == n_qubit and canh > 1:
        bo[f"grid({canh},{canh})"] = CouplingMap.from_grid(canh, canh)
    bo[f"ring({n_qubit})"] = CouplingMap.from_ring(n_qubit)
    bo[f"full({n_qubit})"] = CouplingMap.from_full(n_qubit)
    return bo


def bo_router(cm: CouplingMap, seed: int = 0, gom_lookahead: bool = False) -> dict[str, object]:
    """Các routing pass có sẵn trong Qiskit, để làm baseline.

    ``LookaheadSwap`` **mặc định bị loại**. Đo được trên mạch random dọc ``line(n)``:

    ====  ===========  ==========  ==========
    n     Lookahead    SabreSwap   chậm gấp
    ====  ===========  ==========  ==========
    4       11.7 ms      3.1 ms       4x
    5       45.8 ms      1.8 ms      26x
    6      245.7 ms      2.4 ms     101x
    7      472.7 ms      2.0 ms     240x
    ====  ===========  ==========  ==========

    (Một lần đo; wall-clock dao động ~20% giữa các lần chạy, nhưng bậc độ lớn thì ổn định.)

    Ở n = 6 và 7 nó cho **đúng cùng** số SWAP như ``SabreSwap(decay)``, nên cái giá đó không
    mua được gì; trên mạch 9 qubit (~125 cx) nó chạy quá vài phút. Bật ``gom_lookahead=True``
    nếu muốn so trên mạch nhỏ.
    """
    bo = {
        "BasicSwap": BasicSwap(cm),
        "SabreSwap(basic)": SabreSwap(cm, heuristic="basic", seed=seed),
        "SabreSwap(decay)": SabreSwap(cm, heuristic="decay", seed=seed),
    }
    if gom_lookahead:
        bo["LookaheadSwap"] = LookaheadSwap(cm)
    return bo


def do_mot_lan(qc: QuantumCircuit, cm: CouplingMap, router, layout=None) -> dict:
    """Route một mạch bằng một router, trả về metric của lần đo đó."""
    layout = layout if layout is not None else TrivialLayout(cm)
    pm = PassManager([layout, FullAncillaAllocation(cm), EnlargeWithAncilla(),
                      ApplyLayout(), router, CheckMap(cm)])
    t0 = time.perf_counter()
    out = pm.run(qc)
    giay = time.perf_counter() - t0
    n_swap = out.count_ops().get("swap", 0)
    return {
        "swap": n_swap,
        "depth": out.depth(),
        "hai_qubit": len(out.get_instructions("cx")) + 3 * n_swap,
        "hop_le": bool(pm.property_set["is_swap_mapped"]),
        "giay": giay,
    }


def chay_benchmark(n_qubit: int = 9, do_sau: int = 20,
                   seeds_mach=range(3), seeds_router=range(3),
                   topologies=None, routers=None, kiem_tra: bool = True) -> pd.DataFrame:
    """Quét mạch x topology x router x seed_router. Một dòng = một lần đo.

    ``routers`` là callable ``(cm, seed) -> dict[str, pass]``; mặc định dùng :func:`bo_router`.
    """
    topologies = topologies if topologies is not None else bo_topology(n_qubit)
    routers = routers if routers is not None else bo_router
    rows = []
    for sm in seeds_mach:
        qc = mach_ngau_nhien(n_qubit, do_sau, sm)
        cx_goc = qc.count_ops().get("cx", 0)
        for ten_tp, cm in topologies.items():
            for sr in seeds_router:
                for ten_r, router in routers(cm, sr).items():
                    kq = do_mot_lan(qc, cm, router)
                    if kiem_tra and not kq["hop_le"]:
                        raise AssertionError(
                            f"mạch không hợp lệ: {ten_r} / {ten_tp} / mạch {sm} / seed {sr}")
                    rows.append({"topology": ten_tp, "router": ten_r,
                                 "seed_mach": sm, "seed_router": sr,
                                 "cx_goc": cx_goc, **kq})
    return pd.DataFrame(rows)


def bang_tom_tat(df: pd.DataFrame, chi_so: str = "swap") -> pd.DataFrame:
    """Trung bình ``chi_so`` theo topology x router, lấy mean trên mọi seed."""
    return (df.pivot_table(index="topology", columns="router", values=chi_so,
                           aggfunc="mean", sort=False).round(1))


def nhieu_router(df: pd.DataFrame, chi_so: str = "swap") -> pd.DataFrame:
    """Dao động do **seed của router**, với mạch và topology giữ cố định.

    Đây là con số cần so với "mức cải thiện" của một agent mới: cải thiện nhỏ hơn dao động
    này thì không phân biệt được với nhiễu.
    """
    g = df.groupby(["router", "topology", "seed_mach"], sort=False)[chi_so]
    tung_mach = g.agg(["mean", "std", "min", "max"])
    tung_mach["bien_do_pct"] = ((tung_mach["max"] - tung_mach["min"])
                                / tung_mach["mean"].replace(0, np.nan) * 100)
    return (tung_mach.groupby("router", sort=False)[["std", "bien_do_pct"]]
            .mean().round(2).rename(columns={"std": "std_do_seed",
                                             "bien_do_pct": "bien_do_%_trung_binh"}))


def xep_hang(df: pd.DataFrame, chi_so: str = "swap") -> pd.DataFrame:
    """Xếp hạng router trên mọi mạch, topology và seed."""
    g = (df.groupby("router", sort=False)[chi_so]
           .agg(trung_binh="mean", std="std", tot_nhat="min", te_nhat="max")
           .sort_values("trung_binh"))
    g["so_lan_do"] = df.groupby("router", sort=False)[chi_so].size()
    return g.round(2)


def so_voi_baseline(df: pd.DataFrame, baseline: str = "SabreSwap(decay)",
                    chi_so: str = "swap") -> pd.DataFrame:
    """Phần trăm hơn/kém baseline, ghép theo đúng (mạch, topology, seed).

    Ghép từng cặp thay vì so hai giá trị trung bình - cần thiết vì độ khó của mạch chênh
    nhau nhiều hơn khác biệt giữa các router.
    """
    khoa = ["topology", "seed_mach", "seed_router"]
    goc = df[df["router"] == baseline].set_index(khoa)[chi_so]
    ra = []
    for ten, nhom in df.groupby("router", sort=False):
        v = nhom.set_index(khoa)[chi_so]
        chung = v.index.intersection(goc.index)
        if len(chung) == 0:
            continue
        hieu = (v.loc[chung] - goc.loc[chung])
        mau = goc.loc[chung].replace(0, np.nan)
        ra.append({"router": ten,
                   f"{chi_so}_trung_binh": round(v.loc[chung].mean(), 2),
                   "hon_kem_tuyet_doi": round(hieu.mean(), 2),
                   "hon_kem_%": round((hieu / mau).mean() * 100, 1),
                   "so_cap": len(chung)})
    return pd.DataFrame(ra).sort_values("hon_kem_%").reset_index(drop=True)
