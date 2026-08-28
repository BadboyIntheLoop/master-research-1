"""
Bai 1: Cong Pauli-X (cong NOT luong tu) - viet theo API Qiskit 2.x

Pauli-X lat |0> <-> |1>:

        X = [[0, 1],
             [1, 0]]

    X|0> = |1>          X|1> = |0>

Chay:  uv run 01_pauli_x.py
"""

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2


# ---------------------------------------------------------------------------
# 1. Dung mach: 1 qubit + 1 bit co dien de chua ket qua do
# ---------------------------------------------------------------------------
qc = QuantumCircuit(1, 1)
qc.x(0)            # cong Pauli-X
qc.measure(0, 0)   # do qubit 0 -> bit co dien 0

print("Mach:")
print(qc.draw())           # them output="mpl" neu muon ve bang matplotlib
print()


# ---------------------------------------------------------------------------
# 2. Xem trang thai truoc khi do (khong duoc co lenh measure trong mach)
# ---------------------------------------------------------------------------
qc_khong_do = QuantumCircuit(1)
qc_khong_do.x(0)

print("Ma tran cua X:")
print(Operator(qc_khong_do).data.real)          # [[0 1] [1 0]]

sv = Statevector.from_instruction(qc_khong_do)  # bat dau tu |0>
print("Trang thai sau khi ap X len |0>:", sv)
print("Dang ket:", sv.draw("latex_source"))     # '|1\\rangle'
print()


# ---------------------------------------------------------------------------
# 3. Chay mo phong bang SamplerV2 (cach duoc khuyen dung tu Qiskit 1.0 tro di)
#
#    LUU Y - API cu da bi go bo:
#        from qiskit import execute            # KHONG con
#        execute(qc, backend, shots=1024)      # KHONG con
#        Aer.get_backend("qasm_simulator")     # doi thanh AerSimulator()
# ---------------------------------------------------------------------------
sampler = SamplerV2()
job = sampler.run([qc], shots=1024)
ket_qua = job.result()[0]        # [0] = mach dau tien trong danh sach gui di

# Ket qua nam trong thanh ghi co dien; QuantumCircuit(1, 1) dat ten mac dinh la "c"
counts = ket_qua.data.c.get_counts()
print("Ket qua do (SamplerV2):", counts)   # {'1': 1024}
print()


# ---------------------------------------------------------------------------
# 4. Cach thu hai: goi thang backend AerSimulator
#    Voi backend thi PHAI transpile mach truoc khi run.
# ---------------------------------------------------------------------------
backend = AerSimulator()
qc_da_transpile = transpile(qc, backend)
counts2 = backend.run(qc_da_transpile, shots=1024).result().get_counts()
print("Ket qua do (AerSimulator):", counts2)


# ---------------------------------------------------------------------------
# 5. Ve bieu do cot ra file PNG
# ---------------------------------------------------------------------------
from qiskit.visualization import plot_histogram

fig = plot_histogram(counts)
fig.savefig("01_pauli_x_histogram.png", dpi=150, bbox_inches="tight")
print("\nDa luu bieu do: 01_pauli_x_histogram.png")
