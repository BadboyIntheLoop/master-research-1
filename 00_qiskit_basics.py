"""
Lesson 00 - Qiskit basics: the syntax you will reuse in every later lesson.

Written for Qiskit 2.x. Run it with:

    uv run 00_qiskit_basics.py
"""

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2


def section(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def fmt(probs: dict) -> dict:
    """probabilities_dict() returns numpy types; make them print cleanly."""
    return {str(k): round(float(v), 3) for k, v in probs.items()}


# ---------------------------------------------------------------------------
section("1. Building a circuit")
# ---------------------------------------------------------------------------
# Shortest form: 2 qubits, 2 classical bits.
qc = QuantumCircuit(2, 2)

# Explicit form with named registers - useful once circuits get bigger,
# because the register name is how you read results back out later.
qr = QuantumRegister(2, "q")
cr = ClassicalRegister(2, "c")
qc_named = QuantumCircuit(qr, cr)

print("num_qubits:", qc.num_qubits, " num_clbits:", qc.num_clbits)
print("register names:", qc_named.qregs[0].name, qc_named.cregs[0].name)

# Every qubit starts in |0>. There is no "initialise to zero" call to make.


# ---------------------------------------------------------------------------
section("2. Single-qubit gates")
# ---------------------------------------------------------------------------
demo = QuantumCircuit(1)
demo.x(0)            # Pauli-X : bit flip,   |0> <-> |1>
demo.y(0)            # Pauli-Y : bit + phase flip
demo.z(0)            # Pauli-Z : phase flip, |1> -> -|1>
demo.h(0)            # Hadamard: creates superposition
demo.s(0)            # phase gate, quarter turn (S = sqrt(Z))
demo.t(0)            # T gate, eighth turn (T = sqrt(S))
demo.rx(np.pi / 2, 0)  # rotation around X by an arbitrary angle
demo.ry(np.pi / 4, 0)
demo.rz(np.pi / 8, 0)
print(demo.draw())

# A gate matrix can always be inspected directly.
print("\nH matrix:")
print(np.round(Operator.from_label("H").data, 3))


# ---------------------------------------------------------------------------
section("3. Multi-qubit gates - building a Bell state")
# ---------------------------------------------------------------------------
bell = QuantumCircuit(2)
bell.h(0)            # put qubit 0 into superposition
bell.cx(0, 1)        # CNOT: control = qubit 0, target = qubit 1
print(bell.draw())

# Other common ones:
#   qc.cz(0, 1)      controlled-Z
#   qc.swap(0, 1)    swap two qubits
#   qc.ccx(0, 1, 2)  Toffoli (control, control, target)
#   qc.barrier()     visual/optimisation barrier, not a physical gate

print("\nBell state vector:", Statevector(bell))
print("probabilities:", fmt(Statevector(bell).probabilities_dict()))


# ---------------------------------------------------------------------------
section("4. Measurement and bit ordering")
# ---------------------------------------------------------------------------
m = QuantumCircuit(3, 3)
m.x(0)               # flip ONLY qubit 0
m.measure([0, 1, 2], [0, 1, 2])   # or simply: m.measure_all()

counts = SamplerV2().run([m], shots=100).result()[0].data.c.get_counts()
print("counts:", counts)
print("-> Qiskit is little-endian: qubit 0 is the RIGHTMOST character,")
print("   so flipping qubit 0 of three qubits reads as '001', not '100'.")

# measure_all() appends a barrier and its own register called "meas",
# so the result is read as  result[0].data.meas  instead of  .data.c
m2 = QuantumCircuit(1)
m2.x(0)
m2.measure_all()
print("with measure_all():",
      SamplerV2().run([m2], shots=100).result()[0].data.meas.get_counts())


# ---------------------------------------------------------------------------
section("5. Inspecting a circuit")
# ---------------------------------------------------------------------------
print("depth:     ", bell.depth())      # longest path through the circuit
print("size:      ", bell.size())       # total number of gates
print("count_ops: ", bell.count_ops())  # how many of each gate


# ---------------------------------------------------------------------------
section("6. Parameters - one circuit, many angles")
# ---------------------------------------------------------------------------
theta = Parameter("theta")
p = QuantumCircuit(1)
p.ry(theta, 0)

for value in [0, np.pi / 2, np.pi]:
    bound = p.assign_parameters({theta: value})
    print(f"theta = {value:.4f} -> {fmt(Statevector(bound).probabilities_dict())}")


# ---------------------------------------------------------------------------
section("7. Reusing circuits as building blocks")
# ---------------------------------------------------------------------------
big = QuantumCircuit(4)
big.compose(bell, qubits=[0, 1], inplace=True)   # paste a circuit into another
big.append(bell.to_gate(label="bell"), [2, 3])   # or wrap it as a single gate
print(big.draw())


# ---------------------------------------------------------------------------
section("8. Three ways to get an answer")
# ---------------------------------------------------------------------------
# (a) Statevector - exact maths, no shots, circuit must have NO measurements.
print("(a) Statevector :", Statevector(bell).data.round(3))

# (b) SamplerV2 - simulates repeated measurements, needs measurements.
bell_measured = bell.copy()
bell_measured.measure_all()
sampled = SamplerV2().run([bell_measured], shots=1024).result()[0]
print("(b) SamplerV2   :", sampled.data.meas.get_counts())

# (c) EstimatorV2 - expectation value of an observable, no measurements.
observable = SparsePauliOp("ZZ")
expval = EstimatorV2().run([(bell, observable)]).result()[0].data.evs
print("(c) EstimatorV2 : <ZZ> =", float(expval).__round__(3), "(perfectly correlated)")


# ---------------------------------------------------------------------------
section("9. Transpiling for a backend")
# ---------------------------------------------------------------------------
# A backend only understands its own basis gates. transpile() rewrites the
# circuit into those gates. It is mandatory before backend.run(), and it is
# what you will also do before sending a job to real IBM hardware.
backend = AerSimulator()
transpiled = transpile(bell_measured, backend, optimization_level=2)
# AerSimulator already supports h/cx directly, so nothing changes here. On real
# IBM hardware the basis gates differ and you would see h/cx rewritten.
print("before:", bell_measured.count_ops())
print("after :", transpiled.count_ops())
print("result:", backend.run(transpiled, shots=1024).result().get_counts())


print("\nDone. Next: 01_pauli_x.py")
