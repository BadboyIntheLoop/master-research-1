# Quantum computing notes with Qiskit

Environment managed with [uv](https://docs.astral.sh/uv/).

## Lessons

| File | Topic |
|---|---|
| `00_qiskit_basics.py` | Core syntax: circuits, gates, measurement, bit ordering, primitives, transpile |
| `01_pauli_x.py` | The Pauli-X gate |

## Setup

```bash
uv sync                    # create .venv and install the locked dependencies
uv sync --group notebook   # add JupyterLab when needed
```

## Running

```bash
uv run 00_qiskit_basics.py
uv run 01_pauli_x.py
uv run jupyter lab          # after syncing the notebook group
```

## Managing packages

```bash
uv add qiskit-ibm-runtime    # add a package (updates pyproject.toml + uv.lock)
uv remove matplotlib         # remove a package
uv lock --upgrade            # bump versions within the declared constraints
uv tree                      # inspect the dependency tree
```

Avoid plain `pip install`. Only `uv add` records the change in `pyproject.toml`
and `uv.lock`, which is what makes `uv sync` reproduce the exact same
environment on another machine.

## Old API vs current API (Qiskit 2.x)

| Old (Qiskit 0.x) | Current |
|---|---|
| `from qiskit import execute` | removed - use a primitive or `backend.run()` |
| `execute(qc, backend, shots=1024)` | `SamplerV2().run([qc], shots=1024)` |
| `Aer.get_backend("qasm_simulator")` | `AerSimulator()` from `qiskit_aer` |
| `result.get_counts()` | `result()[0].data.<register_name>.get_counts()` |
| `qiskit.providers.aer` | separate `qiskit-aer` package |
| circuit sent straight to a backend | `transpile(qc, backend)` first, then `backend.run()` |
