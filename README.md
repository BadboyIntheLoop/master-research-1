# Quantum computing notes with Qiskit

Environment managed with [uv](https://docs.astral.sh/uv/).

## Lessons

All fundamentals live in [`00_fundamental/`](00_fundamental/README.md).

| File | Topic |
|---|---|
| `00_fundamental/00_qubit_va_statevector.ipynb` | Qubits, amplitudes, superposition, Bloch sphere, measurement |
| `00_fundamental/01_cong_1_qubit.ipynb` | Single-qubit gates: X, Y, Z, H, S, T, P, RX/RY/RZ |
| `00_fundamental/02_cong_nhieu_qubit.ipynb` | Multi-qubit gates, Bell/GHZ states, entanglement measures |
| `00_fundamental/03_qiskit_workflow.ipynb` | End-to-end workflow: build, inspect, measure, run, transpile |

Then [`01_algorithms/`](01_algorithms/README.md) — all four lessons hang off one idea, phase kickback.

| File | Topic |
|---|---|
| `01_algorithms/00_phase_kickback_va_deutsch.ipynb` | Oracles, phase kickback, Deutsch, Deutsch-Jozsa |
| `01_algorithms/01_bernstein_vazirani.ipynb` | Recover an n-bit secret string in one oracle call |
| `01_algorithms/02_grover.ipynb` | Phase oracle + diffuser, optimal iteration count, multiple solutions |
| `01_algorithms/03_teleportation_va_superdense.ipynb` | No-cloning, teleportation, superdense coding, mid-circuit measurement |

Then [`02_compilation/`](02_compilation/README.md) — the transpiler treated as an optimisation
problem, ending in a `gymnasium` environment for reinforcement-learning research on compilation.

| File | Topic |
|---|---|
| `02_compilation/00_transpiler_pipeline.ipynb` | Hardware constraints, the six pass-manager stages, `DAGCircuit`, front layer, cost functions |
| `02_compilation/01_layout.ipynb` | Coupling maps, interaction graphs, layout passes, `TranspileLayout`, noise-aware layout |
| `02_compilation/02_routing.ipynb` | The routing loop, a router written from scratch, SABRE's heuristics, seed and topology sensitivity |
| `02_compilation/03_synthesis_va_tcount.ipynb` | Euler angles, KAK and the 0/1/2/3-CX classes, Clifford+T, Solovay-Kitaev vs gridsynth, T-count |
| `02_compilation/04_custom_pass_va_benchmark.ipynb` | Writing passes, replacing a stage, building a benchmark harness |
| `02_compilation/05_compilation_as_mdp.ipynb` | Routing as an MDP, a real `gymnasium.Env`, action masking, reward design, baselines |

That folder also ships two reusable modules: `bench.py` (benchmark harness) and `routing_env.py`
(the routing environment, baseline policies and correctness checks).

The notebooks are wired to the shared kernel `Python 3.12 (Shared Venv)` (`~/.venv`) and are
meant to be opened in VS Code. `02_compilation/` additionally needs `gymnasium`; `torch` (CPU
build) is installed there in preparation for the RL agents.

## Course report

[`report/`](report/) holds the LaTeX report for the course "Chuyên đề nghiên cứu 1" built from these
notebooks (in Vietnamese, same layout as the `cv-report` template). The plan and outline are in
[`report/PLAN.md`](report/PLAN.md). Compile with LuaLaTeX:

```bash
cd report && latexmk -pdflua -interaction=nonstopmode -outdir=build report.tex
```

## Setup

```bash
uv sync                    # create .venv and install the locked dependencies
uv sync --group notebook   # add JupyterLab when needed
```

## Running

Default: open `00_fundamental/*.ipynb` in VS Code and pick the `Python 3.12 (Shared Venv)`
kernel. To use this project's own environment instead, sync the notebook group and start
JupyterLab, then switch the notebook to that kernel:

```bash
uv run jupyter lab          # after: uv sync --group notebook
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

Many tutorials online still use the removed Qiskit 0.x API (`execute`, `Aer.get_backend`, ...).
The mapping to the current API, plus the mistakes it usually causes, is in
[`00_fundamental/03_qiskit_workflow.ipynb`](00_fundamental/03_qiskit_workflow.ipynb).
