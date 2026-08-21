# GPU Quantum Interferometry Solver (GQIS)

The GPU Quantum Interferometry Solver (`GQIS`) is a Python/CUDA research
package for fast two-dimensional parameter sweeps of driven open quantum
systems. It converts a symbolic Lindblad master-equation model written with
SymPy into CUDA code, compiles it with CuPy/NVRTC, and runs one independent
parameter point per GPU thread.

The main target is quantum interferometry: dense grids of low-dimensional open-system simulations where a CPU loop over parameter points becomes the bottleneck. The solver is not hard-coded for a two-level system. You provide the Hamiltonian matrix, collapse operators, observable, drive expression, and optional initial density matrix. In principle this can represent any finite-dimensional Lindblad model that fits in GPU memory and has equations small enough for CUDA compilation.

> **Status:** GQIS 0.1.0 is alpha stage research software. The CUDA backend uses
> fixed-step fourth-order Runge-Kutta (RK4) integration on the user-supplied
> time grid. Verify time-step convergence by repeating calculations with
> progressively smaller steps. For important results, compare against a trusted
> reference solver because RK4 is not suitable for every problem, and its
> accuracy and stability depend on the time-step size.

## Why This Tool Exists

Many quantum dynamics tools are excellent for one system, one parameter set, or a moderate number of CPU-parallel jobs. Interferometry often needs millions of independent simulations on a rectangular grid. In that regime, moving only a single trajectory to the GPU is not enough: the parameter sweep itself must live inside the GPU kernel.

`GQIS` is designed for that case. It compiles the model once, sends the parameter grid to the GPU, integrates every grid point in parallel, accumulates the requested observable inside the kernel, and transfers only the final 2D result back to CPU memory for plotting.

## Solver Pipeline

A call with a new symbolic model follows this pipeline:

<table>
  <tr align="center">
    <td><strong>1.</strong> Define <em>H</em>, drives,<br>collapse and observable</td>
    <td>&rarr;</td>
    <td><strong>2.</strong> Build Lindblad<br>master equation</td>
    <td>&rarr;</td>
    <td><strong>3.</strong> Reduce density-matrix<br>equations</td>
    <td>&rarr;</td>
    <td><strong>4.</strong> Simplify and optimize<br>the symbolic RHS</td>
  </tr>
  <tr align="center">
    <td colspan="6"></td>
    <td>&darr;</td>
  </tr>
  <tr align="center">
    <td><strong>8.</strong> Launch the 2D sweep:<br>one thread per point</td>
    <td>&larr;</td>
    <td><strong>7.</strong> Compile and cache<br>with CuPy/NVRTC</td>
    <td>&larr;</td>
    <td><strong>6.</strong> Insert code into<br>the kernel template</td>
    <td>&larr;</td>
    <td><strong>5.</strong> Generate CUDA C<br>for RHS and observable</td>
  </tr>
  <tr align="center">
    <td>&darr;</td>
    <td colspan="6"></td>
  </tr>
  <tr align="center">
    <td><strong>9.</strong> Integrate RK4 and<br>accumulate the observable</td>
    <td>&rarr;</td>
    <td><strong>10.</strong> Return the 2D<br>NumPy result</td>
    <td colspan="4"></td>
  </tr>
</table>

Distinct implementation choices:

- The Hamiltonian and collapse operators are ordinary SymPy matrices, so the physics model remains readable.
- The Lindblad RHS is generated symbolically, then converted into CUDA code instead of being interpreted inside Python.
- Only the independent density-matrix elements are evolved, reducing unnecessary work.
- The CUDA kernel computes the observable during integration, so every time step does not need to be stored in GPU memory.
- When the symbolic model is unchanged, GQIS reuses the generated RHS and compiled kernel; numeric sweep arrays, explicit initial states, and selected runtime constants can change between calls without recompilation, which supports efficient animations.

## Benefits For Massive Parameter Sweeps

- `GQIS` avoids a Python-level loop over millions of parameter points.
- One compiled kernel can evaluate a full 2D interferogram where each thread solves one independent low-dimensional system.
- Dense GPU sweeps are useful for resolving thin resonances, where low-resolution CPU scans can miss structure.
- The workflow stays in Python/SymPy/CuPy while still producing compiled CUDA kernels.
- QuTiP remains the recommended CPU reference for validation, but GQIS is intended for the high-throughput sweep after the model is validated.

This tool is specialized for structured, independent parameter sweeps where a fixed time grid is acceptable after manual convergence confirmation.

## Supported Models And Outputs

The solver is not limited to the included two- and four-level examples. A user supplies:

- any finite-dimensional SymPy Hamiltonian `H`, including one or more symbolic drive placeholders
- a SymPy expression, or a dictionary of expressions, defining the time-dependent drives
- any list of Lindblad collapse operators with dimensions matching `H`
- an output operator whose expectation value is accumulated or sampled
- up to two parameter-sweep axes
- an optional symbolic initial density matrix, symbolic initial-state sweep, or explicit array of initial density matrices

`mesolve_2D` constructs the Lindblad equation, reduces the Hermitian trace-one density matrix to `N*N - 1` independent real variables, generates CUDA expressions for the RHS and observable, inserts them into the packaged CUDA template, compiles with NVRTC, and launches one independent trajectory per GPU thread. The practical system dimension is limited by generated-code size, register pressure, compilation resources, and GPU memory rather than by a hard-coded two-level model.

Output modes include a time-averaged observable, a final observable, the final density matrix, and an optional sampled observable trace. See the [GQIS API reference](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/GQIS_API.md) for every argument and helper function.

GQIS does not interpret physical units or basis labels. Define all quantities,
states, and operators in each model using compatible units and a consistent basis.

## Installation

GQIS requires Python 3.10 or newer, NumPy, SymPy, an NVIDIA CUDA-capable
GPU, and one CuPy distribution matching the CUDA major version. Install the
tested CUDA 12 configuration and plotting examples with:

```bash
pip install "gqis[cuda12,examples]"
```

```python
from gqis import mesolve_2D
```

Use the `cuda11` or `cuda13` extra instead when appropriate. Do not install
multiple CuPy distributions in one environment. CUDA 11 and CUDA 13 extras are
provided but were not tested on the CUDA 12 reference workstation for version
0.1.0. Install optional benchmark dependencies with:

```bash
pip install "gqis[cuda12,examples,benchmarks]"
```

Before the PyPI upload, or to install an exact source revision, use the tagged
GitHub release:

```bash
pip install "gqis[cuda12,examples] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@v0.1.0"
```

From a local clone, install normally or in editable development mode:

```bash
pip install ".[cuda12,examples,benchmarks]"
pip install -e ".[all-cuda12]"
```

Verify the installed package with a small GPU solve:

```bash
gqis-check --installation-test
```

See the [installation and GPU test guide](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/INSTALLATION_TEST.md)
for isolated Conda and `venv` setup, CUDA selection, optional tools, hardware
reporting, and updates. `requirements.txt` is a CUDA 12-oriented environment
recipe; `pyproject.toml` is the package dependency source of truth.

Declared requirements use minimum compatible versions so pip does not reject an
older version unnecessarily. The exact versions below are the version 0.1.0
local test environment; versions outside this tested set may work, but should be
validated with `gqis-check --installation-test` and benchmark `diff` mode.

| Dependency | Declared requirement | Locally tested version |
| --- | --- | --- |
| Python | `>=3.10` | 3.11.7 |
| NumPy | `>=1.24` | 2.4.6 |
| SymPy | `>=1.11` | 1.14.0 |
| CuPy/CUDA | CUDA-specific extra | `cupy-cuda12x` 14.1.1, CUDA runtime 12.9 |
| Matplotlib | `>=3.7` (examples) | 3.11.1 |
| SciPy | `>=1.10` (benchmarks) | 1.16.1 |
| QuTiP | `>=5.0` (benchmarks) | 5.2.0 |
| pytest | `>=8` (tests) | 8.4.2 |
| build | `>=1` (release builds) | 1.5.0 |
| Ruff | `==0.16.2` (development) | 0.16.2 |
| Julia | external optional backend | 1.10.2 |

## Tests

Examples and benchmarks exercise realistic workflows, but they are not a
replacement for automated tests: they are comparatively slow, depend on local
GPU/plotting software, and generally do not assert known numerical answers.

Run the fast package and time-grid tests, without requiring a working GPU:

```bash
pytest -m "not gpu"
```

Run the complete suite, including the CUDA numerical test:

```bash
pytest
```

The GitHub Actions workflow currently tests Python 3.11, builds the source
distribution and wheel, and runs the non-GPU tests. Python 3.10 remains the
declared minimum but is outside the current release test matrix. Numerical
convergence against QuTiP should still be checked before publishing scientific
results or changing the CUDA integration core.

## Quick Start

Installed code should import the packaged interface:

```python
from gqis import mesolve_2D
```

| Script | Demonstration |
| --- | --- |
| `Example_01_two_level_basic.py` | Basic two-level interferogram. |
| `Example_02_four_level_interferogram.py` | Coupled qubit-resonator interferogram. |
| `Example_03_two_level_animation.py` | Two-level parameter animation. |
| `Example_04_four_level_animation.py` | Four-level parameter animation. |
| `Example_05_initial_condition_sweep_gate_fidelity.py` | Initial-state sweep and gate-fidelity comparison. |

Run a tutorial from the repository root, for example:

```bash
python Example_01_two_level_basic.py
```

Tutorial examples use visible GPU-sized grids by default. If your GPU is smaller
or you want a quick test run, reduce the grid in the `user_settings()` block
near the bottom of each script. Examples 01-04 use `simulation_periods` and
`solver_steps_per_period`; Example 05 uses `solver_steps` because each selected
gate has its own duration. A time grid with `N` integration intervals contains
`N + 1` samples, including both `t=0` and the requested final time, and the
current GQIS CUDA backend therefore executes `len(tlist) - 1` fixed-step RK4
updates. `averaging_skip_fraction` is the initial fraction of simulated time
excluded only from a time-averaged output.

In animation examples, `forward_frame_count` sets the number of calculated
forward frames, while `animated_parameter_values` is the corresponding array of
physical parameter values. `pingpong` playback appends those values in reverse
order; each displayed or saved frame currently runs its GPU calculation.

## Benchmarks

Benchmark models:

- `Benchmark_01_two_level.py`: driven qubit.
- `Benchmark_02_four_level_Interferometry.py`: coupled qubit-resonator system.

Available solvers for either benchmark:

- `gpu`: GQIS fixed-step RK4 on CUDA.
- `python_cpu`: fixed-step Python RK4.
- `python_ode_cpu`: adaptive SciPy RK45 on CPU.
- `qutip_cpu`: adaptive QuTiP solver on CPU.
- `julia_gpu`: Julia DifferentialEquations/DiffEqGPU solver.

Both scripts use the same benchmark modes:

| Mode | Solver selection | Result |
| --- | --- | --- |
| `single` | `--solver` | Runs and optionally plots one backend. |
| `diff` | `--solver`, `--solver-b` | Runs two backends, prints both timings plus MSE, RMS, and maximum absolute differences, and optionally plots both maps and their difference. |
| `all` | none | Attempts every backend and skips unavailable optional backends with a console message. It displays maps but does not calculate pairwise errors. |
| `full_benchmark` | `--full-solvers` | Measures timing versus square-grid size, terminates points that exceed the time limit, extrapolates larger points, and saves CSV/PNG results. It does not compare numerical output maps. |

Run either benchmark with its default settings:

```bash
python Benchmark_01_two_level.py
python Benchmark_02_four_level_Interferometry.py
```

Run one GQIS-versus-QuTiP comparison:

```bash
python Benchmark_02_four_level_Interferometry.py --mode diff --solver gpu --solver-b qutip_cpu
```

Full benchmark mode measures powers-of-two square grids. If a solver exceeds
`--bench-solver-time-limit`, that process is terminated before the next measurement.
Larger grids are extrapolated from recent measured scaling. In generated plots,
measured points use circles and extrapolated points use squares with the same color.

The generated CSV files include the hardware, software, physical-model,
time-grid, precision, CPU-divider, sweep-limit, and timing metadata needed to
reproduce the benchmark configuration.

## Benchmark Results

On the reference RTX 3080, GQIS approaches linear scaling with the number of
simulations once fixed launch overhead no longer dominates. The largest
measured `32768 x 32768` runs evaluate `1.07 x 10^9` parameter sets, with
10,240 RK4 steps per trajectory, in about 1 minute 44 seconds for the 2-level
model and 7 minutes 1 second for the 4-level model. Direct QuTiP calculations
at this resolution are estimated to require about 69 days (2 months and 9 days)
and 108 days (3 months
and 18 days), respectively. Those large-grid CPU times are therefore
extrapolated from measured smaller grids.

Across the approximately linear large-grid region from `4096 x 4096` through
`32768 x 32768`, the average point-by-point speedups are about 69,000 times over
QuTiP and 22 times over Julia for the 2-level model, and 24,000 times over QuTiP
and 38 times over Julia for the 4-level model. These averages include
extrapolated values after a reference solver exceeds its practical time limit.
Reaching resolutions that are impractical with a CPU reference is the primary
motivation for GQIS.

### Two-Level Reference

[Timing data (CSV)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_01_full_benchmark.csv) | [Figure file (PNG)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_01_full_benchmark.png)

![Two-level full benchmark](./Benchmark_01_full_benchmark.png)

### Four-Level Reference

[Timing data (CSV)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_02_full_benchmark.csv) | [Figure file (PNG)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_02_full_benchmark.png)

![Four-level full benchmark](./Benchmark_02_full_benchmark.png)

Running `full_benchmark` automatically saves both the CSV timing table and PNG
figure, including when `--no-plot` suppresses the interactive window.

## Solver Fairness Notes

- `gpu` uses the GQIS CUDA backend with a fixed-step RK4 grid.
- `python_cpu` is also a simple fixed-step RK4 reference backend implemented in the benchmark script. It is not Python's default ODE solver and it is not SciPy `solve_ivp`.
- `python_ode_cpu` uses SciPy `solve_ivp` with adaptive RK45 integration. It is the plain Python adaptive ODE reference backend.
- `qutip_cpu` uses QuTiP `mesolve`, which is adaptive internally, but the requested output/coefficient time list still comes from the benchmark settings.
- The fixed-step `python_cpu` divider defaults to `1`, matching the GQIS RK4 step density, while the adaptive `python_ode_cpu` and `qutip_cpu` dividers default to `10` and reduce their requested output/coefficient grid. Use divider `1` for adaptive backends when validating all solvers on the same requested time grid; increase the fixed-step divider only for an intentionally coarser RK4 comparison.
- A time list with `M` samples defines `M - 1` integration intervals.
- `julia_gpu` solves the same trace- and Hermiticity-reduced density-matrix ODE system as GQIS. Its benchmark time is the synchronized Julia solve stage; symbolic and code-generation preparation is excluded from that scaling time and reported separately when available.

Each CSV result row records the grid side, number of simulations, solver,
scaling time, available preparation and calculation times, and whether the
point was measured, extrapolated, or failed. The first GQIS RHS/code-generation
time is stored in the metadata when available.

## Project Layout

- `gqis/` contains the packaged solver, public interface, environment checker,
  and CUDA kernel template.
- `Example_*.py` contains runnable tutorials; `Benchmark_*.py` contains accuracy
  and scaling comparisons.
- `GQIS_API.md` and `INSTALLATION_TEST.md` provide detailed API and
  setup guidance.
- `tests/` and `.github/workflows/ci.yml` contain automated checks.

## Contributing

Bug reports, validation results from other GPUs, documentation corrections, and
focused code contributions are welcome. See [CONTRIBUTING.md](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/CONTRIBUTING.md) before opening an issue or pull request.

## Citation

If you use this software in scientific work, cite the repository metadata from [CITATION.cff](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/CITATION.cff). After a paper or Zenodo archive is available, add the DOI to `CITATION.cff` and cite the archived release for reproducibility.

## License

This project is released under the [MIT License](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/LICENSE).
