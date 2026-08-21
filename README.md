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

Benchmark scripts:

- `Benchmark_01_two_level.py`: two-level GPU, fixed-step Python CPU, adaptive SciPy CPU, QuTiP CPU, and Julia GPU comparison.
- `Benchmark_02_four_level_Interferometry.py`: four-level GPU, fixed-step Python CPU, adaptive SciPy CPU, QuTiP CPU, and Julia GPU comparison.

Both scripts use the same benchmark modes:

| Mode | Solver selection | Result |
| --- | --- | --- |
| `single` | `--solver` | Runs and optionally plots one backend. |
| `diff` | `--solver`, `--solver-b` | Runs two backends, prints both timings plus MSE, RMS, and maximum absolute differences, and optionally plots both maps and their difference. |
| `all` | none | Attempts every backend and skips unavailable optional backends with a console message. It displays maps but does not calculate pairwise errors. |
| `full_benchmark` | `--full-solvers` | Measures timing versus square-grid size, terminates points that exceed the time limit, extrapolates larger points, and saves CSV/PNG results. It does not compare numerical output maps. |

`full` and `full-benchmark` are aliases for `full_benchmark`. A solver name may
also be supplied as the positional argument, for example
`python Benchmark_01_two_level.py qutip_cpu`; this is shorthand for `single`
mode. Command-line options override the values in `user_settings()` near the
bottom of each benchmark. Run either script with `--help` for all options.

Run a single GPU benchmark:

```bash
python Benchmark_01_two_level.py --mode single --solver gpu --nx 512 --ny 512 --no-plot --timings
python Benchmark_02_four_level_Interferometry.py --mode single --solver gpu --nx 256 --ny 256 --no-plot --timings
```

Use `diff` mode to run any two available backends on the same parameter grid.
For numerical validation, the recommended comparison is GQIS against adaptive
QuTiP with divider `1` so both backends receive the full requested time grid:

```bash
python Benchmark_01_two_level.py --mode diff --solver gpu --solver-b qutip_cpu --detuning-points 16 --amplitude-points 16 --qutip-output-density-divider 1 --timings
python Benchmark_02_four_level_Interferometry.py --mode diff --solver gpu --solver-b qutip_cpu --nx 16 --ny 16 --qutip-cpu-num-t-divider 1 --timings
```

Each command displays both interferograms and their difference and prints the
mean-square deviation (`MSE`), root-mean-square deviation (`RMS`), maximum
absolute difference, and solver timings. Increase
`--solver-steps-per-period` until the GPU-versus-QuTiP error is converged; the GQIS
backend uses fixed-step RK4, while QuTiP chooses adaptive internal steps.

Both native Python CPU backends remain available. Substitute `python_cpu` for a
same-grid fixed-step RK4 comparison, or `python_ode_cpu` for the adaptive SciPy
`solve_ivp` comparison. Thus `--solver` and `--solver-b` may be any two of
`gpu`, `python_cpu`, `python_ode_cpu`, `qutip_cpu`, and `julia_gpu`.

Run a full timing sweep and save CSV/PNG output:

```bash
python Benchmark_01_two_level.py --mode full_benchmark --bench-max-side-size 8192 --bench-solver-time-limit 300 --no-plot
python Benchmark_02_four_level_Interferometry.py --mode full_benchmark --bench-max-side-size 8192 --bench-solver-time-limit 300 --no-plot
```

To include the adaptive SciPy backend, add `python_ode_cpu` to `--full-solvers`:

```bash
python Benchmark_01_two_level.py --mode full_benchmark --full-solvers gpu,python_ode_cpu,qutip_cpu,julia_gpu --bench-max-side-size 8192 --no-plot
python Benchmark_02_four_level_Interferometry.py --mode full_benchmark --full-solvers gpu,python_ode_cpu,qutip_cpu,julia_gpu --bench-max-side-size 8192 --no-plot
```

Solver inclusion is user-selectable. Use `--solver` in `single` mode,
`--solver` and `--solver-b` in `diff` mode, or `--full-solvers` in
`full_benchmark` mode. Mode `all` runs every available backend. The default full
performance sweep includes `gpu`, `qutip_cpu`, and `julia_gpu`; QuTiP is the
primary CPU performance and numerical reference, while `python_cpu` is retained
as a transparent fixed-step implementation reference.

Full benchmark mode measures powers-of-two square grids. If a solver exceeds
`--bench-solver-time-limit`, that process is terminated before the next measurement.
To avoid launching a point that is very likely to time out, the benchmark starts
extrapolating when the latest measured time exceeds half the limit and the last
two measured points have a time ratio greater than `0.9 * 4 = 3.6`. There is no
upper bound on this ratio.
Larger points are extrapolated from the last measured point using the log-log slope
between the last two valid measurements, with `log10(time)` versus
`log10(number of simulations)`. In generated plots, measured points use circles
and extrapolated points use squares with the same color.

In `full_benchmark` mode, `--no-plot` suppresses the interactive Matplotlib
window but still saves the PNG figure and CSV table.

The generated CSV files include CPU, GPU, VRAM, OS, Python, GQIS and numerical-package versions, CUDA runtime, and GPU first-RHS/codegen timing metadata.

## Current Benchmark Results

Reference results are generated files rather than manually duplicated README
tables. The CSV files are authoritative: they include machine metadata,
measured/extrapolated status, total time, and separate preparation/calculation
times when a backend reports them. Regenerate them after solver changes before
citing performance.

### Two-Level Reference

[Timing data (CSV)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_01_full_benchmark.csv) | [Figure file (PNG)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_01_full_benchmark.png)

![Two-level full benchmark](./Benchmark_01_full_benchmark.png)

### Four-Level Reference

[Timing data (CSV)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_02_full_benchmark.csv) | [Figure file (PNG)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_02_full_benchmark.png)

![Four-level full benchmark](./Benchmark_02_full_benchmark.png)

## Solver Fairness Notes

- `gpu` uses the GQIS CUDA backend with a fixed-step RK4 grid.
- `python_cpu` is also a simple fixed-step RK4 reference backend implemented in the benchmark script. It is not Python's default ODE solver and it is not SciPy `solve_ivp`.
- `python_ode_cpu` uses SciPy `solve_ivp` with adaptive RK45 integration. It is the plain Python adaptive ODE reference backend.
- `qutip_cpu` uses QuTiP `mesolve`, which is adaptive internally, but the requested output/coefficient time list still comes from the benchmark settings.
- The fixed-step `python_cpu` divider defaults to `1`, giving it the same RK4 integration-step density as the GPU solver. Its independent divider can be increased only for a deliberately coarser fixed-step comparison.
- The adaptive `python_ode_cpu` and `qutip_cpu` dividers default to `10`. They reduce the requested output/coefficient time grid, while the solvers choose internal adaptive steps. Use divider `1` when validating all backends on the same requested time grid.
- A time list with `M` samples defines `M - 1` integration intervals. GQIS and the fixed-step Python RK4 backend average post-step observable samples and exclude the initial state at `t=0`.
- The Julia benchmark path receives the exact trace- and Hermiticity-reduced density-matrix RHS produced by the same `build_reduced_lindblad_rhs` function used by GQIS. Julia adds one accumulator equation to integrate the observable continuously, whereas GQIS forms a post-step sample average; this output calculation can differ slightly on a coarse grid even though the physical density-matrix ODE is identical. The generated scalar Julia RHS uses Float32 literals and global common-subexpression elimination. Julia `prep` is Python/SymPy equation generation, while Julia `calc` is the synchronized `solve` interval and includes first-solve Julia/GPU compilation. Run with `--timings` to also display the complete Julia subprocess duration.

The generated CSV records hardware and software versions in its metadata
header. Each result row records the grid side, number of simulations, solver,
total time, available preparation and calculation times, and whether the point
was measured, extrapolated, or failed. The first GPU RHS/code-generation time
is also stored in the metadata when available.

The scripts currently print, but do not save in the CSV, the simulated duration,
solver steps per period, CPU divider values, and numerical precision. Retain
these settings with the generated files when reporting publication-quality
comparisons.

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
