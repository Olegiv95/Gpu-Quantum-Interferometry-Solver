# GPU Quantum Interferometry Solver (GQIS)

The GPU Quantum Interferometry Solver (`GQIS`) is a Python/CUDA research
package for fast two-dimensional parameter sweeps of driven open quantum
systems. It converts a symbolic Lindblad master-equation model written with
SymPy into CUDA code, compiles it with CuPy/NVRTC, and runs one independent
parameter point per GPU thread.

The main target is quantum interferometry: dense grids of low-dimensional open-system simulations where a CPU loop over parameter points becomes the bottleneck. The solver is not hard-coded for a two-level system. You provide the Hamiltonian matrix, collapse operators, observable, drive expression, and optional initial density matrix. In principle this can represent any finite-dimensional Lindblad model that fits in GPU memory and has equations small enough for CUDA compilation.

## Why This Tool Exists

Many quantum dynamics tools are excellent for one system, one parameter set, or a moderate number of CPU-parallel jobs. Interferometry often needs millions of independent low-dimensional simulations on a rectangular grid. In that regime, moving only a single trajectory to the GPU is not enough: the parameter sweep itself must live inside the GPU kernel.

`GQIS` is designed for that case. It compiles the model once, sends the parameter grid to the GPU, integrates every grid point in parallel, accumulates the requested observable inside the kernel, and transfers only the final 2D result back to CPU memory for plotting.

## Solver Pipeline

This is the principal solver scheme used by the examples and benchmarks:

```mermaid
flowchart TD
    A[User defines H, drive, collapse operators, and an expectation-value operator in SymPy]
    B[Build Lindblad master equation]
    C[Reduce to independent density-matrix equations]
    D[Simplify, substitute constants, and optimize symbolic RHS]
    E[Generate CUDA C code for RHS and observable]
    F[Insert generated code into CUDA kernel template]
    G[Compile with CuPy/NVRTC and cache the kernel]
    H[Launch 2D parameter sweep: one GPU thread per grid point]
    I[Integrate fixed-step RK4 and accumulate the expectation value inside the kernel]
    J[Transfer final 2D map to NumPy and plot with Matplotlib]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J
```

Distinct implementation choices:

- The Hamiltonian and collapse operators are ordinary SymPy matrices, so the physics model remains readable.
- The Lindblad RHS is generated symbolically, then converted into CUDA code instead of being interpreted inside Python.
- Only independent density-matrix variables are evolved, reducing unnecessary work.
- The CUDA kernel computes the observable during integration, so every time step does not need to be stored in GPU memory.
- Runtime constants can be changed for animations without regenerating the full symbolic RHS when the equation structure is unchanged.
- Symbols that remain part of the equation structure participate in the RHS cache key; numeric sweep arrays and explicit initial-state arrays can change without recompiling that structure.

## Benefits For Massive Parameter Sweeps

- `GQIS` avoids a Python-level loop over millions of parameter points.
- One compiled kernel can evaluate a full 2D interferogram where each thread solves one independent low-dimensional system.
- Dense GPU sweeps are useful for thin resonances, where low-resolution CPU scans can miss structure.
- The workflow stays in Python/SymPy/CuPy while still producing compiled CUDA kernels.
- QuTiP remains the recommended CPU reference for validation, but GQIS is intended for the high-throughput sweep after the model and resolution are validated.

This tool is not a replacement for general-purpose quantum solvers. It is specialized for structured, independent parameter sweeps where a fixed time grid is acceptable after convergence checks.

## Supported Models And Outputs

The solver is not limited to the included two- and four-level examples. A user supplies:

- any finite-dimensional SymPy Hamiltonian `H`, including one or more symbolic drive placeholders
- a SymPy expression, or a dictionary of expressions, defining the time-dependent drives
- any list of Lindblad collapse operators with dimensions matching `H`
- an output operator whose expectation value is accumulated or sampled
- up to two parameter-sweep axes
- an optional symbolic initial density matrix, symbolic initial-state sweep, or explicit array of initial reduced density matrices

`mesolve_2D` constructs the Lindblad equation, reduces the Hermitian trace-one density matrix to `N*N - 1` independent real variables, generates CUDA expressions for the RHS and observable, inserts them into the packaged CUDA template, compiles with NVRTC, and launches one independent trajectory per GPU thread. The practical system dimension is limited by generated-code size, register pressure, compilation resources, and GPU memory rather than by a hard-coded two-level model.

Output modes include a time-averaged observable, a final observable, the final reduced density matrix, and an optional sampled observable trace. See the [GQIS API reference](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/GQIS_API.md) for every argument and helper function.

## Units And Basis Conventions

The solver is unit-agnostic. The included examples use dimensionless model
parameters normalized by a selected energy-gap scale. All Hamiltonian
coefficients, drive frequencies, and dissipative rates in one model must use
the same frequency convention. A physical model may use either ordinary
frequency or angular frequency after consistent normalization; mixing the two
introduces an unwanted factor of `2*pi`. The corresponding time variable must
use the reciprocal convention so that every phase argument is dimensionless.

Labels such as `|0>` and `|1>` identify states of the basis used to construct
the Hamiltonian. They are not automatically energy-ground and energy-excited
states. The Hamiltonian determines its instantaneous energy eigenstates, while
collapse operators specify the dissipative transition directions represented
in the model.

## Installation

Core requirements are Python 3.10 or newer, NumPy, SymPy, an NVIDIA CUDA-capable GPU, and exactly one CuPy distribution matching the CUDA major version. Pip cannot detect the CUDA major version and choose a CuPy wheel interactively, so CUDA support is provided through explicit extras.

After the public package is uploaded to PyPI, install the tested CUDA 12
configuration and plotting examples with:

```bash
pip install "gqis[cuda12,examples]"
```

The PyPI distribution and Python import package are both named `gqis`:

```python
from gqis import mesolve_2D
```

Use `cuda11` with CuPy 13 for CUDA 11, or `cuda13` for CUDA 13. Do not install multiple `cupy`, `cupy-cuda11x`, `cupy-cuda12x`, or `cupy-cuda13x` distributions in the same environment. CUDA 11 and CUDA 13 package extras are provided but were not tested on the CUDA 12 reference workstation for version 0.1.0.

Install benchmark dependencies:

```bash
pip install "gqis[cuda12,examples,benchmarks]"
```

Before the PyPI upload, or when an exact source revision is required, install
the tagged public GitHub release directly:

```bash
pip install "gqis[cuda12,examples] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@v0.1.0"
```

From a local repository clone, use `.` as the package source:

```bash
pip install ".[cuda12,examples]"
pip install ".[cuda12,examples,benchmarks]"
pip install -e ".[cuda12,examples]"
```

The first two local commands create standalone installations. Editable mode is
intended only for package development because it remains linked to the source
folder. Install all CUDA 12 development, benchmark, and test dependencies with:

```bash
pip install -e ".[all-cuda12]"
```

See the [installation and GPU smoke-test guide](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/INSTALLATION_AND_SMOKE_TEST.md) for clean Conda and `venv` instructions, CUDA selection, hardware reporting, update commands, and environment deactivation.

Optional external programs are FFmpeg for MP4 export and Julia with `DifferentialEquations`, `DiffEqGPU`, `CUDA`, and `StaticArrays` for Julia GPU comparisons. `requirements.txt` provides a CUDA 12-oriented non-package installation list.

Declared requirements use minimum compatible versions so pip does not reject an
older version unnecessarily. The exact versions below are the version 0.1.0
local test environment; versions outside this tested set may work, but should be
validated with `gqis-check --smoke` and the benchmark `diff` mode.

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

FFmpeg is an optional external executable for MP4 output. The Julia comparison
also requires `DifferentialEquations`, `DiffEqGPU`, `CUDA`, and `StaticArrays`.

CuPy cannot be one unconditional dependency because its binary package name is
CUDA-major-specific. Pip does not prompt for or reliably detect that choice
while resolving package metadata; selecting the matching `cuda11`, `cuda12`, or
`cuda13` extra is explicit and avoids installing conflicting CuPy wheels.

On Windows, installing FFmpeg is not sufficient if its `bin` directory is not
on `PATH`. Confirm the same terminal used to run Python can execute
`where ffmpeg` and `ffmpeg -version`; Matplotlib's `FFMpegWriter` uses that PATH
lookup unless `matplotlib.rcParams["animation.ffmpeg_path"]` is set explicitly.

Check the environment:

```bash
python check_environment.py
```

After package installation, the equivalent command is:

```bash
gqis-check
```

Run a tiny GPU smoke test:

```bash
python check_environment.py --smoke
```

The smoke solve runs in a child process and is terminated after 120 seconds by
default. Change the bound with `--smoke-timeout SECONDS`.

To check Julia benchmark packages as well:

```bash
python check_environment.py --check-julia-packages
```

## Tests

Examples and benchmarks exercise realistic workflows, but they are not a
replacement for automated tests: they are comparatively slow, depend on local
GPU/plotting software, and generally do not assert known numerical answers.

Run the fast package and time-grid tests, without requiring a working GPU:

```bash
pytest -m "not gpu"
```

Run the optional CUDA numerical smoke test as well:

```bash
pytest
```

The GitHub Actions workflow currently tests Python 3.11, builds the source
distribution and wheel, and runs the non-GPU tests. Python 3.10 remains the
declared minimum but is outside the current release test matrix. Numerical
convergence against QuTiP should still be checked before publishing scientific
results or changing the CUDA integration core.

## Tested Local Environment

The reference benchmark files were produced on an RTX 3080 workstation. The environment checker reports the exact machine metadata and the full benchmark CSV files store it at the top of the file.

Version 0.1.0 local test environment:

```text
Python: 3.11.7
OS: Windows 11 Home (25H2, build 26200.9168)
CPU: 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz
GPU: NVIDIA GeForce RTX 3080, compute capability 8.6, memory 10.00 GB
NumPy: 2.4.6
SymPy: 1.14.0
Matplotlib: 3.11.1
CuPy: 14.1.1 (cupy-cuda12x; CUDA runtime 12.9)
SciPy: 1.16.1
QuTiP: 5.2.0
Julia: 1.10.2
```

## Quick Start

Installed code should import the packaged interface:

```python
from gqis import mesolve_2D
```

The former `from gpu_int_tool import mesolve_2D` and
`from GPU_Int_Tool import mesolve_2D` forms remain available as compatibility
imports. New code should use `gqis`.

Run the two-level tutorial:

```bash
python Example_01_two_level_basic.py
```

Run the four-level interferogram tutorial:

```bash
python Example_02_four_level_interferogram.py
```

Run the two-level animation:

```bash
python Example_03_two_level_animation.py
```

Run the four-level animation:

```bash
python Example_04_four_level_animation.py
```

Run the initial-state sweep and gate-fidelity tutorial:

```bash
python Example_05_initial_condition_sweep_gate_fidelity.py
```

Tutorial examples use visible GPU-sized grids by default. If your GPU is smaller
or you want a first smoke run, reduce the grid in the `user_settings()` block
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

![Two-level full benchmark](https://raw.githubusercontent.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/main/Benchmark_01_full_benchmark.png)

### Four-Level Reference

[Timing data (CSV)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_02_full_benchmark.csv) | [Figure file (PNG)](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Benchmark_02_full_benchmark.png)

![Four-level full benchmark](https://raw.githubusercontent.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/main/Benchmark_02_full_benchmark.png)

## Solver Fairness Notes

- `gpu` uses the GQIS CUDA backend with a fixed-step RK4 grid.
- `python_cpu` is also a simple fixed-step RK4 reference backend implemented in the benchmark script. It is not Python's default ODE solver and it is not SciPy `solve_ivp`.
- `python_ode_cpu` uses SciPy `solve_ivp` with adaptive RK45 integration. It is the plain Python adaptive ODE reference backend.
- `qutip_cpu` uses QuTiP `mesolve`, which is adaptive internally, but the requested output/coefficient time list still comes from the benchmark settings.
- The fixed-step `python_cpu` divider defaults to `1`, giving it the same RK4 integration-step density as the GPU solver. Its independent divider can be increased only for a deliberately coarser fixed-step comparison.
- The adaptive `python_ode_cpu` and `qutip_cpu` dividers default to `10`. They reduce the requested output/coefficient time grid, while the solvers choose internal adaptive steps. Use divider `1` when validating all backends on the same requested time grid.
- A time list with `M` samples defines `M - 1` integration intervals. GQIS and the fixed-step Python RK4 backend average post-step observable samples and exclude the initial state at `t=0`.
- The Julia benchmark path receives the exact trace- and Hermiticity-reduced density-matrix RHS produced by the same `build_reduced_lindblad_rhs` function used by GQIS. Julia adds one accumulator equation to integrate the observable continuously, whereas GQIS forms a post-step sample average; this output calculation can differ slightly on a coarse grid even though the physical density-matrix ODE is identical. The generated scalar Julia RHS uses Float32 literals and global common-subexpression elimination. Julia `prep` is Python/SymPy equation generation, while Julia `calc` is the synchronized `solve` interval and includes first-solve Julia/GPU compilation. Run with `--timings` to also display the complete Julia subprocess duration.

For publication-quality comparisons, report:

- hardware and software versions
- grid size and number of simulations
- simulated duration in drive periods and solver steps per period
- CPU divider values
- precision
- preparation/RHS/codegen time
- kernel or solver calculation time
- whether each data point is measured or extrapolated

## Repository Files

- `gqis/solver.py`: packaged symbolic-to-CUDA solver implementation.
- `gqis/N_Level_Kernel.cu`: packaged CUDA kernel template.
- `gqis/__init__.py`: public package interface.
- `gpu_int_tool/` and `GPU_Int_Tool.py`: backward-compatible import shims.
- `GQIS_API.md`: complete function, parameter, and return-value reference.
- `INSTALLATION_AND_SMOKE_TEST.md`: isolated installation and CUDA verification guide.
- [CONTRIBUTING.md](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/CONTRIBUTING.md): issue reports, development setup, scientific validation, and pull-request requirements.
- [CHANGELOG.md](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/CHANGELOG.md): concise user-facing history of significant changes by release.
- `.gitattributes`: cross-platform source line-ending policy.
- `Benchmark_full_tools.py`: shared full-benchmark plotting, extrapolation, and metadata helpers.
- `Example_01_two_level_basic.py`: basic two-level interferogram.
- `Example_02_four_level_interferogram.py`: four-level interferogram.
- `Example_03_two_level_animation.py`: two-level animation using GQIS directly.
- `Example_04_four_level_animation.py`: four-level animation.
- `Example_05_initial_condition_sweep_gate_fidelity.py`: initial-condition sweep and gate-fidelity map.
- `Benchmark_01_two_level.py`: two-level benchmark.
- `Benchmark_02_four_level_Interferometry.py`: four-level benchmark.
- `check_environment.py`: dependency, GPU, CPU, OS, and smoke-test checker.
- `tests/`: fast structural/time-grid tests and an optional CUDA smoke test.
- `.github/workflows/ci.yml`: source-build and non-GPU test workflow.
- `Benchmark_01_full_benchmark.csv` and `Benchmark_02_full_benchmark.csv`: saved reference timing tables.
- `Benchmark_01_full_benchmark.png` and `Benchmark_02_full_benchmark.png`: saved reference timing figures.

## Contributing

Bug reports, validation results from other GPUs, documentation corrections, and
focused code contributions are welcome. See [CONTRIBUTING.md](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/CONTRIBUTING.md) before opening an issue or pull request.

## Citation

If you use this software in scientific work, cite the repository metadata from [CITATION.cff](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/CITATION.cff). After a paper or Zenodo archive is available, add the DOI to `CITATION.cff` and cite the archived release for reproducibility.

## License

This project is released under the [MIT License](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/LICENSE).
