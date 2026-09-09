# GPU Quantum Interferometry Solver (GQIS)

The GPU Quantum Interferometry Solver (`GQIS`) is a Python/NVIDIA CUDA research package for large parameter sweeps of
driven open quantum systems. GQIS evaluates independent simulations in parallel on an NVIDIA graphics processing unit
(GPU), supporting workloads from tutorial-scale examples to parameter grids containing millions or even billions of
simulations.

The physical model is symbolic: the Hamiltonian, drive, collapse operators, and measured operator are written as SymPy
expressions in which selected physical parameters remain named symbols instead of immediately becoming fixed numbers.
GQIS converts this model into the equations and CUDA code used for the parameter sweep.

The same CUDA engine also accepts general SymPy-defined ordinary differential equations through `odesolve_2D`.
This supports non-quantum applications such as nonlinear oscillators and initial-condition maps, with final-state,
time-average and sampled-evolution outputs. See [General ODE Sweeps](#general-ode-sweeps).

## Why GQIS Was Created

High-resolution quantum interferometry requires a parameter sweep that repeats the same time-evolution calculation for
many combinations of physical parameters, with each combination producing one point in a two-dimensional map. Fitting a
model to experimental data often requires the complete map to be recalculated for many candidate parameter sets. The
slowness of these central processing unit (CPU) calculations motivated this project: one sufficiently resolved
interferogram could take from 30 minutes to several hours. Repeating that calculation during parameter fitting could
therefore take days or weeks, while reducing the resolution risked missing narrow interference features.

Many quantum-dynamics packages evaluated during GQIS development were designed primarily to evolve one parameter set
per solver call. Large parameter sweeps consequently required Python code to launch and coordinate many separate solver
calls. Julia was an important exception and provided capable GPU parameter sweeping. However, the tested Julia workflow
required a prepared system of ordinary differential equations (ODEs) and substantial single-threaded CPU preparation
before GPU execution. Structural changes to the Hamiltonian, collapse operators, or measured operator therefore
required the reduced ODE system to be derived and implemented again.

GQIS connects a symbolic open-system model directly to a CUDA kernel optimized for large parameter sweeps. Its symbolic
generator derives the required density-matrix equations, eliminates repeated operations, and precomputes reusable
parameter combinations. The generated equations are then inserted into a compact CUDA kernel that evolves one parameter
set per GPU thread. On the reference NVIDIA GeForce RTX 3080 desktop GPU, representative `2048 x 2048` interferograms
complete in a few seconds,
making repeated parameter studies and animations that vary an additional physical parameter affordable and practical.
Controlled timing comparisons are reported in [Validation And Performance](#validation-and-performance).

## Installation

GQIS requires Python 3.10 or newer, an NVIDIA CUDA-capable GPU, a compatible NVIDIA driver, and CUDA libraries matching
the selected CuPy package. Continuous integration tests Python 3.10 through 3.12. On Windows, GQIS has been tested both
with a locally installed CUDA Toolkit and with CUDA runtime libraries downloaded and installed by pip through CuPy's
`ctk` option.

With a local CUDA 12 Toolkit, install:

```bash
pip install "gqis[cuda12]"
```

Without a local CUDA Toolkit, use pip to install the CUDA 12 runtime libraries and GQIS:

```bash
pip install "cupy-cuda12x[ctk]"
pip install "gqis[cuda12]"
```

The pip-installed CUDA-library method above has been tested with CUDA 12. When using a local Toolkit with another CUDA
major version, replace `cuda12` with `cuda11` or `cuda13`. Verify the core solver installation with:

```bash
gqis-check --installation-test
```

The repository provides reference example scripts demonstrating the solver's main features. To run them, install their
optional dependencies, including Matplotlib:

```bash
pip install "gqis[cuda12,examples]"
```

See the [installation and GPU test
guide](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/INSTALLATION_TEST.md) for optional isolated
environments, troubleshooting, source installs, optional dependencies, tested versions, FFmpeg, Julia, automated tests,
and updates.

## Minimal Use

Import the packaged solver and pass a symbolic model plus one or two numerical sweep axes:

```python
from gqis import mesolve_2D

result = mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays={eps: eps_values, A: amplitude_values},
)
```

Five mandatory positional arguments are:

1. `H`: `N x N` symbolic Hamiltonian, where `N` is the number of simulated quantum levels.
2. `Drive`: SymPy expression for the time-dependent drive, or a dictionary defining multiple time-dependent terms.
3. `Col_Ops`: sequence of `N x N` Lindblad collapse operators representing processes such as relaxation and dephasing.
4. `mean_operator`: `N x N` operator associated with the physical quantity whose expectation value is requested.
5. `tlist`: one-dimensional, uniformly spaced time grid beginning at zero.

If `tlist` contains `M` time samples, the solver performs `M - 1` fixed steps, using fourth-order Runge-Kutta (RK4) by default. The example
above returns one time-averaged expectation value of `mean_operator` for every combination of parameter values from the
two sweep arrays.

See the [complete `mesolve_2D` application programming interface (API)
reference](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/GQIS_API.md) for symbolic constants,
initial-state sweeps, output modes, sampled time traces, kernel reuse, precision, timings, and code-generation controls.

### Choosing A Solver

Both APIs offer `rk4` (default), `lserk4`, `dp5`, `tsit5`, `anas5`, `ab5`, `alshina6` and `dop853`.
Set `solver="tsit5"`, for example, to change the integration method. These are fixed-step implementations:
RK4 is a useful starting point, low-storage LSRK4 reduces work-vector storage, and higher-order methods can
reach tighter accuracy with coarser steps. The [solver comparison table](GQIS_API.md#available-fixed-step-solvers)
summarizes their costs and uses; [Benchmark 03](BENCHMARKS.md#accuracy-calibrated-dividers) compares the time
needed to satisfy chosen error limits.

## General ODE Sweeps

For a model already written as first-order ODEs, supply its derivatives, state symbols and initial values directly.
For example, a driven double-well Duffing oscillator can be swept over damping and drive amplitude:

```python
import numpy as np
import sympy as sp
from gqis import odesolve_2D

x, v, t, damping, amplitude = sp.symbols("x v t damping amplitude", real=True)
result = odesolve_2D(
    [v, -damping*v + x - x**3 + amplitude*sp.cos(t)],
    [x, v], [0.1, 0], np.linspace(0, 20, 4097),
    var_arrays={damping: np.linspace(0.1, 0.5, 64), amplitude: np.linspace(0.1, 0.4, 64)},
    solver="rk4", output_mode="final")
```

`result` contains final position and velocity at each parameter pair, with shape `(64, 64, 2)`.
Use `output_mode="mean"` for averages, or `return_time_trace=True` for sampled evolution.
Both public APIs share the fixed-step integrators and symbolic optimizations; no quantum operators are needed here.

[Example 06](Examples/Example_06_symbolic_ode_sweep.py) evolves a dense Duffing initial-condition cloud live on the GPU
and displays its phase-space flow, with optional MP4 export. The [ODE API reference](GQIS_API.md#odesolve_2d-direct-sympy-ode-sweeps)
describes output shapes and sampling options. See [accuracy calibration](BENCHMARKS.md#accuracy-calibrated-dividers)
for the step-size comparison workflow; its bundled models are Lindblad systems, while a general ODE can be
checked by refining its time grid and comparing with a suitable reference.

For development builds, install the source checkout in [editable mode](INSTALLATION_TEST.md#source-and-development-installation).

## Examples

| Script | Demonstration |
| --- | --- |
| `Examples/Example_01_two_level_basic.py` | Basic two-level interferogram. |
| `Examples/Example_02_four_level_interferogram.py` | Coupled qubit-resonator interferogram. |
| `Examples/Example_03_two_level_animation.py` | Two-level animation that reuses the generated equations and compiled kernel between frames. |
| `Examples/Example_04_four_level_animation.py` | Four-level animation that changes selected physical constants without recompilation. |
| `Examples/Example_05_initial_condition_sweep_gate_fidelity.py` | Initial-state sweep and gate-fidelity comparison. |
| `Examples/Example_06_symbolic_ode_sweep.py` | Live Duffing phase-space flow, GPU rasterization and optional MP4 export through the general ODE API. |

[Example 06](Examples/Example_06_symbolic_ode_sweep.py) demonstrates the general ODE API with a live
Duffing attractor animation. The simulation stays on the GPU, with fast display and optional video export;
see its [animation guide](Examples/Example_06.md) for display settings and timing logs.

Run an example from the repository root:

```bash
python Examples/Example_01_two_level_basic.py
```

The examples print the actual preparation and calculation time for their selected grid, time grid, and physical
parameters. Animation examples print per-frame times, plus total video calculation/export time when saving. Reduce
`grid_size` in the `user_settings()` block for a quicker run or for a GPU with less memory.
Relative output filenames are written to `Examples/results/`. Presentation media in that directory can be committed,
while optional numerical-data exports remain ignored by Git.

<table>
  <tr>
    <td width="50%"><img src="https://raw.githubusercontent.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/main/Examples/results/Example_01_two_level_basic.png" alt="Two-level interferogram"></td>
    <td width="50%"><img src="https://raw.githubusercontent.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/main/Examples/results/Example_02_four_level_interferogram.png" alt="Four-level interferogram"></td>
  </tr>
  <tr align="center">
    <td><strong>Example 01:</strong> two-level interferogram</td>
    <td><strong>Example 02:</strong> coupled qubit-resonator interferogram</td>
  </tr>
</table>

## Supported Models And Outputs

The user supplies the physical model and requested output. A solver call
can contain:

- any finite-dimensional SymPy Hamiltonian, optionally containing named symbols mapped to one or more time-dependent
  drive expressions
- any list of Lindblad collapse operators with the same `N x N` shape as the Hamiltonian
- an operator whose expectation value is averaged, returned at the final time, or sampled over time
- one or two numerical parameter-sweep axes
- a common initial density matrix, a symbolic initial-state sweep, or a numerical initial state supplied for each
  simulation point
- selected physical constants whose numerical values can change without regenerating the ODE system

For the Lindblad master equation of an `N x N` Hermitian, unit-trace density matrix, let $D=N^2-1$ denote the number
of retained real equations. GQIS removes equations made redundant by unit trace and Hermiticity (conjugate symmetry)
and derives a coupled system of $D$ independent real ODEs. Here, independent means that every retained equation is
required and none can be excluded without losing information; the equations remain coupled and are solved together.
Output modes include a time-averaged expectation value, the expectation value at the final time, the final reduced
density matrix, and an optional sampled time trace of the expectation value; see the [API
output-mode
reference](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/GQIS_API.md#initial-state-and-output)
for details. [Example 05](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/Examples/Example_05_initial_condition_sweep_gate_fidelity.py)
uses `output_mode="final_rho"` for final-state gate-fidelity calculations. GQIS does not interpret physical units or basis
labels; define all model quantities in compatible units and one consistent basis.

## Solver Pipeline

GQIS consists of two principal components:

The symbolic generator starts from the Lindblad master equation

```math
\frac{d\rho}{dt} = -i[H(t),\rho]
+ \sum_j \left(C_j\rho C_j^\dagger
- \frac{1}{2}C_j^\dagger C_j\rho
- \frac{1}{2}\rho C_j^\dagger C_j\right),
```

where $\rho$ is the density matrix, $H(t)$ is the time-dependent Hamiltonian, and each $C_j$ is a collapse operator.
The Hamiltonian coefficients and time variable must use mutually consistent units.

1. **Symbolic equation generator:** constructs the Lindblad master equation from the user-defined model, reduces it to
   a coupled system of independent real ODEs, eliminates repeated operations, precomputes parameter-only
   combinations where possible, and emits CUDA C expressions for the right-hand side (RHS), meaning the time
   derivatives in the ODE system.
2. **CUDA execution kernel:** uses the selected fixed-step method, with RK4 as the default and one parameter set per GPU thread. In
   averaged and final-output modes, it retains only the state and intermediate values needed for integration, calculates
   the requested expectation values during evolution, and returns the time average or final reduced density matrix
   without storing the complete time evolution in GPU memory.

Using this notation, denote the $D$ retained density-matrix components, represented by real values, as
$y_1,\ldots,y_D$. GQIS generates the system of ODEs

```math
\begin{cases}
\dfrac{dy_1}{dt} = R_1(t,y_1,\ldots,y_D), \\
\dfrac{dy_2}{dt} = R_2(t,y_1,\ldots,y_D), \\
\qquad\vdots \\
\dfrac{dy_D}{dt} = R_D(t,y_1,\ldots,y_D).
\end{cases}
```

Here, $R_i$ is the generated right-hand side of equation $i$. The vector function $f$ used below combines all $D$
right-hand sides. The fourth-order Runge-Kutta (RK4) update is:

```math
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(t_n,\mathbf{y}_n), \\
\mathbf{k}_2 &= \mathbf{f}\left(t_n+\frac{h}{2},\mathbf{y}_n+\frac{h}{2}\mathbf{k}_1\right), \\
\mathbf{k}_3 &= \mathbf{f}\left(t_n+\frac{h}{2},\mathbf{y}_n+\frac{h}{2}\mathbf{k}_2\right), \\
\mathbf{k}_4 &= \mathbf{f}\left(t_n+h,\mathbf{y}_n+h\mathbf{k}_3\right), \\
\mathbf{y}_{n+1} &= \mathbf{y}_n+\frac{h}{6}
\left(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4\right).
\end{aligned}
```

In this formula, $n$ is the starting time-sample index of the current interval, $y_n$ is the state at time $t_n$, and
$f(t_n,y_n)$ is the vector of all derivatives evaluated at that time and state. A grid of
$M$ time samples contains $t_0,\ldots,t_{M-1}$ and therefore defines $M-1$ integration intervals, each with duration
$h=t_{n+1}-t_n$.

With RK4 selected, each GPU thread applies this update to its own evolution and parameter set.
Other available methods include LSRK4, DP5, Tsit5, Anas5, AB5, Alshina6 and DOP853; see the
[API reference](GQIS_API.md) for their costs and intended uses. For general ODEs, `odesolve_2D`
accepts SymPy derivatives and initial conditions directly, using the same CUDA execution path.

For a new model, these components perform the following pipeline:

<table>
  <tr align="center">
    <td><strong>1.</strong> Define <em>H</em>, drives,<br>collapse operators and observable</td>
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
    <td><strong>7.</strong> Compile and cache<br>with CuPy and the CUDA compiler</td>
    <td>&larr;</td>
    <td><strong>6.</strong> Insert code into<br>the kernel template</td>
    <td>&larr;</td>
    <td><strong>5.</strong> Generate CUDA C<br>for RHS and expectation value</td>
  </tr>
  <tr align="center">
    <td>&darr;</td>
    <td colspan="6"></td>
  </tr>
  <tr align="center">
    <td><strong>9.</strong> Evolve with the selected method and<br>calculate expectation values</td>
    <td>&rarr;</td>
    <td><strong>10.</strong> Return the 2D<br>NumPy result</td>
    <td colspan="4"></td>
  </tr>
</table>

Later calls can reuse the generated RHS and compiled CUDA kernel when the symbolic structure of the model is unchanged,
so they can start from stage 8. Sweep arrays, numerically supplied initial states, and the numerical values of selected
constants deliberately kept symbolic may then change without recompilation. This is particularly useful for animations
that vary one physical parameter between frames. See [Reusing The ODE For
Animations](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/GQIS_API.md#reusing-the-ode-for-animations)
for details.

Important implementation choices are:

- the Hamiltonian and collapse operators are supplied as symbolic SymPy matrices
- a coupled system of independent real ODEs is generated automatically rather than derived and maintained by
  hand
- only independent density-matrix components are evolved after applying unit-trace and Hermiticity relations to remove
  redundant calculations
- only the requested sweep output is transferred back to CPU memory, without storing other intermediate data to save memory

## Validation And Performance

The benchmark scripts support the solver rather than define its interface. They compare GQIS output with trusted CPU
solvers and show how calculation time changes with the size of the parameter grid:

- `Benchmarks/Benchmark_01_two_level.py`: driven qubit model
- `Benchmarks/Benchmark_02_four_level_Interferometry.py`: coupled qubit-resonator model
- `Benchmarks/Benchmark_03_accuracy_timestep_sweep.py`: accuracy versus step size and calculation time for both models

On the reference NVIDIA GeForce RTX 3080 desktop GPU, the largest measured `32768 x 32768` grids contain
1.07 billion independent parameter sets. GQIS(RK4) completed the two-level run in **98.43 seconds** with
10,240 steps per simulation, and the four-level run in **303.85 seconds** with 5,840 steps per simulation.
These time grids were selected through accuracy calibration.
Across grids from `4096 x 4096` to `32768 x 32768`, average point-by-point speedups were approximately
46,900 times and 50,700 times over extrapolated QuTiP timings for the two- and four-level models, respectively.
See the [benchmark results](BENCHMARKS.md#reference-results) for timing definitions and measured/extrapolated status.

The compact GQIS kernel retains the reduced state and RK4 working values instead of storing each complete time
evolution. In the tested large sweeps, this execution design used less video random-access memory (VRAM) than the Julia
comparison solver.

<p align="center">
  <a href="https://raw.githubusercontent.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/main/Benchmarks/results/Benchmark_01_full_benchmark_plot.png">
    <img src="https://raw.githubusercontent.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/main/Benchmarks/results/Benchmark_01_full_benchmark_plot.png" alt="Two-level calculation-time scaling benchmark" width="900">
  </a>
</p>
<p align="center"><em>Two-level scaling reference. Click the figure for the full-resolution result.</em></p>

Comparing every solver or running a full scaling sweep can take considerable time. The scripts print progress, enforce
a configurable solver time limit, save comma-separated values (CSV) data and Portable Network Graphics (PNG) figures,
and mark extrapolated data. See [benchmark validation,
methodology, and complete reference results](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/BENCHMARKS.md)
before interpreting or reproducing these numbers.

Accuracy depends on the selected solver and the step size supplied by the user. The
[Benchmark 03 accuracy results and calibration workflow](BENCHMARKS.md#accuracy-calibrated-dividers)
show how error and calculation time vary with time-grid resolution in the two- and four-level examples.
The same benchmark can help you choose a solver and step size that meet the accuracy needs of your own model.

In the saved 64×64 two-level comparison against QuTiP, fine-grid GQIS FP32 results reached absolute RMS differences
of about `3–4 × 10⁻⁶` and maximum differences of about `3.5 × 10⁻⁵` in the measured observable. These are results
for that model and averaging window, rather than a universal FP32 accuracy floor. See
[precision and Julia comparison notes](BENCHMARKS.md#precision-and-julia-comparison-notes) for roundoff,
stock and modified Julia stepping, and preparation-time definitions.

Start with the [64×64 Benchmark 03 preset](Benchmarks/presets/Benchmark_03_two_level_64.json).
[Reproducing accuracy figures](BENCHMARKS.md#reproducing-accuracy-figures) explains launching presets,
reusing references and rebuilding plots from saved CSV files.

## Project Layout

- `gqis/` contains the solver, public interface, environment checker, and CUDA kernel template.
- `Examples/` contains runnable tutorials; their generated files go to `Examples/results/`.
- `Benchmarks/` contains numerical comparisons and scaling measurements; their generated files go to
  `Benchmarks/results/`.
- `GQIS_API.md`, `INSTALLATION_TEST.md`, and `BENCHMARKS.md` provide detailed guidance.
- `tests/` and `.github/workflows/ci.yml` contain automated checks.

## Contributing

Bug reports, validation results from other GPUs, documentation corrections, and focused code contributions are welcome.
See [CONTRIBUTING.md](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/CONTRIBUTING.md)
before opening an issue or pull request.

## Citation

If you use GQIS in a publication, please cite the software using the metadata in
[CITATION.cff](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/CITATION.cff). A paper citation or
archival digital object identifier (DOI) will be added when available.

## License

This project is released under the [MIT License](https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/blob/main/LICENSE).
