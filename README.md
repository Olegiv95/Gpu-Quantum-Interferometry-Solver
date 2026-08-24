# GPU Quantum Interferometry Solver (GQIS)

The GPU Quantum Interferometry Solver (`GQIS`) is a Python/CUDA research package for high-throughput parameter sweeps
of driven open quantum systems. It targets dense quantum interferometry maps in which every grid point requires an
independent time evolution and conventional CPU loops become impractical at high resolutions. GQIS combines
user-defined open-system SymPy models with massively parallel execution on NVIDIA CUDA, supporting workloads from
tutorial-scale calculations to parameter grids containing millions or even billions of trajectories.

## Why GQIS Was Created

High-resolution quantum interferometry requires a large ensemble of independent time evolutions: every grid point in a
two-dimensional map represents a separate simulation. Fitting a model to experimental data usually requires the complete
map to be recalculated for many candidate parameter sets. In the CPU-based workflow that motivated this project, one
sufficiently resolved interferogram could take from 30 minutes to several hours. Repeated fitting and parameter studies
could therefore take weeks or months, while reducing the grid resolution risked missing narrow interference features.

Many quantum-dynamics software evaluated during GQIS development was designed primarily to evolve one system per solver
call. Large parameter sweeps consequently required Python code to launch and coordinate many independent solver calls.
Julia was an important exception and provided capable GPU parameter sweeping. However, the tested Julia workflow
required a prepared system of independent ODEs as its input and substantial single-threaded CPU preparation before GPU
execution. Structural changes to the Hamiltonian, collapse operators, or observable therefore required the reduced ODE
system to be derived and implemented again.

GQIS was created to connect a symbolic open-system model directly to a sweep-optimized CUDA kernel. The symbolic
generator derives the independent density-matrix equations, eliminates repeated operations, and precomputes reusable
parameter combinations. The generated equations are then inserted into a compact CUDA kernel that evolves one
parameter set per GPU thread. On the reference RTX 3080, representative `2048 x 2048` interferograms complete in
seconds, making repeated parameter studies and animations over an additional model parameter practical. Controlled
timing comparisons are reported in [Validation And Performance](#validation-and-performance).

## Installation

GQIS requires Python 3.10 or newer, an NVIDIA CUDA-capable GPU, and one CuPy distribution matching the CUDA major
version. Install the tested CUDA 12 configuration with plotting support and run the installation test:

```bash
pip install "gqis[cuda12,examples]"
gqis-check --installation-test
```

Use the `cuda11` or `cuda13` extra when appropriate, and do not install multiple CuPy variants in one environment. See
the [installation and GPU test guide](./INSTALLATION_TEST.md) for clean environments, source installs, optional
dependencies, tested versions, FFmpeg, Julia, automated tests, and updates.

## Minimal Use

Import the packaged solver and pass a symbolic model plus one or two numerical sweep axes:

```python
from gqis import mesolve_2D

result = mesolve_2D(
    H, drive_expr, collapse_ops, observable, tlist,
    var_arrays={eps: eps_values, A: amplitude_values},
)
```

The five required positional arguments are:

1. `H`: `N x N` symbolic Hamiltonian, where `N` is the number of simulated quantum levels.
2. `drive_expr`: SymPy drive expression, or a dictionary defining multiple time-dependent terms.
3. `collapse_ops`: sequence of `N x N` Lindblad collapse operators.
4. `observable`: `N x N` operator whose expectation value is requested.
5. `tlist`: one-dimensional, uniformly spaced time grid beginning at zero.

If `tlist` contains `M` time samples, the solver performs `M - 1` RK4 steps. The example above returns one result for
every pair in the two sweep arrays.

See the [complete `mesolve_2D` API reference](./GQIS_API.md) for symbolic constants, initial-state sweeps, output modes,
sampled time traces, kernel reuse, precision, timings, and code-generation controls.

## Examples

| Script | Demonstration |
| --- | --- |
| `Example_01_two_level_basic.py` | Basic two-level interferogram. |
| `Example_02_four_level_interferogram.py` | Coupled qubit-resonator interferogram. |
| `Example_03_two_level_animation.py` | Two-level parameter animation with cached kernel reuse. |
| `Example_04_four_level_animation.py` | Four-level parameter animation with runtime constants. |
| `Example_05_initial_condition_sweep_gate_fidelity.py` | Initial-state sweep and gate-fidelity comparison. |

Run an example from the repository root:

```bash
python Example_01_two_level_basic.py
```

The examples print the actual workload and calculation time for their selected grid, time grid, and physical
parameters. Animation examples print initial-frame and per-frame times, plus total MP4 calculation/export time when
saving. Reduce `grid_size` in the `user_settings()` block for a quicker run or for a GPU with less memory.

<table>
  <tr>
    <td width="50%"><img src="./Example_01_two_level_basic.png" alt="Two-level interferogram"></td>
    <td width="50%"><img src="./Example_02_four_level_interferogram.png" alt="Four-level interferogram"></td>
  </tr>
  <tr align="center">
    <td><strong>Example 01:</strong> two-level interferogram</td>
    <td><strong>Example 02:</strong> coupled qubit-resonator interferogram</td>
  </tr>
</table>

## Supported Models And Outputs

The user supplies the physical model and requested output. A solver call
can contain:

- any finite-dimensional SymPy Hamiltonian, optionally containing symbolic placeholders mapped to one or more
  time-dependent drive expressions
- any list of dimensionally compatible Lindblad collapse operators
- an operator whose expectation value is averaged, returned at the final time, or sampled over time
- one or two numerical parameter-sweep axes
- initial density matrix (fixed, symbolic, swept, or explicitly supplied)
- selected runtime constants that can change without regenerating the system of independent ODEs

From the Lindblad master equation for an `N x N` Hermitian, unit-trace density matrix, GQIS derives `N*N - 1` coupled
real ODEs for the independent density-matrix components. Output modes include a time-averaged observable, final
observable, final reduced density matrix, and an optional sampled observable trace; for details see the [API output-mode
reference](./GQIS_API.md#initial-state-and-output) for details. Please note GQIS does not interpret physical units or basis labels; define all model
quantities in compatible units and one consistent basis.

## Solver Pipeline

GQIS consists of two principal components:

1. **Symbolic equation generator:** constructs the Lindblad master equation from the user-defined model, reduces it to
   independent real ODEs, eliminates repeated operations, precomputes parameter-only combinations where possible, and
   emits CUDA C expressions for the right-hand side and requested observable.
2. **CUDA execution kernel:** uses a minimal fixed-step RK4 implementation with one independent parameter set per GPU
   thread. In averaged and final-output modes, it retains only the state and intermediate values needed for integration,
   accumulates expectation values in place, and returns the requested time average or final reduced density matrix
   without storing the complete trajectory in GPU memory.

For a new model, these components perform the following pipeline:

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

Later calls can reuse the generated equations and compiled CUDA kernel when the symbolic model is unchanged, so they can start from stage 8.
Numerical sweep values, explicit initial states, and selected runtime constants may then change without recompilation.

Important implementation choices are:

- the Hamiltonian and collapse operators are supplied as symbolic SymPy matrices
- independent scalar equations are generated automatically rather than derived and maintained by hand
- only independent density-matrix components are evolved after applying trace and Hermiticity constraints to minimize redundant calcualtions
- only the requested sweep output is transferred back to CPU memory, without storing other intermediate data to save memory

## Validation And Performance

The benchmark scripts support the solver rather than define its interface. They compare GQIS output with trusted CPU
backends and demonstrate scaling on large parameter grids:

- `Benchmark_01_two_level.py`: driven qubit model
- `Benchmark_02_four_level_Interferometry.py`: coupled qubit-resonator model

On the reference RTX 3080, the largest measured `32768 x 32768` grids contain 1.07 billion independent parameter sets,
with 10,240 RK4 steps per trajectory. GQIS completed these runs in about 1 minute 44 seconds for the two-level model and
7 minutes 1 second for the four-level model. Across the approximately linear large-grid region, the average reported
speedups were about 69,000 times and 24,000 times over extrapolated QuTiP timings for the two- and four-level models,
respectively.

The compact GQIS kernel retains the reduced state and RK4 working values instead of storing each complete time
trajectory. In the tested large sweeps, this execution design used less VRAM than the Julia benchmark implementation.

<p align="center">
  <a href="./Benchmark_01_full_benchmark.png">
    <img src="./Benchmark_01_full_benchmark.png" alt="Two-level calculation-time scaling benchmark" width="900">
  </a>
</p>
<p align="center"><em>Two-level scaling reference. Click the figure for the full-resolution result.</em></p>

Comparing every solver or running a full scaling sweep can take considerable time. The scripts print progress, enforce
a configurable solver time limit, save CSV/PNG results, and mark extrapolated data. See [benchmark validation,
methodology, and complete reference results](./BENCHMARKS.md) before interpreting or reproducing these numbers.

> **Numerical accuracy disclaimer:** GQIS 0.1.0 is an alpha research release. The current CUDA backend uses fixed-step
> fourth-order Runge-Kutta (RK4) integration on the user-supplied uniform time grid. Verify time-grid convergence by
> repeating calculations with smaller steps. For important results, compare against a trusted adaptive reference solver
> because RK4 is not suitable for every problem, and its accuracy and stability depend on the time-step size. QuTiP is
> the primary reference used by the included validation benchmarks.

## Project Layout

- `gqis/` contains the solver, public interface, environment checker, and CUDA kernel template.
- `Example_*.py` contains runnable tutorials.
- `Benchmark_*.py` contains numerical comparisons and scaling measurements.
- `GQIS_API.md`, `INSTALLATION_TEST.md`, and `BENCHMARKS.md` provide detailed guidance.
- `tests/` and `.github/workflows/ci.yml` contain automated checks.

## Contributing

Bug reports, validation results from other GPUs, documentation corrections, and focused code contributions are welcome.
See [CONTRIBUTING.md](./CONTRIBUTING.md) before opening an issue or pull request.

## Citation

If you use GQIS in a publication, please cite the software using the metadata in [CITATION.cff](./CITATION.cff). A paper
citation or archival DOI will be added when available.

## License

This project is released under the [MIT License](./LICENSE).
