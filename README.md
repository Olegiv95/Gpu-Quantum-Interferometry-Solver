# GPU Quantum Interferometry Solver (GQIS)

The GPU Quantum Interferometry Solver (`GQIS`) is a Python/CUDA research package for fast two-dimensional parameter
sweeps of driven open quantum systems. It converts a symbolic Lindblad master-equation model written with SymPy into
CUDA code, compiles it with CuPy/NVRTC, and runs one independent parameter point per GPU thread.

GQIS is not hard-coded for a two-level system. The user supplies the Hamiltonian, drive, collapse operators,
observable, and optional initial density matrix. It can represent finite-dimensional Lindblad models whose generated
equations and working data fit the available CUDA compilation resources, registers, and GPU memory.

> **Status:** GQIS 0.1.0 is alpha-stage research software. The CUDA backend uses fixed-step fourth-order Runge-Kutta
> (RK4) integration on the user-supplied time grid. Verify time-grid convergence by repeating calculations with smaller
> steps. For important results, compare against a trusted reference solver because RK4 is not suitable for every
> problem, and its accuracy and stability depend on the time-step size.

## Why GQIS

Quantum interferometry can require millions or billions of independent low-dimensional simulations on a rectangular
parameter grid. In that regime, moving one trajectory to a GPU is not enough: the parameter sweep itself must execute
inside the GPU kernel.

GQIS compiles a readable SymPy model once, integrates one complete trajectory per GPU thread, accumulates the requested
observable inside the kernel, and transfers only the final result to CPU memory. This provides:

- no Python loop over individual parameter points
- symbolic construction of user-defined Lindblad models instead of a fixed built-in system
- trace- and Hermiticity-reduced density-matrix equations to avoid redundant evolution variables
- in-kernel observable accumulation without storing every time sample for every trajectory
- reuse of generated equations and compiled kernels for repeated sweeps, animations, and selected runtime constants
- dense interferograms that can resolve narrow features missed by coarser parameter scans

The method is specialized for independent parameter sweeps where a fixed time grid is acceptable after convergence
validation. QuTiP or another trusted adaptive solver remains the recommended numerical reference.

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

The first five arguments are the `N x N` Hamiltonian, time-dependent drive expression, list of `N x N` collapse
operators, measured `N x N` operator, and uniform time grid. If `tlist` contains `M` time samples, the solver performs
`M - 1` RK4 steps. The example above returns one result for every pair in the two sweep arrays.

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
saving. Reduce `grid_size` in the `user_settings()` block for a quicker run or a GPU with less memory.

[![Two-level GQIS animation preview](./Example_03_two_level_animation_preview.png)](./Example_03_two_level_animation.mp4)

Click the preview to open the two-level parameter animation. MP4 export requires FFmpeg.

## Supported Models And Outputs

A solver call can contain:

- any finite-dimensional SymPy Hamiltonian, including one or more symbolic drive placeholders
- a SymPy drive expression or a dictionary of expressions for multiple time-dependent terms
- any list of dimensionally compatible Lindblad collapse operators
- an operator whose expectation value is averaged, returned at the final time, or sampled over time
- one or two numerical parameter-sweep axes
- a fixed, symbolic, swept, or explicitly supplied initial density matrix
- selected runtime constants that can change without regenerating the symbolic equations

GQIS reduces an `N x N` Hermitian, unit-trace density matrix to `N*N - 1` independent real variables. Output modes
include a time-averaged observable, final observable, final reduced density matrix, and an optional sampled observable
trace. GQIS does not interpret physical units or basis labels; define all model quantities in compatible units and one
consistent basis.

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

Later calls can reuse the generated equations and compiled CUDA kernel when the symbolic model is unchanged. Numerical
sweep values, explicit initial states, and selected runtime constants may then change without recompilation.

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

Comparing every solver or running a full scaling sweep can take considerable time. The scripts print progress, enforce
a configurable solver time limit, save CSV/PNG results, and mark extrapolated data. See [benchmark validation,
methodology, and complete reference results](./BENCHMARKS.md) before interpreting or reproducing these numbers.

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

If you use this software in scientific work, cite the repository metadata from [CITATION.cff](./CITATION.cff). After a
paper or Zenodo archive is available, add its DOI and cite the archived release for reproducibility.

## License

This project is released under the [MIT License](./LICENSE).
