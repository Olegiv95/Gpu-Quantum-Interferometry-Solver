# Changelog

This file records significant user-facing changes. Individual implementation
details remain available in the Git commit history.

The format follows the main ideas of Keep a Changelog, and the project uses
semantic version numbers.

## [Unreleased]

### Planned

- Collect validation and benchmark results from additional NVIDIA GPUs.

## [0.3.0] - 2026-11-27

### Added

- Optional grouped arguments (`consts`, `initial`, `output`, `solver_options`, `codegen` and `diagnostics`) in
  `mesolve_2D` and `odesolve_2D`; see [grouped options](GQIS_API.md#grouped-options).
  Existing direct arguments remain supported.
- `calibrate_timesteps` in `mesolve_2D` and `odesolve_2D` chooses one uniform time step automatically: it
  refines the time grid on a reduced set of sweep points until the estimated RMS and maximum errors of the
  result meet the limits, then calculates the full grid. The accepted step is reused by later calls; see
  [automatic time-step calibration](GQIS_API.md#automatic-time-step-calibration). With `tsit5a` the same
  search tightens `step_rtol` and `step_atol` together instead of the step count. Every example includes a
  one-line switch for it; Example 02 has it on.
- `lserk4f` solver: `lserk4` with the RHS accumulated directly into its residual, storing one state
  vector fewer. Results match `lserk4` to rounding; it reduces register spilling for large systems.
  Switching between `lserk4` and `lserk4f` reuses the generated ODE.
- `tsit5a` solver with adaptive step size (Tsitouras 5(4) with its embedded error estimate) in `mesolve_2D` and `odesolve_2D`: each system
  adapts its step between the `tlist` output points, `mean` outputs integrate over the adaptive steps,
  and time is kept in FP64 by default. Set its tolerances with `adaptive_options`; see
  [adaptive step size](GQIS_API.md#adaptive-step-size-tsit5a). The other solvers' generated code is unchanged.
- `compile_lindblad_rhs` / `compile_ode_rhs`: prepare and optimize the ODE once, without the GPU; both
  solvers accept the result and start from code emission. Per-stage preparation timings; see
  [compiling the ODE once](GQIS_API.md#compiling-the-ode-once).
- `gqis-check --installation-test` checks each stage (symbolic ODE, code generation, GPU solve,
  dissipation, ODE API, step calibration) against exact answers; `--installation-benchmarks-test` tests
  the optional QuTiP, Julia DiffEqGPU and dynamiqs backends.
- [Benchmark 04](BENCHMARKS.md#benchmark-04-hilbert-space-scaling) measures calculation and total time against
  Hilbert-space dimension for qubit-resonator and coupled-qubit models, with GQIS, dynamiqs and Julia DiffEqGPU.

### Changed

- Faster `mesolve_2D` preparation, same generated code: `collect_rho`/`cse_simplify` default to `False`
  and the Lindblad equations are built in parallel from Hilbert-space dimension 10 (`symbolic_workers`).
  5 qubits: 566 s -> 32 s. `odesolve_2D` keeps `cse_simplify=True`.
- Saved Benchmark 01-04 results: preparation and total times reduced by the measured old/new
  preparation difference; calculation times unchanged.
- Lindblad RHS generation simplifies each state coefficient separately. This removes redundant
  multiplications such as `(1/E)*E` and makes symbolic preparation 1.3-2x faster on the tested models;
  the generated equations are mathematically unchanged.
- The CUDA kernel is compiled before its timed launch, so the reported `gpu_kernel_s` (`timings`,
  `return_timing_info`) no longer includes compilation on the first call.
- Benchmarks 01-04 build the Julia DiffEqGPU comparison from the ODE that GQIS compiles, through one shared
  module (`Benchmarks/julia_diffeqgpu.py`). Benchmarks 01/02 now also compile the Julia kernel before the
  timed solve, so new Julia calculation times exclude compilation, as in Benchmarks 03/04.

### Fixed

- Squares of sums, e.g. `(x + 1)**2`, lost their parentheses in the CUDA code (wrong `odesolve_2D`
  results; nested squares did not compile).
- Exact constants (`sp.pi`, `sp.E`, `sp.sqrt(2)`, ...) were printed as C macros that NVRTC does not
  define; they are now FP32 literals.
- The Benchmark 03 plot launchers point to the published result files.

### Removed

- The unused `Benchmarks/Benchmark_01_two_level_basic_julia_gpu.jl` and the committed CUDA kernel cache
  (`.gqis_manual_test_kernel_cache/`, now ignored; Benchmark 03 recreates it).

## [0.2.0] - 2026-09-09

### Added

- General SymPy ODE interface, `odesolve_2D`, with outputs of the final state, the average and the sampled time evolution.
- Absolute initial time (`t_in`) and outputs kept on the GPU (`return_device`) in both solver APIs.
- Selectable fixed-step CUDA integrators alongside the default RK4; see the [API reference](GQIS_API.md).
- Example 06 demonstrates the ODE API and GPU continuation through live Duffing phase-space flow,
  with RGBA display prepared on the GPU, an interactive cap on the frame rate, timing logs and optional MP4 export.
- [Benchmark 03 accuracy sweeps](BENCHMARKS.md#benchmark-03-accuracy-and-convergence), saved references,
  scaling runs calibrated for accuracy, and figures drawn from saved CSV data.

### Changed

- Moved tutorials and benchmarks into `Examples/` and `Benchmarks/`, with outputs in their `results/` folders.
- Separated Example 06 rendering helpers from its ODE demonstration.
- Made all eight GQIS methods selectable in Benchmark 01/02 comparison and scaling modes.
- Reorganized benchmark/API guides and made README links absolute for the PyPI description.
- Removed a hard-coded two-level measurement from the scaling plotter's default four-level plot.

## [0.1.1] - 2026-08-26

### Changed

- Removed the upper limit on the Python version and added Python 3.12 to continuous integration testing.
- Documented the two tested CUDA 12 installation methods: a local CUDA Toolkit and CUDA runtime libraries installed
  through pip with CuPy's `ctk` option.
- Added external results of the installation test from an RTX 4060 Laptop GPU without a system-wide CUDA Toolkit.

## [0.1.0] - 2026-08-25

### Added

- Symbolic reduction of the Lindblad equation and CUDA generation of the right-hand side (RHS) for
  finite-dimensional user-defined Hamiltonians and collapse operators.
- Two-dimensional graphics processing unit (GPU) parameter sweeps with fixed-step fourth-order Runge-Kutta (RK4)
  integration.
- Output modes for the time-averaged expectation value, the final expectation value, the final density
  matrix and a sampled time trace.
- Runtime constants, symbolic RHS caching, and sweeps over initial conditions.
- Two- and four-level examples, animations, gate-fidelity example, and Julia
  comparison solver.
- GPU, native fixed-step Python, adaptive SciPy, and QuTiP benchmark solvers.
- Pairwise benchmark `diff` mode with mean-square deviation (MSE), root-mean-square deviation (RMS), and maximum absolute deviation,
  plus an automatic scaling mode (`full_benchmark`).
- Installable `gqis` package, environment checker, automated tests, public
  documentation, and MIT license.

[Unreleased]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/releases/tag/v0.1.0
