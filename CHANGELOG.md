# Changelog

This file records significant user-facing changes. Individual implementation
details remain available in the Git commit history.

The format follows the main ideas of Keep a Changelog, and the project uses
semantic version numbers.

## [Unreleased]

### Planned

- Collect validation and benchmark results from additional NVIDIA GPUs.

## [0.2.0] - 2026-09-09

### Added

- General SymPy ODE interface, `odesolve_2D`, with final-state, averaged and sampled-evolution outputs.
- Live Duffing phase-space flow with device-resident state, direct GPU-prepared RGBA display,
  an interactive frame-rate cap, timing logs and optional MP4 export.
- Absolute initial time (`t_in`) and device-array outputs (`return_device`) in both solver APIs.
- Selectable fixed-step CUDA integrators alongside the default RK4; see the [API reference](GQIS_API.md).
- [Benchmark 03 accuracy sweeps](BENCHMARKS.md#benchmark-03-accuracy-and-convergence), saved references,
  accuracy-calibrated scaling runs and CSV-based figure generation.

### Changed

- Moved tutorials and benchmarks into `Examples/` and `Benchmarks/`, with outputs in their `results/` folders.
- Separated Example 06 rendering helpers from its ODE demonstration.
- Made all eight GQIS methods selectable in Benchmark 01/02 comparison and scaling modes.
- Reorganized benchmark/API guides and made README links absolute for the PyPI description.
- Removed a hard-coded two-level measurement from the scaling plotter's default four-level plot.

## [0.1.1] - 2026-08-26

### Changed

- Removed the Python upper-version restriction and added Python 3.12 to continuous integration testing.
- Documented the two tested CUDA 12 installation methods: a local CUDA Toolkit and CUDA runtime libraries installed
  through pip with CuPy's `ctk` option.
- Added external installation-test results from an RTX 4060 Laptop GPU without a system-wide CUDA Toolkit.

## [0.1.0] - 2026-08-25

### Added

- Symbolic Lindblad-equation reduction and CUDA right-hand-side (RHS) generation for
  finite-dimensional user-defined Hamiltonians and collapse operators.
- Two-dimensional graphics processing unit (GPU) parameter sweeps with fixed-step fourth-order Runge-Kutta (RK4)
  integration.
- Time-averaged expectation-value, final expectation-value, final-density-matrix, and sampled-trace output modes.
- Runtime constants, symbolic RHS caching, and initial-condition sweeps.
- Two- and four-level examples, animations, gate-fidelity example, and Julia
  comparison solver.
- GPU, native fixed-step Python, adaptive SciPy, and QuTiP benchmark solvers.
- Pairwise benchmark `diff` mode with mean-square deviation (MSE), root-mean-square deviation (RMS), and maximum absolute deviation,
  plus automatic full-benchmark scaling mode.
- Installable `gqis` package, environment checker, automated tests, public
  documentation, and MIT license.

[Unreleased]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/releases/tag/v0.1.0
