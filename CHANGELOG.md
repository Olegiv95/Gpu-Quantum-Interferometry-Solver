# Changelog

This file records significant user-facing changes. Individual implementation
details remain available in the Git commit history.

The format follows the main ideas of Keep a Changelog, and the project uses
semantic version numbers.

## [Unreleased]

### Planned

- Collect validation and benchmark results from additional NVIDIA GPUs.

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

[Unreleased]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/releases/tag/v0.1.0
