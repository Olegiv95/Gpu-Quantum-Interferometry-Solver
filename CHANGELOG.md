# Changelog

This file records significant user-facing changes. Individual implementation
details remain available in the Git commit history.

The format follows the main ideas of Keep a Changelog, and the project uses
semantic version numbers.

## [Unreleased]

### Planned

- Collect validation and benchmark results from additional NVIDIA GPUs.

## [0.1.0] - 2026-08-24

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

[Unreleased]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver/releases/tag/v0.1.0
