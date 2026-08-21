# Contributing to GQIS

We welcome contributions that improve correctness, documentation, GPU
compatibility, or performance. Small fixes and independent validation results
are valuable and remain visible in the Git history.

## Issues

Search existing issues first. Bug and numerical-error reports should include a
minimal reproducing model, relevant console output, the output of `gqis-check`,
model and grid settings, and CUDA/CuPy versions. When relevant, report
convergence after refining the fixed RK4 grid and comparison with QuTiP or
another trusted reference.

Do not share credentials or confidential data. Please open an issue before
implementing a substantial API change or changing a numerical convention.

## Development

Fork the repository, create a branch from `main`, and install an editable
development environment. For CUDA 12:

```bash
python -m pip install --upgrade pip
pip install -e ".[cuda12,examples,benchmarks,test,dev]"
```

Replace `cuda12` with the matching CUDA extra when needed, and install only one
CuPy binary distribution.

Run:

```bash
python -m ruff check .
python -m pytest -m "not gpu"
```

On an NVIDIA CUDA system, also run `python -m pytest` and
`gqis-check --installation-test`.

## Style

Following the [QuTiP contributor guide](https://qutip.readthedocs.io/en/stable/development/contributing.html),
prioritize readability and consistency with surrounding code. Follow PEP 8 and
Ruff; 120 characters is an upper line-length limit, not a target. Preserve
public names unless a breaking change has been discussed. Conventional physics
notation is welcome when documented. Update `GQIS_API.md` when a public
interface changes.

## Numerical Changes and Pull Requests

Keep pull requests focused and explain their purpose, behavior changes, and
validation. Changes to equation generation, time grids, initial states, caching,
or the CUDA kernel need focused tests and, before merging, comparison with QuTiP
in benchmark `diff` mode. Use `--qutip-output-density-divider 1` and a
converged `--solver-steps-per-period`. If CUDA validation is unavailable, state
that so a maintainer can perform it.

For performance changes, report hardware, software versions, grid, precision,
solver steps, and whether timings are measured or extrapolated. Do not commit
caches, environments, videos, or disposable outputs. Update tracked benchmark
CSV and PNG files only when the change intentionally affects them.

Contributions are made available under the project's MIT License. No contributor
license agreement is currently required.
