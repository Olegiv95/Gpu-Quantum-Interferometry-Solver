# Contributing to GQIS

Contributions that improve correctness, reproducibility, documentation, GPU
compatibility, or performance are welcome. GQIS is scientific alpha software,
so numerical evidence and clearly stated assumptions are especially important.

## Before Opening an Issue

Search existing GitHub issues first. For a bug or unexpected numerical result,
include as much of the following information as possible:

- a minimal script or model that reproduces the problem
- the complete exception and console output
- the output of `gqis-check`
- Hamiltonian dimension, collapse operators, output mode, and initial state
- sweep dimensions, simulation periods, solver steps per period, and precision
- whether the result converges when the fixed RK4 time grid is refined
- a QuTiP or other trusted-reference comparison when reporting numerical error

Do not attach confidential data, credentials, access tokens, or unpublished
experimental data that you are not authorized to share.

## Development Environment

Clone the repository and create an isolated Python environment. For the locally
tested CUDA 12 configuration:

```bash
python -m pip install --upgrade pip
pip install -e ".[cuda12,examples,benchmarks,test,dev]"
```

Replace `cuda12` with the matching CUDA extra. Install exactly one CuPy binary
distribution in an environment.

## Code Style

- Preserve existing public names and compatibility imports unless a breaking
  change has been discussed first.
- Follow PEP 8 and the Ruff configuration in `pyproject.toml`; the project line
  length is 120 characters.
- Keep related arguments together where they fit. In solver definitions and
  calls, place physical-model inputs before execution and code-generation
  options rather than forcing every argument onto a separate line.
- For multiline calls, lists, and inline dictionaries, keep the first argument,
  item, or key after the opening delimiter and the closing delimiter after the
  final value whenever the 120-character limit allows. This also applies to
  assigned and nested dictionaries. Only the outer structural dictionary
  introduced directly by `return {` may keep its braces on separate lines.
- Keep physics notation recognizable, but use descriptive names where compact
  symbols would make general-purpose code difficult to understand.
- Add comments for physical conventions or non-obvious numerical decisions,
  not for self-explanatory assignments.
- Keep examples educational and place user-editable settings in the clearly
  marked block near the bottom of each script.

Check Python code with:

```bash
python -m ruff check .
```

## Tests and Numerical Validation

Run CPU-safe automated tests with:

```bash
python -m pytest -m "not gpu"
```

On an NVIDIA CUDA system, run the complete suite and smoke checker:

```bash
python -m pytest
gqis-check --smoke
```

Changes to equation generation, caching, time-grid handling, initial states, or
the CUDA RK4 kernel should also be checked with benchmark `diff` mode against
QuTiP. Use CPU divider `1`, refine `--solver-steps-per-period`, and report MSE,
RMS, and maximum absolute differences. A performance improvement is not sufficient if
the numerical result changes without a documented reason.

For changes that affect performance, include:

- CPU, GPU, VRAM, operating system, Python, CUDA, CuPy, and GQIS versions
- grid dimensions, time steps, precision, and warm-up convention
- separate preparation/code-generation and solver/kernel timings
- whether reported points are measured or extrapolated

## Pull Requests

Keep each pull request focused on one problem. A pull request should:

- explain the problem and the chosen approach
- identify user-visible or numerical behavior changes
- include or update tests when practical
- update the README, API reference, examples, or changelog when behavior changes
- pass Ruff lint, CPU-safe tests, package build, and metadata checks

Use concise commit messages that describe the change. Avoid committing generated
videos, caches, local environments, or disposable benchmark output.

Unless stated otherwise, contributions submitted to this repository are made
available under the project's MIT License. No contributor license agreement is
currently required.
