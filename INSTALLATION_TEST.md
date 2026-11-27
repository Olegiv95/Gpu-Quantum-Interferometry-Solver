# Installation and GPU test

These instructions install the GPU Quantum Interferometry Solver (GQIS), verify its dependencies, and run a small CUDA
calculation. A dedicated environment is not mandatory: install GQIS in an existing compatible environment unless
dependency conflicts occur. Clean environments are useful for troubleshooting, release testing, and reproducible
comparisons across different NVIDIA graphics processing units (GPUs).

## Prerequisites

- Windows 10 or 11, or a supported Linux distribution
- an NVIDIA CUDA-capable GPU with a compatible NVIDIA driver
- a matching CUDA Toolkit, or CUDA runtime libraries installed by pip through CuPy 14's `ctk` option
- Python 3.10 or newer

See [CUDA Version Selection](#cuda-version-selection) for the tested configurations and commands.

Confirm that the NVIDIA driver is visible:

```text
nvidia-smi
```

The command should display the NVIDIA GPU model, driver version, and supported CUDA version. If the command is not
found or cannot communicate with the driver, repair the NVIDIA driver installation before installing GQIS.

## Install In An Existing Environment

Install GQIS in the currently active Python or Conda environment and run the
installation test:

```text
python -m pip install "gqis[cuda12]"
gqis-check --installation-test
```

This installs the release from the Python Package Index (PyPI). To install from GitHub or from a local clone
instead, see [Source And Development Installation](#source-and-development-installation).

The test checks each stage against an exact answer. A successful test ends with output similar to:

```text
GQIS self-test (exact answers for small problems):
  [1/6] Symbolic ODE (Lindblad -> real equations)      PASS  (3 real equations, RHS error 0e+00; 0.15 s)
  [2/6] CUDA code generation                           PASS  (3 CUDA RHS lines; 0.01 s)
  [3/6] Kernel compilation and GPU solve               PASS  (Rabi P = sin^2(A t/2), error 1.2e-07; ...)
  [4/6] Lindblad dissipation                           PASS  (P = 1 - exp(-gamma t), error 6.0e-08; 0.09 s)
  [5/6] ODE API (odesolve_2D)                          PASS  (harmonic oscillator x = cos(w t), error 2.4e-07; 0.05 s)
  [6/6] Automatic time-step calibration                PASS  (4 -> 8 steps, error 1.0e-04; 0.00 s)
Installation test: PASS (6/6 stages)
Environment check: PASS
```

No separate environment is needed when this test passes. If installation or
dependency checks fail, use one of the optional procedures with a clean environment
below to distinguish a GQIS issue from a conflict in the existing environment.

## CUDA Version Selection

The commands above use the tested CUDA 12 configuration. Replace `cuda12` with `cuda11` or `cuda13`
when using a different CUDA major version.

The `cuda12` installation option installs the matching CuPy package. By
default, this installation expects the corresponding CUDA Toolkit libraries
to be available on the system.

Alternatively, CuPy 14 can download and install the required NVIDIA CUDA
runtime libraries through pip when its `ctk` option is used. This avoids a
system-wide CUDA Toolkit, but a compatible NVIDIA driver is still required:

```text
python -m pip install "cupy-cuda12x[ctk]"
python -m pip install "gqis[cuda12]"
gqis-check --installation-test
```

Both configurations have passed the GQIS installation test on Windows: the
local CUDA Toolkit configuration with the reference NVIDIA GeForce RTX 3080
desktop GPU and the pip-based CUDA configuration on an NVIDIA GeForce
RTX 4060 Laptop GPU. The clean `ctk`
test environment occupied approximately 2.5 GB because it included the CUDA
runtime libraries.

Install only one CuPy distribution (`cupy`, `cupy-cuda11x`, `cupy-cuda12x`, or `cupy-cuda13x`)
in an environment because these packages provide the same Python module. CUDA 11 and CUDA 13
installation options have not been tested on the reference desktop system.

## Summary

For a compatible CUDA 12 environment, install `gqis[cuda12]` and run
`gqis-check --installation-test`. The test checks each stage of the solver, from the symbolic equations to the
GPU solution, against exact answers for small problems. It does not establish convergence for your scientific
model; for that, use [automatic time-step calibration](GQIS_API.md#automatic-time-step-calibration), or refine
the time grid and compare with a reference.

The remaining sections cover optional clean environments, extra dependencies,
[source development](#source-and-development-installation),
[automated tests and package builds](#automated-tests-and-package-build), and updates.

## Optional: Clean Conda Environment

Open Anaconda Prompt or Miniconda Prompt and run:

```text
conda create --name gqis-test python=3.11 -y
conda activate gqis-test
python -m pip install --upgrade pip
python -m pip install "gqis[cuda12]"
gqis-check --installation-test
```

Leave the test environment when finished:

```text
conda deactivate
```

To remove the temporary environment later:

```text
conda env remove --name gqis-test
```

## Optional: Standard Python Virtual Environment

In Windows Command Prompt, use a Python executable that is already installed
and visible in that terminal:

```bat
python -m venv "%USERPROFILE%\gqis-test"
"%USERPROFILE%\gqis-test\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install "gqis[cuda12]"
gqis-check --installation-test
deactivate
```

In PowerShell, activate the same environment with:

```powershell
& "$env:USERPROFILE\gqis-test\Scripts\Activate.ps1"
```

On Linux:

```bash
python3 -m venv "$HOME/gqis-test"
source "$HOME/gqis-test/bin/activate"
python -m pip install --upgrade pip
python -m pip install "gqis[cuda12]"
gqis-check --installation-test
deactivate
```

If `python` opens the Microsoft Store or reports that Python was not found,
use Anaconda Prompt instead or invoke the installed Python executable by its
full path.

## Optional Dependencies

Install the dependencies used by the repository's reference example scripts:

```text
python -m pip install "gqis[cuda12,examples]"
```

Install QuTiP and SciPy for the benchmark comparison solvers with:

```text
python -m pip install "gqis[cuda12,examples,benchmarks]"
```

The `cuda12` option is described in [CUDA Version Selection](#cuda-version-selection). Julia is used by the
optional Julia benchmark solver, while FFmpeg is used to save animations as
`.mp4` files. These external programs are not installed by pip; install and
verify them manually if you need those features. Installation and verification
instructions are provided in the [Optional External Programs
section](#optional-external-programs) below.

## Source And Development Installation

To install an exact tagged source revision from GitHub (this requires Git), use:

```text
python -m pip install "gqis[cuda12,examples] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@v0.3.0"
```

From a local clone, install normally or in editable development mode:

```text
python -m pip install ".[cuda12,examples,benchmarks]"
python -m pip install -e ".[all-cuda12]"
```

An editable installation imports the package directly from the clone, so code
changes become available without reinstalling. A normal installation is better
for testing the built package as an end user would receive it.

## Dependency Policy And Tested Versions

`pyproject.toml` is the package dependency source of truth. It declares minimum
compatible versions so pip does not reject an older version unnecessarily.
`requirements.txt` is a CUDA 12-oriented environment recipe.

The versions below record the configuration tested for GQIS 0.3.0; they are not an exclusive list of compatible versions.
Other compatible versions can be used within the declared dependency ranges.

| Dependency | Declared requirement | Tested version |
| --- | --- | --- |
| Python | `>=3.10` | 3.10-3.12 in CI; 3.11.7 locally |
| NumPy | `>=1.24` | 2.4.6 |
| SymPy | `>=1.11` | 1.14.0 |
| CuPy/CUDA | CUDA installation option | 14.2.0 with Toolkit and with `ctk`; CUDA runtime 12.9 |
| Matplotlib | `>=3.7` (examples) | 3.11.1 |
| SciPy | `>=1.10` (benchmarks) | 1.17.1 |
| QuTiP | `>=5.0` (benchmarks) | 5.3.1 |
| pytest | `>=8` (tests) | 9.1.1 |
| build | `>=1` (release builds) | 1.6.0 |
| Ruff | `==0.16.2` (development) | 0.16.2 |
| Julia | external optional solver | 1.10.2 |
| dynamiqs, JAX | external optional solver (Linux or WSL) | dynamiqs 0.3.7, JAX 0.12.0.1 |

## Automated Tests And Package Build

Examples and benchmarks exercise realistic workflows but do not replace
automated tests because they are slower, depend on local GPU/plotting software,
and generally do not assert known numerical answers.

Run the tests that do not require a GPU with:

```text
pytest -m "not gpu"
```

Run the complete test suite, including the CUDA numerical test, with:

```text
pytest
```

Build and check the source distribution and wheel with:

```text
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The GitHub Actions workflow tests Python 3.10, 3.11, and 3.12 and runs the
non-GPU test suite on all three versions. It builds and validates the package
on Python 3.11.

## Optional External Programs

FFmpeg is needed only for `.mp4` video export. On Windows, confirm that the
same terminal used to run Python can execute both `where ffmpeg` and
`ffmpeg -version`. Matplotlib searches `PATH` unless
`matplotlib.rcParams["animation.ffmpeg_path"]` is set explicitly.

The optional Julia GPU benchmark requires Julia plus `DifferentialEquations`,
`DiffEqGPU`, `CUDA`, and `StaticArrays`. Check these packages with:

```text
gqis-check --check-julia-packages
```

The optional dynamiqs comparison in Benchmark 04 requires dynamiqs with JAX; JAX uses the GPU only on Linux,
or on Windows through WSL.

To run a tiny exact test with each optional benchmark backend (QuTiP, Julia DiffEqGPU, dynamiqs), use
`gqis-check --installation-benchmarks-test`. On native Windows the dynamiqs test therefore
reports a warning. A backend that fails gives a warning
naming the benchmarks that need it; it does not fail the check. The first Julia run can take several minutes.

## Record Hardware Information

The environment checker reports Python and dependency versions, the operating system, central processing unit (CPU),
GPU model, CUDA versions, and total GPU memory. Save its complete
output with benchmark results:

```text
gqis-check
```

The benchmark scripts also write the GPU model and `gpu_vram_gb` into their comma-separated values (CSV) metadata and
show the CPU, GPU, and video random-access memory (VRAM) in generated figures.

## Updating GQIS

Update a PyPI installation with:

```text
python -m pip install --upgrade "gqis[cuda12]"
gqis-check --installation-test
```

To test the newest commit of the public `develop` branch instead of a tagged release:

```text
python -m pip install --upgrade --force-reinstall "gqis[cuda12] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@develop"
gqis-check --installation-test
```
