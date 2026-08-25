# Installation and GPU Test

These instructions install the GPU Quantum Interferometry Solver (GQIS), verify its dependencies, and run a small CUDA
calculation. A dedicated environment is not mandatory: install GQIS in an existing compatible environment unless
dependency conflicts occur. Clean environments are useful for troubleshooting, release testing, and reproducible
comparisons across different NVIDIA graphics processing units (GPUs).

## Prerequisites

- Windows 10 or 11, or a supported Linux distribution
- an NVIDIA CUDA-capable GPU with a compatible NVIDIA driver
- a matching CUDA Toolkit, or CUDA components installed through CuPy 14's `ctk` extra
- Python 3.10 or 3.11
- Git when installing directly from GitHub

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

The short Python Package Index (PyPI) command works after the release has been uploaded. To install the tagged public
GitHub release directly, use:

```text
python -m pip install "gqis[cuda12] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@v0.1.0"
gqis-check --installation-test
```

A successful test ends with output similar to:

```text
Installation test: PASS shape=(4, 4)
Environment check: PASS
```

No separate environment is needed when this test passes. If installation or
dependency checks fail, use one of the optional clean-environment procedures
below to distinguish a GQIS issue from a conflict in the existing environment.

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

## CUDA Version Selection

The commands above use the locally tested CUDA 12 configuration:

```text
python -m pip install "gqis[cuda12]"
```

Replace `cuda12` with `cuda11` or `cuda13` when using a different CUDA major
version.

The CUDA extra selects the matching CuPy wheel. By default, that wheel expects
the corresponding CUDA Toolkit compiler/runtime components to be installed on
the system. This is the configuration tested on the reference Windows
workstation.

CuPy 14 can instead install NVIDIA CUDA component wheels through its `ctk`
extra. This avoids a system-wide CUDA Toolkit, but a compatible NVIDIA driver
is still required:

```text
python -m pip install "cupy-cuda12x[ctk]"
python -m pip install "gqis[cuda12]"
gqis-check --installation-test
```

Do not install `cupy`, `cupy-cuda11x`, `cupy-cuda12x`, and `cupy-cuda13x`
together in one environment. CUDA 11 and CUDA 13 extras are provided but have
not been tested on the reference workstation used for version 0.1.0.

## Installation Extras

Install plotting support for the examples with:

```text
python -m pip install "gqis[cuda12,examples]"
```

Install QuTiP and SciPy for the benchmark comparison solvers with:

```text
python -m pip install "gqis[cuda12,examples,benchmarks]"
```

The CUDA extra selects CuPy. The `examples` extra adds Matplotlib, while the
`benchmarks` extra adds Matplotlib, QuTiP, and SciPy. Julia and FFmpeg are
external programs and are not installed by pip.

## Source And Development Installation

Before the PyPI release, or to install an exact tagged source revision, use:

```text
python -m pip install "gqis[cuda12,examples] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@v0.1.0"
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

The versions below are the GQIS 0.1.0 reference environment. Other versions may
work, but they should be checked with `gqis-check --installation-test` and a
numerical comparison before scientific use.

| Dependency | Declared requirement | Locally tested version |
| --- | --- | --- |
| Python | `>=3.10,<3.12` | 3.11.7 |
| NumPy | `>=1.24` | 2.4.6 |
| SymPy | `>=1.11` | 1.14.0 |
| CuPy/CUDA | CUDA-specific extra | `cupy-cuda12x` 14.1.1, CUDA runtime 12.9 |
| Matplotlib | `>=3.7` (examples) | 3.11.1 |
| SciPy | `>=1.10` (benchmarks) | 1.16.1 |
| QuTiP | `>=5.0` (benchmarks) | 5.2.0 |
| pytest | `>=8` (tests) | 8.4.2 |
| build | `>=1` (release builds) | 1.5.0 |
| Ruff | `==0.16.2` (development) | 0.16.2 |
| Julia | external optional solver | 1.10.2 |

## Automated Tests And Package Build

Examples and benchmarks exercise realistic workflows but do not replace
automated tests because they are slower, depend on local GPU/plotting software,
and generally do not assert known numerical answers.

Run the package and time-grid tests that do not require a GPU with:

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

The GitHub Actions workflow tests Python 3.10 and 3.11 and runs the non-GPU test
suite on both versions. It builds and validates the package on Python 3.11.

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

To test the newest public development commit instead of a tagged release:

```text
python -m pip install --upgrade --force-reinstall "gqis[cuda12] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@main"
gqis-check --installation-test
```
