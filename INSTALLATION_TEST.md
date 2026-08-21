# Installation and GPU Test

These instructions create an isolated environment, install GQIS, verify its
dependencies, and run a small CUDA calculation. They are suitable for public
release testing and for comparing behavior across different NVIDIA GPUs.

## Prerequisites

- Windows 10 or 11, or a supported Linux distribution
- an NVIDIA CUDA-capable GPU with a current NVIDIA driver
- Python 3.10 or newer
- Git when installing directly from GitHub

Confirm that the NVIDIA driver is visible:

```text
nvidia-smi
```

## Recommended: Clean Conda Environment

Open Anaconda Prompt or Miniconda Prompt and run:

```text
conda create --name gqis-test python=3.11 -y
conda activate gqis-test
python -m pip install --upgrade pip
pip install "gqis[cuda12]"
gqis-check --installation-test
```

The short PyPI command works after the release has been uploaded. To install
the tagged public GitHub release directly, use:

```text
pip install "gqis[cuda12] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@v0.1.0"
gqis-check --installation-test
```

A successful test ends with output similar to:

```text
Installation test: PASS shape=(4, 4)
Environment check: PASS
```

Leave the test environment when finished:

```text
conda deactivate
```

To remove the temporary environment later:

```text
conda env remove --name gqis-test
```

## Alternative: Standard Python Virtual Environment

In Windows Command Prompt, use a Python executable that is already installed
and visible in that terminal:

```bat
python -m venv "%USERPROFILE%\gqis-test"
"%USERPROFILE%\gqis-test\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install "gqis[cuda12]"
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
pip install "gqis[cuda12]"
gqis-check --installation-test
deactivate
```

If `python` opens the Microsoft Store or reports that Python was not found,
use Anaconda Prompt instead or invoke the installed Python executable by its
full path.

## CUDA Version Selection

The commands above use the locally tested CUDA 12 configuration. Install
exactly one CuPy variant:

- CUDA 11: replace `[cuda12]` with `[cuda11]`
- CUDA 12: use `[cuda12]`
- CUDA 13: replace `[cuda12]` with `[cuda13]`

Do not install `cupy`, `cupy-cuda11x`, `cupy-cuda12x`, and `cupy-cuda13x`
together in one environment. CUDA 11 and CUDA 13 extras are provided but have
not been tested on the reference workstation used for version 0.1.0.

## Optional External Programs

FFmpeg is needed only for MP4 animation export. On Windows, confirm that the
same terminal used to run Python can execute both `where ffmpeg` and
`ffmpeg -version`. Matplotlib searches `PATH` unless
`matplotlib.rcParams["animation.ffmpeg_path"]` is set explicitly.

The optional Julia GPU benchmark requires Julia plus `DifferentialEquations`,
`DiffEqGPU`, `CUDA`, and `StaticArrays`. Check these packages with:

```text
gqis-check --check-julia-packages
```

## Record Hardware Information

The environment checker reports Python and dependency versions, the operating
system, CPU, GPU model, CUDA versions, and total GPU memory. Save its complete
output with benchmark results:

```text
gqis-check
```

The benchmark scripts also write the GPU model and `gpu_vram_gb` into their
CSV metadata and show the CPU, GPU, and VRAM in generated figures.

## Updating GQIS

Update a PyPI installation with:

```text
pip install --upgrade "gqis[cuda12]"
gqis-check --installation-test
```

To test the newest public development commit instead of a tagged release:

```text
pip install --upgrade --force-reinstall "gqis[cuda12] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@main"
gqis-check --installation-test
```
