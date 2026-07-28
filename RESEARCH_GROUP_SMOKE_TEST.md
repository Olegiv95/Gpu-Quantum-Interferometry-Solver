# Private Installation and GPU Smoke Test

These instructions are for research-group members who have been invited as
collaborators to the private GitHub repository. Accept the GitHub invitation
before starting. Each member must authenticate with their own GitHub account;
do not share passwords or access tokens.

## Prerequisites

- Windows 10 or 11 with an NVIDIA CUDA-capable GPU and a current NVIDIA driver
- Git for Windows
- Python 3.10 or newer, preferably through Anaconda or Miniconda
- Access to the private `Gpu-Quantum-Interferometry-Solver` repository

Check that Git and the NVIDIA driver are visible:

```bat
git --version
nvidia-smi
```

## Recommended: Clean Conda Environment

Open **Anaconda Prompt** or **Miniconda Prompt** and run:

```bat
conda create --name gqis-smoke python=3.11 -y
conda activate gqis-smoke
python -m pip install --upgrade pip
pip install "gpu-quantum-interferometry-solver[cuda12] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@main"
gqis-check --smoke
```

Git may open a browser during installation. Sign in with the GitHub account
that was granted repository access and authorize Git Credential Manager.

A successful test ends with output similar to:

```text
Smoke test: PASS shape=(4, 4)
Environment check: PASS
```

Leave the test environment when finished:

```bat
conda deactivate
```

To remove the temporary environment later:

```bat
conda env remove --name gqis-smoke
```

## Alternative: Standard Python Virtual Environment

In Windows Command Prompt:

```bat
python -m venv "%USERPROFILE%\gqis-smoke"
"%USERPROFILE%\gqis-smoke\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install "gpu-quantum-interferometry-solver[cuda12] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@main"
gqis-check --smoke
deactivate
```

In PowerShell, activate the same environment with:

```powershell
& "$env:USERPROFILE\gqis-smoke\Scripts\Activate.ps1"
```

If `python` opens the Microsoft Store or reports that Python was not found,
use Anaconda Prompt instead. Alternatively, invoke the installed Python
executable by its full path when creating the environment.

## CUDA Version Selection

The commands above use the tested CUDA 12 CuPy package. Install exactly one
CuPy variant:

- CUDA 11: replace `[cuda12]` with `[cuda11]`
- CUDA 12: use `[cuda12]`
- CUDA 13: replace `[cuda12]` with `[cuda13]`

Do not install `cupy`, `cupy-cuda11x`, `cupy-cuda12x`, and `cupy-cuda13x`
together in one environment.

## Record Hardware Information

The environment checker reports the CPU, operating system, GPU model, CUDA
versions, and total GPU memory. Save its complete terminal output with any
benchmark results:

```bat
gqis-check
```

The benchmark scripts also write the GPU model and `gpu_vram_gb` into their
CSV metadata and show the CPU, GPU, and VRAM in the generated figure.

## Updating the Private Installation

To install the newest commit from `main` in the test environment:

```bat
pip install --upgrade --force-reinstall "gpu-quantum-interferometry-solver[cuda12] @ git+https://github.com/Olegiv95/Gpu-Quantum-Interferometry-Solver.git@main"
gqis-check --smoke
```

