# GitHub Metadata Files: Short Beginner Tutorial

These files are small text files placed in the root of a GitHub repository. They tell users, Python tools, GitHub, package managers, and citation tools how to understand and use the project.

For this project, the important files are:

- `LICENSE`
- `.gitignore`
- `requirements.txt`
- `pyproject.toml`
- `CITATION.cff`

## `LICENSE`

Purpose:

`LICENSE` tells other people what they are legally allowed to do with your code.

Why it matters:

If there is no license, people should technically assume they do not have permission to reuse, modify, or distribute the code, even if it is public on GitHub.

Common choices:

- `MIT`: simple, permissive, common for research code.
- `BSD-3-Clause`: also permissive, often used in scientific software.
- `Apache-2.0`: permissive, includes explicit patent language.
- `GPL-3.0`: requires derivative software to remain open source under GPL-compatible terms.

Recommendation for GQIS:

Use `MIT` or `BSD-3-Clause` unless you specifically want stronger restrictions. For research software where you want people to cite and reuse the code, `MIT` or `BSD-3-Clause` is usually practical.

Example root location:

```text
LICENSE
```

## `.gitignore`

Purpose:

`.gitignore` tells Git which files should not be committed.

Why it matters:

Your current workspace contains generated data, videos, caches, reports, and old copies. These should not go into the public source repository.

Typical entries for this project:

```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd

.venv/
env/
venv/

*.dat
*.mp4
*_Kernel.cu
*_timings_*.txt

Report/
Reports/
.vscode/
```

Important:

`.gitignore` does not delete files. It only prevents Git from tracking files that are not already committed.

Example root location:

```text
.gitignore
```

## `requirements.txt`

Purpose:

`requirements.txt` lists Python packages needed to run the project.

Why it matters:

Users can install dependencies with:

```bash
pip install -r requirements.txt
```

Possible content for this project:

```text
numpy
sympy
matplotlib
cupy-cuda12x
qutip
scipy
```

Important:

CuPy package names depend on the CUDA version:

- `cupy-cuda11x` for CUDA 11.x
- `cupy-cuda12x` for CUDA 12.x

For this project, it may be better to explain CuPy installation carefully in the README instead of forcing one exact CuPy package for everyone.

Example root location:

```text
requirements.txt
```

## `pyproject.toml`

Purpose:

`pyproject.toml` is the modern Python project configuration file.

It can define:

- package name
- version
- author
- dependencies
- Python version requirement
- build system
- formatter/linter settings

Why it matters:

If you want users to install your tool as a package, they should be able to run:

```bash
pip install .
```

or, during development:

```bash
pip install -e .
```

Simplified example (the repository's real file contains separate CUDA-version
extras and development/test groups):

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "gpu-quantum-interferometry-solver"
version = "0.1.0"
description = "GPU Quantum Interferometry Solver for massive parameter sweeps of finite-dimensional Lindblad systems."
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [
    { name = "Oleh Ivakhnenko" }
]
dependencies = [
    "numpy",
    "sympy"
]

[project.optional-dependencies]
cuda12 = ["cupy-cuda12x"]
examples = ["matplotlib"]
benchmarks = ["qutip", "scipy"]
```

Current GQIS organization:

```text
gqis/
  __init__.py
  solver.py
  N_Level_Kernel.cu
gpu_int_tool/  # compatibility package
tests/
```

The CUDA template must be declared as package data so it is present after a
wheel is installed. CuPy is selected through `cuda11`, `cuda12`, or `cuda13`
because one dependency name cannot safely select the correct CUDA-major wheel.

Example root location:

```text
pyproject.toml
```

## `CITATION.cff`

Purpose:

`CITATION.cff` tells GitHub how users should cite your software.

Why it matters:

GitHub can show a "Cite this repository" button. This is important for scientific software.

Minimal example:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it using the metadata from this file."
title: "GPU Quantum Interferometry Solver (GQIS)"
authors:
  - family-names: "Ivakhnenko"
    given-names: "Oleh"
version: "0.1.0"
date-released: "2026-06-01"
```

Later, if you publish a paper or Zenodo DOI, add:

```yaml
doi: "10.xxxx/zenodo.xxxxxxx"
preferred-citation:
  type: article
  title: "Paper title here"
  authors:
    - family-names: "Ivakhnenko"
      given-names: "Oleh"
  journal: "Journal name"
  year: 2026
```

Example root location:

```text
CITATION.cff
```

## Which Files Are Required?

Minimum public GitHub repository:

- `README.md`
- `LICENSE`
- `.gitignore`
- `requirements.txt`

Recommended for scientific code:

- `CITATION.cff`
- `pyproject.toml`
- `examples/`
- `benchmarks/`
- `tests/`

## Recommended First Version For This Project

For the first GitHub release of GQIS, prepare:

```text
Gpu-Quantum-Interferometry-Solver/
  README.md
  GQIS_API.md
  LICENSE
  .gitignore
  requirements.txt
  pyproject.toml
  MANIFEST.in
  CITATION.cff
  GPU_Int_Tool.py                 # compatibility module for old scripts
  gqis/
    __init__.py
    solver.py
    N_Level_Kernel.cu
  gpu_int_tool/                   # compatibility package for old scripts
  tests/
  Example_01_two_level_basic.py
  Example_02_four_level_interferogram.py
  Example_03_two_level_animation.py
  Example_04_four_level_animation.py
  Benchmark_01_two_level.py
  Benchmark_01_two_level_basic_julia_gpu.jl
  Benchmark_02_four_level_Interferometry.py
```

`pyproject.toml` is now active: it builds the installable package and declares
core and optional dependencies. `requirements.txt` remains a convenient tested
CUDA 12 environment recipe rather than the package metadata source of truth.
