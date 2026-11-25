"""Check the local Python/CUDA environment for GQIS installations."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import site
import shutil
import subprocess
import sys

REQUIRED_PACKAGES = (("numpy", "1.24", "core"), ("sympy", "1.11", "core"), ("cupy", "13",
                                                                            "CUDA support"),
                     )
OPTIONAL_PACKAGES = (("matplotlib", "3.7", "examples and plots"),
                     ("scipy", "1.10", "adaptive CPU benchmarks"), ("qutip", "5.0",
                                                                    "reference CPU benchmarks"),
                     ("pytest", "8", "maintainer tests; not required to run the solver"),
                     ("build", "1",
                      "maintainer wheel/source builds; not required to run the solver"),
                     ("ruff", "0.16.2", "maintainer lint checks; not required to run the solver"))
CUPY_DISTRIBUTIONS = ("cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x")
JULIA_PACKAGES = ("DifferentialEquations", "DiffEqGPU", "CUDA", "StaticArrays")


def _read_windows_registry_value(path: str, name: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            value, _ = winreg.QueryValueEx(key, name)
        return str(value).strip()
    except Exception:
        return ""


def windows_display_version() -> str:
    product = _read_windows_registry_value(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                                           "ProductName")
    display = _read_windows_registry_value(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                                           "DisplayVersion")
    build = _read_windows_registry_value(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                                         "CurrentBuildNumber")
    ubr = _read_windows_registry_value(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "UBR", )

    # Some upgraded systems keep a Windows 10 ProductName even on Windows 11.
    if product.startswith("Windows 10") and build.isdigit() and int(build) >= 22000:
        product = product.replace("Windows 10", "Windows 11", 1)

    if product:
        suffix = []
        if display:
            suffix.append(display)
        if build:
            suffix.append(f"build {build}{'.' + ubr if ubr else ''}")
        return f"{product} ({', '.join(suffix)})" if suffix else product
    return platform.platform()


def cpu_name() -> str:
    return (_read_windows_registry_value(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                                         "ProcessorNameString") or platform.processor()
            or platform.uname().processor or "unknown CPU")


def import_status(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"MISSING ({type(exc).__name__}: {exc})"
    version = getattr(module, "__version__", "unknown")
    if version == "unknown":
        try:
            version = importlib.metadata.version(module_name)
        except importlib.metadata.PackageNotFoundError:
            return False, "MISSING (distribution metadata not found)"
    return True, str(version)


def version_at_least(version: str, minimum: str) -> bool:
    """Compare the numeric release components needed by this checker."""

    def numeric_parts(value: str) -> tuple[int, ...]:
        match = re.match(r"\s*(\d+(?:\.\d+)*)", value)
        return tuple(int(part) for part in match.group(1).split(".")) if match else ()

    current = numeric_parts(version)
    required = numeric_parts(minimum)
    width = max(len(current), len(required))
    return bool(current) and current + (0, ) * (width - len(current)) >= required + (0, ) * (
        width - len(required))


def installed_cupy_distributions() -> list[str]:
    """Return installed CuPy distribution names and versions."""
    found = []
    for name in CUPY_DISTRIBUTIONS:
        try:
            found.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def print_installation_report() -> None:
    """Report whether this package is editable or copied into site-packages."""
    try:
        package = importlib.import_module("gqis")
        package_path = os.path.abspath(package.__file__)
        try:
            distribution = importlib.metadata.distribution("gqis")
        except importlib.metadata.PackageNotFoundError:
            # Support editable installations created before the public package rename.
            distribution = importlib.metadata.distribution("gpu-quantum-interferometry-solver")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else {}
        site_roots = [os.path.abspath(path) for path in site.getsitepackages()]
        site_roots.append(os.path.abspath(site.getusersitepackages()))
        package_in_site = any(os.path.commonpath([os.path.normcase(package_path),
                                                  os.path.normcase(root)]) == os.path.normcase(root)
                              for root in site_roots)
        editable = bool(direct_url.get("dir_info", {}).get("editable"))
        if editable:
            install_mode = "editable source link"
        elif package_in_site:
            install_mode = "standalone/non-editable installation"
        else:
            install_mode = "source/custom-path import (not site-packages)"
        print(f"Package mode: {install_mode}")
        print(f"Package path: {package_path}")
    except Exception as exc:
        print(f"Package mode: unavailable ({type(exc).__name__}: {exc})")


def print_package_report() -> bool:
    """Print required and optional Python package status."""
    print(f"Python: {sys.version.split()[0]}")
    print(f"    OS: {windows_display_version()}")
    print(f"   CPU: {cpu_name()}")
    print_installation_report()
    ok_all = True
    for name, minimum, purpose in REQUIRED_PACKAGES:
        ok, status = import_status(name)
        compatible = ok and version_at_least(status, minimum)
        ok_all = ok_all and compatible
        if ok and not compatible:
            status = f"OUTDATED {status}; requires >={minimum}"
        print(f"{name:>12s}: {status} ({purpose})")
    for name, minimum, purpose in OPTIONAL_PACKAGES:
        ok, status = import_status(name)
        if ok and not version_at_least(status, minimum):
            status = f"OUTDATED {status}; optional extra requires >={minimum}"
        print(f"{name:>12s}: {status} (optional: {purpose})")

    cupy_distributions = installed_cupy_distributions()
    print(f"{'CuPy wheel':>12s}: "
          f"{', '.join(cupy_distributions) if cupy_distributions else 'not detected'}")
    if len(cupy_distributions) > 1:
        print("CuPy wheel check: WARNING multiple CuPy distributions can conflict.")
        ok_all = False

    return ok_all


def print_external_report(*, check_julia_packages: bool) -> None:
    """Report optional FFmpeg and Julia executables and Julia packages."""
    ffmpeg = shutil.which("ffmpeg")
    ffmpeg_candidates = [os.environ.get("IMAGEIO_FFMPEG_EXE", ""),
                         os.path.join(os.environ.get("SystemDrive", "C:"), os.sep,
                            "ffmpeg", "bin", "ffmpeg.exe"),
                         os.path.join(sys.prefix, "Library", "bin", "ffmpeg.exe")]
    ffmpeg_outside_path = next((path for path in ffmpeg_candidates
                                if path and os.path.isfile(path)), None)
    if ffmpeg:
        ffmpeg_status = f"{ffmpeg} (optional: MP4 export)"
    elif ffmpeg_outside_path:
        ffmpeg_status = (
            f"{ffmpeg_outside_path} (installed outside PATH; add "
            f"{os.path.dirname(ffmpeg_outside_path)} to PATH for Matplotlib MP4 export)")
    else:
        ffmpeg_status = "not found (optional: MP4 export)"
    print(f"{'ffmpeg':>12s}: {ffmpeg_status}")

    julia = shutil.which("julia")
    if julia:
        try:
            proc = subprocess.run([julia, "--version"], capture_output=True, text=True, timeout=10,
                                  check=False)
            julia_status = (proc.stdout.strip()
                            if proc.returncode == 0 else f"found but not runnable: {julia}")
        except Exception as exc:
            julia_status = f"found but not runnable: {type(exc).__name__}: {exc}"
    else:
        julia_status = "not found in PATH"
    print(f"{'julia':>12s}: {julia_status} (optional)")
    if not julia:
        print(f"{'Julia pkgs':>12s}: not checked because Julia is unavailable")
    elif not check_julia_packages:
        print(f"{'Julia pkgs':>12s}: not checked; use --check-julia-packages")
    else:
        expression = "using " + ", ".join(JULIA_PACKAGES) + '; println("PASS")'
        try:
            proc = subprocess.run([julia, "--startup-file=no", "-e", expression],
                                  capture_output=True, text=True, timeout=60, check=False)
            status = "PASS" if proc.returncode == 0 else (proc.stderr.strip() or "FAILED")
        except Exception as exc:
            status = f"FAILED ({type(exc).__name__}: {exc})"
        print(f"{'Julia pkgs':>12s}: {status} ({', '.join(JULIA_PACKAGES)})")


def print_cuda_report() -> bool:
    ok, _status = import_status("cupy")
    if not ok:
        print("CUDA check: skipped because CuPy is missing.")
        return False

    import cupy as cp

    try:
        runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
        driver_version = int(cp.cuda.runtime.driverGetVersion())

        def format_cuda_version(value: int) -> str:
            return f"{value // 1000}.{(value % 1000) // 10}"

        print(f"CUDA runtime: {format_cuda_version(runtime_version)}; "
              f"driver API: {format_cuda_version(driver_version)}")
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"CUDA devices: {device_count}")
        for idx in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(idx)
            name = props["name"].decode("utf-8", errors="replace")
            major = props["major"]
            minor = props["minor"]
            total_mem_gb = props["totalGlobalMem"] / (1024**3)
            print(f"  [{idx}] {name}, compute capability {major}.{minor}, "
                  f"memory {total_mem_gb:.2f} GB")
        cupy_major = int(str(cp.__version__).split(".", 1)[0])
        runtime_major = runtime_version // 1000
        if runtime_major >= 13 and cupy_major < 14:
            print("CUDA check: FAILED CUDA 13 requires a CuPy 14 or newer distribution.")
            return False
        return device_count > 0
    except Exception as exc:
        print(f"CUDA check: FAILED ({type(exc).__name__}: {exc})")
        return False


def _stage(number: int, total: int, title: str, check) -> bool:
    """Run one self-test stage; ``check`` returns (passed, detail) or raises."""
    import time

    start = time.perf_counter()
    try:
        passed, detail = check()
    except Exception as exc:
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start
    print(f"  [{number}/{total}] {title:<46s} {'PASS' if passed else 'FAIL'}  ({detail}; {seconds:.2f} s)",
          flush=True)
    return passed


def run_installation_test() -> bool:
    """Check every GQIS stage with small problems that have exact answers."""
    import numpy as np
    import sympy as sp

    from gqis import compile_lindblad_rhs, mesolve_2D, odesolve_2D

    sx = sp.Matrix([[0, 1], [1, 0]])
    excited = sp.diag(0, 1)
    A, Drive, gamma, dummy = sp.symbols("A Drive gamma dummy", real=True)
    one_point = {dummy: np.zeros(1, dtype=np.float32)}
    rabi = (Drive / 2 * sx, A, [], excited)  # Rabi rotation; initial state is level 0

    def symbolic_ode():
        rhs = compile_lindblad_rhs(*rabi, [A, dummy])
        # At rho = |0><0|, d rho/dt = -i[(A/2) sigma_x, rho] has Im(rho_01)' = A/2, the rest 0.
        at_ground = dict(zip(rhs.state_symbols, (1, 0, 0)))
        drive = rhs.drive_map[Drive] if isinstance(rhs.drive_map, dict) else rhs.drive_map
        values = [float(e.subs(Drive, drive).subs({**at_ground, sp.Symbol("ParX", real=True): 2.0}))
                  for e in rhs.equations]
        error = float(np.max(np.abs(np.array(values) - [0.0, 0.0, 1.0])))
        return (rhs.state_size == 3 and error < 1e-12,
                f"{len(rhs.equations)} real equations, RHS error {error:.0e}")

    def code_generation():
        from gqis.solver import _emit_cuda_rhs

        fragments, _, _ = _emit_cuda_rhs(compile_lindblad_rhs(*rabi, [A, dummy]), False, reuse=False)
        lines = [line for line in fragments["drho_lines"] if "d_rho[" in line]
        return len(lines) == 3, f"{len(lines)} CUDA RHS lines"

    def gpu_solve():
        amplitudes = np.array([1.0, 2.0], dtype=np.float32)
        tlist = np.linspace(0.0, np.pi, 257, dtype=np.float32)
        result, info = mesolve_2D(*rabi, tlist, var_arrays={A: amplitudes, **one_point},
                                  output_mode="final", return_timing_info=True)
        exact = np.sin(amplitudes * np.pi / 2) ** 2  # P(excited) = 1 and 0
        error = float(np.max(np.abs(np.real(result[:, 0]) - exact)))
        return error < 1e-4, (f"Rabi P = sin^2(A t/2), error {error:.1e}; kernel build "
                              f"{info['kernel_build_s']:.2f} s, GPU {info['gpu_kernel_s'] * 1e3:.1f} ms")

    def dissipation():
        rates = np.array([0.5, 2.0], dtype=np.float32)
        tlist = np.linspace(0.0, 1.0, 129, dtype=np.float32)
        pump = sp.sqrt(gamma) * sp.Matrix([[0, 0], [1, 0]])
        result = mesolve_2D(sp.zeros(2), 0, [pump], excited, tlist,
                            var_arrays={gamma: rates, **one_point}, output_mode="final")
        error = float(np.max(np.abs(np.real(result[:, 0]) - (1 - np.exp(-rates)))))
        return error < 1e-5, f"P = 1 - exp(-gamma t), error {error:.1e}"

    def ode_api():
        x, v, w = sp.symbols("x v w", real=True)
        frequencies = np.array([1.0, 2.0], dtype=np.float32)
        result = odesolve_2D([v, -w**2 * x], [x, v], [1, 0], np.linspace(0, 2, 257),
                             var_arrays={w: frequencies})
        error = float(np.max(np.abs(result[:, 0, 0] - np.cos(2 * frequencies))))
        return error < 1e-5, f"harmonic oscillator x = cos(w t), error {error:.1e}"

    def calibration():
        tlist = np.linspace(0.0, np.pi, 5, dtype=np.float32)  # deliberately too coarse
        result, info = mesolve_2D(*rabi, tlist, var_arrays={A: np.array([1.0], np.float32), **one_point},
                                  output_mode="final", calibrate_timesteps=True,
                                  return_timing_info=True)
        steps = info["timestep_calibration"]["steps"]
        error = abs(float(np.real(result[0, 0])) - 1.0)
        return steps > 4 and error < 1e-2, f"4 -> {steps} steps, error {error:.1e}"

    stages = [("Symbolic ODE (Lindblad -> real equations)", symbolic_ode),
              ("CUDA code generation", code_generation),
              ("Kernel compilation and GPU solve", gpu_solve),
              ("Lindblad dissipation", dissipation),
              ("ODE API (odesolve_2D)", ode_api),
              ("Automatic time-step calibration", calibration)]
    print("GQIS self-test (exact answers for small problems):", flush=True)
    passed = sum(_stage(i + 1, len(stages), title, check) for i, (title, check) in enumerate(stages))
    print(f"Installation test: {'PASS' if passed == len(stages) else 'FAILED'} "
          f"({passed}/{len(stages)} stages)")
    return passed == len(stages)


_BENCHMARK_BACKENDS = {
    "QuTiP": "Benchmarks 01 and 02 (qutip_cpu) and Benchmark 03 (QuTiP reference)",
    "Julia DiffEqGPU": "Benchmarks 01-04 (julia_gpu_* solvers)",
    "dynamiqs": "Benchmark 04 dynamiqs comparison (GPU only on Linux or WSL)",
}


def run_benchmarks_test(julia_timeout: float = 900.0) -> int:
    """Try each optional benchmark backend on a problem with an exact answer; returns the count
    that work. A missing backend only disables the benchmarks that use it."""
    import time

    import numpy as np

    def qutip_check():
        import qutip

        rates = 1.0
        result = qutip.mesolve(qutip.qzero(2), qutip.basis(2, 0) * qutip.basis(2, 0).dag(),
                               np.linspace(0, 1, 11),
                               c_ops=[np.sqrt(rates) * qutip.basis(2, 1) * qutip.basis(2, 0).dag()],
                               e_ops=[qutip.basis(2, 1) * qutip.basis(2, 1).dag()])
        error = abs(float(np.real(result.expect[0][-1])) - (1 - np.exp(-1.0)))
        return error < 1e-6, f"QuTiP {qutip.__version__}, decay error {error:.1e}"

    def dynamiqs_check():
        if os.name == "nt":  # JAX has no CUDA build for native Windows
            return False, ("dynamiqs runs on the GPU only on Linux or WSL; "
                           "run this check and Benchmark 04's dynamiqs comparison there")
        import dynamiqs as dq
        import jax
        import jax.numpy as jnp

        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        jump = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        result = dq.mesolve(jnp.zeros((2, 2)), [jump], dq.basis_dm(2, 0), jnp.linspace(0, 1, 11),
                            exp_ops=[dq.basis_dm(2, 1)])
        error = abs(float(jnp.real(result.expects[0, -1])) - (1 - np.exp(-1.0)))
        where = "GPU" if gpus else "CPU only (no JAX GPU)"
        return error < 1e-4 and bool(gpus), f"dynamiqs {dq.__version__} on {where}, decay error {error:.1e}"

    def julia_check():
        julia = shutil.which("julia")
        if julia is None:
            return False, "julia not found in PATH"
        script = ("using DifferentialEquations, DiffEqGPU, CUDA, StaticArrays\n"
                  "f(u, p, t) = @SVector [-p[1] * u[1]]\n"
                  "prob = ODEProblem{false}(f, @SVector[1.0f0], (0.0f0, 1.0f0), @SVector[1.0f0])\n"
                  "ep = EnsembleProblem(prob; prob_func=(pr, i, r) -> remake(pr; p=@SVector[Float32(i)]),"
                  " safetycopy=false)\n"
                  "sol = solve(ep, GPUTsit5(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend());"
                  # dt = 1/64 is exact in FP32, so the stock time accumulation ends exactly at t = 1.
                  " trajectories=4, adaptive=false, dt=0.015625f0, save_everystep=false)\n"
                  "err = maximum(abs(sol[i][end][1] - exp(-Float32(i))) for i in 1:4)\n"
                  'println("ERROR=", err)\n')
        proc = subprocess.run([julia, "--startup-file=no", "-e", script], capture_output=True,
                              text=True, timeout=julia_timeout, check=False)
        match = re.search(r"ERROR=([0-9.eE+-]+)", proc.stdout)
        if proc.returncode != 0 or not match:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["no output"]
            return False, f"Julia exit code {proc.returncode}: {tail[0][:120]}"
        error = float(match.group(1))
        return error < 1e-5, f"GPUTsit5 on the GPU, decay error {error:.1e}"

    checks = {"QuTiP": qutip_check, "Julia DiffEqGPU": julia_check, "dynamiqs": dynamiqs_check}
    print("Optional benchmark backends (exact decay e^(-t) solved by each):", flush=True)
    available = 0
    for name, check in checks.items():
        start = time.perf_counter()
        try:
            passed, detail = check()
        except subprocess.TimeoutExpired:
            passed, detail = False, f"timed out after {julia_timeout:g} s"
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {str(exc)[:120]}"
        available += passed
        seconds = time.perf_counter() - start
        print(f"  {name:<16s} {'PASS' if passed else 'WARNING'}  ({detail}; {seconds:.1f} s)", flush=True)
        if not passed:
            print(f"  {'':<16s} needed only for {_BENCHMARK_BACKENDS[name]}")
    print(f"Benchmark backends: {available}/{len(checks)} available (optional)")
    return available


def run_bounded_installation_test(timeout: float) -> bool:
    """Run the CUDA/NVRTC installation test with a hard timeout."""
    try:
        proc = subprocess.run([sys.executable, "-m", "gqis.check_environment",
                               "--installation-test-worker"],
                              capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            print(exc.stdout.decode(errors="replace")
                  if isinstance(exc.stdout, bytes) else exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr.decode(errors="replace")
                  if isinstance(exc.stderr, bytes) else exc.stderr, end="")
        print(f"Installation test: FAILED (exceeded {timeout:g}s; worker terminated)")
        return False

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GQIS dependencies and CUDA availability.")
    parser.add_argument("--installation-test", "--smoke", dest="installation_test",
                        action="store_true", help="run a small GPU installation test")
    parser.add_argument("--installation-test-timeout", "--smoke-timeout",
                        dest="installation_test_timeout", type=float, default=120.0,
                        help="maximum seconds for the GPU/NVRTC installation test")
    parser.add_argument("--installation-test-worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--check-julia-packages", action="store_true",
                        help="import optional Julia benchmark packages")
    parser.add_argument("--installation-benchmarks-test", action="store_true",
                        help="also test the optional benchmark backends (QuTiP, Julia DiffEqGPU, "
                             "dynamiqs); the first Julia run can take several minutes")
    args = parser.parse_args()

    if args.installation_test_worker:
        return 0 if run_installation_test() else 1
    if args.installation_test_timeout <= 0.0:
        parser.error("--installation-test-timeout must be positive")

    packages_ok = print_package_report()
    print_external_report(check_julia_packages=args.check_julia_packages)
    cuda_ok = print_cuda_report()
    installation_ok = (run_bounded_installation_test(args.installation_test_timeout)
                       if args.installation_test else True)
    if args.installation_benchmarks_test:
        run_benchmarks_test()  # optional backends: warnings only, never a failed check

    passed = packages_ok and cuda_ok and installation_ok
    if passed:
        print("Environment check: PASS")
    else:
        print("Environment check: FAIL")
    if sys.stdin.isatty():
        try:
            input("Press ENTER to exit...")
        except EOFError:
            pass
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
