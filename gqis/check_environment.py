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
                                                                            "CUDA backend"),
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


def run_installation_test() -> bool:
    """Run a tiny two-level GPU solve to verify NVRTC/kernel execution."""
    try:
        import numpy as np
        import sympy as sp

        from gqis import mesolve_2D

        N = 2
        Delta, eps, Drive = sp.symbols("Delta eps Drive", real=True)
        A, t = sp.symbols("A t", real=True)
        gamma1_s, gamma2_s = sp.symbols("gamma1_s gamma2_s", real=True, nonnegative=True)

        sz = sp.Matrix([[1, 0], [0, -1]])
        sm = sp.Matrix([[0, 1], [0, 0]])
        sx = sp.Matrix([[0, 1], [1, 0]])
        mean_op = sp.zeros(N)
        mean_op[1, 1] = 1

        H = 0.5 * Delta * sx + 0.5 * Drive * sz
        drive_expr = eps + A * sp.sin(1.14 * t)
        col_ops = [sp.sqrt(gamma1_s) * sm, sp.sqrt(gamma2_s) * sz]

        eps_list = np.linspace(-0.5, 0.5, 4, dtype=np.float32)
        A_list = np.linspace(0.0, 0.5, 4, dtype=np.float32)
        tlist = np.linspace(0.0, 2.0, 32, dtype=np.float32)

        out = mesolve_2D(H, drive_expr, col_ops, mean_op, tlist,
                         var_arrays={eps: eps_list, A: A_list},
                         const_values={Delta: 1.0, gamma1_s: 0.01, gamma2_s: 0.01},
                         timings=True)
        arr = np.asarray(out)
        if arr.shape != (len(eps_list), len(A_list)):
            print(f"Installation test: FAILED unexpected shape {arr.shape}")
            return False
        if not np.all(np.isfinite(arr)):
            print("Installation test: FAILED non-finite output")
            return False
        print(f"Installation test: PASS shape={arr.shape}")
        return True
    except Exception as exc:
        print(f"Installation test: FAILED ({type(exc).__name__}: {exc})")
        return False


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
