"""Shared helpers for benchmark timing sweeps."""

from __future__ import annotations

import csv
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from sympy.printing.julia import JuliaCodePrinter


class _JuliaFloat32Printer(JuliaCodePrinter):
    """Emit Julia floating-point literals that remain in FP32 arithmetic."""

    def _print_Float(self, expr):
        text = super()._print_Float(expr)
        if "e" in text.lower():
            mantissa, exponent = text.lower().split("e", 1)
            return f"{mantissa}f{int(exponent)}"
        return f"{text}f0"


_JULIA_FLOAT32_PRINTER = _JuliaFloat32Printer()


def sympy_to_julia_fp32(expr) -> str:
    """Print a scalar SymPy expression as scalar Julia Float32 code."""
    code = _JULIA_FLOAT32_PRINTER.doprint(expr)
    for broadcast_op, scalar_op in ((".^", "^"), (".*", "*"), ("./", "/"),
                                    (".+", "+"), (".-", "-")):
        code = code.replace(broadcast_op, scalar_op)
    return code


def benchmark_sides(min_side: int, max_side: int) -> list[int]:
    """Powers-of-two side dimensions in [min_side, max_side]."""
    if min_side <= 0 or max_side <= 0 or min_side > max_side:
        raise ValueError("Expected 0 < min_side <= max_side.")
    side = 1
    while side < min_side:
        side *= 2
    sides = []
    while side <= max_side:
        sides.append(side)
        side *= 2
    return sides


def parse_solver_list(text: str, valid_solvers: set[str]) -> tuple[str, ...]:
    solvers = tuple(s.strip() for s in text.split(",") if s.strip())
    unknown = [s for s in solvers if s not in valid_solvers]
    if unknown:
        raise ValueError(f"Unknown solver(s): {unknown}. Valid solvers: {sorted(valid_solvers)}")
    return solvers


def benchmark_output_path(path, *, script_dir) -> Path:
    """Resolve relative generated-output paths inside ``Benchmarks/results``."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(script_dir).resolve() / "results" / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_accuracy_dividers(path, *, expected_problem: str, expected_output: str = "mean",
                           script_dir: Path | None = None) -> dict[str, float]:
    """Load a Benchmark-03 divider file for the expected model and output."""
    requested = Path(path).expanduser()
    candidates = [requested] if requested.is_absolute() else [Path.cwd() / requested]
    if not requested.is_absolute() and script_dir is not None:
        candidates += [Path(script_dir) / "results" / requested,
                       Path(script_dir) / requested, Path(script_dir).parent / requested]
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f"Accuracy-divider file not found: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("format") != "gqis_accuracy_dividers_v1":
        raise ValueError(f"Unsupported accuracy-divider format: {resolved}")
    if payload.get("problem") != expected_problem:
        raise ValueError(f"Accuracy-divider file is for {payload.get('problem')}, not "
                         f"{expected_problem}: {resolved}")
    saved_output = payload.get("comparison_output", "mean")
    if saved_output != expected_output:
        raise ValueError(f"Accuracy-divider file compares {saved_output!r} output, not "
                         f"{expected_output!r}: {resolved}")
    dividers = {str(name): float(value)
                for name, value in payload.get("solver_dividers", {}).items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in dividers.values()):
        raise ValueError(f"Accuracy-divider file contains an invalid divider: {resolved}")
    return dividers


def solver_time_grid_label(solver: str, cfg) -> str:
    """Describe the actual fixed-step or adaptive-output grid for a run."""
    frequency = float(cfg.w if hasattr(cfg, "w") else cfg.w_abs)
    periods = float(cfg.tlist[-1] - cfg.tlist[0]) * frequency / (2.0 * np.pi)
    per_period = cfg.num_steps / periods if periods > 0 else 0.0
    label = "output_intervals" if solver in {"qutip_cpu", "python_ode_cpu"} else "steps"
    return f"{label}={cfg.num_steps} {label}_per_period={per_period:g}"


def accuracy_divider_for_solver(dividers: dict[str, float], solver: str) -> float:
    """Return a calibrated divider or the documented GPU/CPU fallback."""
    aliases = {"gpu": "gqis_rk4", "julia_gpu": "julia_gpu_fp32"}
    key = aliases.get(solver, solver)
    default = 10.0 if solver in {"python_cpu", "python_ode_cpu", "qutip_cpu"} else 1.0
    return float(dividers.get(key, default))


def extrapolate_loglog(history: list[tuple[int, float]], side: int, *, slope_points: int = 2) -> float:
    """Continue an averaged recent log-log slope from the last measured point."""
    valid = [(float(s)**2, float(t)) for s, t in history if s > 0 and np.isfinite(t) and t > 0.0]
    if not valid:
        return np.nan
    n0, t0 = valid[-1]
    if len(valid) == 1:
        slope = 1.0  # fallback: time proportional to number of simulations
    else:
        recent = valid[-max(2, int(slope_points)):]
        log_n = np.log10([point[0] for point in recent])
        log_t = np.log10([point[1] for point in recent])
        dx = np.diff(log_n)
        segment_slopes = np.divide(np.diff(log_t), dx, out=np.full_like(dx, np.nan), where=dx != 0.0)
        finite_slopes = segment_slopes[np.isfinite(segment_slopes)]
        slope = float(np.mean(finite_slopes)) if finite_slopes.size else 1.0
    n = float(side)**2
    return 10.0**(np.log10(t0) + slope * (np.log10(n) - np.log10(n0)))


def should_extrapolate_next(history: list[tuple[int, float]], time_limit: float, *,
                            threshold_fraction: float = 0.5) -> bool:
    """Skip the next side doubling when the last timing ratio exceeds 90% of fourfold."""
    if len(history) < 2 or time_limit <= 0.0:
        return False
    (side_a, time_a), (side_b, time_b) = history[-2:]
    if time_b <= threshold_fraction * time_limit:
        return False
    if side_a <= 0 or time_a <= 0.0 or side_b != 2 * side_a or not np.isfinite(time_b):
        return False
    time_growth = time_b / time_a
    return time_growth > 0.9 * 4.0


def terminate_process_tree(proc, timeout_s: float = 5.0) -> None:
    """Terminate a benchmark worker and all child processes it launched."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                       text=True, check=False)
    else:
        proc.terminate()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.kill()
        proc.join()


def _windows_cpu_name() -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(value).strip()
    except Exception:
        return ""


def _windows_registry_value(path: str, name: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            value, _ = winreg.QueryValueEx(key, name)
        return str(value).strip()
    except Exception:
        return ""


def _os_display_name() -> str:
    if os.name != "nt":
        return f"{platform.system()} {platform.release()} ({platform.machine()})"

    key = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    product = _windows_registry_value(key, "ProductName")
    display = _windows_registry_value(key, "DisplayVersion")
    build = _windows_registry_value(key, "CurrentBuildNumber")
    ubr = _windows_registry_value(key, "UBR")

    if product.startswith("Windows 10") and build.isdigit() and int(build) >= 22000:
        product = product.replace("Windows 10", "Windows 11", 1)

    if not product:
        return f"{platform.system()} {platform.release()} ({platform.machine()})"

    suffix = []
    if display:
        suffix.append(display)
    if build:
        suffix.append(f"build {build}{'.' + ubr if ubr else ''}")
    return f"{product} ({', '.join(suffix)})" if suffix else product


def collect_equipment_info() -> dict[str, str]:
    """Best-effort hardware/software metadata for saved benchmark baselines."""
    cpu = _windows_cpu_name() or platform.processor() or platform.uname().processor or "unknown CPU"
    gpu = "unknown GPU"
    gpu_vram_gb = ""
    cuda_runtime = ""
    cupy_version = ""

    try:
        import cupy as cp

        cupy_version = cp.__version__
        device_id = cp.cuda.runtime.getDevice()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props.get("name", b"")
        gpu = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
        total_mem_gb = props.get("totalGlobalMem", 0) / (1024**3)
        if total_mem_gb > 0:
            gpu_vram_gb = f"{total_mem_gb:.2f}"
        version = cp.cuda.runtime.runtimeGetVersion()
        cuda_runtime = f"{version // 1000}.{(version % 1000) // 10}"
    except Exception as exc:
        gpu = f"unavailable ({type(exc).__name__})"

    metadata = {"timestamp_local": datetime.now().isoformat(timespec="seconds"),
                "cpu": cpu, "gpu": gpu, "os": _os_display_name(),
                "python": sys.version.split()[0], "numpy": np.__version__}
    try:
        from gqis import __version__ as gqis_version

        metadata["gqis"] = gqis_version
    except ImportError:
        pass
    for distribution in ("sympy", "matplotlib", "scipy", "qutip"):
        try:
            metadata[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    if cupy_version:
        metadata["cupy"] = cupy_version
    if gpu_vram_gb:
        metadata["gpu_vram_gb"] = gpu_vram_gb
    if cuda_runtime:
        metadata["cuda_runtime"] = cuda_runtime
    return metadata


def format_equipment_label(metadata: dict[str, str] | None) -> str:
    if not metadata:
        return ""
    label = f"CPU: {metadata.get('cpu', 'unknown')} | GPU: {metadata.get('gpu', 'unknown')}"
    vram = metadata.get("gpu_vram_gb")
    return f"{label} | VRAM: {vram} GB" if vram else label


def print_equipment_info(metadata: dict[str, str] | None = None) -> dict[str, str]:
    """Print benchmark hardware and return the metadata used."""
    metadata = metadata or collect_equipment_info()
    print(f"Benchmark equipment: {format_equipment_label(metadata)}")
    return metadata


def save_benchmark_csv(rows: list[dict], path: Path, *,
                       metadata: dict[str, str] | None = None) -> None:
    columns = ("side_dimension", "number_of_simulations", "solver", "time_s", "prep_s", "calc_s",
               "status",
               )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if metadata:
            for key, value in metadata.items():
                writer.writerow([f"# {key}", value])
            writer.writerow([])
        writer.writerow(columns)
        for row in rows:
            parts = []
            for col in columns:
                val = row.get(col, np.nan)
                parts.append("" if isinstance(val, float) and not np.isfinite(val) else
                             f"{val:.9g}" if isinstance(val, float) else str(val))
            writer.writerow(parts)
    print(f"Saved full benchmark table: {path}")



def read_benchmark_csv(path: Path) -> tuple[list[dict], dict[str, str]]:
    """Read a benchmark CSV written by :func:`save_benchmark_csv`."""
    path = Path(path)
    metadata: dict[str, str] = {}
    rows: list[dict] = []
    header = None
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for values in csv.reader(stream):
            if not values:
                continue
            if values[0].startswith("# "):
                metadata[values[0][2:]] = values[1] if len(values) > 1 else ""
                continue
            if header is None:
                header = values
                continue
            raw = dict(zip(header, values))
            side = int(raw["side_dimension"])
            rows.append({
                "side_dimension": side,
                "number_of_simulations": int(raw.get("number_of_simulations") or side * side),
                "solver": str(raw["solver"]),
                "time_s": float(raw["time_s"]) if raw.get("time_s") else np.nan,
                "prep_s": float(raw["prep_s"]) if raw.get("prep_s") else np.nan,
                "calc_s": float(raw["calc_s"]) if raw.get("calc_s") else np.nan,
                "status": raw.get("status") or "measured",
            })
    if header is None or not rows:
        raise ValueError(f"Benchmark CSV contains no timing rows: {path}")
    return rows, metadata


def resolve_benchmark_csv(path: str | Path, *, script_dir: Path) -> Path:
    """Resolve an existing benchmark CSV from cwd/script/results locations."""
    requested = Path(path).expanduser()
    candidates = ([requested] if requested.is_absolute() else
                  [Path.cwd() / requested, Path(script_dir) / "results" / requested,
                   Path(script_dir) / requested, Path(script_dir).parent / requested])
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")
    return resolved


def merge_benchmark_rows(base_rows: list[dict], replacement_rows: list[dict]) -> list[dict]:
    """Replace rows with the same (solver, side_dimension), preserving solver order."""
    solver_order = list(dict.fromkeys(str(row["solver"]) for row in base_rows))
    merged = {(str(row["solver"]), int(row["side_dimension"])): dict(row)
              for row in base_rows}
    for row in replacement_rows:
        solver = str(row["solver"])
        if solver not in solver_order:
            solver_order.append(solver)
        merged[(solver, int(row["side_dimension"]))] = dict(row)
    solver_rank = {name: index for index, name in enumerate(solver_order)}
    return sorted(merged.values(),
                  key=lambda row: (solver_rank[str(row["solver"])], int(row["side_dimension"])))


def refresh_benchmark_extrapolations(rows: list[dict], *, solvers=None,
                                     slope_points: int = 2) -> None:
    """Refresh only extrapolated rows from the currently measured timing history."""
    selected = set(solvers) if solvers is not None else None
    for solver in dict.fromkeys(str(row["solver"]) for row in rows):
        if selected is not None and solver not in selected:
            continue
        solver_rows = sorted((row for row in rows if str(row["solver"]) == solver),
                             key=lambda row: int(row["side_dimension"]))
        history = [(int(row["side_dimension"]), float(row["time_s"]))
                   for row in solver_rows if row.get("status") == "measured"
                   and np.isfinite(float(row.get("time_s", np.nan)))
                   and float(row["time_s"]) > 0.0]
        for row in solver_rows:
            if row.get("status") != "extrapolated":
                continue
            side = int(row["side_dimension"])
            preceding = [(s, elapsed) for s, elapsed in history if s < side]
            following = [(s, elapsed) for s, elapsed in history if s > side]
            anchors = ([preceding[-1], following[0]] if preceding and following else preceding)
            if not anchors:
                continue
            row["time_s"] = extrapolate_loglog(anchors, side, slope_points=slope_points)


def run_calibrated_csv_update(calibration_file: str | Path, *, problem: str,
                              target_csv: str | Path, runs, script_dir: Path,
                              settings_file: str | Path | None = None,
                              julia_cmd: str = "julia", time_limit: float = 800.0,
                              backup_csv: bool = True,
                              refresh_extrapolated_points: bool = True,
                              update_plot: bool = True, show_plot: bool = True,
                              show_progress: bool = True, fallback_base_steps: int = 256,
                              solver_labels: dict | None = None,
                              show_startup_times: bool = True,
                              show_gpu_preparation: bool = True,
                              show_side_dimensions: bool = True,
                              show_results_table: bool = True,
                              dpi: int = 300) -> Path:
    """Rerun selected calibrated benchmark ranges and update an existing CSV in place.

    Each item in ``runs`` specifies ``solver``, ``min_side`` and ``max_side``.
    CPU and Julia measurements are saved to the target CSV after each completed point.
    Only newly *measured* rows are merged. Existing rows outside the requested ranges are
    untouched; existing extrapolated rows for an updated solver can optionally be refreshed.
    """
    from Benchmark_accuracy_calibration import run_calibrated_sweep

    target_path = resolve_benchmark_csv(target_csv, script_dir=Path(script_dir))
    base_rows, metadata = read_benchmark_csv(target_path)
    run_specs = tuple(runs or ())
    if not run_specs:
        raise ValueError("CSV update requested but no benchmark update runs were configured.")

    replacements: list[dict] = []
    updated_solvers: list[str] = []
    range_labels: list[str] = []
    backup_path = None
    checkpoint_rows = list(base_rows)

    def ensure_backup():
        nonlocal backup_path
        if backup_csv and backup_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = target_path.with_name(
                f"{target_path.stem}_backup_{stamp}{target_path.suffix}")
            shutil.copy2(target_path, backup_path)
            print(f"Backed up previous CSV: {backup_path}")

    def save_checkpoint(row):
        nonlocal checkpoint_rows
        ensure_backup()
        checkpoint_rows = merge_benchmark_rows(checkpoint_rows, [row])
        # Replace the destination only after the complete CSV has been written.
        with tempfile.NamedTemporaryFile(dir=target_path.parent, suffix=".csv",
                                         delete=False) as handle:
            pending_path = Path(handle.name)
        try:
            save_benchmark_csv(checkpoint_rows, pending_path, metadata=metadata)
            pending_path.replace(target_path)
        finally:
            pending_path.unlink(missing_ok=True)
        print(f"Saved {row['solver']} side={row['side_dimension']} to {target_path}")

    with tempfile.TemporaryDirectory(prefix="gqis_benchmark_update_") as temp_dir:
        for index, spec in enumerate(run_specs):
            solver = str(spec["solver"])
            min_side = int(spec["min_side"])
            max_side = int(spec["max_side"])
            if min_side <= 0 or max_side < min_side:
                raise ValueError(f"Invalid update range for {solver}: {min_side}..{max_side}")
            local_limit = float(spec.get("time_limit_s", time_limit))
            temp_stem = Path(temp_dir) / f"update_{index}_{solver}"
            print(f"\nCSV partial update: {solver}, side={min_side}..{max_side}")
            result = run_calibrated_sweep(
                calibration_file,
                problem=problem,
                settings_file=settings_file,
                solvers=(solver,),
                min_side=min_side,
                max_side=max_side,
                time_limit=local_limit,
                output_filename=str(temp_stem),
                julia_cmd=julia_cmd,
                show_plot=False, show_progress=show_progress,
                fallback_base_steps=fallback_base_steps,
                on_measured=(save_checkpoint if (solver in
                             {"python_cpu", "python_ode_cpu", "qutip_cpu"}
                             or solver.startswith("julia_")) else None),
            )

            if isinstance(result, list):
                generated_rows = result
            else:
                generated_csv = Path(f"{temp_stem}.csv")
                if not generated_csv.is_file():
                    raise RuntimeError(
                        f"Calibrated update did not return rows or create {generated_csv}.")
                generated_rows, _ = read_benchmark_csv(generated_csv)

            expected_sides = set(benchmark_sides(min_side, max_side))
            measured_rows = [dict(row) for row in generated_rows
                             if str(row.get("solver")) == solver
                             and int(row.get("side_dimension", -1)) in expected_sides
                             and row.get("status") == "measured"
                             and np.isfinite(float(row.get("time_s", np.nan)))]
            if not measured_rows:
                raise RuntimeError(f"No measured rows were produced for {solver} in "
                                   f"side={min_side}..{max_side}.")
            measured_sides = {int(row["side_dimension"]) for row in measured_rows}
            missing = sorted(expected_sides - measured_sides)
            if missing:
                print(f"{solver}: keeping existing CSV values for unmeasured sides: {missing}")
            replacements.extend(measured_rows)
            checkpoint_rows = merge_benchmark_rows(checkpoint_rows, measured_rows)
            if solver not in updated_solvers:
                updated_solvers.append(solver)
            range_labels.append(f"{solver}:{min_side}-{max_side}")

    merged_rows = merge_benchmark_rows(base_rows, replacements)
    if refresh_extrapolated_points:
        refresh_benchmark_extrapolations(merged_rows, solvers=updated_solvers,
                                         slope_points=2)

    ensure_backup()

    metadata = dict(metadata)
    metadata["partial_update_timestamp_local"] = datetime.now().isoformat(timespec="seconds")
    metadata["partial_update_ranges"] = ";".join(range_labels)
    save_benchmark_csv(merged_rows, target_path, metadata=metadata)
    print(f"Updated measured rows: {len(replacements)}")

    if update_plot:
        solver_order = tuple(dict.fromkeys(str(row["solver"]) for row in merged_rows))
        levels = metadata.get("system_levels", "unknown")
        title = (f"Calculation time scaling for different numerical approaches "
                 f"({levels}-level system)" if levels != "unknown" else
                 "Calculation time scaling for different numerical approaches")
        out_png = target_path.with_suffix(".png")
        plot_benchmark(
            merged_rows, solver_order, out_png, title=title, show=show_plot,
            metadata=metadata, show_startup_times=show_startup_times,
            show_gpu_preparation=show_gpu_preparation, solver_labels=solver_labels,
            show_side_dimensions=show_side_dimensions, show_table=show_results_table,
            dpi=int(dpi),
        )
    return target_path


def _fmt_time(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value == 0.0:
        return "0"
    if 0.01 <= abs(value) < 1000:
        return f"{value:.3g}"
    return f"{value:.2E}"


TIME_REFERENCE_MARKS = ((60.0, "1 minute"), (3600.0, "1 hour"), (86400.0, "1 day"),
                        (604800.0, "1 week"), (2592000.0, "1 month"),
                        )


def add_time_reference_marks(ax: plt.Axes) -> None:
    """Annotate common wall-time levels when they fall inside the visible y-range."""
    ymin, ymax = ax.get_ylim()
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymin <= 0.0:
        return

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for seconds, label in TIME_REFERENCE_MARKS:
        if ymin < seconds < ymax:
            ax.axhline(seconds, color="0.88", linewidth=0.9, zorder=0)
            ax.text(0.012, seconds, label, transform=transform, va="center", ha="left", fontsize=9,
                    color="0.25",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75,
                          "pad": 1.5})


def plot_benchmark(rows: list[dict], solvers: tuple[str, ...], out_png: Path, *, title: str,
                   show: bool, metadata: dict[str, str] | None = None,
                   reference_lines: list[dict] | None = None, show_table: bool = True,
                   dpi: int = 160, show_startup_times: bool = True,
                   show_gpu_preparation: bool = True,
                   preparation_overrides_s: dict | None = None,
                   solver_labels: dict | None = None, show_side_dimensions: bool = True) -> None:
    """Plot log-time scaling; extrapolated data use same color with square markers."""
    sides = sorted({int(r["side_dimension"]) for r in rows})
    fig = plt.figure(figsize=(11, 8.5 if show_table else 6.8))
    if show_table:
        gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.3], hspace=0.15)
        ax = fig.add_subplot(gs[0])
    else:
        ax = fig.add_subplot(111)
    plotted_any = False
    legend_handles = []
    partial_totals = []
    labels = {"gpu": "GQIS(RK4)", "gqis_rk4": "GQIS(RK4)", "qutip_cpu": "QuTiP(CPU)",
              "julia_gpu": "Julia(GPU)", "julia_gpu_fp64": "Julia(GPU FP64)",
              "julia_gpu_fp32": "Julia(GPU FP32 stock)",
              "julia_gpu_fp32_opt": "Julia(GPU FP32 mixed time)",
              "julia_gpu_fp32_fopt": "Julia(GPU FP32)"}
    labels.update(solver_labels or {})

    for solver in solvers:
        solver_rows = [r for r in rows if r["solver"] == solver
                       and r["status"] in {"measured", "extrapolated"}
                       and np.isfinite(r["time_s"])]
        if not solver_rows:
            continue
        solver_rows.sort(key=lambda r: int(r["side_dimension"]))
        measured = [r for r in solver_rows if r["status"] == "measured"]
        extrapolated = [r for r in solver_rows if r["status"] == "extrapolated"]
        plotted_any = True
        marker = "o" if measured else "s"
        linestyle = "-" if measured else "--"
        handle = ax.plot([], [], marker=marker, linestyle=linestyle, linewidth=2.0,
                         markersize=4, label=f"{labels.get(solver, solver)} "
                         f"{'tot time' if solver == 'qutip_cpu' else 'calc time'}")[0]
        color = handle.get_color()
        legend_handles.append(handle)
        is_gqis = solver == "gpu" or solver.startswith("gqis_")
        if show_startup_times and (solver.startswith("julia") or (is_gqis and show_gpu_preparation)):
            recorded = [float(r.get("prep_s", np.nan)) for r in measured]
            recorded = [p for p in recorded if np.isfinite(p) and p >= 0]
            fallback = float(np.mean(recorded)) if recorded else np.nan
            description = "solve + preparation/overhead"
            if is_gqis:
                fallback = float((metadata or {}).get(f"gpu_startup_s_{solver}", "nan"))
                description = "solve + startup (estimate)"
                if not np.isfinite(fallback):
                    fallback = float((metadata or {}).get("gpu_first_rhs_stage_s", "nan"))
                    description = "solve + recorded RHS prep only"
            override = (preparation_overrides_s or {}).get(solver)
            if override is not None:
                fallback = float(override)
                description = "solve + supplied preparation"
            total_x, total_y = [], []
            for row in solver_rows:
                preparation = (fallback if is_gqis or override is not None else
                               float(row.get("prep_s", np.nan)))
                if not np.isfinite(preparation):
                    preparation = fallback
                if np.isfinite(preparation) and preparation >= 0:
                    total_x.append(row["side_dimension"])
                    total_y.append(float(row["time_s"]) + preparation)
            if total_x:
                partial = description == "solve + recorded RHS prep only"
                if partial:
                    partial_totals.append(labels.get(solver, solver))
                legend_handles.append(ax.plot(total_x, total_y, color=color, linestyle=":",
                    linewidth=1.8, label=f"{labels.get(solver, solver)} tot time")[0])
            else:
                print(f"{solver}: startup curve omitted; preparation was not recorded. "
                      "A matching measured value can be supplied with preparation_overrides_s.")
        for left, right in zip(solver_rows, solver_rows[1:]):
            segment_style = "-" if left["status"] == right["status"] == "measured" else "--"
            ax.plot([left["side_dimension"], right["side_dimension"]],
                    [left["time_s"], right["time_s"]], linestyle=segment_style,
                    linewidth=2.0, color=color, label="_nolegend_")
        if measured:
            ax.plot([row["side_dimension"] for row in measured],
                    [row["time_s"] for row in measured], linestyle="None", marker="o",
                    markersize=4, color=color, label="_nolegend_")
        if extrapolated:
            ax.plot([row["side_dimension"] for row in extrapolated],
                    [row["time_s"] for row in extrapolated], linestyle="None", marker="s",
                    markersize=5, color=color, label="_nolegend_")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2E"))
    label = format_equipment_label(metadata)
    ax.set_title(f"{title}\n{label}" if label else title, fontsize=15, color="0.35")
    ax.set_ylabel("Time [s]", fontsize=16)
    ax.set_xlabel("Side dimension")
    ax.grid(True, axis="y", which="major", color="0.85")
    ax.set_xticks(sides)
    ax.set_xticklabels([str(s) for s in sides])

    for ref in reference_lines or []:
        y = float(ref.get("y", np.nan))
        if not np.isfinite(y) or y <= 0.0:
            continue
        ref_line = ax.axhline(y, color=ref.get("color",
                                               "0.25"), linestyle=ref.get("linestyle", ":"),
                              linewidth=float(ref.get("linewidth", 1.6)),
                              label=str(ref.get("label", "reference")))
        legend_handles.append(ref_line)

    add_time_reference_marks(ax)

    if show_table:
        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.axis("off")
        col_labels = ["Simulations"] + [f"{s * s:.2E}" for s in sides]
        table_rows = []
        if show_side_dimensions:
            table_rows.append(col_labels)
            col_labels = ["Side dimension"] + [str(s) for s in sides]
        for solver in solvers:
            row = [labels.get(solver, solver)]
            for side in sides:
                match = next((r for r in rows
                              if r["solver"] == solver
                              and int(r["side_dimension"]) == side), None)
                row.append(_fmt_time(match["time_s"]) if match else "")
            table_rows.append(row)
        if table_rows:
            table = ax_tbl.table(cellText=table_rows, colLabels=col_labels, loc="center",
                                 cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.55)
            # Give the descriptive first column more room than numeric columns.
            ncols = len(col_labels)
            if ncols > 1:
                first_width = min(0.14, 1.7 / ncols)
                data_width = (1.0 - first_width) / (ncols - 1)
                for (row_index, col_index), cell in table.get_celld().items():
                    cell.set_width(first_width if col_index == 0 else data_width)
                    if row_index == 0:
                        cell.set_height(cell.get_height() * 1.12)
    
    legend_y = 0.035
    if plotted_any and legend_handles:
        fig.legend( loc="upper center",bbox_to_anchor=(0.5, 0.925),
                   ncol=min(4, max(1, len(legend_handles))), fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92,
                        bottom=0.0)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"Saved full benchmark figure: {out_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)
