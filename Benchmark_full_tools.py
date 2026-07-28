"""Shared helpers for benchmark timing sweeps."""

from __future__ import annotations

import csv
from datetime import datetime
import os
from pathlib import Path
import platform
import sys

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FormatStrFormatter
import numpy as np


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


def extrapolate_loglog(history: list[tuple[int, float]], side: int) -> float:
    """Continue the last log10(time) vs log10(number of simulations) slope."""
    valid = [(float(s) ** 2, float(t)) for s, t in history if s > 0 and np.isfinite(t) and t > 0.0]
    if not valid:
        return np.nan
    if len(valid) == 1:
        n0, t0 = valid[-1]
        slope = 1.0  # fallback: time proportional to number of simulations
    else:
        n1, t1 = valid[-2]
        n0, t0 = valid[-1]
        dx = np.log10(n0) - np.log10(n1)
        slope = 1.0 if dx == 0.0 else (np.log10(t0) - np.log10(t1)) / dx
    n = float(side) ** 2
    return 10.0 ** (np.log10(t0) + slope * (np.log10(n) - np.log10(n0)))


def _windows_cpu_name() -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
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

    try:
        import cupy as cp

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

    metadata = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "cpu": cpu,
        "gpu": gpu,
        "os": _os_display_name(),
        "python": sys.version.split()[0],
    }
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


def save_benchmark_csv(rows: list[dict], path: Path, *, metadata: dict[str, str] | None = None) -> None:
    columns = ("side_dimension", "number_of_simulations", "solver", "time_s", "prep_s", "calc_s", "status")
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
                parts.append("" if isinstance(val, float) and not np.isfinite(val) else f"{val:.9g}" if isinstance(val, float) else str(val))
            writer.writerow(parts)
    print(f"Saved full benchmark table: {path}")


def _fmt_time(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value == 0.0:
        return "0"
    if 0.01 <= abs(value) < 1000:
        return f"{value:.3g}"
    return f"{value:.2E}"


TIME_REFERENCE_MARKS = (
    (60.0, "1 minute"),
    (3600.0, "1 hour"),
    (86400.0, "1 day"),
    (604800.0, "1 week"),
    (2592000.0, "1 month"),
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
            ax.text(
                0.012,
                seconds,
                label,
                transform=transform,
                va="center",
                ha="left",
                fontsize=9,
                color="0.25",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )


def plot_benchmark(
    rows: list[dict],
    solvers: tuple[str, ...],
    out_png: Path,
    *,
    title: str,
    show: bool,
    metadata: dict[str, str] | None = None,
    reference_lines: list[dict] | None = None,
) -> None:
    """Plot log-time scaling; extrapolated data use same color with square markers."""
    sides = sorted({int(r["side_dimension"]) for r in rows})
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.3], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    plotted_any = False
    legend_handles = []

    for solver in solvers:
        solver_rows = [r for r in rows if r["solver"] == solver and np.isfinite(r["time_s"])]
        if not solver_rows:
            continue
        solver_rows.sort(key=lambda r: int(r["side_dimension"]))
        measured = [r for r in solver_rows if r["status"] == "measured"]
        extrapolated = [r for r in solver_rows if r["status"] == "extrapolated"]
        line = None
        if measured:
            plotted_any = True
            (line,) = ax.plot(
                [r["side_dimension"] for r in measured],
                [r["time_s"] for r in measured],
                marker="o",
                linewidth=2.0,
                markersize=4,
                label=solver,
            )
            legend_handles.append(line)
        if extrapolated:
            plotted_any = True
            color = line.get_color() if line is not None else None
            ex_x = [r["side_dimension"] for r in extrapolated]
            ex_y = [r["time_s"] for r in extrapolated]
            if measured:
                dashed_x = [measured[-1]["side_dimension"], *ex_x]
                dashed_y = [measured[-1]["time_s"], *ex_y]
                ax.plot(dashed_x, dashed_y, linestyle="--", linewidth=2.0, color=color, label="_nolegend_")
                ax.plot(ex_x, ex_y, linestyle="None", marker="s", markersize=5, color=color, label="_nolegend_")
                continue
            ax.plot(
                ex_x,
                ex_y,
                marker="s",
                linestyle="--",
                linewidth=2.0,
                markersize=5,
                color=color,
                label=f"{solver} extrapolated" if line is None else None,
            )

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
        ref_line = ax.axhline(
            y,
            color=ref.get("color", "0.25"),
            linestyle=ref.get("linestyle", ":"),
            linewidth=float(ref.get("linewidth", 1.6)),
            label=str(ref.get("label", "reference")),
        )
        legend_handles.append(ref_line)

    add_time_reference_marks(ax)

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    col_labels = ["Simulations"] + [f"{s*s:.2E}" for s in sides]
    table_rows = []
    for solver in solvers:
        row = [solver]
        for side in sides:
            match = next((r for r in rows if r["solver"] == solver and int(r["side_dimension"]) == side), None)
            row.append(_fmt_time(match["time_s"]) if match else "")
        table_rows.append(row)
    if table_rows:
        table = ax_tbl.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.55)
    if plotted_any and legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(4, max(1, len(legend_handles))),
        )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.11)

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Saved full benchmark figure: {out_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)
