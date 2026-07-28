"""Example 05: GPU initial-condition sweep and gate-fidelity maps.

This tutorial shows two ideas:

1. The two GPU sweep axes do not have to be Hamiltonian parameters. Here they
   are the initial-state angles ``theta`` and ``phi`` on the Bloch sphere.
2. The same symbolic model can test several gate-driving regimes: rectangular
   Rabi pulses, Gaussian-envelope Rabi pulses, and multiple-passage
   Landau–Zener–Stückelberg–Majorana (LZSM) longitudinal sweeps.

The plotted quantity is the gate error ``1 - fidelity`` on a logarithmic color
scale. A working-point sweep can also average the error over all initial states
for several regimes and work-point numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sympy as sp

from gqis import mesolve_2D


I = 1j


LZSM_WORK_POINTS = {
    "LZSM_2pass": {
        "passages": 2,
        "plzsm": 0.5,
        "A": [
            1.19789508, 3.88118069, 5.39762138, 6.57821628, 7.57893276,
            8.46306151, 9.26373092, 10.00085056, 10.68749029, 11.33277605,
            11.94337802, 12.52434648, 13.07961412, 13.61231423,
        ],
    },
    "LZSM_4pass": {
        "passages": 4,
        "plzsm": (2.0 + np.sqrt(2.0)) / 4.0,
        "A": [
            2.71795476, 8.31542225, 11.45922246, 13.91205535, 15.99388096,
            17.83475558, 19.50288678, 21.03933212, 22.47107949, 23.81699341,
            25.09087439, 26.30317939, 27.46205662, 28.57400177,
        ],
    },
    "LZSM_6pass": {
        "passages": 6,
        "plzsm": 0.9330135,
        "A": [
            4.16844324, 12.61592478, 17.3593446, 21.06170386, 24.20477439,
            26.9844783, 29.50360634, 31.82405204, 33.98650882, 36.01942676,
            37.94362208, 39.77486979, 41.5254634, 43.20520413,
        ],
    },
    "LZSM_8pass": {
        "passages": 8,
        "plzsm": 0.961939,
        "A": [
            5.60093253, 16.88662818, 23.22373825, 28.17070808, 32.37070735,
            36.08534036, 39.45187722, 42.55298126, 45.44300603, 48.15995435,
            50.73163348, 53.17911735, 55.5188296, 57.76386472,
        ],
    },
}


@dataclass(frozen=True)
class GatePlan:
    """Concrete pulse settings passed to the symbolic GPU model."""

    gate: str
    regime: str
    angle: float
    delta: float
    gate_time: float
    drive_x: sp.Expr
    drive_y: sp.Expr
    drive_z: sp.Expr
    label: str


def gate_unitary(gate: str, angle: float = np.pi / 2.0) -> np.ndarray:
    """Return the ideal target unitary for the selected gate."""

    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    if gate == "X":
        return sx
    if gate == "Y":
        return sy
    if gate == "Z":
        return sz
    if gate == "Hadamard":
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    if gate == "Phase":
        return np.array(
            [[np.exp(-1j * angle / 2.0), 0], [0, np.exp(1j * angle / 2.0)]],
            dtype=np.complex128,
        )
    raise ValueError(f"Unsupported gate '{gate}'.")


def rotation_axis_for_gate(gate: str) -> tuple[float, float, float, float]:
    """Map a target gate to a simple rotation axis and angle.

    Returns:
        ``(nx, ny, nz, angle)`` where the Hamiltonian is
        ``H = 0.5 * omega * (nx*sx + ny*sy + nz*sz)``.
    """

    if gate == "X":
        return 1.0, 0.0, 0.0, np.pi
    if gate == "Y":
        return 0.0, 1.0, 0.0, np.pi
    if gate == "Z":
        return 0.0, 0.0, 1.0, np.pi
    if gate == "Hadamard":
        return 1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0), np.pi
    if gate == "Phase":
        return 0.0, 0.0, 1.0, np.nan
    raise ValueError(f"Unsupported gate '{gate}'.")


def gaussian_normalization(gate_time: float, sigma_cover: float) -> tuple[float, float]:
    """Return Gaussian center and sigma for a pulse mostly inside [0, gate_time]."""

    center = 0.5 * gate_time
    sigma = gate_time / max(2.0 * float(sigma_cover), 1.0e-9)
    return center, sigma


def lzsm_work_point(regime: str, work_point_number: int, delta: float) -> tuple[float, float, int]:
    """Return ``(A, w, passages)`` for a multiple-passage LZSM regime."""

    info = LZSM_WORK_POINTS[regime]
    idx = int(work_point_number) % len(info["A"])
    A = float(info["A"][idx])
    plzsm = float(info["plzsm"])
    w = -np.pi / (2.0 * np.log(plzsm)) * delta * delta / A
    return A, w, int(info["passages"])


def make_gate_plan(settings: dict, *, regime: str | None = None, work_point_number: int | None = None) -> GatePlan:
    """Build symbolic drive expressions and timing for one gate/regime pair."""

    gate = settings["gate"]
    regime = settings["regime"] if regime is None else regime
    work_point_number = settings["work_point_number"] if work_point_number is None else work_point_number
    delta = float(settings["delta"])
    omega = float(settings["omega"])
    phase_angle = float(settings["phase_angle"])
    t = sp.Symbol("t", real=True)
    nx, ny, nz, default_angle = rotation_axis_for_gate(gate)
    angle = phase_angle if gate == "Phase" else default_angle

    if regime == "rabi":
        gate_time = angle / omega
        amp = omega
        drive_x = float(nx) * amp
        drive_y = float(ny) * amp
        drive_z = float(nz) * amp
        label = f"Rabi rectangular, {gate}"
    elif regime == "rabi_gaussian":
        gate_time = float(settings["gaussian_time_factor"]) * angle / omega
        center, sigma = gaussian_normalization(gate_time, settings["sigma_cover"])
        # Normalize the Gaussian approximately by its full integral. This keeps
        # the pulse area close to the requested gate angle.
        amp = angle / (np.sqrt(2.0 * np.pi) * sigma)
        envelope = sp.Float(amp) * sp.exp(-((t - sp.Float(center)) ** 2) / (2.0 * sp.Float(sigma) ** 2))
        drive_x = sp.Float(nx) * envelope
        drive_y = sp.Float(ny) * envelope
        drive_z = sp.Float(nz) * envelope
        label = f"Rabi Gaussian, {gate}"
    elif regime in LZSM_WORK_POINTS:
        A, w, passages = lzsm_work_point(regime, work_point_number, delta)
        gate_time = (passages / 2.0) * (2.0 * np.pi / w)
        phase = 0.0 if gate != "Y" else -0.5 * np.pi
        longitudinal = sp.Float(A) * sp.cos(sp.Float(w) * t + sp.Float(phase))
        # LZSM model: constant tunnel coupling plus longitudinal periodic drive.
        drive_x = sp.Float(delta)
        drive_y = 0.0
        drive_z = longitudinal
        label = f"{regime}, WP {work_point_number}, {gate}"
    else:
        raise ValueError(f"Unsupported regime '{regime}'.")

    return GatePlan(
        gate=gate,
        regime=regime,
        angle=angle,
        delta=delta,
        gate_time=float(gate_time),
        drive_x=drive_x,
        drive_y=drive_y,
        drive_z=drive_z,
        label=label,
    )


def build_symbolic_model(plan: GatePlan):
    """Create the Hamiltonian, symbolic initial state, and readout operator for GQIS."""

    theta = sp.Symbol("theta", real=True)
    phi = sp.Symbol("phi", real=True)
    DriveX = sp.Symbol("DriveX", real=True)
    DriveY = sp.Symbol("DriveY", real=True)
    DriveZ = sp.Symbol("DriveZ", real=True)

    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    H = sp.Rational(1, 2) * (DriveX * sx + DriveY * sy + DriveZ * sz)
    drive_expr = {DriveX: plan.drive_x, DriveY: plan.drive_y, DriveZ: plan.drive_z}

    c = sp.cos(theta / 2)
    s = sp.sin(theta / 2)
    rho0 = sp.Matrix(
        [
            [c**2, c * s * sp.exp(-sp.I * phi)],
            [c * s * sp.exp(sp.I * phi), s**2],
        ]
    )
    return H, drive_expr, [], sp.eye(2), rho0, theta, phi


def reduced_vector_to_rho(reduced):
    """Reconstruct a 2x2 density matrix from GQIS ``final_rho`` output."""

    rho00 = float(reduced[0])
    rho01 = complex(float(reduced[1]), float(reduced[2]))
    return np.array([[rho00, rho01], [np.conjugate(rho01), 1.0 - rho00]], dtype=np.complex128)


def target_state(theta: float, phi: float, gate: str, angle: float) -> np.ndarray:
    """Return ideal output state after applying the selected target gate."""

    psi_in = np.array([np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)], dtype=np.complex128)
    return gate_unitary(gate, angle) @ psi_in


def fidelity_with_pure_target(rho, psi_target):
    """Compute F = <psi_target|rho|psi_target> for one density matrix."""

    return float(np.clip(np.real(np.vdot(psi_target, rho @ psi_target)), 0.0, 1.0))


def compute_error_map(final_rho, theta_list, phi_list, plan: GatePlan):
    """Convert final density matrices into log-plot-ready gate-error data."""

    fidelity = np.empty((len(theta_list), len(phi_list)), dtype=np.float64)
    for i, theta in enumerate(theta_list):
        for j, phi in enumerate(phi_list):
            rho = reduced_vector_to_rho(final_rho[i, j])
            psi_target = target_state(float(theta), float(phi), plan.gate, plan.angle)
            fidelity[i, j] = fidelity_with_pure_target(rho, psi_target)

    error = np.clip(1.0 - fidelity, 1.0e-12, 1.0)
    theta_weights = np.sin(theta_list)
    avg_fidelity = float(np.average(np.mean(fidelity, axis=1), weights=theta_weights))
    avg_error = float(np.average(np.mean(error, axis=1), weights=theta_weights))
    return error, avg_fidelity, avg_error


def run_gpu_gate_map(settings: dict, *, regime: str | None = None, work_point_number: int | None = None):
    """Run one GPU initial-condition sweep and return the error map."""

    plan = make_gate_plan(settings, regime=regime, work_point_number=work_point_number)
    H, drive_expr, collapse_ops, mean_operator, rho0, theta, phi = build_symbolic_model(plan)
    theta_list = np.linspace(0.0, np.pi, settings["num_theta"], dtype=np.float32)
    phi_list = np.linspace(0.0, 2.0 * np.pi, settings["num_phi"], dtype=np.float32)
    num_rk4_steps = int(settings["num_steps"])
    tlist = np.linspace(0.0, plan.gate_time, num_rk4_steps + 1, dtype=np.float32)
    dt = float(tlist[1] - tlist[0])
    state_step_updates = len(theta_list) * len(phi_list) * num_rk4_steps

    solver_start = time.perf_counter()
    final_rho = mesolve_2D(
        H,
        drive_expr,
        collapse_ops,
        mean_operator,
        tlist,
        var_arrays=None,
        rho0_var_arrays={theta: theta_list, phi: phi_list},
        rho0=rho0,
        output_mode="final_rho",
        timings=bool(settings["timings"]),
    )
    solver_elapsed = time.perf_counter() - solver_start
    error, avg_fidelity, avg_error = compute_error_map(final_rho, theta_list, phi_list, plan)
    return {
        "plan": plan,
        "theta_list": theta_list,
        "phi_list": phi_list,
        "z_list": np.cos(theta_list),
        "error": error,
        "avg_fidelity": avg_fidelity,
        "avg_error": avg_error,
        "num_time_samples": len(tlist),
        "num_rk4_steps": num_rk4_steps,
        "dt": dt,
        "state_step_updates": state_step_updates,
        "solver_elapsed": solver_elapsed,
    }


def plot_error_density(result: dict):
    """Plot initial-condition dependence as log10 error density."""

    phi_list = result["phi_list"]
    z_list = result["z_list"]
    error = result["error"]
    plan = result["plan"]

    fig, ax = plt.subplots(figsize=(8, 5))
    z_plot = z_list[::-1]
    error_plot = error[::-1, :]
    im = ax.imshow(
        error_plot,
        origin="lower",
        aspect="auto",
        extent=[phi_list[0] / np.pi, phi_list[-1] / np.pi, z_plot[0], z_plot[-1]],
        norm=LogNorm(vmin=max(float(np.min(error)), 1.0e-12), vmax=max(float(np.max(error)), 1.0e-10)),
        cmap="magma",
    )
    ax.set_title(f"{plan.label}: 1 - fidelity, average error={result['avg_error']:.3e}")
    ax.set_xlabel("Initial phase / pi")
    ax.set_ylabel("Initial occupation coordinate Z")
    fig.colorbar(im, ax=ax, label="Gate error 1 - F")
    plt.tight_layout()
    plt.show()


def run_working_point_sweep(settings: dict):
    """Sweep regimes and working points, averaging error over initial states."""

    sweep_start = time.time()
    rows = []
    for regime in settings["compare_regimes"]:
        wp_list = settings["work_point_numbers"] if regime in LZSM_WORK_POINTS else [0]
        for wp in wp_list:
            result = run_gpu_gate_map(settings, regime=regime, work_point_number=wp)
            rows.append((regime, wp, result["plan"].gate_time, result["avg_error"], result["avg_fidelity"]))
            print(
                f"{regime:>14s} wp={wp:2d} time={result['plan'].gate_time:9.4g} "
                f"steps={result['num_rk4_steps']} dt={result['dt']:.3e} "
                f"avg_error={result['avg_error']:.4e} avg_fidelity={result['avg_fidelity']:.8f}"
            )
    print(f"Total working-point sweep calculation time: {time.time() - sweep_start:.3f} s")
    return rows


def plot_working_point_sweep(rows):
    """Plot average gate error versus operation time for each regime."""

    fig, ax = plt.subplots(figsize=(8, 5))
    regimes = list(dict.fromkeys(row[0] for row in rows))
    for regime in regimes:
        data = [row for row in rows if row[0] == regime]
        x = [row[2] for row in data]
        y = [row[3] for row in data]
        labels = [row[1] for row in data]
        ax.plot(x, y, marker="o", linestyle="--", label=regime)
        for xx, yy, wp in zip(x, y, labels):
            ax.annotate(str(wp), (xx, yy), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Operation time")
    ax.set_ylabel("Average gate error 1 - F")
    ax.set_title("GPU gate-fidelity working-point sweep")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.show()


def main():
    """Run the configured gate-error map and optional working-point sweep."""

    settings = user_settings()
    if settings["show_initial_condition_density_plot"]:
        calc_start = time.time()
        result = run_gpu_gate_map(settings)
        calc_elapsed = time.time() - calc_start
        print(
            f"{result['plan'].label}: avg_error={result['avg_error']:.4e}, "
            f"avg_fidelity={result['avg_fidelity']:.8f}, "
            f"min_error={np.min(result['error']):.4e}, max_error={np.max(result['error']):.4e}"
        )
        update_rate = result["state_step_updates"] / max(result["solver_elapsed"], 1.0e-15)
        print(
            f"Time grid: {result['num_time_samples']} samples, "
            f"{result['num_rk4_steps']} RK4 steps, dt={result['dt']:.6e}"
        )
        print(
            f"GPU workload: {result['state_step_updates']:.6e} trajectory-step updates, "
            f"solver call={result['solver_elapsed']:.3f} s, "
            f"end-to-end rate={update_rate:.6e} updates/s"
        )
        print(f"Initial-condition density calculation time: {calc_elapsed:.3f} s")
        plot_error_density(result)

    if settings["run_working_point_sweep"]:
        rows = run_working_point_sweep(settings)
        plot_working_point_sweep(rows)


def user_settings() -> dict:
    """User-editable parameters for gate testing."""

    return {
        # Gate to test: "X", "Y", "Z", "Hadamard", or "Phase".
        "gate": "X",
        "phase_angle": np.pi / 2.0,
        # Regime for the initial-condition density plot:
        # "rabi", "rabi_gaussian", "LZSM_2pass", "LZSM_4pass", "LZSM_6pass", "LZSM_8pass".
        "regime": "LZSM_2pass",
        "work_point_number": 2,
        # Regimes and working points for averaged comparison curves.
        "compare_regimes": ["rabi", "rabi_gaussian", "LZSM_2pass", "LZSM_4pass", "LZSM_6pass", "LZSM_8pass"],
        "work_point_numbers": list(range(0, 6)),
        # Initial-state grid. Increase for publication-quality averages.
        "num_theta": int(400),
        "num_phi": int(800),
        # Fixed RK4 integration intervals; the generated tlist has num_steps + 1 samples.
        # Increase if non-finite output or poor convergence appears.
        "num_steps": 1200,
        # Rabi settings.
        "omega": 1.0,
        "gaussian_time_factor": 4.0,
        "sigma_cover": 2.7,
        # LZSM energy-gap scale; all frequencies and energies use the same units.
        "delta": 1.0,
        # Display switches.
        "show_initial_condition_density_plot": True,
        "run_working_point_sweep": False,
        "timings": True,
    }


if __name__ == "__main__":
    main()
