"""Example 02: four-level interferogram (minimal GPU tutorial)."""

import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from gqis import mesolve_2D


def gpu_time_evolution(A_list, eps_list, tlist, delta, w, gamma1, gamma2, kappa, Ap, g1, wr2,
                       averaging_skip_fraction=0.0):
    """Build symbolic model and run GPU solver for one interferogram."""
    # Convert all numerical grids to single precision. The default GPU kernel is
    # FP32 for maximum performance of consumer GPU,
    # so using float32 avoids unnecessary host-to-device conversion work.
    A_list = np.asarray(A_list, dtype=np.float32)  # amplitude sweep, vertical plot axis
    eps_list = np.asarray(eps_list, dtype=np.float32)  # detuning sweep, horizontal plot axis
    tlist = np.asarray(tlist, dtype=np.float32)  # fixed-step integration time grid

    # This example couples a two-level qubit to a two-level truncated resonator.
    N = 4

    # Build symbolic basis/operators ------------------------------------
    # `eps` and `A` are sweep variables; GQIS maps them to ParX/ParY.
    # `Drive` is a placeholder inside the Hamiltonian and is replaced by the
    # explicit time-dependent expression `eps + A*cos(w*t)` during codegen.
    eps, Drive = sp.symbols("eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)

    sz = sp.Matrix([[1, 0], [0, -1]])  # qubit sigma_z
    sp_raise = sp.Matrix([[0, 1], [0, 0]])  # raising operator in this basis convention
    sp_lower = sp.Matrix([[0, 0], [1, 0]])  # lowering operator in this basis convention
    qeye = sp.eye(N - 2)  # identity operator in the resonator subspace

    # Lift qubit and resonator operators into the full tensor-product space.
    sm1 = sp.kronecker_product(sp_lower, qeye)  # qubit lowering operator
    sz1 = sp.kronecker_product(sz, qeye)  # qubit sigma_z operator
    a1 = sp.kronecker_product(qeye, sp_raise)  # resonator annihilation-like operator

    # Hamiltonian + drive ------------------------------------------------
    # h0 is time independent. ht depends on the symbolic Drive placeholder, so
    # the generated CUDA RHS can evaluate it at every RK4 stage.
    hq11 = sz1 / 2
    hp = Ap * (a1.H + a1)  # weak probe term
    hc11u = g1 * delta * (sm1.H * a1 + sm1 * a1.H)
    h0 = hp
    instantaneous_gap = sp.sqrt(delta**2 + Drive**2)
    ht = hc11u / instantaneous_gap + hq11 * (instantaneous_gap - wr2)
    H = h0 + ht
    Drive = eps + A * sp.cos(w * t)  # time-dependent driving signal

    # Collapse operators -------------------------------------------------
    # Lindblad collapse operators: qubit relaxation, qubit dephasing, and
    # resonator relaxation. GQIS converts these symbolic operators into
    # the reduced real-valued density-matrix RHS.
    L_rel = sp.kronecker_product(sp.sqrt(gamma1) * sp_lower, qeye)
    L_phi = sp.kronecker_product(sp.sqrt(gamma2) * sz, qeye)
    L_kappa = sp.sqrt(kappa) * a1
    Col_Ops = [L_rel, L_phi, L_kappa]

    # The solver time-averages Tr(mean_operator*rho). The skip fraction excludes
    # an initial fraction of time to suppress dependence on the initial state.
    # Here the returned expectation value is <a>.
    mean_operator = a1

    # GPU solve ----------------------------------------------------------
    # The leading arguments are the Hamiltonian, explicit drive, collapse
    # operators, measured observable, and fixed-step time grid. Each pair in
    # var_arrays maps one symbolic sweep variable to one GPU grid axis.
    results = mesolve_2D(H, Drive, Col_Ops, mean_operator, tlist,
                         var_arrays={eps: eps_list, A: A_list},
                         timings=False,  # True prints RHS/codegen/kernel timing breakdown.
                         # Initial fraction of time excluded from the time average.
                         warmup_time=averaging_skip_fraction)

    results_np = np.asarray(results)
    return np.abs(np.real(results_np)).T  # transpose so rows=A and columns=eps for imshow


def main() -> None:
    start_time = time.time()
    settings = user_settings()

    # 1) Fixed device/system constants ------------------------------------
    # These define the base physical scales for the four-level model.
    delta = settings["delta"]
    wr2 = settings["wr2"]
    wq1 = settings["wq1"]

    # 2) Simulation parameter set -----------------------------------------
    wd_mhz = settings["wd_mhz"]
    gammaph = settings["gammaph"]
    gamma1 = settings["gamma1"]
    kappa = settings["kappa"]
    Ap = settings["Ap"]
    g1 = settings["g1"]
    simulation_periods = settings["simulation_periods"]
    eps_max = settings["eps_max"]
    A_max_factor = settings["A_max_factor"]
    averaging_skip_fraction = settings["averaging_skip_fraction"]

    # 3) Convert to physical units used by symbolic model -----------------
    # The symbolic model is written in angular-frequency-like units. This block
    # converts the convenient input parameters into the values used by H.
    delta_abs = wq1 * delta
    w = wd_mhz / 1000.0 * delta
    drive_period = 2.0 * np.pi / w
    gamma2 = gamma1 / 2.0 + gammaph
    Ap_abs = Ap * wq1
    g1_abs = g1 * wq1
    A_max_abs = 2.234042553191489 * A_max_factor * wq1
    eps_max_abs = eps_max * wq1

    # 4) Build sweep axes and time grid -----------------------------------
    # Each GPU thread integrates one pair (eps, A). Total work scales as
    # len(eps_list) * len(A_list) * len(tlist).
    grid_size = settings["grid_size"]
    num_steps = int(simulation_periods * settings["solver_steps_per_period"])

    eps_list = np.linspace(-eps_max_abs, eps_max_abs, grid_size,
                           dtype=np.float32)
    A_list = np.linspace(0.0, A_max_abs, grid_size, dtype=np.float32)
    tlist = np.linspace(0.0, simulation_periods * drive_period, num_steps + 1,
                        dtype=np.float32)
    total_solver_steps = grid_size * grid_size * num_steps

    print(f"wd={wd_mhz:.1f} MHz w={w:.6e} gamma1={gamma1:.3e} "
          f"gamma2={gamma2:.3e} kappa={kappa:.3e} "
          f"grid={grid_size}x{grid_size} samples={len(tlist)} steps={num_steps}")
    print(f"Total trajectory-step updates = grid_size**2*steps = {total_solver_steps:.4e}")

    # 5) Run GPU solver ----------------------------------------------------
    # First call includes symbolic RHS generation and CUDA compilation; repeated
    # calls with the same symbolic structure can reuse the cached kernel.
    gpu_start = time.time()
    p_mat = gpu_time_evolution(A_list, eps_list, tlist, delta_abs, w, gamma1, gamma2, kappa,
                               Ap_abs, g1_abs, wr2,
                               averaging_skip_fraction=averaging_skip_fraction)
    print(f"GPU solve time: {time.time() - gpu_start:.2f}s")

    # 6) Plot interferogram map -------------------------------------------
    # Convert the response to dB for display and plot A vertically, eps horizontally.
    map_db = 10.0 * np.log10(np.clip(p_mat, 1e-15, None))

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(map_db, aspect="auto", cmap="jet", origin="lower",
                   extent=[eps_list[0] / wq1, eps_list[-1] / wq1, 0.0, A_max_abs / wq1])
    fig.colorbar(im, ax=ax, label="10*log10(|Re(<a>)|)")
    ax.set_xlabel(r"$\epsilon/\Delta$")
    ax.set_ylabel(r"$A/\Delta$")
    ax.set_title("Four-level interferogram")
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.tight_layout()
    plt.show()

    print(f"Total runtime: {time.time() - start_time:.2f}s")


def user_settings() -> dict:
    """User-editable parameters."""
    wd_mhz, A_max_factor = 500.0, 1.15

    # Alternative regime. Comment the active assignment above and uncomment this line.
    # wd_mhz, A_max_factor = 1500.0, 1.15*1.3

    return {
        # Fixed device/system constants.
        "delta": 1.0,  # dimensionless qubit gap multiplier before conversion
        "wr2": 7.6767,  # resonator frequency in model frequency units
        "wq1": 4.71 * 1.15,  # qubit frequency scale used to convert normalized inputs
        # Simulation parameter set. Keep one active by editing this block.
        "wd_mhz": wd_mhz,  # drive frequency in MHz
        "gammaph": 0.0e-3,  # pure-dephasing rate
        "gamma1": 2.0e-3,  # qubit relaxation rate
        "kappa": 5.0e-3,  # resonator relaxation rate
        "Ap": 0.0002 / 10.0,  # probe amplitude
        "g1": 0.04033970276 / 1.23,  # qubit-resonator coupling
        "simulation_periods": 60,  # simulated duration in drive periods
        "eps_max": 2.09916,  # detuning half-range in units converted by wq1
        "A_max_factor": A_max_factor,  # drive-amplitude range scale
        "averaging_skip_fraction": 0.0,  # initial time fraction excluded from the average
        # Grid and time.
        "grid_size": 512,  # square grid side; total simulations = grid_size**2
        # Increase if mesolve_2D reports non-finite output.
        "solver_steps_per_period": 256,
    }


if __name__ == "__main__":
    main()
