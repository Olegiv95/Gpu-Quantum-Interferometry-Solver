"""Example 02: four-level interferogram (minimal GPU tutorial)."""

import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from gqis import mesolve_2D


def gpu_time_evolution(
    A_list,
    eps_list,
    tlist,
    delta,
    w,
    gamma1,
    gamma2,
    kappa,
    Ap,
    g1,
    nph,
    wr2,
    warmup_time=0.0,
):
    """Build symbolic model and run GPU solver for one interferogram."""
    # Convert all numerical grids to single precision. The default GPU kernel is
    # FP32, so using float32 avoids unnecessary host-to-device conversion work.
    A_list = np.asarray(A_list, dtype=np.float32)      # amplitude sweep, vertical plot axis
    eps_list = np.asarray(eps_list, dtype=np.float32)  # detuning sweep, horizontal plot axis
    tlist = np.asarray(tlist, dtype=np.float32)        # fixed RK4 integration time grid

    # The truncated model contains a two-level qubit plus nph resonator states.
    # With nph=2 this gives the four-level example used below.
    N = nph + 2

    # Build symbolic basis/operators ------------------------------------
    # `eps` and `A` are sweep variables; GQIS maps them to ParX/ParY.
    # `Drive` is a placeholder inside the Hamiltonian and is replaced by the
    # explicit time-dependent expression `eps + A*cos(w*t)` during codegen.
    eps, Drive = sp.symbols("eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)

    sz = sp.Matrix([[1, 0], [0, -1]])        # qubit sigma_z
    sp_raise = sp.Matrix([[0, 1], [0, 0]])   # raising operator in this basis convention
    sp_lower = sp.Matrix([[0, 0], [1, 0]])   # lowering operator in this basis convention
    qeye = sp.eye(N - 2)                     # identity operator in the resonator subspace

    # Lift qubit and resonator operators into the full tensor-product space.
    sm1 = sp.kronecker_product(sp_lower, qeye)  # qubit lowering operator
    sz1 = sp.kronecker_product(sz, qeye)        # qubit sigma_z operator
    a1 = sp.kronecker_product(qeye, sp_raise)   # resonator annihilation-like operator

    # Hamiltonian + drive ------------------------------------------------
    # h0 is time independent. ht depends on the symbolic Drive placeholder, so
    # the generated CUDA RHS can evaluate it at every RK4 stage.
    hq11 = sz1 / 2
    hp = Ap * (a1.H + a1)  # weak probe term
    hc11u = g1 * delta * (sm1.H * a1 + sm1 * a1.H)  # qubit-resonator coupling
    h0 = hp
    ht = hc11u / sp.sqrt(delta**2 + Drive**2) + hq11 * (sp.sqrt(delta**2 + Drive**2) - wr2)
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

    # The solver time-averages Tr(mean_operator*rho). ``warmup_time`` excludes
    # an initial fraction of time to suppress dependence on the initial state.
    # Here the returned expectation value is <a>.
    mean_operator = a1

    # GPU solve ----------------------------------------------------------
    results = mesolve_2D(
        H,                                  # symbolic Hamiltonian
        Drive,                              # explicit time-dependent drive expression
        Col_Ops,                            # symbolic Lindblad collapse operators
        mean_operator,                      # observable to average
        tlist,                              # fixed RK4 time grid
        var_arrays={eps: eps_list, A: A_list},  # 2D sweep variables mapped to GPU threads
        timings=False,                      # True prints RHS/codegen/kernel timing breakdown
        warmup_time=warmup_time,            # initial fraction of time excluded from the time average
    )

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
    wd_hz = settings["wd_hz"]; gammaph = settings["gammaph"]; gamma1 = settings["gamma1"]; kappa = settings["kappa"]; Ap = settings["Ap"]; g1 = settings["g1"]; tr = settings["tr"]; epsmimax = settings["epsmimax"]; amax_factor = settings["amax_factor"]; warmup_time = settings["warmup_time"]

    # 3) Convert to physical units used by symbolic model -----------------
    # The symbolic model is written in angular-frequency-like units. This block
    # converts the convenient input parameters into the values used by H.
    delta_abs = wq1 * delta
    w_abs = wd_hz / 1000.0 * delta
    T_abs = 2.0 * np.pi / w_abs
    gamma2 = gamma1 / 2.0 + gammaph
    Ap = Ap * wq1
    g1 = g1 * wq1
    Amax = 2.234042553191489 * amax_factor * wq1
    epsmimax_abs = epsmimax * wq1

    # 4) Build sweep axes and time grid -----------------------------------
    # Each GPU thread integrates one pair (eps, A). Total work scales as
    # len(eps_list) * len(A_list) * len(tlist).
    nx = settings["nx"]
    ny = nx
    num_steps = int(tr * settings["samples_per_period"])
    Nph = settings["Nph"]

    eps_list = np.linspace(-epsmimax_abs, epsmimax_abs, ny, dtype=np.float32)
    A_list = np.linspace(0.0, Amax, nx, dtype=np.float32)
    tlist = np.linspace(0.0, tr * T_abs, num_steps + 1, dtype=np.float32)
    total_solver_steps = ny * nx * num_steps

    print(
        f"wd={wd_hz:.1f}MHz w={w_abs:.6e} gamma1={gamma1:.3e} gamma2={gamma2:.3e} "
        f"kappa={kappa:.3e} grid={ny}x{nx} samples={len(tlist)} steps={num_steps}"
    )
    print(f"Total trajectory-step updates = ny*nx*steps = {total_solver_steps:.4e}")

    # 5) Run GPU solver ----------------------------------------------------
    # First call includes symbolic RHS generation and CUDA compilation; repeated
    # calls with the same symbolic structure can reuse the cached kernel.
    gpu_start = time.time()
    p_mat = gpu_time_evolution(
        A_list,
        eps_list,
        tlist,
        delta_abs,
        w_abs,
        gamma1,
        gamma2,
        kappa,
        Ap,
        g1,
        Nph,
        wr2,
        warmup_time=warmup_time,
    )
    print(f"GPU solve time: {time.time() - gpu_start:.2f}s")

    # 6) Plot interferogram map -------------------------------------------
    # Convert the response to dB for display and plot A vertically, eps horizontally.
    map_db = 10.0 * np.log10(np.clip(p_mat, 1e-15, None))

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        map_db,
        aspect="auto",
        cmap="jet",
        origin="lower",
        extent=[eps_list[0] / wq1, eps_list[-1] / wq1, 0.0, Amax / wq1],
    )
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
    return {
        # Fixed device/system constants.
        "delta": 1.0,       # qubit gap scale before conversion to model units
        "wr2": 7.6767,      # resonator/reference frequency in model units
        "wq1": 4.71 * 1.15, # conversion factor used for eps/A/delta scaling

        # Simulation parameter set. Keep one active by editing this block.
        "wd_hz": 500.0,             # drive frequency
        "gammaph": 0.0e-3,          # pure dephasing contribution; gamma2 = gamma1/2 + gammaph
        "gamma1": 2.0e-3,           # qubit relaxation rate
        "kappa": 5.0e-3,            # resonator relaxation rate
        "Ap": 0.0002 / 10.0,        # weak probe amplitude
        "g1": 0.04033970276 / 1.23, # qubit-resonator coupling
        "tr": 60,                   # number of driving periods to integrate
        "epsmimax": 2.09916,        # detuning range befor
        "amax_factor": 1.15,        # amplitude range
        "warmup_time": 0.0,         # initial fraction of time excluded from the time average

        # Alternative regime:
        # "wd_hz": 1500.0, "amax_factor": 1.15 * 1.3, "warmup_time": 0.2

        # Grid and time.
        "nx": 512, # square grid side; total simulations = nx*nx
        # Increase samples_per_period if mesolve_2D reports non-finite output.
        "samples_per_period": 256, # fixed RK4 steps per drive period
        "Nph": 2,                  # resonator truncation; total Hilbert size = Nph + 2
    }


if __name__ == "__main__":
    main()
