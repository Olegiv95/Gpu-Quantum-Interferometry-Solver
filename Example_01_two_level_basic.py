"""Example 01: basic two-level interferogram with GQIS ``mesolve_2D``."""

import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from gqis import mesolve_2D


def gpu_time_evolution(A_list, eps_list, tlist, delta, w, gamma1, gamma2):
    """Two-level GPU solver: returns map with rows=A and cols=eps."""

    # Symbolic model ------------------------------------------------------
    # Define the symbolic Hamiltonian parameters.
    Delta, eps, Drive = sp.symbols("Delta eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)
    # Matrices
    sz = sp.Matrix([[1, 0], [0, -1]])  # Sigma_Z
    sm = sp.Matrix([[0, 1], [0, 0]])  # Sigma minus
    sx = sp.Matrix([[0, 1], [1, 0]])  # Sigma X
    Mean_Operator = sp.zeros(2)
    Mean_Operator[1, 1] = 1  # Operator for averaged output

    H = 0.5 * Delta * sx + 0.5 * Drive * sz  # Hamiltonian
    drive_expr = eps + A * sp.sin(float(w) * t)  # Driving signal

    # Define symbolic relaxation and dephasing rates.
    gamma1S, gamma2S = sp.symbols("gamma1S gamma2S", real=True, nonnegative=True)
    col_ops = [sp.sqrt(gamma1S) * sm, sp.sqrt(gamma2S) * sz]  # Define collapse operators

    # The first five arguments define the physical model, observable, and fixed
    # RK4 time grid. Sweep variables remain symbolic during RHS generation and
    # receive their numerical values inside the CUDA kernel.
    # Non-varying constants are substituted during SymPy RHS generation.
    result = mesolve_2D(H, drive_expr, col_ops, Mean_Operator, tlist,
                        var_arrays={eps: eps_list, A: A_list},
                        const_values={Delta: float(delta), gamma1S: float(gamma1),
                                      gamma2S: float(gamma2)},
                        timings=False)  # Display RHS-generation and computation timings when True.
    # Transfer the result to CPU memory and orient it for plotting.
    return np.abs(np.asarray(result)).T


def main() -> None:
    start = time.time()  # Start measuring total runtime
    settings = user_settings()

    delta = settings["delta"]
    w = settings["w"]
    drive_period = 2.0 * np.pi / w
    gammaph = settings["gammaph"]
    gamma1 = settings["gamma1"]
    gamma2 = gamma1 / 2.0 + gammaph
    simulation_periods = settings["simulation_periods"]
    eps_max = settings["eps_max"]
    A_max = settings["A_max"]

    # Grid and time -------------------------------------------------------
    grid_size = settings["grid_size"]
    num_steps = int(simulation_periods * settings["solver_steps_per_period"])
    eps_list = np.linspace(-eps_max, eps_max, grid_size, dtype=np.float32)
    A_list = np.linspace(0.0, A_max, grid_size, dtype=np.float32)
    tlist = np.linspace(0.0, simulation_periods * drive_period, num_steps + 1,
                        dtype=np.float32)

    print(f"w={w:.4f} gamma1={gamma1:.3e} gamma2={gamma2:.3e} "
          f"grid={grid_size}x{grid_size} samples={len(tlist)} steps={num_steps}")
    total_updates = grid_size * grid_size * num_steps
    print(f"Total trajectory-step updates = {grid_size}*{grid_size}*{num_steps} "
          f"= {total_updates:.4e}")

    # Solve ---------------------------------------------------------------
    solve_start = time.time()  # Start measuring solving time
    p_mat = gpu_time_evolution(A_list, eps_list, tlist, delta, w, gamma1, gamma2)
    print(f"GPU solve time: {time.time() - solve_start:.2f}s")  # Print solving time

    # Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(p_mat, aspect="auto", cmap="jet", origin="lower",
                   extent=[eps_list[0] / delta, eps_list[-1] / delta, 0.0, A_max / delta])
    fig.colorbar(im, ax=ax, label="Qubit occupation")
    ax.set_xlabel(r"$\epsilon/\Delta$")
    ax.set_ylabel(r"$A/\Delta$")
    ax.set_title("Two-level interferogram")
    plt.tight_layout()
    print(f"Example calculation and plot preparation time: {time.time() - start:.2f}s")
    plt.show()


def user_settings() -> dict:
    """User-editable parameters."""
    delta = 1.0
    w, simulation_periods = 1.14 * delta, 30

    # Alternative regimes. Comment the active assignment above and uncomment one below.
    # w, simulation_periods = 0.32*delta, 60
    # w, simulation_periods = 5.0*delta, 60
    drive_period = 2.0 * np.pi / w

    return {
        # Parameter set. Keep one active by editing this block.
        "delta": delta,  # two-level energy gap Delta
        "w": w,  # drive angular frequency
        "gammaph": 0.04 / drive_period,  # pure-dephasing rate
        "gamma1": 0.05 / drive_period,  # relaxation rate
        "simulation_periods": simulation_periods,  # simulated duration in drive periods
        "eps_max": 16.0 * w,  # detuning half-range: -eps_max to +eps_max
        "A_max": 16.0 * w,  # maximum drive amplitude
        # Grid and time.
        "grid_size": 512,  # square grid side; total simulations = grid_size**2
        # Increase if mesolve_2D reports non-finite output.
        "solver_steps_per_period": 256,
    }


if __name__ == "__main__":
    main()
