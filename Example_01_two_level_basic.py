"""Example 01: basic two-level interferogram with GQIS ``mesolve_2D``."""

import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from gqis import mesolve_2D


def gpu_time_evolution(A_list, eps_list, tlist, delta, w, gamma1, gamma2):
    """Two-level GPU solver: returns map with rows=A and cols=eps."""

    # Symbolic model ------------------------------------------------------
    Delta, eps, Drive = sp.symbols("Delta eps Drive", real=True)#Define symbolic parameters of Hamiltonian
    A, t = sp.symbols("A t", real=True)
    #Matrices
    sz = sp.Matrix([[1, 0], [0, -1]])#Sigma_Z
    sm = sp.Matrix([[0, 1], [0, 0]])#Sigma minus
    sx = sp.Matrix([[0, 1], [1, 0]])# Sigma X
    Mean_Operator = sp.zeros(2); Mean_Operator[1, 1] = 1 #Operator for averaged output

    H = 0.5 * Delta * sx + 0.5 * Drive * sz #Hamiltonian
    drive_expr = eps + A * sp.sin(float(w) * t) #Driving signal

    gamma1S, gamma2S = sp.symbols("gamma1S gamma2S", real=True, nonnegative=True)# Define Symbolic relaxation and dephasing rates
    col_ops = [sp.sqrt(gamma1S) * sm, sp.sqrt(gamma2S) * sz] #Define collapse operators

    result = mesolve_2D(
        H,              #Hamiltonian.
        drive_expr,     #Driving
        col_ops,        #Collapse operators
        Mean_Operator,  #Operator for averaged output value
        tlist,          # Time Grid
        var_arrays={eps: eps_list, A: A_list}, #Variables that vary for different points integration kept symbolic for RHS creation and replaced with its values only within ODE solver timesteps
        const_values={Delta: float(delta), gamma1S: float(gamma1), gamma2S: float(gamma2)}, #Constants that is not varying will be replaced by its numeric values in SymPy RHS creation
        timings=False #If True display detailed information about RHS creation and Computation timings.
        )
    return np.abs(np.asarray(result)).T#Unload results back to CPU memory and make it float values for later display


def main() -> None:
    start = time.time()#Start measuring total runtime
    settings = user_settings()

    delta = settings["delta"]
    w = settings["w"]
    T = 2.0 * np.pi / w
    gammaph = settings["gammaph"]
    gamma1 = settings["gamma1"]
    gamma2 = gamma1 / 2.0 + gammaph
    tr = settings["tr"]
    epsmimax = settings["epsmimax"]
    amax = settings["amax"]

    # Grid and time -------------------------------------------------------
    nx = settings["nx"] #Square grid size along 1 dimension
    ny = nx
    num_steps = int(tr * settings["samples_per_period"]) # RK4 integration intervals
    eps_list = np.linspace(-epsmimax, epsmimax, nx, dtype=np.float32) #Detuning points grid along X horizontal axis
    A_list = np.linspace(0.0, amax, ny, dtype=np.float32) #Amplitude points grid along Y vertical axis
    tlist = np.linspace(0.0, tr * T, num_steps + 1, dtype=np.float32) # Includes both endpoints

    print(f"w={w:.4f} gamma1={gamma1:.3e} gamma2={gamma2:.3e} "f"grid={nx}x{ny} samples={len(tlist)} steps={num_steps}")
    print(f"Total trajectory-step updates = {nx}*{ny}*{num_steps} = {nx * ny * num_steps:.4e}")
    
    # Solve ---------------------------------------------------------------
    solve_start = time.time() #Start measuring solving time
    p_mat = gpu_time_evolution(A_list, eps_list, tlist, delta, w, gamma1, gamma2)
    print(f"GPU solve time: {time.time() - solve_start:.2f}s") # Print solving time

    # Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(p_mat,aspect="auto",cmap="jet",origin="lower",extent=[eps_list[0] / delta, eps_list[-1] / delta, 0.0, amax / delta])
    fig.colorbar(im, ax=ax, label="Qubit occupation")
    ax.set_xlabel(r"$\epsilon/\Delta$")
    ax.set_ylabel(r"$A/\Delta$")
    ax.set_title("Two-level interferogram")
    plt.tight_layout()
    plt.show()

    print(f"Total runtime: {time.time() - start:.2f}s")#Print total runtime


def user_settings() -> dict:
    """User-editable parameters."""
    delta = 1.0
    w = 1.14 * delta
    T = 2.0 * np.pi / w

    return {
        # Parameter set. Keep one active by editing this block.
        "delta": delta,       # two-level gap scale
        "w": w,               # drive angular frequency
        "gammaph": 0.04 / T,  # pure dephasing rate
        "gamma1": 0.05 / T,   # relaxation rate
        "tr": 30,             # number of driven periods to integrate
        "epsmimax": 16.0 * w, # detuning half-range
        "amax": 16.0 * w,     # drive-amplitude maximum

        # Alternative regimes:
        # "w": 0.32 * delta, "gammaph": 0.02 / T, "gamma1": 0.03 / T, "tr": 60
        # "w": 5.0 * delta, "gammaph": 0.08 / T, "gamma1": 0.08 / T, "tr": 60

        # Grid and time.
        "nx": 512, # square grid side; total simulations = nx*nx
        # Increase samples_per_period if mesolve_2D reports non-finite output.
        "samples_per_period": 256, # fixed RK4 steps per drive period
    }


if __name__ == "__main__":
    main()
