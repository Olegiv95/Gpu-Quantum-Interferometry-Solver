"""Example 03: two-level animation tutorial using GQIS directly."""

import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.animation import FFMpegWriter, FuncAnimation

from gqis import mesolve_2D


def build_two_level_model():
    """Build a reusable symbolic two-level model for animation frames."""
    N = 2

    # `eps` and `A` are the two sweep axes. The other symbols are constants
    # kept symbolic so the same compiled RHS can be reused across animation frames.
    eps, Drive = sp.symbols("eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)
    delta_s, w_s, gamma1_s, gamma2_s = sp.symbols(
        "delta_s w_s gamma1_s gamma2_s", real=True, nonnegative=True
    )

    # Two-level basis operators. The measured quantity is the population
    # P_1 = <1|rho|1> of the selected basis state |1>.
    sz = sp.Matrix([[1, 0], [0, -1]])
    sm = sp.Matrix([[0, 1], [0, 0]])
    sx = sp.Matrix([[0, 1], [1, 0]])
    mean_operator = sp.zeros(N)
    mean_operator[1, 1] = 1

    # Hamiltonian uses a Drive placeholder. `drive_expr` tells GQIS how
    # to evaluate that placeholder at every RK4 stage.
    H = 0.5 * delta_s * sx + 0.5 * Drive * sz
    drive_expr = eps + A * sp.sin(w_s * t)

    # Lindblad collapse operators: relaxation and dephasing.
    col_ops = [sp.sqrt(gamma1_s) * sm, sp.sqrt(gamma2_s) * sz]

    return {
        "H": H,
        "Drive": drive_expr,
        "Col_Ops": col_ops,
        "Mean_Operator": mean_operator,
        "eps_sym": eps,
        "A_sym": A,
        "const_syms": {
            "delta": delta_s,
            "w": w_s,
            "gamma1": gamma1_s,
            "gamma2": gamma2_s,
        },
    }


def build_time_grid(w, periods, samples_per_period):
    """Build a uniform time grid using the current frame drive frequency."""
    # The GPU backend uses fixed-step RK4. The returned grid has one more time
    # sample than integration steps so both endpoints are represented.
    T = 2.0 * np.pi / w
    num_steps = int(periods * samples_per_period)
    return np.linspace(0.0, periods * T, num_steps + 1, dtype=np.float32)


def frame_params(mode, value, base):
    """Apply one animated parameter update and recompute gamma2."""
    # This function changes one physical parameter per frame. gamma2 is derived
    # from relaxation plus pure dephasing and must be recomputed after updates.
    params = dict(base)
    subtitle = "static parameters"

    if mode == "w":
        params["w"] = float(value)
        subtitle = f"w={params['w']:.4f}"
    elif mode == "gamma1":
        params["gamma1"] = float(value)
        subtitle = f"gamma1={params['gamma1']:.4e}"
    elif mode == "gammaph":
        params["gammaph"] = float(value)
        subtitle = f"gammaph={params['gammaph']:.4e}"
    elif mode == "delta":
        params["delta"] = float(value)
        subtitle = f"delta={params['delta']:.4f}"
    elif mode == "same":
        subtitle = "static parameters"
    else:
        raise ValueError(f"Unsupported animation mode: {mode}")

    params["gamma2"] = params["gamma1"] / 2.0 + params["gammaph"]
    return params, subtitle


def gpu_time_evolution(
    A_list,
    eps_list,
    tlist,
    model,
    params,
    *,
    RHSreuse=True,
    warmup_time=0.0,
    timings=False,
):
    """Run one animation frame with runtime constants for fast RHS reuse."""
    # Runtime constants remain symbolic during CUDA code generation and are
    # passed as a small GPU array at launch time. This avoids recompiling the RHS
    # when only constants such as w, gamma1, or gamma2 change between frames.
    const_syms = model["const_syms"]
    const_values = {
        const_syms["delta"]: float(params["delta"]),
        const_syms["w"]: float(params["w"]),
        const_syms["gamma1"]: float(params["gamma1"]),
        const_syms["gamma2"]: float(params["gamma2"]),
    }

    result = mesolve_2D(
        model["H"],                          # symbolic Hamiltonian
        model["Drive"],                      # time-dependent drive expression
        model["Col_Ops"],                    # Lindblad collapse operators
        model["Mean_Operator"],              # observable to average
        tlist,                               # fixed RK4 time grid
        var_arrays={model["eps_sym"]: eps_list, model["A_sym"]: A_list},  # 2D sweep axes
        const_values=const_values,           # values for symbolic constants
        keep_symbolic_consts="all",          # keep constants runtime-adjustable for animation
        RHSreuse=RHSreuse,                   # reuse compiled kernel when symbolic structure matches
        warmup_time=warmup_time,             # initial fraction of time excluded from the time average
        timings=timings,                     # True prints codegen/kernel timing
    )
    return np.abs(np.real(np.asarray(result))).T


def main() -> None:
    total_start = time.time()
    settings = user_settings()

    # Base parameter set --------------------------------------------------
    # These values define the initial physical regime. Individual parameters can
    # be swept by choosing an animation mode below.
    delta = settings["delta"]
    w = settings["w"]
    T = 2.0 * np.pi / w
    gammaph = settings["gammaph"]
    gamma1 = settings["gamma1"]
    gamma2 = gamma1 / 2.0 + gammaph
    periods = settings["periods"]
    epsmimax = settings["epsmimax"]
    amax = settings["amax"]
    warmup_time = settings["warmup_time"]

    # Tutorial defaults. For RTX 3080 benchmark visuals, increase grid_size.
    # Total work per frame scales as grid_size^2 * periods * samples_per_period.
    grid_size = settings["grid_size"]
    samples_per_period = settings["samples_per_period"]
    frame_count = settings["frame_count"]

    eps_list = np.linspace(-epsmimax, epsmimax, grid_size, dtype=np.float32)
    A_list = np.linspace(0.0, amax, grid_size, dtype=np.float32)
    model = build_two_level_model()

    base = {
        "delta": delta,
        "w": w,
        "gamma1": gamma1,
        "gammaph": gammaph,
        "gamma2": gamma2,
    }

    # Animation modes. Keep one active.
    # `varray` contains the parameter values used for successive frames.
    mode = settings["mode"]
    varray = settings["varray"](delta, w, T, frame_count)

    # `pingpong` plays the sweep forward and backward for a looping animation.
    playback_mode = settings["playback_mode"]  # "forward" or "pingpong"
    save_mp4 = settings["save_mp4"]
    video_name = settings["video_name"]
    fps = settings["fps"]
    dpi = settings["dpi"]

    # Build the order of frames. For pingpong mode, avoid duplicating the first
    # endpoint so the loop is visually smoother.
    if playback_mode == "pingpong" and len(varray) > 1:
        frame_indices = np.concatenate(
            [np.arange(len(varray), dtype=np.int32), np.arange(len(varray) - 2, -1, -1, dtype=np.int32)]
        )
    else:
        frame_indices = np.arange(len(varray), dtype=np.int32)

    # Initial frame also compiles the CUDA kernel. Later frames reuse it when
    # only runtime constants change.
    params0, subtitle0 = frame_params(mode, float(varray[0]), base)
    tlist0 = build_time_grid(params0["w"], periods, samples_per_period)
    p0 = gpu_time_evolution(
        A_list,
        eps_list,
        tlist0,
        model,
        params0,
        RHSreuse=True,
        warmup_time=warmup_time,
        timings=False,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(
        p0,
        aspect="auto",
        cmap="jet",
        origin="lower",
        extent=[eps_list[0] / delta, eps_list[-1] / delta, 0.0, amax / delta],
    )
    fig.colorbar(img, ax=ax, label="Qubit occupation")
    ax.set_xlabel(r"$\epsilon/\Delta$")
    ax.set_ylabel(r"$A/\Delta$")
    title = ax.set_title(f"Example 03 | mode={mode} | {subtitle0}")
    plt.tight_layout()

    def update(frame_pos):
        # Matplotlib calls this function once per displayed/saved frame.
        # Each call runs a new 2D GPU sweep for the current parameter value.
        frame_start = time.time()
        frame_idx = int(frame_indices[frame_pos])
        params, subtitle = frame_params(mode, float(varray[frame_idx]), base)
        tlist = build_time_grid(params["w"], periods, samples_per_period)
        p = gpu_time_evolution(
            A_list,
            eps_list,
            tlist,
            model,
            params,
            RHSreuse=True,
            warmup_time=warmup_time,
            timings=False,
        )
        img.set_data(p)
        img.set_clim(float(np.nanmin(p)), float(np.nanmax(p)))
        title.set_text(f"Example 03 | mode={mode} | {subtitle}")
        print(
            f"frame {frame_pos + 1}/{len(frame_indices)} calc={time.time() - frame_start:.2f}s",
            end="\r",
            flush=True,
        )
        return [img, title]

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=20, blit=False)

    if save_mp4:
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1)
        ani.save(video_name, writer=writer, dpi=dpi)
        print(f"\nSaved: {video_name}")

    plt.show()
    print(f"Total runtime: {time.time() - total_start:.2f}s")


def user_settings() -> dict:
    """User-editable parameters."""
    delta = 1.0
    w = 1.14 * delta
    T = 2.0 * np.pi / w

    return {
        # Base physical parameters.
        "delta": delta,       # two-level energy gap 
        "w": w,               # drive angular frequency
        "gammaph": 0.04 / T,  # pure dephasing rate
        "gamma1": 0.05 / T,   # relaxation rate
        "periods": 20,        # driven periods for simulation
        "epsmimax": 16.0 * w, # detuning range
        "amax": 16.0 * w,     # drive-amplitude range 0 to x*w
        "warmup_time": 0.0,   # initial fraction of time excluded from the time average

        # Grid, time, and animation size.
        "grid_size": 512, # square grid side per animation frame
        # Increase samples_per_period if mesolve_2D reports non-finite output.
        "samples_per_period": 250, # fixed RK4 steps per drive period
        "frame_count": 30,         # number of parameter values before optional pingpong mirroring

        # Animation mode. Keep one active.
        "mode": "w", # animated parameter: w, gamma1, gammaph, delta, same
        "varray": lambda delta, w, T, frame_count: np.linspace(0.8 * w, 1.6 * w, frame_count, dtype=np.float32), # values for mode
        # "mode": "gamma1",
        # "varray": lambda delta, w, T, frame_count: np.linspace(0.01 / T, 0.15 / T, frame_count, dtype=np.float32),
        # "mode": "gammaph",
        # "varray": lambda delta, w, T, frame_count: np.linspace(0.0, 0.12 / T, frame_count, dtype=np.float32),
        # "mode": "delta",
        # "varray": lambda delta, w, T, frame_count: np.linspace(0.7 * delta, 1.3 * delta, frame_count, dtype=np.float32),
        # "mode": "same",
        # "varray": lambda delta, w, T, frame_count: np.linspace(0.0, 1.0, frame_count, dtype=np.float32),

        # Playback/output.
        "playback_mode": "pingpong", # "forward" or "pingpong" animation
        "save_mp4": True,           # False shows interactive animation only without saving to file
        "video_name": "Example_03_two_level_animation.mp4", # output filename when save_mp4=True
        "fps": 5,                    # saved video frames per second
        "dpi": 120,                  # saved video resolution scale
    }


if __name__ == "__main__":
    main()
