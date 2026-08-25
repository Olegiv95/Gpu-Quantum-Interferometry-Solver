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
    delta_s, w_s, gamma1_s, gamma2_s = sp.symbols("delta_s w_s gamma1_s gamma2_s", real=True,
                                                  nonnegative=True)

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
        "const_syms": {"delta": delta_s, "w": w_s, "gamma1": gamma1_s,
                       "gamma2": gamma2_s},
    }


def build_time_grid(w, simulation_periods, solver_steps_per_period):
    """Build a uniform time grid using the current frame drive frequency."""
    # The GPU backend uses fixed-step RK4. The returned grid has one more time
    # sample than integration steps so both endpoints are represented.
    drive_period = 2.0 * np.pi / w
    num_steps = int(simulation_periods * solver_steps_per_period)
    return np.linspace(0.0, simulation_periods * drive_period, num_steps + 1,
                       dtype=np.float32)


def frame_params(animated_parameter, value, base):
    """Apply one animated parameter update and recompute qubit decoherence."""
    # This function changes one physical parameter per frame. Decoherence is derived
    # from relaxation plus pure dephasing and must be recomputed after updates.
    params = dict(base)
    subtitle = "static parameters"

    if animated_parameter == "w":
        params["w"] = float(value)
        subtitle = f"w={params['w']:.4f}"
    elif animated_parameter == "gamma1":
        params["gamma1"] = float(value)
        subtitle = f"gamma1={params['gamma1']:.4e}"
    elif animated_parameter == "gammaph":
        params["gammaph"] = float(value)
        subtitle = f"gammaph={params['gammaph']:.4e}"
    elif animated_parameter == "delta":
        params["delta"] = float(value)
        subtitle = f"delta={params['delta']:.4f}"
    elif animated_parameter == "same":
        subtitle = "static parameters"
    else:
        raise ValueError(f"Unsupported animated_parameter: {animated_parameter}")

    params["gamma2"] = params["gamma1"] / 2.0 + params["gammaph"]
    return params, subtitle


def gpu_time_evolution(A_list, eps_list, tlist, model, params, *, RHSreuse=True,
                       averaging_skip_fraction=0.0, timings=False):
    """Run one animation frame with runtime constants for fast RHS reuse."""
    # Runtime constants remain symbolic during CUDA code generation and are
    # passed as a small GPU array at launch time. This avoids recompiling the RHS
    # when only constants such as w, gamma1, or gamma2 change between frames.
    const_syms = model["const_syms"]
    const_values = {const_syms["delta"]: float(params["delta"]),
                    const_syms["w"]: float(params["w"]),
                    const_syms["gamma1"]: float(params["gamma1"]),
                    const_syms["gamma2"]: float(params["gamma2"])}

    # The leading arguments define the physical model, averaged observable, and
    # fixed-step grid. Runtime-adjustable constants let animation frames reuse the
    # generated RHS and compiled CUDA kernel.
    result = mesolve_2D(model["H"], model["Drive"], model["Col_Ops"], model["Mean_Operator"], tlist,
                        var_arrays={model["eps_sym"]: eps_list, model["A_sym"]: A_list},
                        const_values=const_values, keep_symbolic_consts="all", RHSreuse=RHSreuse,
                        warmup_time=averaging_skip_fraction, timings=timings)
    return np.abs(np.real(np.asarray(result))).T


def main() -> None:
    total_start = time.time()
    settings = user_settings()

    # Base parameter set --------------------------------------------------
    # These values define the initial physical regime. One parameter can be
    # varied across frames using the animation settings below.
    delta = settings["delta"]
    w = settings["w"]
    gammaph = settings["gammaph"]
    gamma1 = settings["gamma1"]
    gamma2 = gamma1 / 2.0 + gammaph
    simulation_periods = settings["simulation_periods"]
    eps_max = settings["eps_max"]
    A_max = settings["A_max"]
    averaging_skip_fraction = settings["averaging_skip_fraction"]

    # Tutorial defaults. For RTX 3080 benchmark visuals, increase grid_size.
    # Total work scales as grid_size**2 * simulation_periods * solver_steps_per_period.
    grid_size = settings["grid_size"]
    solver_steps_per_period = settings["solver_steps_per_period"]
    forward_frame_count = int(settings["forward_frame_count"])

    eps_list = np.linspace(-eps_max, eps_max, grid_size, dtype=np.float32)
    A_list = np.linspace(0.0, A_max, grid_size, dtype=np.float32)
    model = build_two_level_model()

    base = {"delta": delta, "w": w, "gamma1": gamma1, "gammaph": gammaph,
            "gamma2": gamma2}

    # The animated parameter name and its numerical value in every forward frame
    # are separate settings. Pingpong playback adds the reverse frames later.
    animated_parameter = settings["animated_parameter"]
    parameter_values = np.asarray(settings["animated_parameter_values"], dtype=np.float32)
    if forward_frame_count < 1:
        raise ValueError("forward_frame_count must be at least 1.")
    if parameter_values.ndim != 1 or len(parameter_values) != forward_frame_count:
        raise ValueError("animated_parameter_values must be a one-dimensional array with "
                         "forward_frame_count elements.")

    # `pingpong` plays the sweep forward and backward for a looping animation.
    playback_mode = settings["playback_mode"]  # "forward" or "pingpong"
    if playback_mode not in {"forward", "pingpong"}:
        raise ValueError("playback_mode must be 'forward' or 'pingpong'.")
    save_mp4 = settings["save_mp4"]
    video_filename = settings["video_filename"]
    video_fps = settings["video_fps"]
    video_dpi = settings["video_dpi"]
    ffmpeg_preset = settings["ffmpeg_preset"]
    ffmpeg_crf = settings["ffmpeg_crf"]

    # Build the frame order. Pingpong avoids repeating the turnaround endpoint;
    # the initial endpoint returns as the final frame for smooth looping.
    if playback_mode == "pingpong" and len(parameter_values) > 1:
        frame_indices = np.concatenate([np.arange(len(parameter_values), dtype=np.int32),
                                        np.arange(len(parameter_values) - 2, -1, -1,
                                                  dtype=np.int32)])
    else:
        frame_indices = np.arange(len(parameter_values), dtype=np.int32)

    num_steps_per_frame = int(simulation_periods * solver_steps_per_period)
    print(f"Animation workload: grid={grid_size}x{grid_size}, "
          f"calculated_frames={len(frame_indices)}, solver_steps_per_frame={num_steps_per_frame}")

    # Initial frame also compiles the CUDA kernel. Later frames reuse it when
    # only runtime constants change.
    initial_frame_start = time.time()
    params0, subtitle0 = frame_params(animated_parameter, float(parameter_values[0]), base)
    tlist0 = build_time_grid(params0["w"], simulation_periods, solver_steps_per_period)
    p0 = gpu_time_evolution(A_list, eps_list, tlist0, model, params0, RHSreuse=True,
                            averaging_skip_fraction=averaging_skip_fraction, timings=False)
    print(f"Initial frame calculation time: {time.time() - initial_frame_start:.2f}s")

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(p0, aspect="auto", cmap="jet", origin="lower",
                    extent=[eps_list[0] / delta, eps_list[-1] / delta, 0.0, A_max / delta])
    fig.colorbar(img, ax=ax, label="Qubit occupation")
    ax.set_xlabel(r"$\epsilon/\Delta$")
    ax.set_ylabel(r"$A/\Delta$")
    title = ax.set_title(f"Example 03 | animated={animated_parameter} | {subtitle0}")
    plt.tight_layout()

    def update(frame_pos):
        # Matplotlib calls this function once per displayed/saved frame.
        # Each call runs a new 2D GPU sweep for the current parameter value.
        frame_start = time.time()
        frame_idx = int(frame_indices[frame_pos])
        value = float(parameter_values[frame_idx])
        params, subtitle = frame_params(animated_parameter, value, base)
        tlist = build_time_grid(params["w"], simulation_periods, solver_steps_per_period)
        p = gpu_time_evolution(A_list, eps_list, tlist, model, params, RHSreuse=True,
                               averaging_skip_fraction=averaging_skip_fraction, timings=False)
        img.set_data(p)
        img.set_clim(float(np.nanmin(p)), float(np.nanmax(p)))
        title.set_text(f"Example 03 | animated={animated_parameter} | {subtitle}")
        print(f"frame {frame_pos + 1}/{len(frame_indices)} calc={time.time() - frame_start:.2f}s",
              end="\r", flush=True)
        return [img, title]

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=20, blit=False)

    if save_mp4:
        writer = FFMpegWriter(fps=video_fps, codec="libx264", bitrate=-1,
                              extra_args=["-preset", ffmpeg_preset, "-crf",
                                          str(ffmpeg_crf), "-pix_fmt", "yuv420p"])
        ani.save(video_filename, writer=writer, dpi=video_dpi)
        print(f"\nSaved: {video_filename}")
        print(f"Animation calculation and MP4 export time: {time.time() - total_start:.2f}s")
    else:
        print(f"Interactive animation setup time: {time.time() - total_start:.2f}s; "
              "per-frame calculation times are printed during playback.")

    plt.show()


def user_settings() -> dict:
    """User-editable physical, numerical, and animation settings."""
    delta = 1.0
    w = 1.14 * delta
    T = 2.0 * np.pi / w  # drive period

    # Animation parameter sweep. The array contains one value for each forward
    # frame. Pingpong playback produces 2*forward_frame_count - 1 total frames.
    forward_frame_count = 30
    animated_parameter, frame_values = "w", np.linspace(0.8 * w, 1.6 * w, forward_frame_count)

    # Alternative animation sweeps. Comment the active assignment above and uncomment one below.
    # animated_parameter, frame_values = "gamma1", np.linspace(0.01/T, 0.15/T, forward_frame_count)
    # animated_parameter, frame_values = "gammaph", np.linspace(0.0, 0.12/T, forward_frame_count)
    # animated_parameter, frame_values = "delta", np.linspace(0.7*delta, 1.3*delta, forward_frame_count)
    # animated_parameter, frame_values = "same", np.zeros(forward_frame_count)

    return {
        # Base physical parameters.
        "delta": delta,  # minimum energy gap
        "w": w,  # drive angular frequency
        "gammaph": 0.04 / T,  # pure-dephasing rate
        "gamma1": 0.05 / T,  # relaxation rate
        "simulation_periods": 20,  # simulated duration per frame in drive periods
        "eps_max": 16.0 * w,  # detuning half-range
        "A_max": 16.0 * w,  # maximum drive amplitude
        "averaging_skip_fraction": 0.0,  # initial time fraction excluded from the average
        # Numerical grid and integration settings.
        "grid_size": 512,  # square grid side per animation frame
        # Increase if mesolve_2D reports non-finite output.
        "solver_steps_per_period": 250,
        # Animation parameter and values.
        "forward_frame_count": forward_frame_count,  # before optional reverse playback
        "animated_parameter": animated_parameter,  # defined before return block
        "animated_parameter_values": np.asarray(frame_values, dtype=np.float32),
        # Playback and output settings.
        "playback_mode": "pingpong",  # "forward" or "pingpong" animation
        "save_mp4": True,  # False shows interactive animation only without saving to file
        "video_filename": "Example_03_two_level_animation.mp4",
        "video_fps": 5,  # playback rate, not the number of calculated frames
        "video_dpi": 120,  # saved video resolution scale
        "ffmpeg_preset": "medium",  # x264 speed/compression preset
        "ffmpeg_crf": 23,  # x264 quality; lower is higher quality/larger file
    }


if __name__ == "__main__":
    main()
