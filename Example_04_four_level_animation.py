"""Example 04: four-level animation tutorial using the GQIS backend."""

import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.animation import FFMpegWriter, FuncAnimation

from gqis import mesolve_2D


def tanh_space(start, stop, num, stretch_factor=3.0):
    """Smoothly dense-at-edges sweep, useful for kappa scans."""
    # Linear sweeps can miss sharp edge behavior. This helper gives more points
    # near the endpoints while still covering the full interval.
    x = np.linspace(-stretch_factor, stretch_factor, num, dtype=np.float32)
    y = np.tanh(x)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return start + (stop - start) * y


def build_four_level_model():
    """Build symbolic four-level model once and reuse across frames."""
    N = 4

    # `eps` and `A` are the two sweep axes. The remaining symbols are constants
    # that can be kept runtime-adjustable between animation frames.
    eps, Drive = sp.symbols("eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)
    (delta_c, w_c, gamma1_c, gamma2_c, kappa_c, Ap_c, g1_c, wr2_c,
     wpr_c) = sp.symbols("delta_c w_c gamma1_c gamma2_c kappa_c Ap_c g1_c wr2_c wpr_c",
                         real=True)

    sz = sp.Matrix([[1, 0], [0, -1]])  # qubit sigma_z
    sp_raise = sp.Matrix([[0, 1], [0, 0]])  # raising operator in this basis convention
    sp_lower = sp.Matrix([[0, 0], [1, 0]])  # lowering operator in this basis convention
    qeye = sp.eye(N - 2)  # identity in the resonator subspace

    # Lift qubit and resonator operators into the full Hilbert space.
    sm1 = sp.kronecker_product(sp_lower, qeye)
    sz1 = sp.kronecker_product(sz, qeye)
    a1 = sp.kronecker_product(qeye, sp_raise)
    p0 = sp.zeros(N)
    p0[0, 0] = 1  # projector onto the selected qubit basis state |0>

    # Hamiltonian terms. The Drive placeholder is evaluated from drive_expr at
    # each RK4 stage in the generated CUDA kernel.
    hq11 = sz1 / 2
    hr11 = (wr2_c - wpr_c) * a1.H * a1
    hp = Ap_c * (a1.H + a1)
    hc11u = g1_c * delta_c * (sm1.H * a1 + sm1 * a1.H)
    H = (hr11 + hp + hc11u / sp.sqrt(delta_c**2 + Drive**2) + hq11 *
         (sp.sqrt(delta_c**2 + Drive**2) - wr2_c))
    drive_expr = eps + A * sp.cos(w_c * t)

    # Lindblad collapse operators: qubit relaxation, qubit dephasing, and
    # resonator relaxation.
    col_ops = [sp.kronecker_product(sp.sqrt(gamma1_c) * sp_lower, qeye),
               sp.kronecker_product(sp.sqrt(gamma2_c) * sz, qeye),
               sp.sqrt(kappa_c) * a1]

    return {
        "H": H,
        "Drive": drive_expr,
        "Col_Ops": col_ops,
        "Mean_Response": a1,
        "Mean_Population": p0,
        "eps_sym": eps,
        "A_sym": A,
        "const_syms": {"delta": delta_c, "w": w_c, "gamma1": gamma1_c,
                       "gamma2": gamma2_c, "kappa": kappa_c, "Ap": Ap_c, "g1": g1_c,
                       "wr2": wr2_c, "wpr": wpr_c},
    }


def build_time_grid(delta, wd_mhz, simulation_periods, solver_steps_per_period):
    """Convert drive frequency and build uniform time list."""
    # GQIS uses fixed-step RK4. Include both endpoints by returning one
    # more time sample than integration steps.
    w = wd_mhz / 1000.0 * delta
    drive_period = 2.0 * np.pi / w
    num_steps = int(simulation_periods * solver_steps_per_period)
    tlist = np.linspace(0.0, simulation_periods * drive_period, num_steps + 1,
                        dtype=np.float32)
    return w, drive_period, tlist


def frame_params(animated_parameter, value, base, wq1):
    """Apply the selected animation parameter to the base parameters."""
    # Only one physical parameter is changed per animation frame. Keeping the
    # symbolic model fixed lets GQIS reuse the compiled RHS where possible.
    params = dict(base)
    subtitle = "Static parameters"

    if animated_parameter == "wd_mhz":
        params["wd_mhz"] = float(value)
        subtitle = f"wd={params['wd_mhz']:.1f} MHz"
    elif animated_parameter == "gamma1":
        params["gamma1"] = float(value)
        params["gamma2"] = params["gamma1"] / 2.0 + params["gammaph"]
        subtitle = f"gamma1={float(value):.4f}*Delta"
    elif animated_parameter == "gamma2":
        params["gamma2"] = float(value)
        subtitle = f"gamma2={float(value):.4f}*Delta"
    elif animated_parameter == "gammaph":
        params["gammaph"] = float(value)
        params["gamma2"] = params["gamma1"] / 2.0 + params["gammaph"]
        subtitle = f"gammaph={float(value):.4f}*Delta"
    elif animated_parameter == "kappa":
        params["kappa"] = float(value)
        subtitle = f"kappa={float(value):.4f}*Delta"
    elif animated_parameter == "g1":
        params["g1"] = float(value) * wq1 / 1.23
        subtitle = f"g={float(value):.4f}*Delta"
    elif animated_parameter == "Ap":
        params["Ap"] = float(value) * wq1
        subtitle = f"Ap={float(value):.4f}*Delta"
    elif animated_parameter == "same":
        subtitle = "Static parameters"
    else:
        raise ValueError(f"Unsupported animated_parameter '{animated_parameter}'.")

    return params, subtitle


def gpu_time_evolution(A_list, eps_list, tlist, model, params, output_mode, animated_parameter, *,
                       RHSreuse=True, averaging_skip_fraction=0.0, timings=False, fp64=False):
    """Run one 2D GPU sweep for the four-level model."""
    # Values in const_values either get folded into the generated RHS or, for
    # selected sweep modes below, remain runtime constants so animation frames do
    # not require full symbolic regeneration.
    const_syms = model["const_syms"]
    const_values = {const_syms["delta"]: float(params["delta"]),
                    const_syms["w"]: float(params["w"]),
                    const_syms["gamma1"]: float(params["gamma1"]),
                    const_syms["gamma2"]: float(params["gamma2"]),
                    const_syms["kappa"]: float(params["kappa"]),
                    const_syms["Ap"]: float(params["Ap"]),
                    const_syms["g1"]: float(params["g1"]),
                    const_syms["wr2"]: float(params["wr2"]),
                    const_syms["wpr"]: float(params["wr2"])}

    # Constants listed here are passed at runtime for the corresponding animated
    # parameter. Other constants are substituted before CUDA code generation.
    parameter_to_runtime_names = {"wd_mhz": {"w"}, "gamma1": {"gamma1", "gamma2"},
                                  "gamma2": {"gamma2"}, "gammaph": {"gamma2"},
                                  "kappa": {"kappa"}, "g1": {"g1"}, "Ap": {"Ap"},
                                  "same": {"g1"}}
    keep_names = parameter_to_runtime_names.get(animated_parameter, set())
    keep_syms = {const_syms[name] for name in keep_names}
    runtime_consts = {s: const_values[s] for s in keep_syms}

    # Choose which observable is averaged over time.
    if output_mode == "response":
        mean_op = model["Mean_Response"]
    elif output_mode == "qubit_population":
        mean_op = model["Mean_Population"]
    else:
        raise ValueError("output_mode must be 'response' or 'qubit_population'.")

    # The leading arguments define the physical model, averaged observable, and
    # fixed-step grid. Only constants changed by the animated parameter stay
    # symbolic at runtime, allowing the generated RHS and kernel to be reused.
    result = mesolve_2D(model["H"], model["Drive"], model["Col_Ops"], mean_op, tlist,
                        var_arrays={model["eps_sym"]: eps_list, model["A_sym"]: A_list},
                        const_values=const_values, RHSreuse=RHSreuse, runtime_consts=runtime_consts,
                        keep_symbolic_consts=keep_syms if keep_syms else None, output_mode="mean",
                        warmup_time=averaging_skip_fraction, timings=timings, fp64=fp64)
    return np.abs(np.real(np.asarray(result))).T


def main() -> None:
    start_total = time.time()
    settings = user_settings()

    # Fixed model constants ------------------------------------------------
    # These define the base physical scales for the four-level model.
    delta = settings["delta"]
    wq1 = settings["wq1"]
    wr2 = settings["wr2"]

    # Base physical parameters --------------------------------------------
    # These values define the model before one selected parameter is animated.
    wd_mhz = settings["wd_mhz"]
    gammaph = settings["gammaph"]
    gamma1 = settings["gamma1"]
    gamma2 = gamma1 / 2.0 + gammaph
    kappa = settings["kappa"]
    Ap = settings["Ap"]
    g1 = settings["g1"]
    simulation_periods = settings["simulation_periods"]
    eps_max = settings["eps_max"]
    A_max = settings["A_max"]
    averaging_skip_fraction = settings["averaging_skip_fraction"]

    # Match Example_02 units before entering the symbolic model.
    delta_abs = wq1 * delta
    Ap_abs = Ap * wq1
    g1_abs = g1 * wq1
    eps_max_abs = eps_max * wq1
    A_max_abs = A_max * wq1

    # Sweep and animation settings ----------------------------------------
    # Total work scales as grid_size**2 * simulation_periods * solver_steps_per_period.
    forward_frame_count = int(settings["forward_frame_count"])
    animated_parameter = settings["animated_parameter"]
    parameter_values = np.asarray(settings["animated_parameter_values"], dtype=np.float32)
    if forward_frame_count < 1:
        raise ValueError("forward_frame_count must be at least 1.")
    if parameter_values.ndim != 1 or len(parameter_values) != forward_frame_count:
        raise ValueError("animated_parameter_values must be a one-dimensional array with "
                         "forward_frame_count elements.")

    output_mode = settings["output_mode"]  # "response" or "qubit_population"
    playback_mode = settings["playback_mode"]  # "forward" or "pingpong"
    if playback_mode not in {"forward", "pingpong"}:
        raise ValueError("playback_mode must be 'forward' or 'pingpong'.")
    solver_steps_per_period = settings["solver_steps_per_period"]
    grid_size = settings["grid_size"]
    save_mp4 = settings["save_mp4"]
    video_filename = settings["video_filename"]
    video_fps = settings["video_fps"]
    video_dpi = settings["video_dpi"]
    ffmpeg_preset = settings["ffmpeg_preset"]
    ffmpeg_crf = settings["ffmpeg_crf"]

    base = {"delta": float(delta_abs), "wd_mhz": float(wd_mhz),
            "gammaph": float(gammaph), "gamma1": float(gamma1), "gamma2": float(gamma2),
            "kappa": float(kappa), "Ap": float(Ap_abs), "g1": float(g1_abs),
            "wr2": float(wr2)}

    # Build static axes and symbolic model once ---------------------------
    # The symbolic model is reused for all frames; only selected constants change.
    eps_list = np.linspace(-eps_max_abs, eps_max_abs, grid_size, dtype=np.float32)
    A_list = np.linspace(0.0, A_max_abs, grid_size, dtype=np.float32)
    model = build_four_level_model()

    # Pingpong avoids repeating the turnaround endpoint; the initial endpoint
    # returns as the final frame for smooth looping.
    if playback_mode == "pingpong" and len(parameter_values) > 1:
        frame_indices = np.concatenate([np.arange(len(parameter_values), dtype=np.int32),
                                        np.arange(len(parameter_values) - 2, -1, -1,
                                                  dtype=np.int32)])
    else:
        frame_indices = np.arange(len(parameter_values), dtype=np.int32)

    # Initial frame --------------------------------------------------------
    # This call may include symbolic RHS generation and CUDA compilation. Later
    # frames can reuse the compiled RHS when only runtime constants change.
    params0, subtitle0 = frame_params(animated_parameter, float(parameter_values[0]), base,
                                      wq1)
    w0, _, tlist0 = build_time_grid(delta=delta, wd_mhz=params0["wd_mhz"],
                                    simulation_periods=simulation_periods,
                                    solver_steps_per_period=solver_steps_per_period)
    params0["w"] = w0
    map0 = gpu_time_evolution(A_list, eps_list, tlist0, model, params0, output_mode,
                              animated_parameter, RHSreuse=True,
                              averaging_skip_fraction=averaging_skip_fraction, timings=False,
                              fp64=False)

    if output_mode == "response":
        img0 = 10.0 * np.log10(np.clip(map0, 1.0e-30, None))
        cbar_label = r"$|S_{21}|\ (\mathrm{dB})$"
    else:
        img0 = map0
        cbar_label = "Qubit population"

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(img0, aspect="auto",
                    cmap="jet" if output_mode == "response" else "viridis", origin="lower",
                    extent=[eps_list[0] / wq1, eps_list[-1] / wq1, 0.0, A_max_abs / wq1])
    fig.colorbar(img, ax=ax, label=cbar_label)
    ax.set_xlabel(r"$\epsilon/\Delta$")
    ax.set_ylabel(r"$A/\Delta$")
    title = ax.set_title(f"Example 04 | animated={animated_parameter} | {subtitle0}")
    plt.tight_layout()

    def update(frame_pos):
        # Matplotlib calls this once per frame. Each call computes a full 2D GPU
        # sweep for the current value of the animated parameter.
        frame_start = time.time()
        frame_idx = int(frame_indices[frame_pos])
        value = float(parameter_values[frame_idx])
        params, subtitle = frame_params(animated_parameter, value, base, wq1)
        w_now, _, tlist = build_time_grid(delta=delta, wd_mhz=params["wd_mhz"],
                                          simulation_periods=simulation_periods,
                                          solver_steps_per_period=solver_steps_per_period)
        params["w"] = w_now
        map_lin = gpu_time_evolution(A_list, eps_list, tlist, model, params, output_mode,
                                     animated_parameter, RHSreuse=True,
                                     averaging_skip_fraction=averaging_skip_fraction,
                                     timings=False, fp64=False)

        if output_mode == "response":
            img_data = 10.0 * np.log10(np.clip(map_lin, 1.0e-30, None))
        else:
            img_data = map_lin

        img.set_data(img_data)
        img.set_clim(float(np.nanmin(img_data)), float(np.nanmax(img_data)))
        title.set_text(f"Example 04 | animated={animated_parameter} | {subtitle}")
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

    plt.show()
    print(f"Total runtime: {time.time() - start_total:.2f}s")


def user_settings() -> dict:
    """User-editable physical, numerical, and animation settings."""

    wd_mhz, A_max = 500.0, 2.234042553191489 * 1.15

    # Alternative regime. Comment the active assignment above and uncomment this line.
    # wd_mhz, A_max = 1500.0, 2.234042553191489*1.15*1.3

    # Animation parameter sweep. The array contains one value for each forward
    # frame. Pingpong playback produces 2*forward_frame_count - 1 total frames.
    forward_frame_count = 25
    animated_parameter, frame_values = "g1", np.linspace(1.0e-7, 0.1, forward_frame_count)

    # Alternative animation sweeps. Comment the active assignment above and
    # uncomment exactly one complete assignment below.
    # animated_parameter, frame_values = "wd_mhz", np.linspace(500.0, 1500.0, forward_frame_count)
    # animated_parameter, frame_values = "gamma1", np.linspace(1.0e-3, 0.5, forward_frame_count)
    # animated_parameter, frame_values = "gammaph", np.linspace(1.0e-3, 0.5, forward_frame_count)
    # animated_parameter, frame_values = "kappa", tanh_space(5.0e-3, 8.0, forward_frame_count)
    # animated_parameter, frame_values = "Ap", np.linspace(1.0e-4, 0.3, forward_frame_count)
    # animated_parameter, frame_values = "same", np.zeros(forward_frame_count)

    return {
        # Fixed model constants.
        "delta": 1.0,  # dimensionless qubit gap multiplier before conversion
        "wq1": 4.71 * 1.15,  # qubit frequency scale used to convert normalized inputs
        "wr2": 7.6767,  # resonator frequency in model frequency units
        # Base physical parameter set.
        # These are unscaled input values, matching Example_02; main() converts
        # them to the absolute units used by the symbolic model.
        "wd_mhz": wd_mhz,  # drive frequency in MHz
        "gammaph": 1.0e-3,  # pure-dephasing rate
        "gamma1": 2.0e-3,  # qubit relaxation rate
        "kappa": 5.0e-3,  # resonator relaxation rate
        "Ap": 0.0002 / 10.0,  # probe amplitude
        "g1": 0.04033970276 / 1.23,  # qubit-resonator coupling
        "simulation_periods": 60,  # simulated duration per frame in drive periods
        "eps_max": 2.09916,  # detuning half-range in units converted by wq1
        "A_max": A_max,  # maximum normalized drive amplitude
        "averaging_skip_fraction": 0.0,  # initial time fraction excluded from the average
        # Animation parameter and values.
        "forward_frame_count": forward_frame_count,  # before optional reverse playback
        "animated_parameter": animated_parameter,
        "animated_parameter_values": np.asarray(frame_values, dtype=np.float32),
        # Numerical and output settings.
        "output_mode": "response",  # "response" or "qubit_population"
        "playback_mode": "pingpong",  # "forward" or "pingpong"
        # Increase if mesolve_2D reports non-finite output.
        "solver_steps_per_period": 250,
        "grid_size": 512,  # square grid side per animation frame
        "save_mp4": False,  # False shows interactive animation only
        "video_filename": "Example_04_four_level_animation.mp4",
        "video_fps": 5,  # playback rate, not the number of calculated frames
        "video_dpi": 120,  # saved video resolution scale
        "ffmpeg_preset": "medium",  # x264 speed/compression preset
        "ffmpeg_crf": 23,  # x264 quality; lower is higher quality/larger file
    }


if __name__ == "__main__":
    main()
