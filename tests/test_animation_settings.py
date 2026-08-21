import numpy as np

import Example_01_two_level_basic as example_01
import Example_02_four_level_interferogram as example_02
import Example_03_two_level_animation as example_03
import Example_04_four_level_animation as example_04
import Example_05_initial_condition_sweep_gate_fidelity as example_05


def test_animation_settings_separate_frame_count_from_parameter_values():
    for module in (example_03, example_04):
        settings = module.user_settings()
        values = np.asarray(settings["animated_parameter_values"])

        assert settings["forward_frame_count"] >= 1
        assert values.ndim == 1
        assert len(values) == settings["forward_frame_count"]
        assert settings["animated_parameter"]
        assert {"frame_count", "mode", "varray"}.isdisjoint(settings)


def test_interferogram_examples_use_consistent_time_grid_names():
    legacy_names = {"tr", "periods", "samples_per_period", "rk4_steps_per_period",
                    "warmup_time"}
    for module in (example_01, example_02, example_03, example_04):
        settings = module.user_settings()
        assert "simulation_periods" in settings
        assert "solver_steps_per_period" in settings
        assert legacy_names.isdisjoint(settings)


def test_examples_keep_conventional_physics_parameter_names():
    verbose_physics_names = {"energy_gap", "drive_angular_frequency", "pure_dephasing_rate",
                             "qubit_relaxation_rate", "qubit_decoherence_rate",
                             "resonator_relaxation_rate", "probe_amplitude",
                             "qubit_resonator_coupling", "resonator_frequency",
                             "qubit_frequency_scale"}
    for module in (example_01, example_02, example_03, example_04, example_05):
        assert verbose_physics_names.isdisjoint(module.user_settings())


def test_four_level_rate_animation_preserves_decoherence_definition():
    base = {"delta": 1.0, "wd_mhz": 500.0, "gammaph": 0.03, "gamma1": 0.10,
            "gamma2": 0.08, "kappa": 0.01, "Ap": 0.001, "g1": 0.02, "wr2": 7.0}

    params, _ = example_04.frame_params("gamma1", 0.40, base, wq1=1.0)
    assert np.isclose(params["gamma2"], 0.40 / 2.0 + base["gammaph"])

    params, _ = example_04.frame_params("gammaph", 0.20, base, wq1=1.0)
    assert np.isclose(params["gamma2"], base["gamma1"] / 2.0 + 0.20)

    params, _ = example_04.frame_params("gamma2", 0.70, base, wq1=1.0)
    assert np.isclose(params["gamma2"], 0.70)
