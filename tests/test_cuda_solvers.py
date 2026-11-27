import pytest

from gqis.cuda_solvers import available_solvers, get_solver_spec, normalize_solver_name


EXPECTED_SOLVERS = ("rk4", "lserk4", "lserk4f", "dp5", "ab5", "anas5", "tsit5",
                    "alshina6", "dop853", "tsit5a")


def test_solver_names_and_aliases():
    assert available_solvers() == EXPECTED_SOLVERS
    assert normalize_solver_name("RK4") == "rk4"
    assert normalize_solver_name("RK45") == "dp5"
    assert normalize_solver_name("Anastassi-Simos") == "anas5"
    assert normalize_solver_name("Alshina") == "alshina6"
    assert normalize_solver_name("DOP-853") == "dop853"
    with pytest.raises(ValueError, match="Unsupported solver"):
        normalize_solver_name("not_a_solver")


def test_each_cuda_fragment_contains_only_its_selected_solver():
    markers = {name: f"// GQIS solver: {get_solver_spec(name, step_size=0.1).label}"
               for name in EXPECTED_SOLVERS}
    for name in EXPECTED_SOLVERS:
        source = get_solver_spec(name, step_size=0.1).source
        lines = [line.strip() for line in source.splitlines()]
        assert lines.count(markers[name]) == 1
        assert "#SOLVER_CODE#" not in source
        assert "#SOLVER_UNROLL#" in source
        for other_name, marker in markers.items():
            if other_name != name:
                assert marker not in lines


def test_reported_steady_rhs_counts_include_fsal_reuse():
    assert get_solver_spec("rk4").rhs_evaluations == 4
    assert get_solver_spec("lserk4").rhs_evaluations == 5
    for name in ("dp5", "anas5", "tsit5"):
        assert get_solver_spec(name, step_size=0.1).rhs_evaluations == 6
    assert get_solver_spec("ab5").rhs_evaluations == 1
    assert get_solver_spec("alshina6").rhs_evaluations == 7
    assert get_solver_spec("dop853").rhs_evaluations == 12


def test_high_order_sources_use_compressed_storage_without_extra_rhs_work():
    alshina = get_solver_spec("alshina6").source
    dop853 = get_solver_spec("dop853").source
    assert "5 reused work vectors" in alshina
    assert "6 reused work vectors" in dop853
    assert alshina.count("compute_drho(") == 8  # initial + 6 stages + endpoint reuse
    assert dop853.count("compute_drho(") == 13  # initial + 11 stages + endpoint reuse
    assert alshina.count("compute_drives(") == 7
    assert dop853.count("compute_drives(") == 12


def test_fitted_solver_parameters_are_runtime_values():
    anas = get_solver_spec("anas5", frequency=1.14, step_size=0.01)
    assert len(anas.parameters) == 1
    assert "solver_param0" in anas.source


def test_tsit5a_options_are_validated_and_only_accepted_by_tsit5a():
    spec = get_solver_spec("tsit5a", step_size=0.1, adaptive_options={"step_rtol": 1e-6})
    assert spec.parameters[:3] == (1e-6, 1e-5, 64.0)
    assert "adaptive_steps[" not in spec.source  # count_steps is off by default
    with pytest.raises(ValueError, match="only to solver='tsit5a'"):
        get_solver_spec("tsit5", step_size=0.1, adaptive_options={"step_rtol": 1e-6})
    with pytest.raises(ValueError, match="Unknown adaptive_options"):
        get_solver_spec("tsit5a", adaptive_options={"rtol": 1e-6})
