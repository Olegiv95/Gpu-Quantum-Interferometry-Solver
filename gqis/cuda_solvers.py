"""Fixed-step CUDA solver fragments inserted into the common GQIS kernel."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


@dataclass(frozen=True)
class SolverSpec:
    """Metadata and generated CUDA source for one fixed-step solver."""

    name: str
    label: str
    order: int
    rhs_evaluations: int
    source: str
    parameters: tuple[float, ...] = ()


def _float_literal(value: float) -> str:
    """Return a CUDA FP32 literal; the FP64 code path removes the suffix."""
    if value == 0.0:
        return "0.0f"
    if value == 1.0:
        return "1.0f"
    if value == -1.0:
        return "-1.0f"
    return f"{value:.17g}f"


def _combination_lines(target: str, terms: list[tuple[float, str]], *,
                       declaration: bool, indent: str) -> list[str]:
    """Emit an FMA chain for one scalar linear combination."""
    prefix = "float " if declaration else ""
    nonzero = [(coefficient, source) for coefficient, source in terms if coefficient]
    if not nonzero:
        return [f"{indent}{prefix}{target} = 0.0f;"]
    lines = [f"{indent}{prefix}{target} = 0.0f;"]
    for coefficient, source in nonzero:
        lines.append(f"{indent}{target} = fmaf({_float_literal(coefficient)}, "
                     f"{source}, {target});")
    return lines


def _rk5_source(label: str, c: tuple[float, ...],
                a: tuple[tuple[float, ...], ...], b: tuple[float, ...]) -> str:
    """Generate a six-stage fifth-order RK body with five scratch N-arrays.

    RHS evaluations are the primary cost in GQIS, so this implementation keeps
    all six RK stages and avoids any derivative recomputation.  Four derivative
    buffers let k1..k4 stay live through stage four.  In the normal stage-four
    update loop, the only combination still needed by stage six is folded into
    ``k_tmp``; after Y5 is materialized the old k1..k4 values can be overwritten.

    This costs five supplementary arrays including ``rho_tmp`` but avoids the
    extra compression pass required by a generic four-array realization.

    Drive values are reused whenever consecutive stages have the same c value.
    If the final stage has c=1, its endpoint Drive/Static values are also reused
    directly for the next step's pipelined k1, exactly as in the RK4 path.
    """
    if len(b) != 6 or len(c) != 6 or len(a) != 6:
        raise ValueError("The compact RK5 generator requires a six-stage tableau")
    if any(len(a[index]) != index for index in range(6)):
        raise ValueError("Invalid explicit six-stage RK tableau")

    d21 = a[1][0] - b[0]
    d31, d32 = a[2][0] - b[0], a[2][1] - b[1]
    d41, d42, d43 = a[3][0] - b[0], a[3][1] - b[1], a[3][2] - b[2]
    d51, d52, d53, d54 = tuple(a[4][j] - b[j] for j in range(4))
    d61, d62, d63, d64, d65 = tuple(a[5][j] - b[j] for j in range(5))

    lines: list[str] = []

    def append_drive_for_stage(stage: int) -> None:
        """Update drives only when this stage uses a new time abscissa."""
        if abs(c[stage] - c[stage - 1]) > 5.0e-14:
            lines.append(
                f"                compute_drives(ParX, ParY, fmaf({_float_literal(c[stage])}, dt, t), "
                "Drive, Static, Const_arr);"
            )

    lines.extend([
        f"            // GQIS solver: {label}",
        "            // Six-stage, fifth-order fixed-step RK.",
        "            // Five scratch N-arrays; exactly six RHS evaluations/step.",
        "            // No extra compression pass: k1..k4 stay live through stage four.",
        "            float k_tmp[N];",       # k1 -> stage-6 combination -> k6 / next k1
        "            float k_accum1[N];",    # k2 -> k5
        "            float k_accum2[N];",    # k3
        "            float k_accum3[N];",    # k4
        "            float rho_tmp[N];",     # current stage state
        "",
        "            compute_drives(ParX, ParY, 0.0f, Drive, Static, Const_arr);",
        "            compute_drho(rho, ParX, ParY, Drive, Static, k_tmp, Const_arr);",
        "            for (int step = 0; step < num_steps; ++step)",
        "            {",
        "                const float t = (float)step * dt;",
        "",
        "                // k1 is pipelined in k_tmp. Accumulate b1 and form Y2.",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k1 = k_tmp[i];",
    ])
    if b[0]:
        lines.append(f"                    rho[i] = fmaf(dt * {_float_literal(b[0])}, k1, rho[i]);")
    lines.extend(_combination_lines(
        "stage_sum", [(d21, "k1")], declaration=True, indent="                    "))
    lines.extend([
        "                    rho_tmp[i] = fmaf(dt, stage_sum, rho[i]);",
        "                }",
    ])
    append_drive_for_stage(1)
    lines.extend([
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_accum1, Const_arr);",
        "",
        "                // k2: accumulate b2 and form Y3.",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k1 = k_tmp[i];",
        "                    const float k2 = k_accum1[i];",
    ])
    if b[1]:
        lines.append(f"                    rho[i] = fmaf(dt * {_float_literal(b[1])}, k2, rho[i]);")
    lines.extend(_combination_lines(
        "stage_sum", [(d31, "k1"), (d32, "k2")],
        declaration=True, indent="                    "))
    lines.extend([
        "                    rho_tmp[i] = fmaf(dt, stage_sum, rho[i]);",
        "                }",
    ])
    append_drive_for_stage(2)
    lines.extend([
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_accum2, Const_arr);",
        "",
        "                // k3: accumulate b3 and form Y4.",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k1 = k_tmp[i];",
        "                    const float k2 = k_accum1[i];",
        "                    const float k3 = k_accum2[i];",
    ])
    if b[2]:
        lines.append(f"                    rho[i] = fmaf(dt * {_float_literal(b[2])}, k3, rho[i]);")
    lines.extend(_combination_lines(
        "stage_sum", [(d41, "k1"), (d42, "k2"), (d43, "k3")],
        declaration=True, indent="                    "))
    lines.extend([
        "                    rho_tmp[i] = fmaf(dt, stage_sum, rho[i]);",
        "                }",
    ])
    append_drive_for_stage(3)
    lines.extend([
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_accum3, Const_arr);",
        "",
        "                // k4: form Y5 directly. In the same loop fold k1..k4 into",
        "                // the single old-derivative combination still needed by Y6.",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k1 = k_tmp[i];",
        "                    const float k2 = k_accum1[i];",
        "                    const float k3 = k_accum2[i];",
        "                    const float k4 = k_accum3[i];",
    ])
    if b[3]:
        lines.append(f"                    rho[i] = fmaf(dt * {_float_literal(b[3])}, k4, rho[i]);")
    lines.extend(_combination_lines(
        "stage_sum", [(d51, "k1"), (d52, "k2"), (d53, "k3"), (d54, "k4")],
        declaration=True, indent="                    "))
    lines.extend(_combination_lines(
        "future6", [(d61, "k1"), (d62, "k2"), (d63, "k3"), (d64, "k4")],
        declaration=True, indent="                    "))
    lines.extend([
        "                    rho_tmp[i] = fmaf(dt, stage_sum, rho[i]);",
        "                    k_tmp[i] = future6;",
        "                }",
        "",
    ])
    append_drive_for_stage(4)
    lines.extend([
        "                // k1..k4 are dead after Y5/future6, so reuse k_accum1 for k5.",
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_accum1, Const_arr);",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k5 = k_accum1[i];",
    ])
    if b[4]:
        lines.append(f"                    rho[i] = fmaf(dt * {_float_literal(b[4])}, k5, rho[i]);")
    if d65:
        lines.append(f"                    k_tmp[i] = fmaf({_float_literal(d65)}, k5, k_tmp[i]);")
    lines.extend([
        "                    rho_tmp[i] = fmaf(dt, k_tmp[i], rho[i]);",
        "                }",
        "",
    ])
    append_drive_for_stage(5)
    lines.extend([
        "                // future6 is consumed, so reuse k_tmp for k6.",
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_tmp, Const_arr);",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
    ])
    if b[5]:
        lines.append(f"                    rho[i] = fmaf(dt * {_float_literal(b[5])}, k_tmp[i], rho[i]);")
    lines.extend(["                }"])

    if abs(c[5] - 1.0) > 5.0e-14:
        lines.append("                compute_drives(ParX, ParY, t + dt, Drive, Static, Const_arr);")
    lines.extend([
        "                if (step + 1 < num_steps)",
        "                    compute_drho(rho, ParX, ParY, Drive, Static, k_tmp, Const_arr);",
        "                #MEAN_LINE#",
        "            }",
    ])
    return "\n".join(lines)


def _compressed_rk_source(label: str, order: int, c: tuple[float, ...],
                          a: tuple[tuple[float, ...], ...], b: tuple[float, ...],
                          pivot: int) -> str:
    """Generate a fixed-step explicit RK body with liveness-based vector reuse.

    At ``pivot`` the still-live early derivatives are folded into one pending
    combination for each remaining stage.  Each combination buffer is reused
    by that stage's derivative as soon as it is consumed.  This avoids retaining
    every RK stage while adding no RHS evaluations or derivative recomputation.
    """
    stages = len(b)
    if len(c) != stages or len(a) != stages or not 1 < pivot < stages - 1:
        raise ValueError("Invalid compressed explicit-RK tableau or pivot")
    if any(len(a[index]) != index for index in range(stages)):
        raise ValueError("Invalid explicit RK tableau")
    d = tuple(tuple(a[i][j] - b[j] for j in range(i)) for i in range(stages))
    active = {0: 0}
    free: list[int] = []
    slot_count = 1
    records = []

    def allocate() -> int:
        nonlocal slot_count
        if free:
            slot = min(free)
            free.remove(slot)
            return slot
        slot = slot_count
        slot_count += 1
        return slot

    # Ordinary liveness reuse before the compression point.
    for stage in range(1, pivot):
        before = dict(active)
        dead = [j for j in active
                if not any(d[future][j] for future in range(stage + 1, stages))]
        for j in dead:
            free.append(active.pop(j))
        slot = allocate()
        active[stage] = slot
        records.append((stage, before, slot))

    pivot_before = dict(active)
    future_stages = tuple(range(pivot + 1, stages))
    available_slots = sorted(set(active.values()) | set(free))
    while len(available_slots) < len(future_stages) + 1:
        available_slots.append(slot_count)
        slot_count += 1
    future_slots = dict(zip(future_stages, available_slots))
    pivot_slot = available_slots[len(future_stages)]
    active = {pivot: pivot_slot}
    pending = dict(future_slots)
    records.append((pivot, pivot_before, pivot_slot))

    # After compression, consume each pending combination and immediately reuse
    # its slot.  Derivatives die as soon as no later stage needs them.
    post_records = []
    for stage in range(pivot + 1, stages):
        before = dict(active)
        base_slot = pending.pop(stage)
        dead = [j for j in active
                if not any(d[future][j] for future in range(stage + 1, stages))]
        freed = {active[j] for j in dead}
        for j in dead:
            del active[j]
        slot = min({base_slot, *freed})
        active[stage] = slot
        post_records.append((stage, before, base_slot, slot))

    lines = [
        f"            // GQIS solver: {label}",
        f"            // {stages}-stage, order-{order} fixed-step explicit RK main formula.",
        "            // Embedded error estimates, dense output, and adaptive control are omitted.",
        f"            // {slot_count + 1} reused work vectors; no derivative recomputation.",
        "            float " + ", ".join(f"k_slot{slot}[N]" for slot in range(slot_count)) + ";",
        "            float rho_tmp[N];",
        "",
        "            compute_drives(ParX, ParY, 0.0f, Drive, Static, Const_arr);",
        "            compute_drho(rho, ParX, ParY, Drive, Static, k_slot0, Const_arr);",
        "            for (int step = 0; step < num_steps; ++step)",
        "            {",
        "                const float t = (float)step * dt;",
    ]

    def append_stage(stage: int, before: dict[int, int], output_slot: int,
                     base_slot: int | None = None) -> None:
        previous = stage - 1
        lines.extend([
            "                #SOLVER_UNROLL#",
            "                for (int i = 0; i < N; ++i)",
            "                {",
        ])
        for derivative, slot in sorted(before.items()):
            lines.append(f"                    const float k{derivative + 1} = k_slot{slot}[i];")
        if b[previous]:
            lines.append(f"                    rho[i] = fmaf(dt * {_float_literal(b[previous])}, "
                         f"k{previous + 1}, rho[i]);")
        terms = []
        if base_slot is not None:
            terms.append((1.0, f"k_slot{base_slot}[i]"))
        terms.extend((d[stage][j], f"k{j + 1}") for j in sorted(before))
        lines.extend(_combination_lines("stage_sum", terms, declaration=True,
                                        indent="                    "))
        lines.append("                    rho_tmp[i] = fmaf(dt, stage_sum, rho[i]);")
        if stage == pivot:
            for future, slot in future_slots.items():
                terms = [(d[future][j], f"k{j + 1}") for j in sorted(before)]
                lines.extend(_combination_lines(f"future{future + 1}", terms,
                                                declaration=True,
                                                indent="                    "))
                lines.append(f"                    k_slot{slot}[i] = future{future + 1};")
        lines.extend([
            "                }",
            f"                compute_drives(ParX, ParY, fmaf({_float_literal(c[stage])}, dt, t), "
            "Drive, Static, Const_arr);",
            f"                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_slot{output_slot}, "
            "Const_arr);",
        ])

    for stage, before, slot in records:
        append_stage(stage, before, slot)
    for stage, before, base_slot, slot in post_records:
        append_stage(stage, before, slot, base_slot)

    final_slot = active[stages - 1]
    lines.extend([
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        f"                    rho[i] = fmaf(dt * {_float_literal(b[-1])}, "
        f"k_slot{final_slot}[i], rho[i]);",
        "                }",
    ])
    if abs(c[-1] - 1.0) > 5.0e-14:
        lines.append("                compute_drives(ParX, ParY, t + dt, Drive, Static, Const_arr);")
    lines.extend([
        "                if (step + 1 < num_steps)",
        f"                    compute_drho(rho, ParX, ParY, Drive, Static, k_slot{final_slot}, "
        "Const_arr);",
        "                #MEAN_LINE#",
        "            }",
    ])
    return "\n".join(lines)


def _anas5_source() -> str:
    """Return the fitted Anastassi-Simos method with five scratch N-arrays."""
    b1, b3, b4, b5, b6 = 23 / 216, 63 / 136, 9 / 56, 1000 / 3213, -1 / 24
    d21 = 1 / 10 - b1
    d31, d32 = -2 / 9 - b1, 5 / 9
    d41, d42, d43 = 28 / 9 - b1, -40 / 9, 2.0 - b3
    d51, d52 = -11277 / 8000 - b1, 171 / 80
    d53, d54 = -459 / 2000 - b3, 3213 / 8000 - b4
    lines = [
        "            // GQIS solver: Anas5",
        "            // Fixed-step frequency-fitted Anastassi-Simos method; six RHS evaluations/step.",
        "            // solver_param0 is fitted once on the host from v=solver_frequency*dt.",
        "            // The method is fifth order when that frequency is accurate, otherwise fourth order.",
        "            float k_tmp[N], k_accum1[N], k_accum2[N], k_accum3[N], rho_tmp[N];",
        "            compute_drives(ParX, ParY, 0.0f, Drive, Static, Const_arr);",
        "            compute_drho(rho, ParX, ParY, Drive, Static, k_tmp, Const_arr);",
        "            for (int step = 0; step < num_steps; ++step)",
        "            {",
        "                const float t = (float)step * dt;",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k1 = k_tmp[i];",
        f"                    rho[i] = fmaf(dt * {_float_literal(b1)}, k1, rho[i]);",
        f"                    rho_tmp[i] = fmaf(dt * {_float_literal(d21)}, k1, rho[i]);",
        "                }",
        "                compute_drives(ParX, ParY, fmaf(0.1f, dt, t), Drive, Static, Const_arr);",
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_accum1, Const_arr);",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
    ]
    lines.extend(_combination_lines("stage_sum", [(d31, "k_tmp[i]"), (d32, "k_accum1[i]")],
                                    declaration=True, indent="                    "))
    lines.extend([
        "                    rho_tmp[i] = fmaf(dt, stage_sum, rho[i]);",
        "                }",
        "                compute_drives(ParX, ParY, fmaf(0.3333333333333333f, dt, t), Drive, Static, Const_arr);",
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_accum2, Const_arr);",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k3 = k_accum2[i];",
        f"                    rho[i] = fmaf(dt * {_float_literal(b3)}, k3, rho[i]);",
    ])
    lines.extend(_combination_lines(
        "stage_sum", [(d41, "k_tmp[i]"), (d42, "k_accum1[i]"), (d43, "k3")],
        declaration=True, indent="                    "))
    lines.extend([
        "                    rho_tmp[i] = fmaf(dt, stage_sum, rho[i]);",
        "                }",
        "                compute_drives(ParX, ParY, fmaf(0.6666666666666667f, dt, t), Drive, Static, Const_arr);",
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_accum3, Const_arr);",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k4 = k_accum3[i];",
        f"                    rho[i] = fmaf(dt * {_float_literal(b4)}, k4, rho[i]);",
    ])
    lines.extend(_combination_lines(
        "stage_sum", [(d51, "k_tmp[i]"), (d52, "k_accum1[i]"),
                      (d53, "k_accum2[i]"), (d54, "k4")],
        declaration=True, indent="                    "))
    lines.extend([
        "                    rho_tmp[i] = fmaf(dt, stage_sum, rho[i]);",
        "                    float future6 = 0.0f;",
        f"                    future6 = fmaf({_float_literal(-4.0 - b1)}, k_tmp[i], future6);",
        "                    future6 = fmaf(5.0f, k_accum1[i], future6);",
        f"                    future6 = fmaf((-{_float_literal(b3)} + 1.89f * solver_param0), k_accum2[i], future6);",
        f"                    future6 = fmaf((-{_float_literal(b4)} - 2.295f * solver_param0), k4, future6);",
        "                    future6 = fmaf(-0.595f * solver_param0, k_tmp[i], future6);",
        "                    k_tmp[i] = future6;",
        "                }",
        "                compute_drives(ParX, ParY, fmaf(0.9f, dt, t), Drive, Static, Const_arr);",
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_accum1, Const_arr);",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        "                    const float k5 = k_accum1[i];",
        f"                    rho[i] = fmaf(dt * {_float_literal(b5)}, k5, rho[i]);",
        f"                    k_tmp[i] = fmaf((solver_param0 - {_float_literal(b5)}), k5, k_tmp[i]);",
        "                    rho_tmp[i] = fmaf(dt, k_tmp[i], rho[i]);",
        "                }",
        "                compute_drives(ParX, ParY, t + dt, Drive, Static, Const_arr);",
        "                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_tmp, Const_arr);",
        "                #SOLVER_UNROLL#",
        "                for (int i = 0; i < N; ++i)",
        "                {",
        f"                    rho[i] = fmaf(dt * {_float_literal(b6)}, k_tmp[i], rho[i]);",
        "                }",
        "                if (step + 1 < num_steps)",
        "                    compute_drho(rho, ParX, ParY, Drive, Static, k_tmp, Const_arr);",
        "                #MEAN_LINE#",
        "            }",
    ])
    return "\n".join(lines)


def _rk4_source() -> str:
    """Return the existing compact classical RK4 implementation unchanged."""
    return r"""            // GQIS solver: RK4
            const float dt2 = dt * 0.5f;
            const float dt6 = dt / 6.0f;
            float k_tmp[N];
            float rho_tmp[N];
            float accum[N];

            compute_drives(ParX, ParY, 0.0f, Drive, Static, Const_arr);
            compute_drho(rho, ParX, ParY, Drive, Static, k_tmp, Const_arr);
            for (int step = 0; step < num_steps; ++step)
            {
                float t_mid = ((float)step + 0.5f) * dt;
                compute_drives(ParX, ParY, t_mid, Drive, Static, Const_arr);
                #SOLVER_UNROLL#
                for (int i = 0; i < N; ++i)
                {
                    rho_tmp[i] = fmaf(dt2, k_tmp[i], rho[i]);
                    accum[i] = k_tmp[i];
                }
                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_tmp, Const_arr);
                #SOLVER_UNROLL#
                for (int i = 0; i < N; ++i)
                {
                    rho_tmp[i] = fmaf(dt2, k_tmp[i], rho[i]);
                    accum[i] = fmaf(2.0f, k_tmp[i], accum[i]);
                }
                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_tmp, Const_arr);
                t_mid += dt2;
                compute_drives(ParX, ParY, t_mid, Drive, Static, Const_arr);
                #SOLVER_UNROLL#
                for (int i = 0; i < N; ++i)
                {
                    rho_tmp[i] = fmaf(dt, k_tmp[i], rho[i]);
                    accum[i] = fmaf(2.0f, k_tmp[i], accum[i]);
                }
                compute_drho(rho_tmp, ParX, ParY, Drive, Static, k_tmp, Const_arr);
                #SOLVER_UNROLL#
                for (int i = 0; i < N; ++i)
                {
                    accum[i] += k_tmp[i];
                    rho[i] = fmaf(dt6, accum[i], rho[i]);
                }
                if (step + 1 < num_steps)
                    compute_drho(rho, ParX, ParY, Drive, Static, k_tmp, Const_arr);
                #MEAN_LINE#
            }"""


def _lserk4_source() -> str:
    """Return a compact Carpenter-Kennedy five-stage low-storage RK4 body."""
    a = (0.0, -567301805773 / 1357537059087, -2404267990393 / 2016746695238,
         -3550918686646 / 2091501179385, -1275806237668 / 842570457699)
    b = (1432997174477 / 9575080441755, 5161836677717 / 13612068292357,
         1720146321549 / 2090206949498, 3134564353537 / 4481467310338,
         2277821191437 / 14882151754819)
    c = (0.0, 1432997174477 / 9575080441755, 2526269341429 / 6820363962896,
         2006345519317 / 3224310063776, 2802321613138 / 2924317926251)
    lines = ["            // GQIS solver: LSRK4",
             "            // Carpenter-Kennedy five-stage fourth-order 2N-storage scheme.",
             "            float residual[N], k_tmp[N];",
             "            compute_drives(ParX, ParY, 0.0f, Drive, Static, Const_arr);",
             "            for (int step = 0; step < num_steps; ++step)",
             "            {",
             "                const float t = (float)step * dt;"]
    for stage in range(5):
        if stage > 0:
            lines.extend([
                f"                compute_drives(ParX, ParY, fmaf({_float_literal(c[stage])}, dt, t), "
                "Drive, Static, Const_arr);",
            ])
        lines.extend([
            "                compute_drho(rho, ParX, ParY, Drive, Static, k_tmp, Const_arr);",
            "                #SOLVER_UNROLL#",
            "                for (int i = 0; i < N; ++i)",
            "                {",
        ])
        if stage == 0:
            lines.append("                    residual[i] = dt * k_tmp[i];")
        else:
            lines.append(f"                    residual[i] = fmaf({_float_literal(a[stage])}, "
                         "residual[i], dt * k_tmp[i]);")
        lines.extend([f"                    rho[i] = fmaf({_float_literal(b[stage])}, "
                      "residual[i], rho[i]);", "                }"])
    if abs(c[-1] - 1.0) > 5.0e-14:
        lines.append("                compute_drives(ParX, ParY, t + dt, Drive, Static, Const_arr);")
    lines.extend([
        "                #MEAN_LINE#",
        "            }",
    ])
    return "\n".join(lines)

def _ab5_start_step(step: int, derivative_in: str, derivative_out: str) -> str:
    """Generate one RK4 startup step, reusing the new history slot as k_tmp."""
    return f"""            if (step < num_steps)
            {{
                const float t = (float)step * dt;
                compute_drives(ParX, ParY, fmaf(0.5f, dt, t), Drive, Static, Const_arr);
                #SOLVER_UNROLL#
                for (int i = 0; i < N; ++i)
                {{
                    rho_tmp[i] = fmaf(dt2, {derivative_in}[i], rho[i]);
                    accum[i] = {derivative_in}[i];
                }}
                compute_drho(rho_tmp, ParX, ParY, Drive, Static, {derivative_out}, Const_arr);
                #SOLVER_UNROLL#
                for (int i = 0; i < N; ++i)
                {{
                    rho_tmp[i] = fmaf(dt2, {derivative_out}[i], rho[i]);
                    accum[i] = fmaf(2.0f, {derivative_out}[i], accum[i]);
                }}
                compute_drho(rho_tmp, ParX, ParY, Drive, Static, {derivative_out}, Const_arr);
                #SOLVER_UNROLL#
                for (int i = 0; i < N; ++i)
                {{
                    rho_tmp[i] = fmaf(dt, {derivative_out}[i], rho[i]);
                    accum[i] = fmaf(2.0f, {derivative_out}[i], accum[i]);
                }}
                compute_drives(ParX, ParY, t + dt, Drive, Static, Const_arr);
                compute_drho(rho_tmp, ParX, ParY, Drive, Static, {derivative_out}, Const_arr);
                #SOLVER_UNROLL#
                for (int i = 0; i < N; ++i)
                {{
                    accum[i] += {derivative_out}[i];
                    rho[i] = fmaf(dt6, accum[i], rho[i]);
                }}
                if (step + 1 < num_steps)
                    compute_drho(rho, ParX, ParY, Drive, Static, {derivative_out}, Const_arr);
                #MEAN_LINE#
                ++step;
            }}"""


def _ab5_source() -> str:
    """Return AB5 with RK4 startup and circular derivative-history reuse."""
    histories = (
        ("f4", "f3", "f2", "f1", "f0", "f0"),
        ("f0", "f4", "f3", "f2", "f1", "f1"),
        ("f1", "f0", "f4", "f3", "f2", "f2"),
        ("f2", "f1", "f0", "f4", "f3", "f3"),
        ("f3", "f2", "f1", "f0", "f4", "f4"),
    )
    coefficients = (1901.0 / 720.0, -1387.0 / 360.0, 109.0 / 30.0,
                    -637.0 / 360.0, 251.0 / 720.0)
    cases = []
    for case, (*sources, destination) in enumerate(histories):
        case_lines = [f"                    case {case}:",
                      "                    {", "                        #SOLVER_UNROLL#",
                      "                        for (int i = 0; i < N; ++i)",
                      "                        {", "                            float sum = 0.0f;"]
        for coefficient, source in zip(coefficients, sources):
            case_lines.append(f"                            sum = fmaf("
                              f"{_float_literal(coefficient)}, {source}[i], sum);")
        case_lines.extend([
            "                            rho[i] = fmaf(dt, sum, rho[i]);",
            "                        }",
            "                        compute_drives(ParX, ParY, t + dt, Drive, Static, "
            "Const_arr);",
            "                        if (step + 1 < num_steps)",
            f"                            compute_drho(rho, ParX, ParY, Drive, Static, "
            f"{destination}, Const_arr);",
            "                        break;",
            "                    }",
        ])
        cases.extend(case_lines)
    startup = "\n".join(_ab5_start_step(i, f"f{i}", f"f{i + 1}") for i in range(4))
    return f"""            // GQIS solver: AB5
            // Fifth-order Adams-Bashforth with four RK4 startup steps.
            // f1...f4 double as the RK4 derivative scratch while each history slot is built.
            const float dt2 = dt * 0.5f;
            const float dt6 = dt / 6.0f;
            float f0[N], f1[N], f2[N], f3[N], f4[N];
            float rho_tmp[N], accum[N];
            compute_drives(ParX, ParY, 0.0f, Drive, Static, Const_arr);
            compute_drho(rho, ParX, ParY, Drive, Static, f0, Const_arr);
            int step = 0;
{startup}
            for (; step < num_steps; ++step)
            {{
                const float t = (float)step * dt;
                switch ((step - 4) % 5)
                {{
{chr(10).join(cases)}
                }}
                #MEAN_LINE#
            }}"""


_TSIT5_C = (0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0)
_TSIT5_A = (
    (),
    (0.161,),
    (-0.008480655492356989, 0.335480655492357),
    (2.8971530571054935, -6.359448489975075, 4.3622954328695815),
    (5.325864828439257, -11.748883564062828, 7.4955393428898365,
     -0.09249506636175525),
    (5.86145544294642, -12.92096931784711, 8.159367898576159,
     -0.071584973281401, -0.028269050394068383),
)
_TSIT5_B = (0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742,
            -3.290069515436081, 2.324710524099774)

_DP5_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0)
_DP5_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
)
_DP5_B = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)

_ALSHINA6_C = (0.0, 4 / 7, 5 / 7, 6 / 7, 0.276393202250021,
               0.7236067977499789, 1.0)
_ALSHINA6_A = (
    (),
    (4 / 7,),
    (1.0267857142857142, -0.3125),
    (0.9349206349206349, 0.2777777777777778, -0.35555555555555557),
    (0.18002567144208434, 0.14737940683961614, 0.016070163293314288,
     -0.0670820393249937),
    (-0.07979765856603138, 0.025290591998992584, -0.3516326202266813,
     0.3207294901687516, 0.8090169943749475),
    (0.4988599356197352, -0.8633499941930429, 1.6778122846668349,
     -1.2682372542187894, -0.42705098312484235, 1.381966011250105),
)
_ALSHINA6_B = (1 / 12, 0.0, 0.0, 0.0, 5 / 12, 5 / 12, 1 / 12)

_DOP853_C = (0.0, 0.05260015195876773, 0.0789002279381516,
             0.1183503419072274, 0.2816496580927726, 1 / 3, 0.25,
             0.3076923076923077, 0.6512820512820513, 0.6, 6 / 7, 1.0)
_DOP853_A = (
    (),
    (0.05260015195876773,),
    (0.0197250569845379, 0.0591751709536137),
    (0.02958758547680685, 0.0, 0.08876275643042054),
    (0.2413651341592667, 0.0, -0.8845494793282861, 0.924834003261792),
    (0.037037037037037035, 0.0, 0.0, 0.17082860872947386,
     0.12546768756682242),
    (0.037109375, 0.0, 0.0, 0.17025221101954405, 0.06021653898045596,
     -0.017578125),
    (0.03709200011850479, 0.0, 0.0, 0.17038392571223998,
     0.10726203044637328, -0.015319437748624402, 0.008273789163814023),
    (0.6241109587160757, 0.0, 0.0, -3.3608926294469414,
     -0.868219346841726, 27.59209969944671, 20.154067550477894,
     -43.48988418106996),
    (0.47766253643826434, 0.0, 0.0, -2.4881146199716677,
     -0.590290826836843, 21.230051448181193, 15.279233632882423,
     -33.28821096898486, -0.020331201708508627),
    (-0.9371424300859873, 0.0, 0.0, 5.186372428844064,
     1.0914373489967295, -8.149787010746927, -18.52006565999696,
     22.739487099350505, 2.4936055526796523, -3.0467644718982196),
    (2.273310147516538, 0.0, 0.0, -10.53449546673725,
     -2.0008720582248625, -17.9589318631188, 27.94888452941996,
     -2.8589982771350235, -8.87285693353063, 12.360567175794303,
     0.6433927460157636),
)
_DOP853_B = (0.054293734116568765, 0.0, 0.0, 0.0, 0.0,
             4.450312892752409, 1.8915178993145003, -5.801203960010585,
             0.3111643669578199, -0.1521609496625161, 0.20136540080403034,
             0.04471061572777259)

def _fitted_solver_parameters(name: str, frequency: float | None,
                              step_size: float | None) -> tuple[float, ...]:
    """Evaluate the fixed-step Anas5(w) coefficient from v=w*dt on the host."""
    if name != "anas5" or step_size is None:
        return ()
    if not mp.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("anas5 requires a positive finite step_size")
    frequency = 1.0 if frequency is None else float(frequency)
    if not mp.isfinite(frequency) or frequency < 0.0:
        raise ValueError("solver_frequency must be a finite non-negative angular frequency")
    with mp.workdps(60):
        v = mp.mpf(str(frequency)) * mp.mpf(str(step_size))
        if v == 0:
            return (float(mp.mpf(-800) / 1071),)
        tangent = mp.tan(v)
        a43 = mp.mpf(2)
        numerator = (-a43 * v**5 + 6 * tangent * v**4 + 24 * v**3
                     - 72 * tangent * v**2 - 144 * v + 144 * tangent)
        denominator = v**5 * (a43 * tangent * v + 12 - 10 * a43)
        return (float((-mp.mpf(8000) / 1071) * numerator / denominator),)


_SOLVERS = {
    "rk4": SolverSpec("rk4", "RK4", 4, 4, _rk4_source()),
    "lserk4": SolverSpec("lserk4", "LSRK4", 4, 5, _lserk4_source()),
    "dp5": SolverSpec("dp5", "DP5", 5, 6,
                      _rk5_source("DP5", _DP5_C, _DP5_A, _DP5_B)),
    "ab5": SolverSpec("ab5", "AB5", 5, 1, _ab5_source()),
    "anas5": SolverSpec("anas5", "Anas5", 5, 6, _anas5_source()),
    "tsit5": SolverSpec("tsit5", "Tsit5", 5, 6,
                        _rk5_source("Tsit5", _TSIT5_C, _TSIT5_A, _TSIT5_B)),
    "alshina6": SolverSpec(
        "alshina6", "Alshina6", 6, 7,
        _compressed_rk_source("Alshina6", 6, _ALSHINA6_C, _ALSHINA6_A,
                              _ALSHINA6_B, pivot=4)),
    "dop853": SolverSpec(
        "dop853", "DOP853", 8, 12,
        _compressed_rk_source("DOP853", 8, _DOP853_C, _DOP853_A,
                              _DOP853_B, pivot=7)),
}

_ALIASES = {
    "classical_rk4": "rk4",
    "carpenter_kennedy": "lserk4",
    "carpenter_kennedy_2n54": "lserk4",
    "dormand_prince": "dp5",
    "dormand_prince_5": "dp5",
    "rk45": "dp5",
    "anastassi5": "anas5",
    "anastassi_simos": "anas5",
    "adams_bashforth_5": "ab5",
    "alshina": "alshina6",
    "dormand_prince_853": "dop853",
    "dop_853": "dop853",
}


def normalize_solver_name(solver: str) -> str:
    """Return the canonical lower-case solver name or raise ValueError."""
    name = str(solver).strip().lower().replace(" ", "_").replace("-", "_")
    name = _ALIASES.get(name, name)
    if name not in _SOLVERS:
        available = ", ".join(_SOLVERS)
        raise ValueError(f"Unsupported solver '{solver}'. Use one of: {available}")
    return name


def get_solver_spec(solver: str, *, frequency: float | None = None,
                    step_size: float | None = None) -> SolverSpec:
    """Return normalized metadata and the sole CUDA fragment to compile."""
    spec = _SOLVERS[normalize_solver_name(solver)]
    parameters = _fitted_solver_parameters(spec.name, frequency, step_size)
    return SolverSpec(spec.name, spec.label, spec.order, spec.rhs_evaluations,
                      spec.source, parameters)


def available_solvers() -> tuple[str, ...]:
    """Return canonical public solver option names in display order."""
    return tuple(_SOLVERS)
