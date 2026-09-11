using DifferentialEquations
using DiffEqGPU
using CUDA
using StaticArrays
using DelimitedFiles

function bloch_rhs(u, p, t)
    x = u[1]
    y = u[2]
    z = u[3]
    s = u[4]
    A = p[1]
    eps = p[2]
    delta = p[3]
    w = p[4]
    gamma1 = p[5]
    gamma2_eff = p[6]
    warmup_t = p[7]
    drive = eps + A * sin(w * t)
    dx = -drive * y - gamma2_eff * x
    dy = drive * x - delta * z - gamma2_eff * y
    dz = delta * y - gamma1 * (z - 1.0f0)
    p_exc = 0.5f0 * (1.0f0 - z)
    ds = (t >= warmup_t) ? p_exc : 0.0f0
    return @SVector [dx, dy, dz, ds]
end

function main()
    if length(ARGS) < 15
        println(stderr, "Expected 15 args.")
        exit(2)
    end
    out_csv = ARGS[1]
    nx = parse(Int, ARGS[2])
    ny = parse(Int, ARGS[3])
    num_t = parse(Int, ARGS[4])
    dt = parse(Float64, ARGS[5])
    eps_min = parse(Float64, ARGS[6])
    eps_max = parse(Float64, ARGS[7])
    A_min = parse(Float64, ARGS[8])
    A_max = parse(Float64, ARGS[9])
    delta = parse(Float64, ARGS[10])
    w = parse(Float64, ARGS[11])
    gamma1 = parse(Float64, ARGS[12])
    gamma2 = parse(Float64, ARGS[13])
    warmup_steps = parse(Int, ARGS[14])
    t0 = parse(Float64, ARGS[15])

    eps_list = collect(Float32, range(Float32(eps_min), stop=Float32(eps_max), length=nx))
    A_list = collect(Float32, range(Float32(A_min), stop=Float32(A_max), length=ny))

    gamma2_eff = Float32(0.5f0 * Float32(gamma1) + 2.0f0 * Float32(gamma2))
    warmup_t = Float32(t0 + warmup_steps * dt)

    params = Vector{SVector{7, Float32}}(undef, nx * ny)
    k = 1
    for j in 1:ny
        Aj = A_list[j]
        for i in 1:nx
            epsi = eps_list[i]
            params[k] = @SVector [Aj, epsi, Float32(delta), Float32(w), Float32(gamma1), gamma2_eff, warmup_t]
            k += 1
        end
    end

    u0 = @SVector Float32[0.0f0, 0.0f0, 1.0f0, 0.0f0]
    tf = t0 + dt * max(num_t - 1, 0)
    prob = ODEProblem{false}(bloch_rhs, u0, (Float32(t0), Float32(tf)), params[1])
    prob_func = (pr, i, repeat) -> remake(pr; p=params[i])
    eprob = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)

    backend = CUDA.CUDABackend()
    alg = DiffEqGPU.EnsembleGPUKernel(backend)
    sol = solve(
        eprob,
        GPUTsit5(),
        alg;
        trajectories=length(params),
        adaptive=false,
        dt=Float32(dt),
        save_everystep=false
    )

    denom = Float32(max(num_t - 1 - warmup_steps, 1)) * Float32(dt)
    out = zeros(Float32, ny, nx)
    for idx in 1:length(params)
        j = Int(fld(idx - 1, nx)) + 1
        i = Int(mod(idx - 1, nx)) + 1
        s_final = sol[idx][end][4]
        out[j, i] = s_final / denom
    end

    writedlm(out_csv, out, ',')
end

main()
