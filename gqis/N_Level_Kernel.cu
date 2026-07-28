// Canonical CUDA kernel template packaged with GQIS.
#define N ${N_DECL}
#define NUM_STATICS ${NUM_STATICS_DECL}
#define NUM_DRIVES ${NUM_DRIVES_DECL}

extern "C" {

// Optional knobs to influence register pressure.
#ifndef DRIVES_NOINLINE
#define DRIVES_NOINLINE 0
#endif
#if DRIVES_NOINLINE
#define DRIVES_ATTR __device__ __noinline__
#else
#define DRIVES_ATTR __device__ __forceinline__
#endif

#ifndef DRHO_NOINLINE
#define DRHO_NOINLINE 0
#endif
#if DRHO_NOINLINE
#define DRHO_ATTR __device__ __noinline__
#else
#define DRHO_ATTR __device__ __forceinline__
#endif

#ifndef KERNEL_LAUNCH_BOUNDS
#define KERNEL_LAUNCH_BOUNDS
#endif

// Compute thread-local static expressions once per parameter point.
DRIVES_ATTR void compute_static_terms(
    float ParX,
    float ParY,
    float* __restrict__ Static_arr,
    const float* __restrict__ Const_arr
)
{
    // SymPy generated lines
    #INSERT_STATICS#
}

// Compute time-dependent drive signals once per stage-time and reuse across multiple RHS evals.
DRIVES_ATTR void compute_drives(
    float ParX,
    float ParY,
    float t,
    float* __restrict__ Drive_arr,
    const float* __restrict__ Static_arr,
    const float* __restrict__ Const_arr
)
{
    // SymPy generated lines
    #INSERT_DRIVES#
}

// Force-inline to help the compiler with liveness/reg allocation across the RK stages.
DRHO_ATTR void compute_drho(
    const float* __restrict__ rho,
    float ParX,
    float ParY,
    const float* __restrict__ Drive_arr,
    const float* __restrict__ Static_arr,
    float* __restrict__ d_rho,
    const float* __restrict__ Const_arr
)
{
    // SymPy generated lines
    #INSERT_DRHO#
    //d_rho[0]=-2*Delta*rho[2]-(rho[0]-1.0f)*gamma1S;
    //d_rho[1]=2*Drive*rho[2]-rho[1]*gamma2S;
    //d_rho[2]=2*Delta*rho[0]-2*Drive*rho[1]-rho[2]*gamma2S;
}

__global__ KERNEL_LAUNCH_BOUNDS void time_evolution_kernel(
    const float dt,
    const int num_ParX,
    const int num_ParY,
    const int num_steps,
    const int warmup_steps,
    const float* __restrict__ ParX_list,
    const float* __restrict__ ParY_list, 
    #CONST_ARG_DECL#
    #RHO0_ARG_DECL#
    float2* __restrict__ results // averaged population of level 1
)   
{
    const int idx_ParX = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx_ParY = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx_ParX >= num_ParX || idx_ParY >= num_ParY) return;

    const float dt2 = dt * 0.5f;
    const float dt6 = dt / 6.0f;

        const float ParX = ParX_list[idx_ParX];

            const float ParY = ParY_list[idx_ParY]; 
            const int result_idx = idx_ParX * num_ParY + idx_ParY;

            float rho[N];
            #INIT_RHO#
            
            float2 avg = make_float2(0.0f, 0.0f);

            float k_tmp[N];
            float rho_tmp[N];
            float accum[N];
            float Static[NUM_STATICS];
            float Drive[NUM_DRIVES];

            compute_static_terms(ParX, ParY, Static, Const_arr);
            compute_drives(ParX, ParY, 0.0f, Drive, Static, Const_arr);
            compute_drho(rho,ParX,ParY, Drive, Static, k_tmp, Const_arr);
            for (int step = 0; step < num_steps; ++step)
            {   
                // Derive stage times from the integer step to avoid accumulated
                // float32 drift in long, phase-sensitive integrations.
                float t_mid = ((float)step + 0.5f) * dt;
                compute_drives(ParX, ParY, t_mid, Drive, Static, Const_arr);

                // --- k2 ---
                //#pragma unroll
                for (int i = 0; i < N; ++i)
                {
                    rho_tmp[i] = fmaf(dt2, k_tmp[i], rho[i]);
                    accum[i] = k_tmp[i];
                }
                compute_drho(rho_tmp,ParX,ParY, Drive, Static, k_tmp, Const_arr);
                
                // --- k3 ---
                //#pragma unroll
                for (int i = 0; i < N; ++i)
                { 
                    rho_tmp[i] = fmaf(dt2, k_tmp[i], rho[i]);
                    accum[i] = fmaf(2.0f , k_tmp[i],accum[i]);
                }
                // same stage time as k2 (center)
                compute_drho(rho_tmp,ParX,ParY, Drive, Static, k_tmp, Const_arr);
                // --- k4 ---
                //#pragma unroll
                t_mid +=dt2;
                compute_drives(ParX, ParY, t_mid, Drive, Static, Const_arr);
                for (int i = 0; i < N; ++i)
                {
                    rho_tmp[i] = fmaf(dt, k_tmp[i], rho[i]);
                    accum[i] = fmaf(2.0f , k_tmp[i],accum[i]);
                }
                
                compute_drho(rho_tmp,ParX,ParY, Drive, Static, k_tmp, Const_arr);

                // update rho += dt/6*(k1 + 2*k2 + 2*k3 + k4)
                //#pragma unroll
                for (int i = 0; i < N; ++i)
                {
                    accum[i] += k_tmp[i];
                    rho[i] = fmaf(dt6, accum[i], rho[i]);
                }

                // Classical RK4 is not FSAL: k4 cannot be reused as the next
                // step's k1 because it was evaluated at rho + dt*k3, not at
                // the updated RK4 state.
                if (step + 1 < num_steps)
                {
                    compute_drho(rho,ParX,ParY, Drive, Static, k_tmp, Const_arr);
                }

                #MEAN_LINE#    
            }
            #FINAL_LINE#
            #RESULTS_LINE#
            }

}
