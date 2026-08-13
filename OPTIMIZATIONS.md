# Optimizations Done & Tried So Far

Ranked from the earliest to the latest.

## SIMD Optimization

TinyTensor uses [`rten-simd`](https://github.com/robertknight/rten/tree/main/rten-simd) to accelerate broadcast binary operations. Contiguous values are processed using the best SIMD instruction set available at runtime, while scalar broadcasting and arbitrary-stride fallbacks preserve the original tensor semantics.

In an Instruments profiling run on Apple Silicon, adding this SIMD path reduced end-to-end inference time from approximately **4m 3s to 1m 5s**, a **\~3.7× speedup**. The share of samples attributed to `compute_linear_layer` fell from **26% to 5.8%**.

These figures are from a single profiling comparison rather than a controlled benchmark. Actual gains depend on the model workload, tensor shapes, CPU, build configuration, and proportion of operations eligible for SIMD.

## Parallelism & Weights Fusion in `compute_swiglu`

In the profiles collected for this experiment, `compute_swiglu` accounted for 48.6–48.7% of the instrumented inference time, making it the largest single hotspot. I evaluated two ways to reduce its cost. The first fuses the gate and up projection weights into one linear operation, then splits the result before applying the gate. Because both projections consume the same hidden state, this preserves the SwiGLU computation while allowing one GEMM to perform both projections.

The fused implementation performed roughly the same as the original implementation. I initially considered the tensor split on the fly in `compute_swiglu` a possible source of overhead, so I also pre-fused the gate and up weights while loading the model. That still produced no material improvement, indicating that neither the split nor the timing of the weight fusion was the limiting factor.

I also ran the independent gate and up projections concurrently without fusion. This regressed the observed steady-state performance: average latency increased from 430.71 ms/token to 449.27 ms/token, median latency rose from 404.49 ms/token to 415.88 ms/token, P95 latency increased from 695.82 ms/token to 737.05 ms/token, and throughput fell from 2.32 to 2.23 tokens per second. The `compute_swiglu` total similarly increased from 21.11 s to 23.08 s. Time to first token fell from 988.84 ms to 807.95 ms in this single run, but that does not establish an end-to-end improvement. The runs emitted 96 and 101 tokens respectively, so they are not a fully controlled comparison.

This outcome is consistent with the existing matrix multiplication implementation: each GEMM already requests Rayon parallelism using `std::thread::available_parallelism()`. Spawning additional threads for the two projections therefore makes concurrent GEMMs compete for the same CPU resources, cache, and memory bandwidth instead of providing additive parallelism. Neither fusion nor outer projection-level parallelism provided a significant improvement in this implementation, so I moved on to other optimization opportunities.

## SIMD Optimization for GEMM

In the profiling run that prompted this investigation, `matrix_multiply` represented approximately **41%** of the observed performance cost, making GEMM an obvious candidate for further work. However, TinyTensor already passes `f32` input slices and an `f32` destination buffer directly to the resolved [`gemm` 0.19.0](https://github.com/sarah-quinones/gemm) dependency. It also requests Rayon parallelism using `std::thread::available_parallelism()`. The hot path therefore enters `gemm`'s specialized `f32` implementation rather than a naïve scalar triple loop.

The dependency contains explicit architecture-specific SIMD microkernels and runtime feature dispatch; it does not rely solely on compiler auto-vectorization. Its x86 FMA implementation uses 256-bit `__m256` operations, processing **eight `f32` values per vector**, while the opt-in `x86-v4` feature provides AVX-512 kernels with **16 `f32` lanes**. On the AArch64 Apple Silicon target used for the profiling work, the standard implementation uses four-lane NEON `f32` vectors and fused multiply-add operations. At initialization, `gemm` selects the best supported implementation for the current CPU and retains a scalar fallback.

The project now also enables `gemm`'s `experimental-apple-amx` feature. On a supported native Apple Silicon target (only M1, M2 and M3, no beyond), the crate can select its AMX microkernel instead of NEON; otherwise, it falls back to NEON path.

After enabling AMX on an M3 Macbook Air, the performance share of `matrix_multiply` drops from ~41% to ~37%. A new in-house SIMD GEMM remains a low-priority duplicate of work already done by the dependency. The 41% profile share bounds the possible end-to-end benefit.
