# Optimizations Done & Tried So Far

Ranked from the earliest to the latest. Each title is followed by their commit head or branch.

## Original - Branch `candle-based-implementation`

No optimizations. Just a simple engine implemented with `candle`. Good for study the core inference logics and algorithms.

## Hand-written Tensor - Commit Head `6b23eebc20e3391e907684f9a4a7dd1515ac7f9f`

Implemented a tensor for the engine. The performance is on par with the `candle` cpu backend.

## SIMD Optimization - Commit Head `d3478247cd482324d52d2effdd25dcf44da9a084`

TinyTensor uses [`rten-simd`](https://github.com/robertknight/rten/tree/main/rten-simd) to accelerate broadcast binary operations. Contiguous values are processed using the best SIMD instruction set available at runtime, while scalar broadcasting and arbitrary-stride fallbacks preserve the original tensor semantics.

In an Instruments profiling run on Apple Silicon, adding this SIMD path reduced end-to-end inference time from approximately **4m 3s to 1m 5s**, a **\~3.7× speedup**. The share of samples attributed to `compute_linear_layer` fell from **26% to 5.8%**.

These figures are from a single profiling comparison rather than a controlled benchmark. Actual gains depend on the model workload, tensor shapes, CPU, build configuration, and proportion of operations eligible for SIMD.

## Parallelism & Weights Fusion in `compute_swiglu` - Never committed

In the profiles collected for this experiment, `compute_swiglu` accounted for 48.6–48.7% of the instrumented inference time, making it the largest single hotspot. I evaluated two ways to reduce its cost. The first fuses the gate and up projection weights into one linear operation, then splits the result before applying the gate. Because both projections consume the same hidden state, this preserves the SwiGLU computation while allowing one GEMM to perform both projections.

The fused implementation performed roughly the same as the original implementation. I initially considered the tensor split on the fly in `compute_swiglu` a possible source of overhead, so I also pre-fused the gate and up weights while loading the model. That still produced no material improvement, indicating that neither the split nor the timing of the weight fusion was the limiting factor.

I also ran the independent gate and up projections concurrently without fusion. This regressed the observed steady-state performance: average latency increased from 430.71 ms/token to 449.27 ms/token, median latency rose from 404.49 ms/token to 415.88 ms/token, P95 latency increased from 695.82 ms/token to 737.05 ms/token, and throughput fell from 2.32 to 2.23 tokens per second. The `compute_swiglu` total similarly increased from 21.11 s to 23.08 s. Time to first token fell from 988.84 ms to 807.95 ms in this single run, but that does not establish an end-to-end improvement. The runs emitted 96 and 101 tokens respectively, so they are not a fully controlled comparison.

This outcome is consistent with the existing matrix multiplication implementation: each GEMM already requests Rayon parallelism using `std::thread::available_parallelism()`. Spawning additional threads for the two projections therefore makes concurrent GEMMs compete for the same CPU resources, cache, and memory bandwidth instead of providing additive parallelism. Neither fusion nor outer projection-level parallelism provided a significant improvement in this implementation, so I moved on to other optimization opportunities.

## SIMD Optimization for GEMM - Commit Head `d9982f885d38e418fbaf8a1c359e2d546b8b705d`

In the profiling run that prompted this investigation, `matrix_multiply` represented approximately **41%** of the observed performance cost, making GEMM an obvious candidate for further work. However, TinyTensor already passes `f32` input slices and an `f32` destination buffer directly to the resolved [`gemm` 0.19.0](https://github.com/sarah-quinones/gemm) dependency. It also requests Rayon parallelism using `std::thread::available_parallelism()`. The hot path therefore enters `gemm`'s specialized `f32` implementation rather than a naïve scalar triple loop.

The dependency contains explicit architecture-specific SIMD microkernels and runtime feature dispatch; it does not rely solely on compiler auto-vectorization. Its x86 FMA implementation uses 256-bit `__m256` operations, processing **eight `f32` values per vector**, while the opt-in `x86-v4` feature provides AVX-512 kernels with **16 `f32` lanes**. On the AArch64 Apple Silicon target used for the profiling work, the standard implementation uses four-lane NEON `f32` vectors and fused multiply-add operations. At initialization, `gemm` selects the best supported implementation for the current CPU and retains a scalar fallback.

The project now also enables `gemm`'s `experimental-apple-amx` feature. On a supported native Apple Silicon target (only M1, M2 and M3, no beyond), the crate can select its AMX microkernel instead of NEON; otherwise, it falls back to NEON path.

After enabling AMX on an M3 Macbook Air, the performance share of `matrix_multiply` drops from ~41% to ~37%. A new in-house SIMD GEMM remains a low-priority duplicate of work already done by the dependency. The 41% profile share bounds the possible end-to-end benefit.

## KV Cache - Commit Head `41bc7a455382b0a906b03d24a287a956b2f6be4b`

Traditionally, a transformer model will accept a full text's tokens for predicting the next token. If we want the model to generate coherent text, we need to append the newly generated token to the full text's tokens and then pass all of them into the model again for the next token. This means that every time we generate a new token, the model recomputes QKV tensors across all layers, and the amount of computation keeps growing. Before I implemented the KV cache, the generation speed was on average about 1.x–2.x tokens per second on an M3 MacBook Air.

To properly compute attention and thus generate the next token, we need a Q tensor for each layer of the current generation loop, and K and V tensors for each layer from both the current and previous loops. Q does not need to survive into the next generation loop, while K and V can be reused. So for each layer of the current generation loop, instead of recomputing K and V from the start, we simply compute the current K and V, retrieve the previous KVs, and concatenate the previous KVs with the current KVs. This significantly cuts down the time spent repeatedly computing the KVs. This is why we call this method a KV cache. After I implemented the KV cache, generation throughput increased by roughly 3×, reaching around 6.x tokens per second.

From this attempt, I realized that shape is one of the most important things in tensor-related algorithms. While debugging my KV cache, I found that most of the bugs were related to mismatched shapes. Programming syntax issues were negligible by comparison. This further showed me the importance of developing sensitivity to shapes when working with tensors.
