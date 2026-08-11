# Optimizations Done So Far

## SIMD optimization

TinyTensor uses [`rten-simd`](https://github.com/robertknight/rten/tree/main/rten-simd) to accelerate broadcast binary operations. Contiguous values are processed using the best SIMD instruction set available at runtime, while scalar broadcasting and arbitrary-stride fallbacks preserve the original tensor semantics.

In an Instruments profiling run on Apple Silicon, adding this SIMD path reduced end-to-end inference time from approximately **4m 3s to 1m 5s**, a **\~3.7× speedup**. The share of samples attributed to `compute_linear_layer` fell from **26% to 5.8%**.

These figures are from a single profiling comparison rather than a controlled benchmark. Actual gains depend on the model workload, tensor shapes, CPU, build configuration, and proportion of operations eligible for SIMD.
