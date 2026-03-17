# Metal Threadgroup Barrier Bug Repro

This repository is a repro for a bug in the Metal/Swift shader compilation pipeline on Apple Silicon involving `threadgroup_barrier` and pipeline specialization via `maxTotalThreadsPerThreadgroup`.

The repro is contained in `main.swift`, which builds two shader variants that differ only in their use of `threadgroup_barrier` flags. One version uses `threadgroup_barrier(mem_flags::mem_threadgroup)`, while the other uses `threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)`.

The important trigger is not the dispatched thread count. The kernel is always dispatched with `threadsPerThreadgroup=25`, and there is also an `if (tid >= 25) return;` guard at the top of the shader so only the first 25 lanes ever participate in the computation. In other words, the shader never has 32 or more active threads doing useful work, so changing the pipeline's `maxTotalThreadsPerThreadgroup` from 33 to 32 should not matter semantically for this kernel.

However, it does matter in practice. The repro builds each shader twice: once with `maxTotalThreadsPerThreadgroup=33`, and once with `maxTotalThreadsPerThreadgroup=32`. With `maxTotalThreadsPerThreadgroup=33`, both barrier variants produce correct results. With `maxTotalThreadsPerThreadgroup=32`, the variant using only `mem_flags::mem_threadgroup` produces incorrect output, while the variant using `mem_flags::mem_device | mem_flags::mem_threadgroup` still produces correct results.

That makes this repo a repro for a bug whose trigger appears to be pipeline specialization on `maxTotalThreadsPerThreadgroup`, even though the kernel itself never uses 32 active threads.

To run this repro, use the following command

```bash
swift main.swift
```

Additionally, the repro can be used to dump the generated shader artifacts for inspection, which may be useful for debugging the issue. To do this, use the following command:

```bash
swift main.swift --dump-runtime-artifacts
```

This will also make a copy of stdout and record other metadata about the environment that may be useful for debugging.
