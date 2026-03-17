# Metal Threadgroup Barrier Bug Repro

This repository contains a reproducible case on Apple Silicon where compute kernel output changes based on the pipeline's `maxTotalThreadsPerThreadgroup` specialization when using `threadgroup_barrier(mem_flags::mem_threadgroup)`.

The repro is contained in `main.swift`. It builds two shader variants that differ only in their barrier flags:

- `threadgroup_barrier(mem_flags::mem_threadgroup)`
- `threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)`

Each variant is compiled twice: once with `maxTotalThreadsPerThreadgroup=33`, and once with `maxTotalThreadsPerThreadgroup=32`.

The kernel is always dispatched with `threadsPerThreadgroup=25`, and it also contains an `if (tid >= 25) return;` guard at the top of the shader. As a result, only the first 25 lanes ever participate in the computation. From the kernel's perspective, no execution path requires 32 active threads.

Observed behavior:

- With `maxTotalThreadsPerThreadgroup=33`, both shader variants produce correct results.
- With `maxTotalThreadsPerThreadgroup=32`, the variant using only `mem_flags::mem_threadgroup` produces incorrect output.
- With `maxTotalThreadsPerThreadgroup=32`, the variant using `mem_flags::mem_device | mem_flags::mem_threadgroup` continues to produce correct output.

Expected behavior:

- Reducing `maxTotalThreadsPerThreadgroup` from 33 to 32 should not change the result for this kernel, because only 25 threads ever participate in the computation.

This suggests an issue related to pipeline specialization on `maxTotalThreadsPerThreadgroup` in the presence of `threadgroup_barrier(mem_flags::mem_threadgroup)`.

To run this repro, use the following command

```bash
swift main.swift
```

Additionally, the repro can be used to dump the generated shader artifacts for inspection, which may be useful for debugging the issue. To do this, use the following command:

```bash
swift main.swift --dump-runtime-artifacts
```

This also writes a copy of stdout and environment metadata that may be useful for debugging.
