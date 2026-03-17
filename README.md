# Metal Threadgroup Barrier Bug Repro

This repository contains a reproducible case on Apple Silicon where compute kernel output changes based on the pipeline's `maxTotalThreadsPerThreadgroup` specialization when using `threadgroup_barrier(mem_flags::mem_threadgroup)`.

The repro is contained in `main.swift`. It builds two shader variants that differ only in their barrier flags:

- `threadgroup_barrier(mem_flags::mem_threadgroup)`
- `threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)`

Each variant is compiled twice: once with `maxTotalThreadsPerThreadgroup=33`, and once with `maxTotalThreadsPerThreadgroup=32`.

The kernel is always dispatched with `threadsPerThreadgroup=25`, and it also contains an `if (tid >= 25) return;` guard at the top of the shader. As a result, only the first 25 lanes ever participate in the computation. From the kernel's perspective, no execution path requires 32 active threads.

That is why changing `maxTotalThreadsPerThreadgroup` from `33` to `32` should not change the result. In both cases, the pipeline limit remains greater than or equal to the actual launched threadgroup size of 25, and the shader immediately exits any lane with `tid >= 25`. No code path in the kernel depends on lanes 25 through 31 existing, and no dispatch in this repro ever asks Metal to run more than 25 threads per threadgroup. So the 33 vs 32 specialization should be observationally equivalent for this kernel.

Apple's WWDC20 session [Bring your Metal app to Apple silicon Macs](https://developer.apple.com/videos/play/wwdc2020/10631/) also supports the expectation that `mem_flags::mem_threadgroup` should be sufficient for this kernel. In the session's threadgroup-memory synchronization example, Apple describes threadgroup memory as the shared state that "needs to be properly synchronized for correct ordering" and shows the correct implementation using `threadgroup_barrier(mem_flags::mem_threadgroup)` for cross-thread communication through threadgroup memory. That matches this repro's communication pattern: the synchronized shared state is the `threadgroup` array, not a `device` buffer.

Because this repro only uses barriers to order accesses to threadgroup memory, adding `mem_flags::mem_device` should not be necessary to make the kernel correct. The `mem_device | mem_threadgroup` variant is included here as a comparison point only.

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

To show that this is not dependent on a large dispatch, you can also run the smallest useful case with a single threadgroup:

```bash
swift main.swift 1 1
```

This keeps `threadsPerThreadgroup=25`, but launches only `groups=1` and runs a single comparison iteration. The failure still reproduces there, which helps show that the issue is tied to the pipeline specialization rather than to overall grid size.

Additionally, the repro can be used to dump the generated shader artifacts for inspection, which may be useful for debugging the issue. To do this, use the following command:

```bash
swift main.swift --dump-runtime-artifacts
```

This also writes a copy of stdout and environment metadata that may be useful for debugging. A copy of these files from the machine where this repro was developed (a Macbook Pro with an M2 Pro chip) is included in the `artifacts_m2_pro` directory for reference. When the repro is run on said machine, it produces the following output:

```
Metal barrier repro

[config]
groups: 875
iterations: 3
reference: fence baseline
dump_runtime_artifacts: no
artifact_dir: disabled

[environment]
macos_version: Version 15.7.4 (Build 24G517)
macos_semver: 15.7.4
kernel_version: Darwin 24.6.0 (Darwin Kernel Version 24.6.0: Mon Jan 19 21:56:28 PST 2026; root:xnu-11417.140.69.708.3~1/RELEASE_ARM64_T6020)
process_arch: arm64
machine_arch: arm64
machine_model: Mac14,9
metal_compiler_version: Apple metal version 32023.864 (metalfe-32023.864)
rosetta_translated: no
physical_memory: 32 GB
device_name: Apple M2 Pro
device_registry_id: 4294968568
device_low_power: no
device_headless: no
device_removable: no
device_has_unified_memory: yes
device_location: builtIn
max_threads_per_threadgroup: 1024x1024x1024
recommended_max_working_set_size: 21.33 GB
read_write_texture_support: 2
argument_buffers_support: 1
supports_64bit_float: unavailable (not exposed by current SDK)

[features]
gpu_families: apple1, apple2, apple3, apple4, apple5, apple6, apple7, apple8, mac1, mac2

[variants]
fence baseline: threadLimited=false requestedMaxThreads=33 pipelineMaxThreads=33 executionWidth=32 staticThreadgroupMemory=1008
barrier baseline: threadLimited=false requestedMaxThreads=33 pipelineMaxThreads=33 executionWidth=32 staticThreadgroupMemory=1008
fence thread_limited: threadLimited=true requestedMaxThreads=32 pipelineMaxThreads=32 executionWidth=32 staticThreadgroupMemory=1008
barrier thread_limited: threadLimited=true requestedMaxThreads=32 pipelineMaxThreads=32 executionWidth=32 staticThreadgroupMemory=1008

fence baseline: PASS
barrier baseline: PASS
fence thread_limited: PASS
barrier thread_limited: FAIL mismatches=109227 mismatchPercent=99.86% firstIndex=0 firstGroup=0 firstLane=0 expected=(-27.249022, -8.928854) actual=(-28.246582, -7.701342) delta=(-0.997561, 1.227512)

overall: FAIL
```
