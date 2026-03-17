# Metal Threadgroup Barrier Bug Repro

This repository is a repro for a bug in the Swift compiler's handling of `threadgroup_barrier` when compiling for Apple Silicon for execution with only 1 threadgroup.

The repro is contained in `main.swift`, which builds two shader variants that differ only in their use of `threadgroup_barrier` flags. One version uses `threadgroup_barrier(mem_flags::mem_threadgroup)`, while the other uses `threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)`. The shader consists of calculations that utilize shared memory to transfer data between threads, and there is an if statement at the start of the shader limiting execution of the code to only the first 25 threads (which means that only one threadgroup is ever active in all invocations of the shader variants). When the shader is executed with `maxTotalThreadsPerThreadgroup=33` (which is one above the 32 wide threadgroups on apple silicon), both variants run and produce the correct results. However, when `maxTotalThreadsPerThreadgroup=32`, only the shader with the `mem_flags::mem_device` flag added to the barrier produce correct results, while the shader using only `mem_flags::mem_threadgroup` produces wrong output seemingly due to lack of proper data access synchropnization (or some other overly agressive compiler optimization that produces incorrect behavior).

To run this repro, use the following command

```bash
swift main.swift
```

Additionally, the repro can be used to dump the generated shader artifacts for inspection, which may be useful for debugging the issue. To do this, use the following command:

```bash
swift main.swift --dump-runtime-artifacts
```

This will also make a copy of stdout and record other metadata about the environment that may be useful for debugging.