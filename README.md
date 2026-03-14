# Metal Barrier Repro

Run:

```bash
swift -module-cache-path /tmp/swift-module-cache main.swift
```

Optional:

```bash
swift -module-cache-path /tmp/swift-module-cache main.swift 875 3
```

To dump compiled artifacts for each variant from the Swift repro:

```bash
swift -module-cache-path /tmp/swift-module-cache main.swift 64 3 --dump-runtime-artifacts
```

You can also choose a custom output directory:

```bash
swift -module-cache-path /tmp/swift-module-cache main.swift 64 3 --dump-runtime-artifacts --artifact-dir artifacts/m2pro-run
```

To compile emitted `.metal` sources into `.air`/`.metallib` and produce best-effort disassembly for an existing artifact directory:

```bash
./disassemble.sh --artifact-dir artifacts/m2pro-run
```

Arguments are:

1. threadgroup count, default `875`
2. iteration count, default `3`

Flags are:

- `--dump-runtime-artifacts`: writes per-variant shader source and runtime binary archives from Swift
- `--artifact-dir <path>`: output directory for dumped artifacts, default `artifacts`

What it does:

- builds two shader variants inside `main.swift`
- runs each one through both pipeline paths
- uses `fence baseline` as the reference output
- fails if any other run differs from that reference

When `--dump-runtime-artifacts` is enabled the Swift repro creates one directory per variant containing:

- `<variant>.metal`: source emitted by the Swift repro
- `<variant>.binarchive`: runtime binary archive emitted through `MTLBinaryArchive`
- `metadata.txt`: pipeline properties such as `threadExecutionWidth`

After that, `./disassemble.sh --artifact-dir <path>` adds:

- `<variant>.air`: AIR emitted by `xcrun metal`
- `<variant>.metallib`: library emitted by `xcrun metallib`
- `<variant>.S`: best-effort `metal-objdump` disassembly, when available

On a buggy compiler, `barrier thread_limited` should fail while the other three lines pass.
When the bug is fixed, all four lines should pass and the script exits `0`.
