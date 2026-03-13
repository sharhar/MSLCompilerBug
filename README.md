# Metal Barrier Repro

Run:

```bash
./run.sh
```

Optional:

```bash
./run.sh 875 3
```

To dump compiled artifacts for each variant:

```bash
./run.sh 64 3 --dump-cli-artifacts
```

You can also choose a custom output directory:

```bash
./run.sh 64 3 --dump-cli-artifacts --artifact-dir artifacts/m2pro-run
```

Arguments are:

1. threadgroup count, default `875`
2. iteration count, default `3`

Flags are:

- `--dump-cli-artifacts`: writes per-variant shader source and runtime artifacts from Swift, then runs `xcrun metal`, `xcrun metallib`, and best-effort disassembly from Bash
- `--artifact-dir <path>`: output directory for dumped artifacts, default `artifacts/latest`

What it does:

- builds two shader variants inside `main.swift`
- runs each one through both pipeline paths
- uses `fence baseline` as the reference output
- fails if any other run differs from that reference

When `--dump-cli-artifacts` is enabled it also creates one directory per variant containing:

- `<variant>.metal`: source emitted by the Swift repro
- `<variant>.air`: AIR emitted by `xcrun metal` from `run.sh`
- `<variant>.metallib`: library emitted by `xcrun metallib` from `run.sh`
- `<variant>.binarchive`: runtime binary archive emitted through `MTLBinaryArchive`
- `<variant>.S`: best-effort `metal-objdump` disassembly emitted from `run.sh`, when available
- `metadata.txt`: pipeline properties such as `threadExecutionWidth`

On a buggy compiler, `barrier thread_limited` should fail while the other three lines pass.
When the bug is fixed, all four lines should pass and the script exits `0`.
