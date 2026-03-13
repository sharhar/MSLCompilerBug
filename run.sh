#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

dump_cli_artifacts=0
artifact_dir="artifacts/latest"
swift_args=()
artifact_dir_set=0

while (($# > 0)); do
  case "$1" in
    --dump-cli-artifacts)
      dump_cli_artifacts=1
      shift
      ;;
    --artifact-dir)
      artifact_dir="${2:?--artifact-dir requires a path}"
      artifact_dir_set=1
      shift 2
      ;;
    *)
      swift_args+=("$1")
      shift
      ;;
  esac
done

if ((artifact_dir_set || dump_cli_artifacts)); then
  swift_args+=("--artifact-dir" "$artifact_dir")
fi

if ((dump_cli_artifacts)); then
  swift_args+=("--dump-runtime-artifacts")
fi

swift_status=0
set +e
swift -module-cache-path /tmp/swift-module-cache main.swift "${swift_args[@]}"
swift_status=$?
set -e

compile_cli_artifacts() {
  local root="$1"
  local variant_dir source air metallib asm

  shopt -s nullglob
  for variant_dir in "$root"/*; do
    [[ -d "$variant_dir" ]] || continue

    source=("$variant_dir"/*.metal)
    [[ ${#source[@]} -eq 1 ]] || continue

    air="${source[0]%.metal}.air"
    metallib="${source[0]%.metal}.metallib"
    asm="${source[0]%.metal}.S"

    xcrun metal -c "${source[0]}" -o "$air"
    xcrun metallib "$air" -o "$metallib"

    if xcrun -f metal-objdump >/dev/null 2>&1; then
      if ! xcrun metal-objdump -disassemble "$metallib" >"$asm" 2>"$variant_dir/metal-objdump.log"; then
        if ! xcrun metal-objdump --disassemble "$metallib" >"$asm" 2>"$variant_dir/metal-objdump.log"; then
          xcrun metal-objdump -d "$metallib" >"$asm" 2>"$variant_dir/metal-objdump.log"
        fi
      fi
    else
      cat >"$asm" <<EOF
metal-objdump was not found via xcrun on this machine.
Target metallib: $metallib
EOF
    fi
  done
}

cli_status=0
if ((dump_cli_artifacts)); then
  set +e
  compile_cli_artifacts "$artifact_dir"
  cli_status=$?
  set -e
fi

if ((swift_status != 0)); then
  exit "$swift_status"
fi

if ((cli_status != 0)); then
  exit "$cli_status"
fi
