#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

artifact_dir="artifacts"

while (($# > 0)); do
  case "$1" in
    --artifact-dir)
      artifact_dir="${2:?--artifact-dir requires a path}"
      shift 2
      ;;
    *)
      echo "usage: ./disassemble.sh [--artifact-dir <path>]" >&2
      exit 2
      ;;
  esac
done

disassemble_artifacts() {
  local root="$1"
  local variant_dir source air metallib asm objdump_log

  if [[ ! -d "$root" ]]; then
    echo "error: artifact directory not found: $root" >&2
    return 1
  fi

  shopt -s nullglob
  for variant_dir in "$root"/*; do
    [[ -d "$variant_dir" ]] || continue

    source=("$variant_dir"/*.metal)
    [[ ${#source[@]} -eq 1 ]] || continue

    air="${source[0]%.metal}.air"
    metallib="${source[0]%.metal}.metallib"
    asm="${source[0]%.metal}.S"
    objdump_log="$variant_dir/metal-objdump.log"

    xcrun metal -c "${source[0]}" -o "$air"
    xcrun metallib "$air" -o "$metallib"

    if xcrun -f metal-objdump >/dev/null 2>&1; then
      if ! xcrun metal-objdump -disassemble "$metallib" >"$asm" 2>"$objdump_log"; then
        if ! xcrun metal-objdump --disassemble "$metallib" >"$asm" 2>"$objdump_log"; then
          xcrun metal-objdump -d "$metallib" >"$asm" 2>"$objdump_log"
        fi
      fi
    else
      cat >"$asm" <<EOF
metal-objdump was not found via xcrun on this machine.
Target metallib: $metallib
EOF
    fi

    echo "processed: $variant_dir"
  done
}

disassemble_artifacts "$artifact_dir"
