#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Runs the cases of cases.tsv with sdpa_repro_main (CPU and GPU) and prints one
# PASS / FAIL line per case.
#
#   tools/repro_sdpa_transposed_metal/run.sh [options] <LiteRT checkout> [<out dir>]
#
# Links this directory into the checkout as tensor/examples/sdpa_transposed_repro,
# builds //tensor/examples/sdpa_transposed_repro:sdpa_repro_main with bazelisk
# (or bazel), and runs every case with the checkout's GPU accelerator library
# (litert/prebuilt/macos_arm64 on macOS). Exit code 0 = every case PASS,
# 1 = at least one case FAIL, 2 = a setup, build or run error.
# Options: --no-build, --bin <binary>, --cases <file>, --only <id,id,...>,
# --dry-run (check the setup and print the commands; build and run nothing).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=1
BIN=""
CASES=$HERE/cases.tsv
ONLY=""
DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --no-build) BUILD=0; shift ;;
    --bin) BIN=$2; shift 2 ;;
    --cases) CASES=$2; shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
if [ $# -lt 1 ]; then
  sed -n '3,14p' "$0"
  exit 2
fi
LITERT=$(cd "$1" && pwd) || exit 2
OUT=${2:-$PWD/sdpa_repro_out}
TARGET=//tensor/examples/sdpa_transposed_repro:sdpa_repro_main
LINK=$LITERT/tensor/examples/sdpa_transposed_repro
[ -n "$BIN" ] || BIN=$LITERT/bazel-bin/tensor/examples/sdpa_transposed_repro/sdpa_repro_main
# Absolute paths: every case runs from its own directory.
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
case $BIN in /*) ;; *) BIN=$PWD/$BIN ;; esac

case $(uname -s) in
  Darwin) GPULIB=$LITERT/litert/prebuilt/macos_arm64; GPUFILE=libLiteRtMetalAccelerator.dylib ;;
  *) echo "run.sh knows the GPU library location for macOS only" >&2; exit 2 ;;
esac
[ -f "$CASES" ] || { echo "no cases file: $CASES" >&2; exit 2; }
[ -d "$LITERT/tensor" ] || { echo "not a LiteRT checkout (no tensor/): $LITERT" >&2; exit 2; }
# A Git LFS pointer instead of the library: the checkout needs `git lfs pull`.
size=$(wc -c < "$GPULIB/$GPUFILE" 2>/dev/null | tr -d ' ')
[ -n "$size" ] || size=0
if [ "$size" -lt 1000000 ]; then
  echo "$GPULIB/$GPUFILE is missing or a Git LFS pointer ($size bytes); run git lfs pull in the checkout" >&2
  exit 2
fi
if [ -e "$LINK" ] && [ "$(cd "$LINK" && pwd -P)" != "$(cd "$HERE" && pwd -P)" ]; then
  echo "$LINK exists and is not this directory" >&2
  exit 2
fi

if [ $DRY = 1 ]; then
  echo "dry run: checkout $LITERT ($(git -C "$LITERT" rev-parse --short HEAD 2>/dev/null)), out $OUT"
  echo "dry run: link $LINK -> $HERE ($([ -e "$LINK" ] && echo present || echo to create))"
  echo "dry run: build $TARGET ($([ $BUILD = 1 ] && echo yes || echo skipped)), binary $BIN"
  echo "dry run: GPU library $GPULIB/$GPUFILE ($size bytes)"
fi

if [ $DRY = 0 ]; then
  [ -e "$LINK" ] || ln -s "$HERE" "$LINK" || exit 2
  if [ $BUILD = 1 ]; then
    BZ=$(command -v bazelisk || command -v bazel) || { echo "no bazelisk or bazel" >&2; exit 2; }
    (cd "$LITERT" && "$BZ" build "$TARGET") || { echo "build failed" >&2; exit 2; }
  fi
  [ -x "$BIN" ] || { echo "no binary: $BIN" >&2; exit 2; }
  mkdir -p "$OUT" || exit 2
  {
    echo "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null), macOS $(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
    echo "LiteRT checkout: $(git -C "$LITERT" rev-parse HEAD 2>/dev/null)"
    echo "GPU library: $GPUFILE sha256 $(shasum -a 256 "$GPULIB/$GPUFILE" | cut -d' ' -f1)"
    echo "binary: sha256 $(shasum -a 256 "$BIN" | cut -d' ' -f1)"
  } > "$OUT/env.txt"
  cat "$OUT/env.txt"
fi

printf '%-4s %-6s %-11s %-24s %-10s %s\n' case result "max|c-g|" "first over (h,s,d)" "gpu runs" "what changes"
n=0; fails=0; errors=0
TAB=$(printf '\t')
[ $DRY = 0 ] && printf 'case\tresult\tmax_abs_cpu_gpu\tfirst_over\tgpu_runs_max_diff\tdescription\tflags\n' > "$OUT/summary.tsv"
while IFS="$TAB" read -r id desc flags; do
  case $id in ''|'#'*) continue ;; esac
  if [ -n "$ONLY" ]; then
    case $ONLY in *",$id,"*) ;; *) continue ;; esac
  fi
  dir=$OUT/$id
  if [ $DRY = 1 ]; then
    echo "$id: (cd $dir && DYLD_LIBRARY_PATH=$GPULIB $BIN $flags --label=$id --tflite_path=$dir/model.tflite --dump_dir=$dir)"
    continue
  fi
  mkdir -p "$dir"
  # shellcheck disable=SC2086
  (cd "$dir" && DYLD_LIBRARY_PATH=$GPULIB "$BIN" $flags --label="$id" \
      --tflite_path="$dir/model.tflite" --dump_dir="$dir" < /dev/null) \
      > "$dir/log.txt" 2> "$dir/stderr.txt"
  rc=$?
  n=$((n + 1))
  case $rc in
    0) result=PASS ;;
    2) result=FAIL; fails=$((fails + 1)) ;;
    *) result=ERROR; errors=$((errors + 1)) ;;
  esac
  maxd=$(sed -n 's/^cpu vs gpu run 1: max_abs_diff \([^,]*\),.*/\1/p' "$dir/log.txt")
  first=$(sed -n 's/^first element over the line (output order): head \([0-9]*\), token \([0-9]*\), dim \([0-9]*\):.*/\1,\2,\3/p' "$dir/log.txt")
  runs=$(sed -n 's/^gpu run [0-9]* vs float64 reference: .*, vs gpu run 1: \(.*\)$/\1/p' "$dir/log.txt" | sort -g | tail -1)
  printf '%-4s %-6s %-11s %-24s %-10s %s\n' "$id" "$result" "${maxd:--}" "${first:--}" "${runs:--}" "$desc"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$result" "${maxd:--}" "${first:--}" "${runs:--}" "$desc" "$flags" >> "$OUT/summary.tsv"
  if [ $result = ERROR ]; then
    echo "     rc $rc: $(tail -3 "$dir/stderr.txt" | tr '\n' ' ' | cut -c1-300)"
  fi
done < "$CASES"

[ $DRY = 1 ] && exit 0
echo "overall: $n cases, $((n - fails - errors)) PASS, $fails FAIL, $errors ERROR (logs and dumps in $OUT)"
[ $errors -gt 0 ] && exit 2
[ $fails -gt 0 ] && exit 1
exit 0
