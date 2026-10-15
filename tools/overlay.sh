#!/usr/bin/env bash
# Links models/<name>/graph into a LiteRT checkout as tensor/examples/<name> and prints the Bazel
# build line and the macOS run lines for the graph's main binary.
# usage: tools/overlay.sh <LiteRT checkout> <name>
# exit 0 = linked (or already the same link); 1 = something else is at tensor/examples/<name>;
#      2 = usage, no checkout, no models/<name>/graph/BUILD, or no cc_binary in it
set -u
[ $# -eq 2 ] || { echo "usage: $0 <LiteRT checkout> <name>"; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
LITERT=$(cd "$1" 2>/dev/null && pwd) || { echo "ERROR: no directory $1"; exit 2; }
NAME=$2
SRC="$REPO/models/$NAME/graph"
DST="$LITERT/tensor/examples/$NAME"

[ -d "$LITERT/tensor/examples" ] || { echo "ERROR: $LITERT has no tensor/examples (not a LiteRT checkout?)"; exit 2; }
[ -f "$SRC/BUILD" ] || { echo "ERROR: $SRC/BUILD not found"; exit 2; }

if [ -L "$DST" ] && [ "$(readlink "$DST")" = "$SRC" ]; then
  echo "already linked: $DST -> $SRC"
elif [ -e "$DST" ] || [ -L "$DST" ]; then
  echo "ERROR: $DST exists and is not a link to $SRC (it is: $(readlink "$DST" || echo 'a directory or file')); remove it first"
  exit 1
else
  ln -s "$SRC" "$DST" || exit 1
  echo "linked: $DST -> $SRC"
fi

# cc_binary targets in the graph's BUILD; the main one is the first named *_main, else the first one.
TARGETS=$(awk '/cc_binary\(/{b=1} b && /name *= *"/{match($0, /"[^"]+"/); print substr($0, RSTART+1, RLENGTH-2); b=0}' "$SRC/BUILD")
[ -n "$TARGETS" ] || { echo "ERROR: no cc_binary in $SRC/BUILD"; exit 2; }
MAIN=$(printf '%s\n' $TARGETS | grep -m1 '_main$' || printf '%s\n' $TARGETS | head -1)
OTHERS=$(printf '%s\n' $TARGETS | grep -vx "$MAIN" | tr '\n' ' ')
BIN="bazel-bin/tensor/examples/$NAME/$MAIN"

cat <<EOF

build (from the checkout; on macOS the .bazelrc applies build:macos by itself):
  cd $LITERT
  bazel build //tensor/examples/$NAME:$MAIN
${OTHERS:+  other cc_binary targets: $OTHERS
}
run on the CPU (from the checkout):
  $BIN --accelerator=cpu <model flags>

run on Metal (macOS arm64, from the checkout; the accelerator dylib ships in the target's runfiles):
  DYLD_LIBRARY_PATH=\$PWD/$BIN.runfiles/litert_prebuilts/macos_arm64 \\
    $BIN --accelerator=gpu <model flags>
  Metal computes in fp16 unless the GPU options ask for fp32: set the precision explicitly and label every number.
EOF
