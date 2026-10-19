#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Links this directory's files and the graph sources of ../graph into a
# LiteRT-LM checkout as tensor_api/qwen3-0.6b (one link per file: the graph's
# own BUILD names LiteRT's workspace and is not linked). No file of the
# checkout is changed.
#
#   models/qwen3-0.6b/litert_lm/overlay.sh <LiteRT-LM checkout>
set -eu
if [ $# -ne 1 ]; then
  echo "usage: $0 <LiteRT-LM checkout>" >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
graph=$(cd "$here/../graph" && pwd)
dir="$1/tensor_api/qwen3-0.6b"
mkdir -p "$dir"
for f in BUILD tensor_api_executor.h tensor_api_executor.cc \
    tensor_api_backend.cc greedy_litertlm_resources.cc; do
  ln -sfn "$here/$f" "$dir/$f"
done
for f in qwen3_config.h qwen3_graph.h qwen3_graph.cc qwen3_weights.h \
    qwen3_weights.cc; do
  ln -sfn "$graph/$f" "$dir/$f"
done
echo "linked $dir"
