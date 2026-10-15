#!/usr/bin/env bash
# Runs every line of checks/fixtures/expected.tsv (the script and its arguments, a tab, the expected
# exit code) from the repo root and compares the exit code.
# usage: bash tools/selftest.sh        (PYTHON=<interpreter> to pick one; numpy is needed for block_match)
# exit 0 = SELFTEST PASS n/n; 1 = SELFTEST FAIL with the failing lines; 2 = no expected.tsv
cd "$(dirname "$0")/.." || exit 2
PY=${PYTHON:-python3}
TSV=checks/fixtures/expected.tsv
[ -f "$TSV" ] || { echo "SELFTEST ERROR: $TSV not found"; exit 2; }
n=0; failed=()
while IFS=$'\t' read -r cmd want || [ -n "$cmd" ]; do
  case "$cmd" in ''|'#'*) continue ;; esac
  n=$((n + 1))
  # shellcheck disable=SC2086  # the arguments are space-separated paths without spaces
  out=$("$PY" $cmd 2>&1); got=$?
  if [ "$got" != "$want" ]; then
    failed+=("$cmd -> exit $got, expected $want")
    printf '%s\n' "$out" | sed 's/^/    /' | tail -5
  fi
done < "$TSV"
if [ ${#failed[@]} -gt 0 ]; then
  for f in "${failed[@]}"; do echo "SELFTEST FAIL: $f"; done
  exit 1
fi
echo "SELFTEST PASS $n/$n"
