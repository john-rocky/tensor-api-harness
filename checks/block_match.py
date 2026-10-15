#!/usr/bin/env python3
"""Per-tensor output match between a graph's dumps and the reference dumps.

usage: python3 checks/block_match.py --got <dir> --ref <dir> --manifest <manifest.json>

The manifest names each tensor, its file (the same name in both directories)
and its shape, plus the bar written down before the run:

  {"bar": {"max_abs": 1e-3, "cosine": 0.9999},
   "tensors": [{"name": "block_00", "file": "block_00.f32", "shape": [1, 8, 1024]},
               {"name": "logits", "file": "logits.npy", "shape": [1, 8, 151936],
                "bar": {"max_abs": 2e-2}}]}

A tensor's own "bar" overrides the keys it names. Files are raw little-endian
float32 (`.f32`) or numpy `.npy`. A tensor passes when max|got - ref| <= max_abs
and cosine(got, ref) >= cosine (flattened, float64); a non-finite value fails.

Exit codes: 0 every tensor passes; 1 a tensor fails (the first one is named);
2 usage, a bad manifest, a missing file or an element-count mismatch (named).
"""

import argparse
import json
import math
import os
import sys

import numpy as np

BAR_KEYS = ("max_abs", "cosine")


class InputError(Exception):
    pass


def load(path, shape):
    if not os.path.isfile(path):
        raise InputError(f"missing file {path}")
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    else:
        arr = np.fromfile(path, dtype="<f4")
    want = math.prod(shape)
    if arr.size != want:
        raise InputError(f"{path} has {arr.size} elements, the manifest shape {shape} needs {want}")
    return arr.astype(np.float64).reshape(shape)


def read_manifest(path):
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except OSError as e:
        raise InputError(f"cannot read manifest {path}: {e.strerror}")
    except json.JSONDecodeError as e:
        raise InputError(f"manifest {path} is not JSON: {e}")
    if not isinstance(m, dict) or not isinstance(m.get("tensors"), list) or not m["tensors"]:
        raise InputError(f"manifest {path} needs a non-empty \"tensors\" list")
    default = m.get("bar", {})
    if not isinstance(default, dict):
        raise InputError("manifest \"bar\" must be an object")
    out, names = [], set()
    for k, t in enumerate(m["tensors"]):
        if not isinstance(t, dict) or not all(key in t for key in ("name", "file", "shape")):
            raise InputError(f"manifest tensor #{k} needs name, file and shape")
        shape = t["shape"]
        if not isinstance(shape, list) or not all(isinstance(d, int) and d > 0 for d in shape):
            raise InputError(f"tensor {t['name']}: shape must be a list of positive integers")
        if t["name"] in names:
            raise InputError(f"tensor {t['name']} appears twice in the manifest")
        names.add(t["name"])
        bar = dict(default)
        bar.update(t.get("bar", {}))
        bar = {key: bar[key] for key in BAR_KEYS if key in bar}
        if not bar or not all(isinstance(v, (int, float)) for v in bar.values()):
            raise InputError(f"tensor {t['name']} has no numeric bar (max_abs and/or cosine)")
        out.append((t["name"], t["file"], shape, bar))
    return out


def compare(got, ref):
    if not (np.all(np.isfinite(got)) and np.all(np.isfinite(ref))):
        return float("nan"), float("nan")
    max_abs = float(np.max(np.abs(got - ref)))
    g, r = got.ravel(), ref.ravel()
    ng, nr = float(np.linalg.norm(g)), float(np.linalg.norm(r))
    if ng == 0.0 or nr == 0.0:
        cosine = 1.0 if ng == nr else 0.0
    else:
        cosine = float(np.dot(g, r) / (ng * nr))
    return max_abs, cosine


def passes(max_abs, cosine, bar):
    if math.isnan(max_abs) or math.isnan(cosine):
        return False
    if "max_abs" in bar and max_abs > bar["max_abs"]:
        return False
    if "cosine" in bar and cosine < bar["cosine"]:
        return False
    return True


def bar_text(bar):
    return " ".join(f"{k}{'<=' if k == 'max_abs' else '>='}{bar[k]:g}" for k in BAR_KEYS if k in bar)


def main():
    ap = argparse.ArgumentParser(description="Per-tensor output match against reference dumps.")
    ap.add_argument("--got", required=True, help="directory with the graph's dumps")
    ap.add_argument("--ref", required=True, help="directory with the reference dumps")
    ap.add_argument("--manifest", required=True, help="manifest JSON: bar and tensors")
    args = ap.parse_args()   # exits 2 on a usage error

    try:
        tensors = read_manifest(args.manifest)
        pairs, errors = [], []
        for name, fname, shape, bar in tensors:
            try:
                pairs.append((name, shape, bar,
                              load(os.path.join(args.got, fname), shape),
                              load(os.path.join(args.ref, fname), shape)))
            except InputError as e:
                errors.append(f"{name}: {e}")
        if errors:
            raise InputError("; ".join(errors))
    except InputError as e:
        print(f"BLOCK MATCH ERROR: {e}")
        return 2

    width = max(len("tensor"), max(len(p[0]) for p in pairs))
    print(f"{'tensor':<{width}}  {'shape':<20} {'max_abs':>11} {'cosine':>12}  {'bar':<30} result")
    first_fail = None
    for name, shape, bar, got, ref in pairs:
        max_abs, cosine = compare(got, ref)
        ok = passes(max_abs, cosine, bar)
        print(f"{name:<{width}}  {str(shape):<20} {max_abs:>11.4e} {cosine:>12.8f}  {bar_text(bar):<30} "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok and first_fail is None:
            first_fail = (name, max_abs, cosine)
    if first_fail:
        name, max_abs, cosine = first_fail
        print(f"BLOCK MATCH FAIL: first {name} max_abs {max_abs:.4e} cosine {cosine:.8f}")
        return 1
    print(f"BLOCK MATCH PASS {len(pairs)}/{len(pairs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
