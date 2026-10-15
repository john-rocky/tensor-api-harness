#!/usr/bin/env python3
"""Name the fastest form that passes the output match and compare it with the floor.

usage: python3 checks/speed_floor.py --timing <timing.json>

  {"floor": {"form": "cpu fp32", "ms_per_token": 120.0},
   "forms": [{"form": "metal fp16", "backend": "metal", "precision": "fp16",
              "ms_per_token": 20.0, "match": "PASS"}, ...]}

The floor is the CPU fp32 run of the same graph. Every measured form stays in
"forms", slower and failing ones included; the table prints them all.

Exit codes: 0 the fastest form with match PASS is faster than the floor;
1 no form passes the match, or the fastest passing form is not faster;
2 usage or a file that is not JSON of that form.
"""

import argparse
import json
import sys


class InputError(Exception):
    pass


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise InputError(f"cannot read {path}: {e.strerror}")
    except json.JSONDecodeError as e:
        raise InputError(f"{path} is not JSON: {e}")
    if not isinstance(data, dict):
        raise InputError(f"{path} must hold an object with \"floor\" and \"forms\"")
    floor = data.get("floor")
    if not isinstance(floor, dict) or not isinstance(floor.get("form"), str) or not number(floor.get("ms_per_token")):
        raise InputError("\"floor\" needs a string \"form\" and a positive \"ms_per_token\"")
    forms = data.get("forms")
    if not isinstance(forms, list) or not forms:
        raise InputError("\"forms\" must be a non-empty list")
    for k, f in enumerate(forms):
        if not isinstance(f, dict):
            raise InputError(f"form #{k} must be an object")
        for key in ("form", "backend", "precision"):
            if not isinstance(f.get(key), str):
                raise InputError(f"form #{k} needs a string \"{key}\"")
        if not number(f.get("ms_per_token")):
            raise InputError(f"form {f['form']} needs a positive \"ms_per_token\"")
        if f.get("match") not in ("PASS", "FAIL"):
            raise InputError(f"form {f['form']}: \"match\" must be \"PASS\" or \"FAIL\"")
    return floor, forms


def main():
    ap = argparse.ArgumentParser(description="Fastest passing form against the CPU fp32 floor.")
    ap.add_argument("--timing", required=True, help="timing JSON: floor and forms")
    args = ap.parse_args()   # exits 2 on a usage error
    try:
        floor, forms = read(args.timing)
    except InputError as e:
        print(f"SPEED FLOOR ERROR: {e}")
        return 2

    forms = sorted(forms, key=lambda f: f["ms_per_token"])
    w = max(len("form"), max(len(f["form"]) for f in forms))
    print(f"{'form':<{w}}  {'backend':<10} {'precision':<10} {'ms/token':>10}  match")
    for f in forms:
        print(f"{f['form']:<{w}}  {f['backend']:<10} {f['precision']:<10} {f['ms_per_token']:>10.2f}  {f['match']}")
    passing = [f for f in forms if f["match"] == "PASS"]
    floor_ms = floor["ms_per_token"]
    if not passing:
        print(f"fastest passing: none (floor {floor['form']} {floor_ms:g} ms/token)")
        print("SPEED FLOOR FAIL: no form passes the match")
        return 1
    best = passing[0]
    print(f"fastest passing: {best['form']} {best['ms_per_token']:g} ms/token (floor {floor_ms:g})")
    if best["ms_per_token"] < floor_ms:
        print(f"SPEED FLOOR PASS: {best['form']} {best['ms_per_token']:g} ms/token "
              f"< floor {floor['form']} {floor_ms:g} ms/token")
        return 0
    print(f"SPEED FLOOR FAIL: fastest passing {best['form']} {best['ms_per_token']:g} ms/token "
          f"is not below the floor {floor['form']} {floor_ms:g} ms/token")
    return 1


if __name__ == "__main__":
    sys.exit(main())
