#!/usr/bin/env python3
"""Token-id match of greedy-decoded text against the reference, prompt by prompt.

usage: python3 checks/text_match.py --got <ids.json> --ref <ids.json>

Both files: {"prompts": [{"id": "p0", "ids": [9707, 11, ...], "text": "..."}]}
("text" is optional). Every reference prompt must appear in --got with the
same ids at every position; prompts only in --got are listed and ignored.

Exit codes: 0 every prompt equal; 1 a prompt differs or is missing from --got;
2 usage or a file that is not JSON of that form.
"""

import argparse
import json
import sys


class InputError(Exception):
    pass


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise InputError(f"cannot read {path}: {e.strerror}")
    except json.JSONDecodeError as e:
        raise InputError(f"{path} is not JSON: {e}")
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if not isinstance(prompts, list):
        raise InputError(f"{path} needs a \"prompts\" list")
    out = {}
    for k, p in enumerate(prompts):
        if not isinstance(p, dict) or not isinstance(p.get("id"), str) or not isinstance(p.get("ids"), list):
            raise InputError(f"{path} prompt #{k} needs a string \"id\" and an \"ids\" list")
        if not all(isinstance(t, int) and not isinstance(t, bool) for t in p["ids"]):
            raise InputError(f"{path} prompt {p['id']}: ids must be integers")
        if p["id"] in out:
            raise InputError(f"{path} prompt {p['id']} appears twice")
        out[p["id"]] = p["ids"]
    return out


def first_mismatch(got, ref):
    for i in range(max(len(got), len(ref))):
        g = got[i] if i < len(got) else None
        r = ref[i] if i < len(ref) else None
        if g != r:
            return i, g, r
    return None


def main():
    ap = argparse.ArgumentParser(description="Token-id match against the reference.")
    ap.add_argument("--got", required=True, help="the graph's ids JSON")
    ap.add_argument("--ref", required=True, help="the reference ids JSON")
    args = ap.parse_args()   # exits 2 on a usage error
    try:
        got, ref = read(args.got), read(args.ref)
        if not ref:
            raise InputError(f"{args.ref} has no prompts")
    except InputError as e:
        print(f"TEXT MATCH ERROR: {e}")
        return 2

    failed = 0
    for pid, ref_ids in ref.items():
        if pid not in got:
            print(f"{pid} missing from {args.got}")
            failed += 1
            continue
        mm = first_mismatch(got[pid], ref_ids)
        if mm is None:
            print(f"{pid} equal {len(ref_ids)} tokens")
        else:
            i, g, r = mm
            print(f"{pid} first mismatch at {i}: got {'(end)' if g is None else g} ref {'(end)' if r is None else r}")
            failed += 1
    for pid in got:
        if pid not in ref:
            print(f"{pid} only in {args.got} (ignored)")
    if failed:
        print(f"TEXT MATCH FAIL: {failed} prompts")
        return 1
    print(f"TEXT MATCH PASS {len(ref)}/{len(ref)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
