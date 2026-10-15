#!/usr/bin/env python3
"""Refuse an iteration record that is missing one of the five close sections.

usage: python3 tools/iteration_close_check.py <ITERATION_N.md> [<ITERATION_N.md> ...]

A section is a markdown heading outside code fences (any level, an optional
"N." prefix) whose text starts with one of the headings of
iterations/ITERATION_CLOSE.md (case-insensitive), followed by at least one
non-blank line before the next heading of the same or a higher level. "none"
counts as content.

Exit codes: 0 every file passes; 1 a section is missing or empty; 2 usage or
an unreadable file.
"""

import re
import sys

SECTIONS = ["Corrections", "Dead ends", "Op and delegate observations", "Environment", "Files left"]
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(?:\d+[.)]\s*)?(.*?)\s*#*\s*$")


def norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def check(path):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        print(f"FAIL {path}: cannot read: {e.strerror}")
        return 2
    heads, fence = [], False   # (line index, level, text)
    for i, line in enumerate(lines):
        if FENCE.match(line):
            fence = not fence
            continue
        m = None if fence else HEADING.match(line)
        if m:
            heads.append((i, len(m.group(1)), m.group(2)))
    missing, empty = [], []
    for key in SECTIONS:
        hits = [n for n, (_, _, text) in enumerate(heads) if norm(text).startswith(norm(key))]
        if not hits:
            missing.append(key)
            continue
        ok = False
        for n in hits:
            i, level, _ = heads[n]
            end = next((j for j, lv, _ in heads[n + 1:] if lv <= level), len(lines))
            body = [x for x in lines[i + 1:end] if x.strip() and not FENCE.match(x)]
            if body:
                ok = True
                break
        if not ok:
            empty.append(key)
    if missing or empty:
        print(f"FAIL {path}: missing={missing} empty={empty}")
        return 1
    print(f"PASS iteration close: {path}")
    return 0


def main(argv):
    if len(argv) < 2 or any(a in ("-h", "--help") for a in argv[1:]):
        print("usage: python3 tools/iteration_close_check.py <ITERATION_N.md> [<ITERATION_N.md> ...]")
        return 2
    return max(check(p) for p in argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
