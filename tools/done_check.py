#!/usr/bin/env python3
"""Refuse "done" while a skill table still has rows to work on.

usage: python3 tools/done_check.py <skill_table.md> [<skill_table.md> ...]

Reads every markdown table (outside code fences) whose header has a `status`
and an `evidence` column. The status cell of each row is one of:

  tried                      the evidence cell names at least one path that exists
  not applicable: <reason>   reason of 12 characters or more
  blocked: <cause>           cause of 12 characters or more
  not yet

Evidence paths resolve against the table's directory, the repo root, or as
absolute paths.

Exit codes (the largest one wins when several files are given):
  0  DONE       every row is `tried` or `not applicable`
  1  NOT DONE   rows still `not yet` (listed)
  3  BLOCKED    no `not yet` left, but `blocked` rows remain (listed)
  2  ERROR      usage, unreadable file, no table, a wrong status word, or a
                `tried` row without an existing evidence path (the row is named)
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_REASON = 12
STATUSES = ("tried", "not applicable", "blocked", "not yet")
FENCE = re.compile(r"^\s*(```|~~~)")
SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def split_row(line):
    """Splits a markdown table line into stripped cells; `\\|` stays in a cell."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.replace("\\|", "|").strip() for c in cells]


def read_tables(lines):
    """Yields (header cells, [(line number, cells), ...]) for each table outside code fences."""
    fence = False
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if FENCE.match(line):
            fence = not fence
            i += 1
            continue
        if (not fence and line.lstrip().startswith("|") and i + 1 < n
                and SEPARATOR.match(lines[i + 1])):
            header = [c.lower() for c in split_row(line)]
            rows = []
            j = i + 2
            while j < n and lines[j].lstrip().startswith("|"):
                rows.append((j + 1, split_row(lines[j])))
                j += 1
            yield header, rows
            i = j
            continue
        i += 1


def evidence_paths(cell):
    """Candidate paths named in an evidence cell: link targets and backticked or bare tokens."""
    found = LINK.findall(cell)
    rest = LINK.sub(" ", cell)
    for tok in re.split(r"[\s,;]+", rest):
        tok = tok.strip("`'\"()<>")
        if tok and tok.strip("./~"):   # ".", "..", "/" alone name no evidence
            found.append(tok)
    return [p for p in found if p.strip("./~")]


def path_exists(token, table_dir):
    candidates = [token]
    m = re.match(r"^(.*?)(:\d+(-\d+)?)$", token)   # path:line or path:line-line
    if m:
        candidates.append(m.group(1))
    for c in candidates:
        c = os.path.expanduser(c)
        if os.path.isabs(c):
            if os.path.exists(c):
                return True
            continue
        if os.path.exists(os.path.join(table_dir, c)) or os.path.exists(os.path.join(REPO_ROOT, c)):
            return True
    return False


def parse_status(cell):
    """-> (status, detail) or (None, message)."""
    s = cell.strip().strip("`").strip()
    low = s.lower()
    if low in ("tried", "not yet"):
        return low, ""
    for word in ("not applicable", "blocked"):
        if low.startswith(word):
            rest = s[len(word):].lstrip()
            if not rest.startswith(":"):
                return None, f"`{word}` needs a colon and the {'reason' if word == 'not applicable' else 'cause'}"
            detail = rest[1:].strip()
            if len(detail) < MIN_REASON:
                return None, (f"`{word}:` needs a {'reason' if word == 'not applicable' else 'cause'} "
                              f"of {MIN_REASON} characters or more, got {len(detail)}")
            return word, detail
    shown = s if s else "(empty)"
    return None, f"status `{shown}` is not one of: tried | not applicable: <reason> | blocked: <cause> | not yet"


def row_label(header, cells, index):
    """`#<n> <row name>`: the `#` column and the `row` column when the table has them."""
    def cell(name):
        if name in header:
            k = header.index(name)
            return cells[k] if k < len(cells) else ""
        return ""
    num = cell("#") or str(index)
    name = cell("row") or (cells[1] if len(cells) > 1 and "#" in header else (cells[0] if cells else ""))
    return f"#{num} {name}".rstrip()


def check(path):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        print(f"ERROR {path}: cannot read: {e.strerror}")
        return 2
    table_dir = os.path.dirname(os.path.abspath(path))
    counts = {s: 0 for s in STATUSES}
    errors, not_yet, blocked = [], [], []
    n_tables, index = 0, 0
    for header, rows in read_tables(lines):
        if "status" not in header or "evidence" not in header:
            continue
        n_tables += 1
        si, ei = header.index("status"), header.index("evidence")
        for lineno, cells in rows:
            index += 1
            label = row_label(header, cells, index)
            status_cell = cells[si] if si < len(cells) else ""
            evidence_cell = cells[ei] if ei < len(cells) else ""
            status, detail = parse_status(status_cell)
            if status is None:
                errors.append(f"line {lineno} {label}: {detail}")
                continue
            counts[status] += 1
            if status == "tried":
                paths = evidence_paths(evidence_cell)
                if not paths:
                    errors.append(f"line {lineno} {label}: `tried` with an empty evidence cell")
                elif not any(path_exists(p, table_dir) for p in paths):
                    errors.append(f"line {lineno} {label}: `tried` but no evidence path exists: {evidence_cell}")
            elif status == "not yet":
                not_yet.append(label)
            elif status == "blocked":
                blocked.append(f"{label} — {detail}")
    if n_tables == 0:
        print(f"ERROR {path}: no table with `status` and `evidence` columns")
        return 2
    print(f"{path}: " + ", ".join(f"{s} {counts[s]}" for s in STATUSES))
    if errors:
        for e in errors:
            print(f"ERROR {path}: {e}")
        return 2
    if not_yet:
        print(f"NOT DONE {path}: {len(not_yet)} rows not yet")
        for label in not_yet:
            print(f"  {label}")
        return 1
    if blocked:
        print(f"BLOCKED {path}: {len(blocked)} rows blocked")
        for label in blocked:
            print(f"  {label}")
        return 3
    print(f"DONE {path}")
    return 0


def main(argv):
    if len(argv) < 2 or any(a in ("-h", "--help") for a in argv[1:]):
        print("usage: python3 tools/done_check.py <skill_table.md> [<skill_table.md> ...]")
        return 2
    return max(check(p) for p in argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
