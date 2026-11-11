#!/usr/bin/env python3
"""Draw the figure at the top of README.md from models/qwen3-0.6b/results/speed.md.

    python3 tools/readme_figure.py

Writes docs/images/forms-light.svg and docs/images/forms-dark.svg, and prints the sentence the figure's alt text uses.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "models" / "qwen3-0.6b" / "results" / "speed.md"
OUT = ROOT / "docs" / "images"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "pass": "#2a78d6", "fail": "#eb6834"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "pass": "#3987e5", "fail": "#d95926"},
}
KEYS = [("pass", "All 48 tokens matched PyTorch"), ("fail", "Tokens differed")]
TITLE = "Qwen3-0.6B on an Apple M4 Max: the {n} variants in the timing table"
NOTE = "Time per decode token, measured in one session."
WIDTH, LABELS, ROW, BAR, VALUE_ROOM = 860, 240, 28, 16, 76


def forms():
    """Return [(label, ms or None, matched)], read from the table in the speed record."""
    if not SOURCE.is_file():
        sys.exit(f"readme_figure: {SOURCE} not found")
    rows = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 11 or not re.match(r"(cpu|metal) ", cells[0]):
            continue
        label = re.sub(r"^cpu", "CPU", re.sub(r"^metal", "Metal", cells[0]))
        ms = None if cells[4] == "—" else float(cells[4])
        rows.append((label, ms, cells[8] == "PASS (48/48)"))
    if not rows:
        sys.exit(f"readme_figure: no forms in {SOURCE}")
    return sorted(rows, key=lambda r: (r[1] is None, r[1] or 0))


def text(x, y, body, size, fill, weight="400", anchor="start"):
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{body}</text>'


def draw(rows, theme, desc):
    c = THEMES[theme]
    unit = (WIDTH - LABELS - VALUE_ROOM) / max(ms for _, ms, _ in rows if ms is not None)
    title = TITLE.format(n=len(rows))
    parts = [text(0, 28, title, 20, c["text"], "600"), text(0, 52, NOTE, 14, c["muted"])]
    x = 0
    for key, label in KEYS:
        parts.append(f'<rect x="{x}" y="66" width="14" height="14" rx="4" fill="{c[key]}"/>')
        parts.append(text(x + 20, 78, label, 14, c["muted"]))
        x += 260
    y = 110
    for label, ms, matched in rows:
        parts.append(text(0, y + 13, label, 14, c["text"]))
        if ms is None:
            parts.append(text(LABELS, y + 13, "did not run", 14, c["muted"]))
        else:
            w = ms * unit
            parts.append(f'<rect x="{LABELS}" y="{y:g}" width="{w:g}" height="{BAR}" rx="4" fill="{c["pass" if matched else "fail"]}"/>')
            parts.append(text(LABELS + w + 8, y + 13, f"{ms:g} ms", 14, c["text"], "600"))
        y += ROW
    height = y + 4
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{title}</title>\n<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    rows = forms()
    timed = [r for r in rows if r[1] is not None]
    best = next(r for r in timed if r[2])
    desc = (f"{TITLE.format(n=len(rows))}. {len(rows)} variants: {sum(1 for r in rows if r[2])} matched PyTorch, {sum(1 for r in timed if not r[2])} did not match, {len(rows) - len(timed)} did not run. "
            f"The fastest variant that matched is {best[0]} at {best[1]:g} ms per token.")
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"forms-{theme}.svg").write_text(draw(rows, theme, desc), encoding="utf-8")
    print(desc)


if __name__ == "__main__":
    main()
