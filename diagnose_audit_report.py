#!/usr/bin/env python3
"""Characterise what the auditor found, from the -audit-report.jsonl files.

The audit summary says how many blocks failed. It does not say whether a failure is a
systematic offset that one fix would clear, or scattered noise that needs the generating
prompt changed. This script answers that, one check at a time.

    python diagnose_audit_report.py <dir-or-file> [...]

With no arguments it globs *-audit-report.jsonl under the current directory.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(paths):
    """Read every report line from the given files or directories."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.rglob("*-audit-report.jsonl")))
        else:
            files.append(p)
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["_file"] = f.name
                rows.append(r)
    return rows, files


def pct(n, d):
    return "%5.2f%%" % (100.0 * n / d) if d else "    - "


def bar(n, total, width=28):
    return "#" * int(round(width * n / total)) if total else ""


def section(title):
    print("\n" + "-" * 74)
    print(title)
    print("-" * 74)


def check_breakdown(rows):
    """Which checks fire, split by the label the block carries."""
    section("findings by security level")
    levels = sorted({r.get("security", "?") for r in rows})
    codes = Counter()
    for r in rows:
        codes.update(set(r.get("errors", [])))
    if not codes:
        print("no findings")
        return
    head = "  %-42s" % "check" + "".join("%12s" % l for l in levels) + "%10s" % "total"
    print(head)
    for code, _ in codes.most_common():
        cells = []
        for l in levels:
            n = sum(1 for r in rows if r.get("security") == l and code in r.get("errors", []))
            d = sum(1 for r in rows if r.get("security") == l)
            cells.append("%12s" % pct(n, d))
        print("  %-42s%s%10d" % (code[:42], "".join(cells), codes[code]))


def quote_quality(rows):
    """How close are the quotes that failed the verbatim test?"""
    vals = [r["best_overlap"] for r in rows
            if "quote_loose_match" in r.get("errors", []) and "best_overlap" in r]
    if not vals:
        return
    section("quote_loose_match: overlap between the quote and the cited message (%d blocks)" % len(vals))
    buckets = Counter()
    for v in vals:
        buckets[min(int(v * 10) / 10.0, 0.9)] += 1
    for b in sorted(buckets):
        print("  %.1f - %.1f  %6d  %s" % (b, b + 0.1, buckets[b], bar(buckets[b], len(vals))))
    print("\n  high overlap means the model reworded a real quote; low means it cited the wrong text.")


def wrapper_discipline(rows):
    """Is the wrapper test being written at all, and in the right place?"""
    section("wrapper discipline")
    tot = len(rows)
    missing = sum(1 for r in rows if "no_wrapper_test" in r.get("errors", []))
    numbered = sum(1 for r in rows if "numbered_citation" in r.get("errors", []))
    print("  never names the wrapper        %6d  %s" % (missing, pct(missing, tot)))
    print("  numbers a message anyway       %6d  %s" % (numbered, pct(numbered, tot)))
    print("  states the wrapper             %6d  %s" % (tot - missing, pct(tot - missing, tot)))
    by_level = defaultdict(lambda: [0, 0])
    for r in rows:
        lv = r.get("security", "?")
        by_level[lv][1] += 1
        if {"no_wrapper_test", "numbered_citation"} & set(r.get("errors", [])):
            by_level[lv][0] += 1
    print("\n  by label:")
    for lv in sorted(by_level):
        b, t = by_level[lv]
        print("    %-12s %6d blocks  %s skip it or number a message" % (lv, t, pct(b, t)))


def label_vs_position(rows):
    """The shortcut that matters most: can the label be guessed from the preceding turn?"""
    section("label vs the turn the call follows")
    table = defaultdict(Counter)
    for r in rows:
        table[r.get("prev_turn") or "none"][r.get("security", "?")] += 1
    for prev in sorted(table):
        tot = sum(table[prev].values())
        parts = ", ".join("%s %d (%s)" % (k, n, pct(n, tot))
                          for k, n in table[prev].most_common())
        print("  follows %-6s %6d calls -> %s" % (prev, tot, parts))
        top = table[prev].most_common(1)[0][1]
        if tot and top / tot > 0.95:
            print("      the preceding turn alone predicts the label %s of the time" % pct(top, tot))


def main():
    paths = sys.argv[1:] or ["."]
    rows, files = load(paths)
    if not rows:
        print("no report rows found under: %s" % ", ".join(str(p) for p in paths))
        return 1
    print("read %d blocks from %d report file(s)" % (len(rows), len(files)))
    for f in files:
        print("  %s" % f.name)
    check_breakdown(rows)
    quote_quality(rows)
    wrapper_discipline(rows)
    label_vs_position(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
