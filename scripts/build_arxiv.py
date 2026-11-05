#!/usr/bin/env python
"""Builds the arXiv upload from the local paper/ folder: main.tex with
full-line comments removed (arXiv publishes the source), the compiled
main.bbl (arXiv does not run BibTeX reliably), and only the figures the
paper includes. Compiles the bundle once in a temporary copy to prove it
is self-contained. Writes release/arxiv/ and release/arxiv.zip.

Run `make paper` first so main.bbl is current.

Usage:
    python scripts/build_arxiv.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def strip_comments(tex: str) -> str:
    """Drops full-line comments; keeps `\\%` and comments after code."""
    out = [line for line in tex.splitlines() if not line.lstrip().startswith("%")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    parser.add_argument("--out", type=Path, default=Path("release/arxiv"))
    args = parser.parse_args()

    tex = (args.paper / "main.tex").read_text(encoding="utf-8")
    bbl = args.paper / "main.bbl"
    if not bbl.exists() or bbl.stat().st_mtime < (args.paper / "references.bib").stat().st_mtime:
        raise SystemExit("main.bbl is missing or older than references.bib; run `make paper`")
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    (args.out / "main.tex").write_text(strip_comments(tex), encoding="utf-8", newline="\n")
    shutil.copy2(bbl, args.out / "main.bbl")
    figures = INCLUDE.findall(tex)
    for fig in figures:
        dst = args.out / fig
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.paper / fig, dst)

    # Prove the bundle compiles on its own.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "arxiv"
        shutil.copytree(args.out, work)
        for _ in range(2):
            run = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=work,
                capture_output=True,
                text=True,
            )
        log = (work / "main.log").read_text(encoding="utf-8", errors="replace")
        if run.returncode != 0 or "undefined" in log.lower():
            raise SystemExit(f"bundle does not compile cleanly; see {work / 'main.log'}")
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    archive = shutil.make_archive(str(args.out), "zip", root_dir=args.out)
    n = len(figures) + 2
    print(f"{n} files, compiles to {pages.group(1) if pages else '?'} pages -> {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
