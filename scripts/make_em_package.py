"""Build the LaTeX source package for Elsevier Editorial Manager (EM).

EM requirements this script enforces (Elsevier support article "How to submit a LaTeX
file in Editorial Manager"):
  * no subfolders: every file sits at the top level, and \\includegraphics paths are
    rewritten to bare file names;
  * every referenced figure and the bibliography (.bib, plus .bbl if available) is included;
  * source file names differ from the PDF file names uploaded separately;
  * only standard packages (elsarticle, stix, amsmath, ... are on EM's TeX installation).

The manuscript source is the CLEAN version (\\markchangesfalse): EM builds the reviewer
PDF from it. Upload the marked PDF (main.pdf) separately as "Revised Manuscript with
Changes Marked".

Usage (from the repository root, after a full local build of paper/main.tex):
    python scripts/make_em_package.py
Output: paper/em_submission/  (flat folder)  and  paper/em_submission.zip (flat zip)
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "em_submission"
JOB = "RoAS_manuscript"          # source name, deliberately different from main.pdf

TEXT_ITEMS = {
    "highlights.tex": "RoAS_highlights.txt",
}


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def main() -> None:
    raw = (PAPER / "main.tex").read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    tex = raw.replace("\r\n", "\n")

    # 1. clean version
    if "\\markchangestrue" not in tex:
        fail("\\markchangestrue not found in main.tex")
    tex = tex.replace("\\markchangestrue", "\\markchangesfalse", 1)

    # 2. no unresolved placeholders
    pending = re.findall(r"\\pending\{([^{}]+)\}", tex)
    pending = [p for p in pending if p != "..."]
    if pending:
        fail(f"unfilled \\pending placeholders: {pending}")

    # 3. flatten figure paths
    figures = []

    def flatten(m: re.Match) -> str:
        opts, path = m.group(1) or "", m.group(2)
        src = PAPER / path
        if not src.is_file():
            fail(f"figure not found: {path}")
        figures.append(src)
        return f"\\includegraphics{opts}{{{src.name}}}"

    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", flatten, tex)
    tex = re.sub(r"\\graphicspath\{.*?\}\s*\n", "", tex)
    names = [f.name for f in figures]
    if len(set(names)) != len(names):
        fail("two figures share a file name; rename one before flattening")

    # 4. bibliography: every cited key must exist
    bib_match = re.search(r"\\bibliography\{([^}]+)\}", tex)
    if not bib_match:
        fail("no \\bibliography command")
    bib_name = bib_match.group(1).split(",")[0].strip()
    bib_path = PAPER / f"{bib_name}.bib"
    bib = bib_path.read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cited = {k.strip() for grp in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", tex) for k in grp.split(",")}
    missing = sorted(cited - keys)
    if missing:
        fail(f"cited keys missing from {bib_path.name}: {missing}")

    # 5. write the flat package
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    (OUT / f"{JOB}.tex").write_bytes(tex.replace("\n", newline).encode("utf-8"))
    shutil.copy2(bib_path, OUT / bib_path.name)
    for fig in figures:
        shutil.copy2(fig, OUT / fig.name)

    # .bbl: only if it is newer than both main.tex and the .bib, i.e. from a fresh build
    bbl = PAPER / "main.bbl"
    bbl_note = "not included (no up-to-date main.bbl; EM will run BibTeX on the .bib)"
    if bbl.is_file() and bbl.stat().st_mtime >= max((PAPER / "main.tex").stat().st_mtime, bib_path.stat().st_mtime):
        bbl_text = bbl.read_text(encoding="utf-8")
        absent = [k for k in cited if f"{{{k}}}" not in bbl_text]
        if absent:
            bbl_note = f"not included (main.bbl lacks {absent}; rebuild locally first)"
        else:
            shutil.copy2(bbl, OUT / f"{JOB}.bbl")
            bbl_note = f"{JOB}.bbl included"

    # highlights as a plain-text file (Elsevier asks for an editable highlights file)
    for src_name, dst_name in TEXT_ITEMS.items():
        src = PAPER / src_name
        if src.is_file():
            items = re.findall(r"\\item\s+(.*)", src.read_text(encoding="utf-8"))
            items = [i.strip().replace("\\%", "%") for i in items]
            (OUT.parent / dst_name).write_text("Highlights\n\n" + "\n".join(f"- {i}" for i in items) + "\n",
                                               encoding="utf-8")

    # 6. flat zip (no folder inside)
    zpath = PAPER / "em_submission.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, arcname=f.name)

    print(f"Wrote {OUT.relative_to(ROOT)} and {zpath.relative_to(ROOT)}")
    print(f"  {JOB}.tex (clean version, figure paths flattened)")
    print(f"  {bib_path.name} ({len(cited)} cited keys, all present)")
    print(f"  figures: {', '.join(names)}")
    print(f"  bibliography: {bbl_note}")
    print(f"  highlights: paper/{TEXT_ITEMS['highlights.tex']}")


if __name__ == "__main__":
    main()
