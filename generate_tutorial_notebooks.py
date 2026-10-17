"""Build notebooks from reviewed Markdown lessons; no network or model calls."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FENCE = re.compile(r"^```\{code-cell\} python(?: (live|remote))?\n(.*?)^```\s*$", re.M | re.S)


def notebook(source, relative):
    cells = []

    def add(kind, text, tag=None):
        text = text.strip() + "\n"
        if not text.strip():
            return
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
        cell["id"] = hashlib.sha256(f"{relative}:{len(cells)}:{text}".encode()).hexdigest()[:12]
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            cell["metadata"]["tags"] = [tag or "offline"]
            if tag:
                flag = "RUN_REMOTE" if tag == "remote" else "RUN_LIVE"
                text = (f"import os\nif os.getenv('{flag}') != '1':\n"
                        f"    print('Skipped: set {flag}=1 after reading this lesson’s setup.')\nelse:\n"
                        + "".join("    " + line if line.strip() else line for line in text.splitlines(True)))
                cell["source"] = text.splitlines(True)
        cells.append(cell)

    pos = 0
    for match in FENCE.finditer(source):
        add("markdown", source[pos:match.start()])
        add("code", match[2], match[1])
        pos = match.end()
    add("markdown", source[pos:])
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "course": {"source": str(relative), "reviewed": "2026-09-27"},
    }, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated notebooks differ")
    parser.add_argument("--match", default="*.md", help="Only (re)generate lessons matching this glob")
    args = parser.parse_args()
    changed = []
    for source in sorted((ROOT / "lessons").rglob(args.match)):
        if source.stem.isupper():
            continue  # STYLE.md and READMEs are author docs, not lessons
        relative = source.relative_to(ROOT / "lessons")
        target = ROOT / "notebooks" / relative.with_suffix(".ipynb")
        content = json.dumps(notebook(source.read_text(), relative), indent=2, ensure_ascii=False) + "\n"
        if not target.exists() or target.read_text() != content:
            changed.append(str(target.relative_to(ROOT)))
            if not args.check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
    print(f"{len(changed)} notebooks {'out of date' if args.check else 'updated'}")
    if args.check and changed:
        raise SystemExit("\n".join(changed))


if __name__ == "__main__":
    main()
