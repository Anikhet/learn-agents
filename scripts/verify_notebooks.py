"""Validate and execute each notebook in a fresh kernel, without live services."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="*.ipynb")
    args = parser.parse_args()
    # Never enable provider calls or remote trace upload during course validation.
    for key in list(os.environ):
        if key.endswith("API_KEY") or key.startswith("LANGSMITH_") or key.startswith("LANGCHAIN_"):
            os.environ.pop(key)
    os.environ.update(RUN_LIVE="0", RUN_REMOTE="0", LANGSMITH_TRACING="false",
                      LANGCHAIN_TRACING_V2="false")
    # Use the same interpreter as this runner, rather than an unrelated global kernel.
    with tempfile.TemporaryDirectory(prefix="course-kernels-") as kernel_root:
        kernel = Path(kernel_root) / "kernels" / "course"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Course verification", "language": "python"}))
        os.environ["JUPYTER_PATH"] = kernel_root + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        results = []
        for path in sorted((ROOT / "notebooks").rglob(args.match)):
            started = time.monotonic()
            try:
                nb = nbformat.read(path, as_version=4)
                nbformat.validate(nb)
                with tempfile.TemporaryDirectory(prefix="course-lab-") as work:
                    NotebookClient(nb, timeout=120, kernel_name="course",
                                   resources={"metadata": {"path": work}}).execute()
                result = {"notebook": str(path.relative_to(ROOT)), "status": "passed"}
            except Exception as exc:
                result = {"notebook": str(path.relative_to(ROOT)), "status": "failed", "error": str(exc)}
            result["seconds"] = round(time.monotonic() - started, 2)
            results.append(result)
            print(f"{result['status'].upper()} {path.name}", flush=True)
            if result["status"] == "failed":
                print(result["error"], flush=True)
        report = ROOT / "docs" / "verification.json"
        report.write_text(json.dumps({"mode": "offline; live and remote cells skipped",
                                     "python": sys.version, "results": results}, indent=2) + "\n")
        if not results or any(r["status"] == "failed" for r in results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
