"""Print the vendored source revisions as JSON (vendor/revisions.json).

An sdist carries no git metadata, so the sdist workflow records the
revisions here and cmake/build_info.cmake embeds them in every wheel built
from it. Run from the repository root:

    python scripts/source_revisions.py > vendor/revisions.json
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = {
    "NeMo-Speech.cpp": ROOT / "vendor" / "NeMo-Speech.cpp",
    "llama.cpp": ROOT / "vendor" / "NeMo-Speech.cpp" / "llama.cpp",
    "sentencepiece": ROOT / "vendor" / "sentencepiece",
}


def _git(directory: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def revisions() -> dict:
    result = {}
    for name, directory in SOURCES.items():
        result[name] = {
            "commit": _git(directory, "rev-parse", "HEAD"),
            "describe": _git(directory, "describe", "--tags", "--always"),
        }
    return result


if __name__ == "__main__":
    json.dump(revisions(), sys.stdout, indent=2)
    sys.stdout.write("\n")
