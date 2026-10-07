"""Generate the notes of a GitHub release (used by .github/workflows/release.yml).

Everything about the build comes from the wheels themselves: each one carries
nemo_speech/_build_info.json (see cmake/build_info.cmake). The change list
comes from GitHub's generated release notes, which group merged PRs by label
(.github/release.yml); a first final release combines those of its
pre-releases.

    python scripts/release_notes.py --tag v0.1.0 --dist dist > notes.md

Needs git (with tags) and, for the change list, an authenticated `gh`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

PRERELEASE = re.compile(r"(a|b|rc|dev)\d*$")
SOURCE_REPOS = {
    "NeMo-Speech.cpp": "NVIDIA/NeMo-Speech.cpp",
    "llama.cpp": "ggml-org/llama.cpp",
    "sentencepiece": "google/sentencepiece",
}
SOURCE_NAMES = {"NeMo-Speech.cpp": "NeMo-Speech.cpp", "llama.cpp": "llama.cpp", "sentencepiece": "SentencePiece"}
# Minimum NVIDIA driver branch for the CUDA releases the wheels are built with.
DRIVERS = {"12": "R570", "13": "R580"}


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def previous_tag(tag: str) -> Optional[str]:
    """The release before `tag`: the previous final release for a final tag,
    the previous release of any kind for a pre-release."""
    tags = run("git", "tag", "--list", "v*", "--merged", tag, "--sort=-creatordate").splitlines()
    final = not PRERELEASE.search(tag)
    for candidate in tags:
        if candidate == tag or (final and PRERELEASE.search(candidate)):
            continue
        return candidate
    return None


def changelog_ranges(
    tag: str, previous: Optional[str], tags: List[str]
) -> List[Tuple[str, Optional[str]]]:
    """(tag, previous tag) pairs whose generated notes together cover `tag`.

    `tags` are the release tags merged into `tag`, oldest first. Without a
    previous tag, GitHub starts the notes at the latest release before `tag`,
    so the first final release after pre-releases would list only the changes
    since the last of them; it gets one range per earlier release instead.
    """
    if previous:
        return [(tag, previous)]
    chain = [t for t in tags if t != tag] + [tag]
    return [(t, chain[i - 1] if i else None) for i, t in enumerate(chain)]


def changelog_categories() -> List[str]:
    """Category titles of the generated notes, in .github/release.yml order."""
    config = pathlib.Path(__file__).resolve().parents[1] / ".github" / "release.yml"
    return re.findall(r"^\s*- title: (.+?)\s*$", config.read_text(encoding="utf-8"), re.M)


def merge_changes(bodies: List[str], categories: List[str], full_changelog: str) -> str:
    """Combine generated notes: each category's entries, then new contributors."""
    entries: Dict[str, List[str]] = {}
    contributors: List[str] = []
    for body in bodies:
        section, category = "", ""
        for line in body.splitlines():
            if line.startswith("## "):
                section, category = line[3:].strip(), ""
            elif line.startswith("### "):
                category = line[4:].strip()
            elif line.startswith("* "):
                if section == "New Contributors":
                    target = contributors
                else:
                    target = entries.setdefault(category, [])
                if line not in target:
                    target.append(line)
    order = [c for c in categories if c in entries] + [c for c in entries if c not in categories]
    out = ["## What's Changed"]
    for category in order:
        if category:
            out.append(f"### {category}")
        out += entries[category]
    if contributors:
        out += ["", "## New Contributors", *contributors]
    out += ["", "", f"**Full Changelog**: {full_changelog}"]
    return "\n".join(out)


def generated_notes(repo: str, tag: str, previous: Optional[str]) -> str:
    args = ["gh", "api", f"repos/{repo}/releases/generate-notes", "-f", f"tag_name={tag}"]
    if previous:
        args += ["-f", f"previous_tag_name={previous}"]
    return json.loads(run(*args))["body"].strip()


def generated_changes(repo: str, tag: str, previous: Optional[str]) -> str:
    tags = run("git", "tag", "--list", "v*", "--merged", tag, "--sort=creatordate").splitlines()
    ranges = changelog_ranges(tag, previous, tags)
    try:
        bodies = [generated_notes(repo, t, p) for t, p in ranges]
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError) as e:
        print(f"warning: generated notes unavailable ({e})", file=sys.stderr)
        return "_The change list could not be generated; see the commit history._"
    if len(bodies) == 1:
        return bodies[0]
    return merge_changes(bodies, changelog_categories(), f"https://github.com/{repo}/commits/{tag}")


def read_wheel(path: pathlib.Path) -> Tuple[dict, Dict[str, List[str]]]:
    with zipfile.ZipFile(path) as z:
        info = json.loads(z.read("nemo_speech/_build_info.json"))
        metadata_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        metadata: Dict[str, List[str]] = {}
        for line in z.read(metadata_name).decode("utf-8").splitlines():
            if not line:
                break  # headers end at the first blank line
            key, _, value = line.partition(": ")
            metadata.setdefault(key, []).append(value)
    return info, metadata


def platform_name(wheel: pathlib.Path) -> str:
    tag = wheel.name[: -len(".whl")].split("-")[-1]
    if tag.startswith("win_amd64"):
        return "Windows x64"
    if tag.startswith("macosx"):
        m = re.match(r"macosx_(\d+)_(\d+)_(\w+)", tag)
        return f"macOS {m.group(3)} ({m.group(1)}.{m.group(2)}+)" if m else "macOS"
    m = re.search(r"manylinux_(\d+)_(\d+)_(\w+)$", tag)
    if m:
        return f"Linux {m.group(3)} (glibc {m.group(1)}.{m.group(2)}+)"
    return tag


def variant_name(info: dict) -> str:
    variant = info.get("variant", "default")
    if variant.startswith("cu"):
        toolkit = info.get("cuda", {}).get("toolkit", "")
        return f"CUDA {'.'.join(toolkit.split('.')[:2])}" if toolkit else variant
    if variant == "vulkan":
        return "Vulkan"
    return "CPU + Metal" if info.get("backends", {}).get("metal") else "CPU"


def cuda_targets(architectures: str) -> str:
    """Describe a built CMAKE_CUDA_ARCHITECTURES list in compute capabilities.

    ``-real`` entries are native code for that capability, ``-virtual``
    entries are PTX that the driver compiles for any later GPU, and a bare
    entry is both. PTX of architecture- or family-specific targets (``120a``,
    ``120f``) does not run on later GPUs, so only plain targets count as PTX.
    """
    native: List[Tuple[int, int]] = []
    ptx: List[Tuple[int, int]] = []
    for entry in filter(None, architectures.split(";")):
        m = re.fullmatch(r"(\d+)(\d)([af]?)(?:-(real|virtual))?", entry)
        if not m:
            return architectures.replace(";", ", ")
        capability = (int(m.group(1)), int(m.group(2)))
        if m.group(4) != "virtual":
            native.append(capability)
        if m.group(4) != "real" and not m.group(3):
            ptx.append(capability)
    text = ", ".join(f"{major}.{minor}" for major, minor in sorted(native))
    text = f"compute capability {text}" if text else ""
    if ptx:
        major, minor = min(ptx)
        newer = f"newer GPUs through PTX ({major}.{minor}+)"
        text = f"{text}; {newer}" if text else newer
    return text


def gpu_notes(info: dict) -> str:
    cuda = info.get("cuda")
    if cuda:
        major = cuda.get("toolkit", "").split(".")[0]
        targets = cuda_targets(cuda.get("architectures", ""))
        return f"NVIDIA driver {DRIVERS.get(major, 'for CUDA ' + major)}+; {targets}"
    if info.get("variant") == "vulkan":
        return "GPU driver with Vulkan"
    return ""


def python_versions(metadata: Dict[str, List[str]]) -> str:
    versions = [
        c.rsplit(":: ", 1)[1]
        for c in metadata.get("Classifier", [])
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    ]
    requires = metadata.get("Requires-Python", ["?"])[0]
    return f"{', '.join(versions)} (Requires-Python `{requires}`)" if versions else requires


def upstream_pin(tag: str) -> Optional[str]:
    try:
        return run("git", "rev-parse", f"{tag}:vendor/NeMo-Speech.cpp")
    except subprocess.CalledProcessError:
        return None


def notes(tag: str, dist: pathlib.Path, repo: str) -> str:
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels in {dist}")
    built = [(w, *read_wheel(w)) for w in wheels]
    sources = built[0][1].get("sources", {})
    for wheel, info, _ in built[1:]:
        if info.get("sources") != sources:
            raise SystemExit(f"{wheel.name} was built from different sources than {wheels[0].name}")

    previous = previous_tag(tag)
    out = [f"Unofficial Python bindings for NeMo-Speech.cpp, {tag}.", ""]

    out += ["## Changes", "", generated_changes(repo, tag, previous), ""]

    out += ["## Vendored sources", "", "| Source | Revision | Commit |", "|---|---|---|"]
    for key, revision in sources.items():
        commit = revision.get("commit", "")
        url = f"https://github.com/{SOURCE_REPOS.get(key, key)}/commit/{commit}"
        out.append(f"| {SOURCE_NAMES.get(key, key)} | {revision.get('describe', '')} | [`{commit[:12]}`]({url}) |")
    current = sources.get("NeMo-Speech.cpp", {}).get("commit")
    old = upstream_pin(previous) if previous else None
    if previous and old and current:
        if old == current:
            out += ["", f"NeMo-Speech.cpp is unchanged since {previous}."]
        else:
            out += [
                "",
                f"NeMo-Speech.cpp changes since {previous}: "
                f"https://github.com/{SOURCE_REPOS['NeMo-Speech.cpp']}/compare/{old}...{current}",
            ]
    out.append("")

    out += [
        "## Wheels",
        "",
        f"Python: {python_versions(built[0][2])}. Every wheel is `py3-none`, so one file serves all of them.",
        "",
        "The x86-64 wheels (Linux x86_64 and Windows x64, every variant) need a CPU with AVX2, "
        "FMA, F16C and BMI2 (Intel Haswell, AMD Zen or newer).",
        "",
        "| Variant | Platform | Requirements | File |",
        "|---|---|---|---|",
    ]
    order = {"CPU": 0, "CPU + Metal": 0, "Vulkan": 1}
    for wheel, info, _ in sorted(built, key=lambda b: (order.get(variant_name(b[1]), 2), variant_name(b[1]), platform_name(b[0]))):
        out.append(f"| {variant_name(info)} | {platform_name(wheel)} | {gpu_notes(info)} | `{wheel.name}` |")
    out.append("")

    base = f"https://github.com/{repo}/releases/download/{tag}"
    example = next((w for w, i, _ in built if i.get("variant", "").startswith("cu")), wheels[0])
    out += [
        "## Install",
        "",
        "All variants share the distribution name `nemo-speech`, so install the file for your "
        "platform and backend directly (a version specifier with `-f` would let pip pick a GPU "
        "variant). For example:",
        "",
        "```bash",
        f"pip install {base}/{example.name}",
        "```",
        "",
        "GPU wheels bundle their runtime libraries (CUDA runtime, cuBLAS) and need only the GPU "
        "driver. `nemo_speech.build_info()` reports how an installation was built. "
        "`SHA256SUMS` lists the checksums of all files.",
        "",
        "## Known issues",
        "",
        f"Upstream issues worked around in this release: https://github.com/{repo}/issues?q=label%3Aupstream",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist", type=pathlib.Path, required=True)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    args = parser.parse_args()
    repo = args.repo or re.sub(
        r"^.*github\.com[:/]|\.git$", "", run("git", "remote", "get-url", "origin")
    )
    sys.stdout.write(notes(args.tag, args.dist, repo))


if __name__ == "__main__":
    main()
