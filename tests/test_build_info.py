"""nemo_speech.build_info() and the release-notes helpers."""

import importlib.util
import pathlib
import re
from importlib.metadata import version

import pytest

import nemo_speech

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_build_info_describes_this_build():
    info = nemo_speech.build_info()
    assert info["version"] == version("nemo-speech")
    assert info["variant"] == "default" or re.fullmatch(r"vulkan|cu\d+", info["variant"])
    assert set(info["backends"]) == {"cuda", "vulkan", "metal"}
    assert info["components"] == {"asr": True, "diar": True, "tts": True, "nmt": True}
    assert info["tts_tokenizers"] == {"ja": True, "zh": True}
    for name in ("NeMo-Speech.cpp", "llama.cpp", "sentencepiece"):
        assert re.fullmatch(r"[0-9a-f]{40}", info["sources"][name]["commit"]), name


def test_variant_matches_backend():
    info = nemo_speech.build_info()
    variant = info["variant"]
    if variant.startswith("cu"):
        assert info["backends"]["cuda"]
        major_minor = "".join(info["cuda"]["toolkit"].split(".")[:2])
        assert variant == f"cu{major_minor}"
        # Recorded as built: the patched ggml-cuda compiles plain 100, 110 and
        # 12X as their architecture-specific "a" forms.
        archs = info["cuda"]["architectures"]
        assert not re.search(r"(^|;)(100|110|12\d)(-real|-virtual)?(;|$)", archs), archs
        assert version("nemo-speech").endswith(f"+{variant}")
    elif variant == "vulkan":
        assert info["backends"]["vulkan"]
        assert version("nemo-speech").endswith("+vulkan")
    else:
        assert not info["backends"]["cuda"] and not info["backends"]["vulkan"]
        assert "+" not in version("nemo-speech")


@pytest.fixture(scope="module")
def release_notes():
    path = REPO_ROOT / "scripts" / "release_notes.py"
    if not path.exists():
        pytest.skip("scripts/ not available")
    spec = importlib.util.spec_from_file_location("release_notes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("nemo_speech-0.1.0-py3-none-win_amd64.whl", "Windows x64"),
        ("nemo_speech-0.1.0-py3-none-macosx_13_0_arm64.whl", "macOS arm64 (13.0+)"),
        (
            "nemo_speech-0.1.0+cu128-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "Linux x86_64 (glibc 2.28+)",
        ),
        ("nemo_speech-0.1.0-py3-none-manylinux_2_28_aarch64.whl", "Linux aarch64 (glibc 2.28+)"),
    ],
)
def test_platform_names(release_notes, filename, expected):
    assert release_notes.platform_name(pathlib.Path(filename)) == expected


def test_variant_names(release_notes):
    assert release_notes.variant_name({"variant": "default", "backends": {"metal": False}}) == "CPU"
    assert release_notes.variant_name({"variant": "default", "backends": {"metal": True}}) == "CPU + Metal"
    assert release_notes.variant_name({"variant": "vulkan"}) == "Vulkan"
    assert release_notes.variant_name({"variant": "cu130", "cuda": {"toolkit": "13.0.88"}}) == "CUDA 13.0"


@pytest.mark.parametrize(
    "architectures, expected",
    [
        (
            "75-real;80-real;86-real;89-real;90-real;100a-real;120a-real;90-virtual",
            "compute capability 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0; newer GPUs through PTX (9.0+)",
        ),
        ("86-real", "compute capability 8.6"),
        ("75-real;120a", "compute capability 7.5, 12.0"),
        ("90", "compute capability 9.0; newer GPUs through PTX (9.0+)"),
        ("native", "native"),
    ],
)
def test_cuda_targets(release_notes, architectures, expected):
    assert release_notes.cuda_targets(architectures) == expected


@pytest.mark.parametrize(
    "tag, previous, tags, expected",
    [
        # A later final release and any pre-release: one range from `previous`.
        ("v0.2.0", "v0.1.0", ["v0.1.0", "v0.2.0rc1", "v0.2.0"], [("v0.2.0", "v0.1.0")]),
        ("v0.1.0rc2", "v0.1.0rc1", ["v0.1.0rc1", "v0.1.0rc2"], [("v0.1.0rc2", "v0.1.0rc1")]),
        # The first release of all: GitHub starts at the first commit.
        ("v0.1.0rc1", None, ["v0.1.0rc1"], [("v0.1.0rc1", None)]),
        # The first final release: one range per pre-release, so the list does
        # not start at the last pre-release.
        (
            "v0.1.0",
            None,
            ["v0.1.0rc1", "v0.1.0rc2", "v0.1.0"],
            [("v0.1.0rc1", None), ("v0.1.0rc2", "v0.1.0rc1"), ("v0.1.0", "v0.1.0rc2")],
        ),
    ],
)
def test_changelog_ranges(release_notes, tag, previous, tags, expected):
    assert release_notes.changelog_ranges(tag, previous, tags) == expected


def test_merge_changes(release_notes):
    first = "\n".join([
        "<!-- Release notes generated using configuration in .github/release.yml at v0.1.0rc1 -->",
        "",
        "## What's Changed",
        "### Documentation",
        "* docs by @a in #2",
        "### Features",
        "* feature one by @a in #1",
        "",
        "## New Contributors",
        "* @a made their first contribution in #1",
        "",
        "**Full Changelog**: https://example.invalid/commits/v0.1.0rc1",
    ])
    second = "\n".join([
        "## What's Changed",
        "### Fixes",
        "* fix by @b in #3",
        "### Features",
        "* feature two by @a in #4",
        "",
        "**Full Changelog**: https://example.invalid/compare/v0.1.0rc1...v0.1.0",
    ])
    categories = release_notes.changelog_categories()
    assert categories[:3] == ["Breaking changes", "Features", "Fixes"]
    merged = release_notes.merge_changes([first, second, ""], categories, "https://example.invalid/commits/v0.1.0")
    assert merged == "\n".join([
        "## What's Changed",
        "### Features",
        "* feature one by @a in #1",
        "* feature two by @a in #4",
        "### Fixes",
        "* fix by @b in #3",
        "### Documentation",
        "* docs by @a in #2",
        "",
        "## New Contributors",
        "* @a made their first contribution in #1",
        "",
        "",
        "**Full Changelog**: https://example.invalid/commits/v0.1.0",
    ])


def test_prerelease_pattern(release_notes):
    assert release_notes.PRERELEASE.search("v0.1.0rc1")
    assert release_notes.PRERELEASE.search("v0.2.0a2")
    assert not release_notes.PRERELEASE.search("v0.1.0")
