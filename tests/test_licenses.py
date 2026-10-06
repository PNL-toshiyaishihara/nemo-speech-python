"""License texts for everything a wheel bundles."""

import pathlib
import sys

import nemo_speech

PACKAGE = pathlib.Path(nemo_speech.__file__).resolve().parent
THIRD_PARTY = PACKAGE / "licenses" / "third_party"


def _bundled_names():
    """Library file names in the package and in the repair step's directory."""
    names = []
    for directory in (PACKAGE / "lib", PACKAGE.parent / "nemo_speech.libs"):
        if directory.is_dir():
            names += [p.name.lower() for p in directory.iterdir()]
    return names


def test_compiler_runtime_notices():
    if sys.platform == "win32":
        assert (THIRD_PARTY / "msvc-runtime" / "NOTICE.txt").is_file()
    elif sys.platform.startswith("linux"):
        for name in ("NOTICE.txt", "COPYING3", "COPYING.RUNTIME"):
            assert (THIRD_PARTY / "gcc-runtime" / name).is_file(), name


def test_bundled_cuda_libraries_ship_the_eula():
    if any("cublas" in name or "cudart" in name for name in _bundled_names()):
        assert (THIRD_PARTY / "cuda" / "EULA.txt").is_file()
