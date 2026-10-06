"""Locate and load the NeMo-Speech.cpp shared libraries."""

from __future__ import annotations

import ctypes
import functools
import os
import pathlib
import sys

_PACKAGE_LIB_DIR = pathlib.Path(__file__).resolve().parent / "lib"

# Keeps os.add_dll_directory() registrations alive for the process lifetime.
_dll_directory_handles: list = []


def library_dir() -> pathlib.Path:
    """Directory searched for the shared libraries.

    ``NEMO_SPEECH_LIB_PATH`` overrides the copy bundled with the package, e.g.
    to use a local NeMo-Speech.cpp build or install prefix.
    """
    override = os.environ.get("NEMO_SPEECH_LIB_PATH")
    return pathlib.Path(override) if override else _PACKAGE_LIB_DIR


def _candidate_names(stem: str) -> list[str]:
    if sys.platform == "win32":
        return [f"{stem}.dll"]
    if sys.platform == "darwin":
        return [f"lib{stem}.1.dylib", f"lib{stem}.dylib"]
    return [f"lib{stem}.so.1", f"lib{stem}.so"]


def _dll_directories(directory: pathlib.Path, overridden: bool) -> list:
    """Directories whose DLLs the C ABI DLLs in ``directory`` may import.

    The dependent DLLs (ggml*, and in a wheel the CUDA runtime and cuBLAS
    copied in by the repair step) live beside the C ABI DLL. Only a library
    directory given through NEMO_SPEECH_LIB_PATH, such as a local CUDA build,
    also gets the toolkit's DLLs (``bin``, or ``bin\\x64`` from CUDA 13 on).
    A wheel must not: the search order among added directories is
    unspecified, so an installed toolkit's older cuBLAS could be picked
    instead of the bundled one.
    """
    directories = [directory]
    cuda_path = os.environ.get("CUDA_PATH")
    if overridden and cuda_path:
        directories += [pathlib.Path(cuda_path) / "bin", pathlib.Path(cuda_path) / "bin" / "x64"]
    return [d for d in directories if d.is_dir()]


def _register_dll_directories(directory: pathlib.Path) -> None:
    overridden = bool(os.environ.get("NEMO_SPEECH_LIB_PATH"))
    for d in _dll_directories(directory, overridden):
        _dll_directory_handles.append(os.add_dll_directory(str(d)))


def _missing_cuda_driver_hint(directory: pathlib.Path) -> str:
    """Explain a load failure of a CUDA build on a machine without the driver."""
    if not any(directory.glob("*ggml-cuda*")):
        return ""
    driver = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        ctypes.CDLL(driver)
        return ""
    except OSError:
        return (
            f" This is a CUDA build of nemo-speech, which needs the NVIDIA driver "
            f"({driver}); it was not found. Install an NVIDIA driver, or install "
            "the CPU wheel instead."
        )


@functools.lru_cache(maxsize=None)
def load_library(stem: str) -> ctypes.CDLL:
    """Load ``stem`` (e.g. ``"nemo_speech_asr_c"``) from :func:`library_dir`."""
    directory = library_dir()
    if sys.platform == "win32":
        _register_dll_directories(directory)
    candidates = [directory / name for name in _candidate_names(stem)]
    for path in candidates:
        if path.exists():
            try:
                return ctypes.CDLL(str(path))
            except OSError as e:
                hint = _missing_cuda_driver_hint(directory)
                raise OSError(f"failed to load {path}: {e}{hint}") from e
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"NeMo-Speech.cpp library '{stem}' not found (tried: {tried}). "
        "Reinstall the package, or set NEMO_SPEECH_LIB_PATH to the directory "
        "containing the library."
    )
