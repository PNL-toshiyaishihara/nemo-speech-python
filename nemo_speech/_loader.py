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


def _register_dll_directories(directory: pathlib.Path) -> None:
    # The dependent DLLs (ggml*, the C++ runtime libraries) live beside the
    # C ABI DLL. CUDA builds additionally need the toolkit runtime.
    directories = [directory]
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        directories.append(pathlib.Path(cuda_path) / "bin")
    for d in directories:
        if d.is_dir():
            _dll_directory_handles.append(os.add_dll_directory(str(d)))


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
                raise OSError(f"failed to load {path}: {e}") from e
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"NeMo-Speech.cpp library '{stem}' not found (tried: {tried}). "
        "Reinstall the package, or set NEMO_SPEECH_LIB_PATH to the directory "
        "containing the library."
    )
