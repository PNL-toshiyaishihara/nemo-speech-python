"""Library loading helpers (no models)."""

import ctypes

import pytest

from nemo_speech import _loader
from nemo_speech.capi import ctypes_function_for


def test_missing_symbol_names_the_symbol():
    bind = ctypes_function_for(ctypes.pythonapi)
    with pytest.raises(ImportError, match="does not export nemo_speech_no_such_function"):

        @bind("nemo_speech_no_such_function", [], None)
        def nemo_speech_no_such_function():
            ...


def test_cuda_toolkit_only_for_overridden_library_dirs(monkeypatch, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    toolkit = tmp_path / "cuda"
    (toolkit / "bin" / "x64").mkdir(parents=True)
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    # Bundled libraries (a wheel): only their own directory.
    assert _loader._dll_directories(lib, overridden=False) == [lib]
    # NEMO_SPEECH_LIB_PATH (e.g. a local CUDA build): the toolkit too, with
    # CUDA 13's bin\x64.
    assert _loader._dll_directories(lib, overridden=True) == [
        lib,
        toolkit / "bin",
        toolkit / "bin" / "x64",
    ]
