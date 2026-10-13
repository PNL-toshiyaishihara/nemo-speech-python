"""ggml loads backends from libggml's own directory, not the working directory.

``ggml_backend_load_all`` (reached through TTS) loads every ggml-named library
it finds, which runs that library's initializer. It must therefore not look in
the current working directory, where an attacker could plant one.
patches/llama.cpp/ggml-load-backends-from-library-dir.patch makes the scan look
beside libggml instead.

The test compiles a tiny library whose initializer drops a marker file, so a
planted library's code running is observable. It needs a C compiler and is
skipped without one; no models are needed.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

import nemo_speech
from nemo_speech import _loader

pytestmark = pytest.mark.skipif(
    not nemo_speech.build_info().get("llama_cpp_patched"),
    reason="the fix ships as a llama.cpp patch (NEMO_SPEECH_GGML_PATCHED)",
)

MARKER = "backend_initializer_ran"

_MARKER_SOURCE = r"""
#if defined(_WIN32)
#include <windows.h>
static void write_marker(void) {
    HANDLE f = CreateFileA("%(marker)s", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f != INVALID_HANDLE_VALUE) { DWORD n; WriteFile(f, "x", 1, &n, NULL); CloseHandle(f); }
}
BOOL WINAPI DllMain(HINSTANCE h, DWORD reason, LPVOID r) { if (reason == DLL_PROCESS_ATTACH) write_marker(); return TRUE; }
#else
#include <stdio.h>
__attribute__((constructor)) static void write_marker(void) {
    FILE *f = fopen("%(marker)s", "w"); if (f) { fputs("x", f); fclose(f); }
}
#endif
"""


def _compiler() -> list[str] | None:
    if sys.platform == "win32":
        for cc in ("clang", "clang-cl", "cl"):
            if shutil.which(cc):
                return [cc]
        return None
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return [cc]
    return None


def _compile_marker(out_path, tmp_path):
    """Compile a shared library whose initializer writes MARKER into the cwd."""
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler to build the marker library")
    src = tmp_path / "marker.c"
    src.write_text(_MARKER_SOURCE % {"marker": MARKER}, encoding="ascii")
    if cc[0] in ("cl",):
        cmd = cc + ["/nologo", "/LD", str(src), f"/Fe:{out_path}"]
    elif cc[0] in ("clang-cl",):
        cmd = cc + ["/LD", str(src), f"/Fe:{out_path}"]
    else:  # gcc/clang, and clang on Windows
        cmd = cc + ["-shared", str(src), "-o", str(out_path)]
        if sys.platform != "win32":
            cmd.insert(1, "-fPIC")
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        pytest.skip(f"could not build the marker library: {proc.stderr or proc.stdout}")


# Load a C ABI library the normal way (which pulls in libggml and, on Windows,
# registers the DLL directories), then call ggml_backend_load_all on the libggml
# core library, found by file name (its SONAME is not what _loader expects).
_LOAD_ALL = textwrap.dedent(
    """
    import ctypes, sys
    from nemo_speech import _loader
    _loader.load_library("nemo_speech_asr_c")
    directory = _loader.library_dir()
    if sys.platform == "win32":
        names = ["ggml.dll"]
    elif sys.platform == "darwin":
        names = ["libggml.1.dylib", "libggml.dylib"]
    else:
        names = ["libggml.so.0", "libggml.so"]
    core = next(directory / n for n in names if (directory / n).exists())
    ctypes.CDLL(str(core)).ggml_backend_load_all()
    """
)


def _run_load_all(cwd, env):
    env = {k: v for k, v in env.items() if k != "PYTHONPATH"}  # import the installed package
    subprocess.run([sys.executable, "-c", _LOAD_ALL], cwd=cwd, env=env, check=True,
                   capture_output=True, text=True)


def _lib_name(stem: str) -> str:
    return f"{stem}.dll" if sys.platform == "win32" else f"lib{stem}.so"


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS bundles .dylib, not the scanned .so")
def test_planted_backend_in_cwd_is_not_loaded(tmp_path):
    # Named to match ggml's scan for the "rpc" backend.
    planted = tmp_path / _lib_name("ggml-rpc-planted")
    _compile_marker(planted, tmp_path)
    _run_load_all(tmp_path, os.environ)
    assert not (tmp_path / MARKER).exists(), \
        "a ggml-named library in the working directory was loaded and its initializer ran"


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS bundles .dylib, not the scanned .so")
def test_explicit_backend_path_is_still_honored(tmp_path):
    # Positive control: proves the marker is observable when a library is loaded,
    # and that GGML_BACKEND_PATH (an explicit opt-in) keeps working. The name is
    # not a scan match, so only GGML_BACKEND_PATH can load it.
    probe = tmp_path / _lib_name("probe")
    _compile_marker(probe, tmp_path)
    _run_load_all(tmp_path, dict(os.environ, GGML_BACKEND_PATH=str(probe)))
    assert (tmp_path / MARKER).exists(), "GGML_BACKEND_PATH library was not loaded"
