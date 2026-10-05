"""Non-ASCII paths and the C runtime's narrow file APIs (Windows).

These exercise the Universal CRT ``fopen`` directly, the same narrow API that
NeMo-Speech.cpp's TTS loaders call. On a system whose ANSI code page is not
UTF-8 (GitHub's Windows runners use 1252, a default Japanese Windows 932) the
first test reproduces the failure and the second shows the fix; on a UTF-8
system both pass trivially and the forced test covers the switching logic.
"""

import ctypes
import sys

import pytest

from nemo_speech import _paths
from nemo_speech.capi import utf8_file_paths

windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows C runtime behavior")


def _crt():
    crt = ctypes.CDLL("ucrtbase")
    crt.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    crt.fopen.restype = ctypes.c_void_p
    crt.fclose.argtypes = [ctypes.c_void_p]
    crt.setlocale.argtypes = [ctypes.c_int, ctypes.c_char_p]
    crt.setlocale.restype = ctypes.c_char_p
    crt._configthreadlocale.argtypes = [ctypes.c_int]
    crt._configthreadlocale.restype = ctypes.c_int
    crt.___lc_codepage_func.restype = ctypes.c_uint
    return crt


def _open_utf8(path) -> bool:
    crt = _crt()
    fp = crt.fopen(str(path).encode("utf-8"), b"rb")
    if fp:
        crt.fclose(fp)
    return bool(fp)


@pytest.fixture
def non_ascii_file(tmp_path):
    directory = tmp_path / "日本語のモデル"
    directory.mkdir()
    path = directory / "model.bin"
    path.write_bytes(b"\0")
    return path


def test_no_op_off_windows_and_reentrant():
    with utf8_file_paths():
        with utf8_file_paths():
            pass


@windows
def test_legacy_code_page_rejects_utf8_paths(non_ascii_file):
    if _paths._ansi_code_page() == _paths._CP_UTF8:
        pytest.skip("the system ANSI code page is already UTF-8")
    assert not _open_utf8(non_ascii_file)


@windows
def test_utf8_paths_open_inside_the_context(non_ascii_file):
    with utf8_file_paths():
        assert _open_utf8(non_ascii_file)


@windows
def test_context_switches_and_restores_the_thread_locale(monkeypatch):
    # Force the switch even on a UTF-8 system.
    monkeypatch.setattr(_paths, "_ansi_code_page", lambda: 932)
    crt = _crt()
    mode = crt._configthreadlocale(0)
    ctype = crt.setlocale(_paths._LC_CTYPE, None)
    with utf8_file_paths():
        # What both the UCRT path conversion and std::filesystem consult.
        assert crt.___lc_codepage_func() == _paths._CP_UTF8
    assert crt._configthreadlocale(0) == mode
    assert crt.setlocale(_paths._LC_CTYPE, None) == ctype


@windows
def test_other_threads_keep_their_locale(monkeypatch):
    import threading

    monkeypatch.setattr(_paths, "_ansi_code_page", lambda: 932)
    crt = _crt()
    outside = crt.___lc_codepage_func()
    seen = []
    inside = threading.Event()
    release = threading.Event()

    def hold():
        with utf8_file_paths():
            inside.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    assert inside.wait(5)
    seen.append(crt.___lc_codepage_func())
    release.set()
    t.join()
    assert seen == [outside]
