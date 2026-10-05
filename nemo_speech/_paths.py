"""UTF-8 file paths for the C runtime on Windows.

The C ABI takes paths as UTF-8 (see ``_common.fsencode_or_none``). Most of
NeMo-Speech.cpp opens files through ``ggml_fopen``, which converts UTF-8 to
UTF-16, but some components use plain ``fopen`` / ``std::ifstream`` /
``std::filesystem::path`` from a ``char*``: the MagpieTTS model, the NanoCodec
model, the TTS tokenizer assets and the ASR profanity list. The Universal CRT
interprets those narrow paths in the ANSI code page (932 on a default Japanese
Windows) unless the thread's C locale is UTF-8, so any non-ASCII path -- for
example a model cache under a Japanese user name -- fails to open.

:func:`utf8_file_paths` switches the calling thread to a per-thread UTF-8 C
locale for the duration of a block, which makes the UCRT and the MSVC STL treat
those paths as UTF-8 as well. Other threads and the process-wide locale are not
touched. It is a no-op off Windows and when the ANSI code page already is UTF-8
(the "Beta: Use Unicode UTF-8" system setting).

8.3 short names are not a substitute: a name of up to eight bytes in the OEM
code page (such as a four-kanji user name) gets no separate short name.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
from typing import Iterator, Optional

__all__ = ["utf8_file_paths"]

_CP_UTF8 = 65001
_LC_CTYPE = 2
_QUERY_PER_THREAD_LOCALE = 0
_ENABLE_PER_THREAD_LOCALE = 1


def _ansi_code_page() -> int:
    return ctypes.WinDLL("kernel32").GetACP()


def _ucrt() -> Optional[ctypes.CDLL]:
    """The Universal CRT that Python and the NeMo-Speech.cpp libraries share."""
    try:
        crt = ctypes.CDLL("ucrtbase")
    except OSError:
        return None
    crt._configthreadlocale.argtypes = [ctypes.c_int]
    crt._configthreadlocale.restype = ctypes.c_int
    crt.setlocale.argtypes = [ctypes.c_int, ctypes.c_char_p]
    crt.setlocale.restype = ctypes.c_char_p
    return crt


@contextlib.contextmanager
def utf8_file_paths() -> Iterator[None]:
    """Let narrow C runtime file APIs accept UTF-8 paths on this thread.

    Wrap native calls that open files (the ``*_create`` functions) in it when
    using :mod:`nemo_speech.capi` directly; the high-level classes already do.
    """
    if sys.platform != "win32" or _ansi_code_page() == _CP_UTF8:
        yield
        return
    crt = _ucrt()
    if crt is None:
        yield
        return
    previous_mode = crt._configthreadlocale(_ENABLE_PER_THREAD_LOCALE)
    try:
        previous_ctype = crt.setlocale(_LC_CTYPE, None)
        # NULL on a UCRT without UTF-8 locales (before Windows 10 1803).
        switched = crt.setlocale(_LC_CTYPE, b".UTF8") is not None
        try:
            yield
        finally:
            if switched and previous_ctype is not None:
                crt.setlocale(_LC_CTYPE, previous_ctype)
    finally:
        crt._configthreadlocale(previous_mode)
