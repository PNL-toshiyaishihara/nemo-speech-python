"""ctypes mirror of ``include/nemo_speech/nmt.h`` (exported by nemo_speech_nmt_c)."""

from __future__ import annotations

from ctypes import POINTER, c_char_p, c_int, c_int32, c_size_t, c_void_p

from .._loader import load_library
from . import SizedStructure, ctypes_function_for

_lib = load_library("nemo_speech_nmt_c")
ctypes_function = ctypes_function_for(_lib)

# ---- Opaque handles ----

nemo_speech_nmt_translator_p = c_void_p
nemo_speech_nmt_result_p = c_void_p

# ---- Enums ----

nemo_speech_nmt_status = c_int
NEMO_SPEECH_NMT_OK = 0
NEMO_SPEECH_NMT_ERROR_INVALID_ARGUMENT = 1
NEMO_SPEECH_NMT_ERROR_OUT_OF_MEMORY = 2
NEMO_SPEECH_NMT_ERROR_RUNTIME = 3

# ---- Startup config ----


class nemo_speech_nmt_backend_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("gpu", c_int32),  # -1 = CPU
    ]


class nemo_speech_nmt_model_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("path", c_char_p),
        ("n_ctx", c_int32),  # decode context length in tokens; 0 = default
    ]


class nemo_speech_nmt_generation_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("max_new_tokens", c_int32),  # cap per input text; 0 = default
    ]


class nemo_speech_nmt_pool_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("contexts", c_int32),  # concurrent decode contexts; 0 = default
    ]


class nemo_speech_nmt_translator_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("backend", POINTER(nemo_speech_nmt_backend_config)),
        ("model", POINTER(nemo_speech_nmt_model_config)),
        ("generation", POINTER(nemo_speech_nmt_generation_config)),
        ("pool", POINTER(nemo_speech_nmt_pool_config)),
    ]


# ---- Translator ----


@ctypes_function(
    "nemo_speech_nmt_create",
    [POINTER(nemo_speech_nmt_translator_config), POINTER(nemo_speech_nmt_translator_p)],
    nemo_speech_nmt_status,
)
def nemo_speech_nmt_create(cfg, out, /) -> int: ...


@ctypes_function("nemo_speech_nmt_destroy", [nemo_speech_nmt_translator_p], None)
def nemo_speech_nmt_destroy(translator, /) -> None: ...


@ctypes_function(
    "nemo_speech_nmt_translate",
    [
        nemo_speech_nmt_translator_p,
        POINTER(c_char_p),
        c_size_t,
        c_char_p,
        c_char_p,
        POINTER(nemo_speech_nmt_result_p),
    ],
    nemo_speech_nmt_status,
)
def nemo_speech_nmt_translate(translator, texts, n_texts, source_language, target_language, out, /) -> int:
    """Translate each text; languages are codes ("en") or a pair tag ("en-de")."""
    ...


# ---- Result accessors ----


@ctypes_function("nemo_speech_nmt_result_count", [nemo_speech_nmt_result_p], c_size_t)
def nemo_speech_nmt_result_count(result, /) -> int: ...


@ctypes_function("nemo_speech_nmt_result_text", [nemo_speech_nmt_result_p, c_size_t], c_char_p)
def nemo_speech_nmt_result_text(result, i, /) -> bytes: ...


@ctypes_function(
    "nemo_speech_nmt_result_language", [nemo_speech_nmt_result_p, c_size_t], c_char_p
)
def nemo_speech_nmt_result_language(result, i, /) -> bytes: ...


@ctypes_function("nemo_speech_nmt_result_destroy", [nemo_speech_nmt_result_p], None)
def nemo_speech_nmt_result_destroy(result, /) -> None: ...


# ---- Misc ----


@ctypes_function("nemo_speech_nmt_last_error", [], c_char_p)
def nemo_speech_nmt_last_error() -> bytes: ...


@ctypes_function("nemo_speech_nmt_version", [], c_char_p)
def nemo_speech_nmt_version() -> bytes: ...
