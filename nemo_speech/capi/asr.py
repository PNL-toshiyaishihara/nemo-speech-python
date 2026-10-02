"""ctypes mirror of ``include/nemo_speech/asr.h`` (exported by nemo_speech_asr_c).

Keep this file in declaration order with the header; tests/test_capi_abi.py
checks that every struct field and exported function is present.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int32,
    c_size_t,
    c_void_p,
)

from .._loader import load_library
from . import SizedStructure, ctypes_function_for

_lib = load_library("nemo_speech_asr_c")
ctypes_function = ctypes_function_for(_lib)

# ---- Opaque handles ----

nemo_speech_asr_recognizer_p = c_void_p
nemo_speech_asr_stream_p = c_void_p
nemo_speech_asr_result_p = c_void_p

# ---- Enums ----

nemo_speech_asr_status = c_int
NEMO_SPEECH_ASR_OK = 0
NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT = 1
NEMO_SPEECH_ASR_ERROR_OUT_OF_MEMORY = 2
NEMO_SPEECH_ASR_ERROR_RUNTIME = 3
NEMO_SPEECH_ASR_ERROR_CANCELLED = 4

nemo_speech_asr_decoder_kind = c_int
NEMO_SPEECH_ASR_DECODER_GREEDY = 0
NEMO_SPEECH_ASR_DECODER_FLASHLIGHT = 1

# ---- Startup config ----


class nemo_speech_asr_backend_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("gpu", c_int32),  # -1 = CPU
    ]


class nemo_speech_asr_model_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("path", c_char_p),
        ("name", c_char_p),  # optional; NULL/"" = derived from model
    ]


class nemo_speech_asr_streaming_config(SizedStructure):
    """All fields are applied when this struct is passed (no zero = default)."""

    _fields_ = [
        ("size", c_size_t),
        ("chunk_size", c_float),  # s (CTC buffered window)
        ("ctc_left_padding", c_float),  # s
        ("ctc_right_padding", c_float),  # s
        ("rnnt_right_context", c_int32),  # encoder frames; -1 = model default
    ]


class nemo_speech_asr_decoder_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("kind", nemo_speech_asr_decoder_kind),
        ("flashlight_lm", c_char_p),
        ("flashlight_lexicon", c_char_p),
        ("flashlight_tokenizer", c_char_p),
        ("beam_size", c_int32),
        ("beam_size_token", c_int32),
        ("beam_threshold", c_double),
        ("lm_weight", c_double),
        ("word_insertion_score", c_double),
        ("max_boost", c_double),  # 0 keeps the default (10.0)
    ]


class nemo_speech_asr_vad_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("model_path", c_char_p),  # empty/NULL = no VAD
        ("enable_masking", c_bool),
        ("onset", c_float),
        ("offset", c_float),
    ]


class nemo_speech_asr_endpointing_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("enable", c_bool),
        ("vad_based", c_bool),
        ("stop_history_eou_ms", c_int32),
    ]


class nemo_speech_asr_postproc_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("profanity_list_path", c_char_p),
        ("itn_model_dir", c_char_p),
        ("pnc_model_path", c_char_p),
    ]


class nemo_speech_asr_diar_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("model_path", c_char_p),  # empty/NULL = diarization not available
        ("chunk_frames", c_int32),
        ("right_context_frames", c_int32),
        ("left_context_frames", c_int32),  # 0 is valid; < 0 = default
        ("fifo_frames", c_int32),
        ("spkcache_frames", c_int32),
        ("update_period_frames", c_int32),
    ]


class nemo_speech_asr_batching_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("enable", c_bool),
        ("max_batch_size", c_int32),
        ("max_queue_delay_us", c_int32),  # 0 is valid; < 0 = default
        ("max_queue_depth", c_int32),
        ("ingress_cohort_delay_us", c_int32),  # 0 is valid; < 0 = default
        ("state_arena_slots", c_int32),
    ]


class nemo_speech_asr_recognizer_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("backend", POINTER(nemo_speech_asr_backend_config)),
        ("model", POINTER(nemo_speech_asr_model_config)),
        ("streaming", POINTER(nemo_speech_asr_streaming_config)),
        ("decoder", POINTER(nemo_speech_asr_decoder_config)),
        ("vad", POINTER(nemo_speech_asr_vad_config)),
        ("endpointing", POINTER(nemo_speech_asr_endpointing_config)),
        ("postproc", POINTER(nemo_speech_asr_postproc_config)),
        ("diar", POINTER(nemo_speech_asr_diar_config)),
        ("batching", POINTER(nemo_speech_asr_batching_config)),
    ]


# ---- Per-request options ----


class nemo_speech_asr_speech_context(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("phrases", POINTER(c_char_p)),
        ("phrase_count", c_size_t),
        ("boost", c_float),
    ]


class nemo_speech_asr_recognition_options(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("request_id", c_char_p),
        ("language_code", c_char_p),  # NULL/"" = auto/default
        ("interim_results", c_bool),
        ("enable_word_time_offsets", c_bool),
        ("enable_automatic_punctuation", c_bool),
        ("verbatim_transcripts", c_bool),
        ("profanity_filter", c_bool),
        ("stop_history_eou_ms", c_int32),  # <= 0 = server default
        ("speech_contexts", POINTER(nemo_speech_asr_speech_context)),
        ("speech_context_count", c_size_t),
        ("max_alternatives", c_int32),  # <= 1 = 1-best
        ("enable_speaker_diarization", c_bool),
        ("max_speaker_count", c_int32),
    ]


@ctypes_function(
    "nemo_speech_asr_recognition_options_default", [], nemo_speech_asr_recognition_options
)
def nemo_speech_asr_recognition_options_default() -> nemo_speech_asr_recognition_options:
    """Options with ``size`` set and the library defaults filled in."""
    ...


# ---- Recognizer ----


@ctypes_function(
    "nemo_speech_asr_create",
    [POINTER(nemo_speech_asr_recognizer_config), POINTER(nemo_speech_asr_recognizer_p)],
    nemo_speech_asr_status,
)
def nemo_speech_asr_create(cfg, out, /) -> int:
    """Create a recognizer; the config is copied."""
    ...


@ctypes_function("nemo_speech_asr_destroy", [nemo_speech_asr_recognizer_p], None)
def nemo_speech_asr_destroy(recognizer, /) -> None: ...


@ctypes_function(
    "nemo_speech_asr_recognize_f32",
    [
        nemo_speech_asr_recognizer_p,
        POINTER(nemo_speech_asr_recognition_options),
        POINTER(c_float),
        c_size_t,
        c_int32,
        POINTER(nemo_speech_asr_result_p),
    ],
    nemo_speech_asr_status,
)
def nemo_speech_asr_recognize_f32(recognizer, options, samples, n_samples, sample_rate, out, /) -> int:
    """Offline recognition of mono float32 audio (8-96 kHz; 0 = model rate)."""
    ...


# ---- Streaming ----


@ctypes_function(
    "nemo_speech_asr_streaming_recognize",
    [
        nemo_speech_asr_recognizer_p,
        POINTER(nemo_speech_asr_recognition_options),
        POINTER(nemo_speech_asr_stream_p),
    ],
    nemo_speech_asr_status,
)
def nemo_speech_asr_streaming_recognize(recognizer, options, out, /) -> int: ...


@ctypes_function(
    "nemo_speech_asr_stream_push_f32",
    [nemo_speech_asr_stream_p, POINTER(c_float), c_size_t, c_int32],
    nemo_speech_asr_status,
)
def nemo_speech_asr_stream_push_f32(stream, samples, n_samples, sample_rate, /) -> int:
    """Buffer audio without decoding; nemo_speech_asr_stream_next drives decoding."""
    ...


@ctypes_function(
    "nemo_speech_asr_stream_force_endpoint", [nemo_speech_asr_stream_p], nemo_speech_asr_status
)
def nemo_speech_asr_stream_force_endpoint(stream, /) -> int: ...


@ctypes_function(
    "nemo_speech_asr_stream_finish", [nemo_speech_asr_stream_p], nemo_speech_asr_status
)
def nemo_speech_asr_stream_finish(stream, /) -> int: ...


@ctypes_function(
    "nemo_speech_asr_stream_next",
    [nemo_speech_asr_stream_p, POINTER(nemo_speech_asr_result_p)],
    nemo_speech_asr_status,
)
def nemo_speech_asr_stream_next(stream, out, /) -> int:
    """Pull one result; ``*out`` is NULL when more audio is needed."""
    ...


@ctypes_function("nemo_speech_asr_stream_close", [nemo_speech_asr_stream_p], None)
def nemo_speech_asr_stream_close(stream, /) -> None: ...


# ---- Result accessors (memory owned by the result; valid until destroy) ----


@ctypes_function("nemo_speech_asr_result_is_final", [nemo_speech_asr_result_p], c_bool)
def nemo_speech_asr_result_is_final(result, /) -> bool: ...


@ctypes_function("nemo_speech_asr_result_audio_processed", [nemo_speech_asr_result_p], c_float)
def nemo_speech_asr_result_audio_processed(result, /) -> float: ...


@ctypes_function("nemo_speech_asr_result_channel_tag", [nemo_speech_asr_result_p], c_int32)
def nemo_speech_asr_result_channel_tag(result, /) -> int: ...


@ctypes_function(
    "nemo_speech_asr_result_alternative_count", [nemo_speech_asr_result_p], c_size_t
)
def nemo_speech_asr_result_alternative_count(result, /) -> int: ...


@ctypes_function(
    "nemo_speech_asr_result_transcript", [nemo_speech_asr_result_p, c_size_t], c_char_p
)
def nemo_speech_asr_result_transcript(result, alt, /) -> bytes: ...


@ctypes_function(
    "nemo_speech_asr_result_confidence", [nemo_speech_asr_result_p, c_size_t], c_float
)
def nemo_speech_asr_result_confidence(result, alt, /) -> float: ...


@ctypes_function(
    "nemo_speech_asr_result_word_count", [nemo_speech_asr_result_p, c_size_t], c_size_t
)
def nemo_speech_asr_result_word_count(result, alt, /) -> int: ...


@ctypes_function(
    "nemo_speech_asr_result_word_text",
    [nemo_speech_asr_result_p, c_size_t, c_size_t],
    c_char_p,
)
def nemo_speech_asr_result_word_text(result, alt, i, /) -> bytes: ...


@ctypes_function(
    "nemo_speech_asr_result_word_start_time",
    [nemo_speech_asr_result_p, c_size_t, c_size_t],
    c_int32,
)
def nemo_speech_asr_result_word_start_time(result, alt, i, /) -> int:
    """Word start in milliseconds."""
    ...


@ctypes_function(
    "nemo_speech_asr_result_word_end_time",
    [nemo_speech_asr_result_p, c_size_t, c_size_t],
    c_int32,
)
def nemo_speech_asr_result_word_end_time(result, alt, i, /) -> int:
    """Word end in milliseconds."""
    ...


@ctypes_function(
    "nemo_speech_asr_result_word_confidence",
    [nemo_speech_asr_result_p, c_size_t, c_size_t],
    c_float,
)
def nemo_speech_asr_result_word_confidence(result, alt, i, /) -> float: ...


@ctypes_function(
    "nemo_speech_asr_result_word_speaker_tag",
    [nemo_speech_asr_result_p, c_size_t, c_size_t],
    c_int32,
)
def nemo_speech_asr_result_word_speaker_tag(result, alt, i, /) -> int:
    """1-based speaker id when diarization was requested; 0 = untagged."""
    ...


@ctypes_function(
    "nemo_speech_asr_result_language_count", [nemo_speech_asr_result_p, c_size_t], c_size_t
)
def nemo_speech_asr_result_language_count(result, alt, /) -> int: ...


@ctypes_function(
    "nemo_speech_asr_result_language_code",
    [nemo_speech_asr_result_p, c_size_t, c_size_t],
    c_char_p,
)
def nemo_speech_asr_result_language_code(result, alt, i, /) -> bytes: ...


@ctypes_function("nemo_speech_asr_result_destroy", [nemo_speech_asr_result_p], None)
def nemo_speech_asr_result_destroy(result, /) -> None: ...


# ---- Misc ----


@ctypes_function("nemo_speech_asr_last_error", [], c_char_p)
def nemo_speech_asr_last_error() -> bytes:
    """Thread-local message of the most recent failed call on this thread."""
    ...


@ctypes_function("nemo_speech_asr_version", [], c_char_p)
def nemo_speech_asr_version() -> bytes: ...
