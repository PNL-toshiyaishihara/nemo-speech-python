"""ctypes mirror of ``include/nemo_speech/tts.h`` (exported by nemo_speech_tts)."""

from __future__ import annotations

from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int32,
    c_size_t,
    c_uint8,
    c_uint64,
    c_void_p,
)

from .._loader import load_library
from . import SizedStructure, ctypes_function_for

_lib = load_library("nemo_speech_tts")
ctypes_function = ctypes_function_for(_lib)

# ---- Opaque handles ----

nemo_speech_tts_synthesizer_p = c_void_p

# ---- Enums ----

nemo_speech_tts_status = c_int
NEMO_SPEECH_TTS_OK = 0
NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT = 1
NEMO_SPEECH_TTS_ERROR_OUT_OF_MEMORY = 2
NEMO_SPEECH_TTS_ERROR_RUNTIME = 3
NEMO_SPEECH_TTS_ERROR_CANCELLED = 4

nemo_speech_tts_backend_preference = c_int
NEMO_SPEECH_TTS_BACKEND_AUTO = 0
NEMO_SPEECH_TTS_BACKEND_CPU = 1
NEMO_SPEECH_TTS_BACKEND_CUDA = 2

nemo_speech_tts_uma_mode = c_int
NEMO_SPEECH_TTS_UMA_AUTO = 0
NEMO_SPEECH_TTS_UMA_OFF = 1
NEMO_SPEECH_TTS_UMA_ON = 2

nemo_speech_tts_longform_mode = c_int
NEMO_SPEECH_TTS_LONGFORM_AUTO = 0
NEMO_SPEECH_TTS_LONGFORM_OFF = 1
NEMO_SPEECH_TTS_LONGFORM_ON = 2

# ---- Startup config ----


class nemo_speech_tts_model_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("magpie_model", c_char_p),
        ("codec_model", c_char_p),
        # Optional for synthesize_tokens; required for synthesize_text.
        ("tokenizer_model_dir", c_char_p),
        ("text_normalizer_model_dir", c_char_p),
    ]


class nemo_speech_tts_runtime_config(SizedStructure):
    """Start from nemo_speech_tts_runtime_config_default(); all fields apply."""

    _fields_ = [
        ("size", c_size_t),
        ("speaker", c_int32),
        ("threads", c_int32),  # 0 = min(8, hardware threads)
        ("codec_threads", c_int32),  # 0 = threads
        ("seed", c_int32),
        ("steps", c_int32),
        ("top_k", c_int32),
        ("chunk_frames", c_int32),
        ("codec_queue_depth", c_int32),
        ("codec_history_frames", c_int32),
        ("codec_future_frames", c_int32),
        ("window_ms", c_int32),
        ("temperature", c_float),
        ("override_temperature", c_bool),
        ("cfg_scale", c_float),
        ("override_cfg_scale", c_bool),
        ("use_cfg", c_bool),
        ("use_local_transformer", c_bool),
        ("use_kv_cache", c_bool),
        ("use_stateful_codec", c_bool),
        ("codec_cpu", c_bool),
        ("flush_partial_chunk", c_bool),
        ("verbose", c_bool),
        ("lt_backend", nemo_speech_tts_backend_preference),
        ("sampling_backend", nemo_speech_tts_backend_preference),
        ("uma_mode", nemo_speech_tts_uma_mode),
        ("longform_mode", nemo_speech_tts_longform_mode),
        ("lt_fp32", c_bool),
    ]


class nemo_speech_tts_synthesizer_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("model", POINTER(nemo_speech_tts_model_config)),
        ("runtime", POINTER(nemo_speech_tts_runtime_config)),
        ("default_language_code", c_char_p),  # NULL/"" = en-US
        ("default_voice_name", c_char_p),  # speaker name, model.name, or index
    ]


# ---- Per-request options ----


class nemo_speech_tts_synthesis_options(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("request_id", c_char_p),
        ("language_code", c_char_p),  # NULL/"" = en-US
        ("speaker", c_int32),  # < 0 = synthesizer default
        ("seed", c_int32),  # < 0 = synthesizer default
        ("steps", c_int32),  # <= 0 = synthesizer default
        ("top_k", c_int32),  # <= 0 = synthesizer default
        ("temperature", c_float),
        ("override_temperature", c_bool),
        ("cfg_scale", c_float),
        ("override_cfg_scale", c_bool),
        ("voice_name", c_char_p),  # ignored when speaker >= 0
        ("output_sample_rate", c_int32),  # 0 = model rate; else 8000..model rate
    ]


# Little-endian signed 16-bit mono PCM; return False to cancel.
nemo_speech_tts_pcm_callback = CFUNCTYPE(c_bool, POINTER(c_uint8), c_size_t, c_void_p)


class nemo_speech_tts_synthesis_stats(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("sample_rate", c_int32),
        ("generated_frames", c_int32),
        ("chunks", c_int32),
        ("e2e_chunks", c_int32),
        ("samples_written", c_uint64),
        ("tokenizer_ms", c_double),
        ("encoder_ms", c_double),
        ("audio_s", c_double),
        ("elapsed_s", c_double),
        ("rtf", c_double),
        ("rtfx", c_double),
        ("ttfa_ms", c_double),
        ("icl_avg_ms", c_double),
        ("icl_min_ms", c_double),
        ("icl_max_ms", c_double),
        ("decoder_audio_s", c_double),
        ("decoder_elapsed_s", c_double),
        ("decoder_rtfx", c_double),
        ("decoder_ttft_ms", c_double),
        ("decoder_itl_avg_ms", c_double),
        ("decoder_itl_min_ms", c_double),
        ("decoder_itl_max_ms", c_double),
        ("decoder_itl_p95_ms", c_double),
        ("decoder_itl_p99_ms", c_double),
        ("codec_audio_s", c_double),
        ("codec_elapsed_s", c_double),
        ("codec_rtfx", c_double),
        ("codec_ttfa_ms", c_double),
        ("codec_icl_avg_ms", c_double),
        ("codec_icl_min_ms", c_double),
        ("codec_icl_max_ms", c_double),
        ("codec_icl_p95_ms", c_double),
        ("codec_icl_p99_ms", c_double),
        ("e2e_ttfa_ms", c_double),
        ("e2e_icl_avg_ms", c_double),
        ("e2e_icl_min_ms", c_double),
        ("e2e_icl_max_ms", c_double),
        ("e2e_icl_p95_ms", c_double),
        ("e2e_icl_p99_ms", c_double),
        ("e2e_rtfx", c_double),
    ]


@ctypes_function(
    "nemo_speech_tts_runtime_config_default", [], nemo_speech_tts_runtime_config
)
def nemo_speech_tts_runtime_config_default() -> nemo_speech_tts_runtime_config: ...


@ctypes_function(
    "nemo_speech_tts_synthesis_options_default", [], nemo_speech_tts_synthesis_options
)
def nemo_speech_tts_synthesis_options_default() -> nemo_speech_tts_synthesis_options: ...


@ctypes_function(
    "nemo_speech_tts_synthesis_stats_default", [], nemo_speech_tts_synthesis_stats
)
def nemo_speech_tts_synthesis_stats_default() -> nemo_speech_tts_synthesis_stats: ...


# ---- Synthesizer ----


@ctypes_function(
    "nemo_speech_tts_create",
    [POINTER(nemo_speech_tts_synthesizer_config), POINTER(nemo_speech_tts_synthesizer_p)],
    nemo_speech_tts_status,
)
def nemo_speech_tts_create(cfg, out, /) -> int: ...


@ctypes_function("nemo_speech_tts_destroy", [nemo_speech_tts_synthesizer_p], None)
def nemo_speech_tts_destroy(synthesizer, /) -> None: ...


@ctypes_function("nemo_speech_tts_sample_rate", [nemo_speech_tts_synthesizer_p], c_int32)
def nemo_speech_tts_sample_rate(synthesizer, /) -> int: ...


@ctypes_function("nemo_speech_tts_speaker_count", [nemo_speech_tts_synthesizer_p], c_int32)
def nemo_speech_tts_speaker_count(synthesizer, /) -> int: ...


@ctypes_function(
    "nemo_speech_tts_speaker_name", [nemo_speech_tts_synthesizer_p, c_size_t], c_char_p
)
def nemo_speech_tts_speaker_name(synthesizer, i, /) -> bytes: ...


@ctypes_function(
    "nemo_speech_tts_synthesize_text",
    [
        nemo_speech_tts_synthesizer_p,
        POINTER(nemo_speech_tts_synthesis_options),
        c_char_p,
        nemo_speech_tts_pcm_callback,
        c_void_p,
        POINTER(nemo_speech_tts_synthesis_stats),
    ],
    nemo_speech_tts_status,
)
def nemo_speech_tts_synthesize_text(synthesizer, options, text, callback, user_data, stats_out, /) -> int:
    """Requires a synthesizer created with model.tokenizer_model_dir."""
    ...


@ctypes_function(
    "nemo_speech_tts_synthesize_tokens",
    [
        nemo_speech_tts_synthesizer_p,
        POINTER(nemo_speech_tts_synthesis_options),
        POINTER(c_int32),
        c_size_t,
        nemo_speech_tts_pcm_callback,
        c_void_p,
        POINTER(nemo_speech_tts_synthesis_stats),
    ],
    nemo_speech_tts_status,
)
def nemo_speech_tts_synthesize_tokens(
    synthesizer, options, tokens, token_count, callback, user_data, stats_out, /
) -> int: ...


# ---- Misc ----


@ctypes_function("nemo_speech_tts_last_error", [], c_char_p)
def nemo_speech_tts_last_error() -> bytes: ...


@ctypes_function("nemo_speech_tts_version", [], c_char_p)
def nemo_speech_tts_version() -> bytes: ...
