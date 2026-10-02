"""ctypes mirror of ``include/nemo_speech/diar.h``.

Exported by the ASR library (nemo_speech_asr_c); status codes and
``nemo_speech_asr_last_error`` are shared with :mod:`nemo_speech.capi.asr`.
"""

from __future__ import annotations

from ctypes import POINTER, Structure, c_char_p, c_double, c_float, c_int32, c_int64, c_size_t, c_void_p

from . import SizedStructure
from .asr import ctypes_function, nemo_speech_asr_status

# ---- Opaque handles ----

nemo_speech_diar_model_p = c_void_p
nemo_speech_diar_stream_p = c_void_p

# ---- Structs ----


class nemo_speech_diar_model_config(SizedStructure):
    _fields_ = [
        ("size", c_size_t),
        ("model_path", c_char_p),  # Sortformer GGUF (required)
        ("gpu", c_int32),  # -1 = CPU
        ("preset", c_char_p),  # NULL/"" = model-specific low-latency default
        # Geometry overrides in 80 ms frames: <= 0 keeps the preset value,
        # except left context where < 0 keeps it and 0 is a valid value.
        ("chunk_frames", c_int32),
        ("right_context_frames", c_int32),
        ("left_context_frames", c_int32),
        ("fifo_frames", c_int32),
        ("spkcache_frames", c_int32),
        ("update_period_frames", c_int32),
    ]


class nemo_speech_diar_segmentation_config(SizedStructure):
    """Zero/negative fields keep the library defaults."""

    _fields_ = [
        ("size", c_size_t),
        ("onset", c_float),
        ("offset", c_float),
        ("pad_onset_sec", c_double),
        ("pad_offset_sec", c_double),
        ("min_gap_sec", c_double),
        ("min_duration_sec", c_double),
    ]


class nemo_speech_diar_segment(Structure):
    _fields_ = [
        ("start_time", c_double),  # seconds
        ("end_time", c_double),  # seconds
        ("speaker", c_int32),  # 1-based
    ]


# ---- Model ----


@ctypes_function(
    "nemo_speech_diar_create",
    [POINTER(nemo_speech_diar_model_config), POINTER(nemo_speech_diar_model_p)],
    nemo_speech_asr_status,
)
def nemo_speech_diar_create(cfg, out, /) -> int: ...


@ctypes_function("nemo_speech_diar_destroy", [nemo_speech_diar_model_p], None)
def nemo_speech_diar_destroy(model, /) -> None: ...


@ctypes_function("nemo_speech_diar_num_speakers", [nemo_speech_diar_model_p], c_int32)
def nemo_speech_diar_num_speakers(model, /) -> int: ...


@ctypes_function("nemo_speech_diar_seconds_per_frame", [nemo_speech_diar_model_p], c_double)
def nemo_speech_diar_seconds_per_frame(model, /) -> float: ...


# ---- Streaming ----


@ctypes_function(
    "nemo_speech_diar_stream_open",
    [nemo_speech_diar_model_p, POINTER(nemo_speech_diar_stream_p)],
    nemo_speech_asr_status,
)
def nemo_speech_diar_stream_open(model, out, /) -> int: ...


@ctypes_function(
    "nemo_speech_diar_stream_push_f32",
    [nemo_speech_diar_stream_p, POINTER(c_float), c_size_t, c_int32],
    nemo_speech_asr_status,
)
def nemo_speech_diar_stream_push_f32(stream, samples, n_samples, sample_rate, /) -> int: ...


@ctypes_function(
    "nemo_speech_diar_stream_finish", [nemo_speech_diar_stream_p], nemo_speech_asr_status
)
def nemo_speech_diar_stream_finish(stream, /) -> int: ...


@ctypes_function("nemo_speech_diar_stream_close", [nemo_speech_diar_stream_p], None)
def nemo_speech_diar_stream_close(stream, /) -> None: ...


# ---- Offline (stateless) ----


@ctypes_function(
    "nemo_speech_diar_offline_f32",
    [nemo_speech_diar_model_p, POINTER(c_float), c_size_t, c_int32, POINTER(nemo_speech_diar_stream_p)],
    nemo_speech_asr_status,
)
def nemo_speech_diar_offline_f32(model, samples, n_samples, sample_rate, out, /) -> int:
    """Full-attention pass; *out is a finished job (close with stream_close)."""
    ...


# ---- Results ----


@ctypes_function("nemo_speech_diar_frame_count", [nemo_speech_diar_stream_p], c_int64)
def nemo_speech_diar_frame_count(stream, /) -> int: ...


@ctypes_function("nemo_speech_diar_frame_probs_start", [nemo_speech_diar_stream_p], c_int64)
def nemo_speech_diar_frame_probs_start(stream, /) -> int: ...


@ctypes_function(
    "nemo_speech_diar_frame_probs",
    [nemo_speech_diar_stream_p, POINTER(c_float), c_size_t],
    nemo_speech_asr_status,
)
def nemo_speech_diar_frame_probs(stream, out, capacity_floats, /) -> int: ...


@ctypes_function(
    "nemo_speech_diar_segments",
    [
        nemo_speech_diar_stream_p,
        POINTER(nemo_speech_diar_segmentation_config),
        POINTER(nemo_speech_diar_segment),
        c_size_t,
        POINTER(c_size_t),
    ],
    nemo_speech_asr_status,
)
def nemo_speech_diar_segments(stream, cfg, out, capacity, count, /) -> int:
    """Two-call pattern: out=NULL returns the count, then fill a buffer."""
    ...
