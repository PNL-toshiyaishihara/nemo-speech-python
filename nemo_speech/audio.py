"""Minimal WAV input without third-party dependencies.

Reads integer PCM WAV (8/16/24/32-bit) through the standard library. For
float WAV, FLAC, MP3 and so on, use a library such as ``soundfile`` and pass
the samples to the recognizer directly.
"""

from __future__ import annotations

import wave
from typing import Tuple

import numpy as np

from ._common import PathLike, pcm_to_f32


def _decode_frames(raw: bytes, width: int) -> np.ndarray:
    if width == 1:
        return pcm_to_f32(np.frombuffer(raw, dtype=np.uint8))
    if width == 2:
        return pcm_to_f32(np.frombuffer(raw, dtype="<i2"))
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        return v.astype(np.float32) / float(0x800000)
    if width == 4:
        return pcm_to_f32(np.frombuffer(raw, dtype="<i4"))
    raise ValueError(f"unsupported WAV sample width: {width} bytes")


def load_wav(path: PathLike) -> Tuple[np.ndarray, int]:
    """Read ``path`` as ``(mono float32 samples, sample_rate)``.

    Multi-channel files are downmixed by averaging the channels.
    """
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    samples = _decode_frames(raw, width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), rate


def save_wav(path: PathLike, samples: "np.typing.ArrayLike", sample_rate: int) -> None:
    """Write mono samples as 16-bit PCM WAV (floats are clipped to [-1, 1])."""
    a = np.asarray(samples)
    if a.dtype != np.int16:
        a = np.round(np.clip(pcm_to_f32(a), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(a.astype("<i2").tobytes())
