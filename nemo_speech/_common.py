"""Helpers shared by the high-level wrappers."""

from __future__ import annotations

import os
from typing import Callable, Optional, Union

import numpy as np

PathLike = Union[str, "os.PathLike[str]"]

# Shared by every C ABI family (asr/diar, nmt, tts).
STATUS_OK = 0


class NemoSpeechError(RuntimeError):
    """A NeMo-Speech.cpp call returned a non-OK status."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"{message} (status {status})" if message else f"status {status}")


def status_checker(last_error: Callable[[], Optional[bytes]]) -> Callable[[int], None]:
    """Return a function raising NemoSpeechError for a non-OK status.

    ``last_error`` is the family's thread-local ``*_last_error`` accessor; it
    must be read on the same thread right after the failing call.
    """

    def check(status: int) -> None:
        if status != STATUS_OK:
            raise NemoSpeechError(status, decode(last_error()))

    return check


def fsencode_or_none(path: Optional[PathLike]) -> Optional[bytes]:
    """Encode a path for the C ABI (UTF-8 on Windows, as ggml_fopen expects)."""
    if path is None:
        return None
    return os.fsencode(os.fspath(path))


def decode(raw: Optional[bytes]) -> str:
    return raw.decode("utf-8", "replace") if raw else ""


def pcm_to_f32(a: np.ndarray) -> np.ndarray:
    """Convert PCM samples of any shape to float32, scaling integers to [-1, 1]."""
    if a.dtype.kind == "f":
        return a.astype(np.float32, copy=False)
    if a.dtype.kind == "i":
        return a.astype(np.float32) / float(-np.iinfo(a.dtype).min)
    if a.dtype.kind == "u":
        half = float(np.iinfo(a.dtype).max + 1) / 2.0
        return (a.astype(np.float32) - half) / half
    raise TypeError(f"unsupported audio dtype {a.dtype}")


def as_mono_f32(audio: "np.typing.ArrayLike") -> np.ndarray:
    """Return ``audio`` as a contiguous 1-D float32 array.

    Integer PCM is scaled by its full-scale value. Multi-channel input is
    rejected rather than silently downmixed.
    """
    a = np.asarray(audio)
    if a.ndim == 2 and 1 in a.shape:
        a = a.reshape(-1)
    if a.ndim != 1:
        raise ValueError(
            f"audio must be mono (1-D), got shape {a.shape}; "
            "downmix first, e.g. audio.mean(axis=1)"
        )
    return np.ascontiguousarray(pcm_to_f32(a), dtype=np.float32)
