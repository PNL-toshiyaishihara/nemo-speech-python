"""Helpers shared by the high-level wrappers."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any, Callable, Iterator, Optional, Set, Union

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


class NativeHandle:
    """A native object whose destruction waits for the calls using it.

    Wrap every native call in :meth:`use`. :meth:`close` refuses new calls,
    waits for running ones (which may be on other threads, with the GIL
    released), closes the dependents registered with :meth:`adopt` (streams
    opened on the object), and only then destroys the object. Dependents are
    held here rather than by their Python wrappers, so even when the garbage
    collector finalizes a model and its streams in an arbitrary order, every
    stream is closed before its model.
    """

    def __init__(self, handle: Any, destroy: Callable[[Any], None], name: str) -> None:
        self._handle = handle
        self._destroy = destroy
        self._name = name
        self._cond = threading.Condition()
        self._calls = 0
        self._closed = False
        self._dependents: Set["NativeHandle"] = set()
        self._parent: Optional["NativeHandle"] = None

    @contextlib.contextmanager
    def use(self) -> Iterator[Any]:
        """Yield the raw handle; :meth:`close` waits until the block exits."""
        with self._cond:
            if self._closed:
                raise RuntimeError(f"{self._name} is closed")
            self._calls += 1
        try:
            yield self._handle
        finally:
            with self._cond:
                self._calls -= 1
                if not self._calls:
                    self._cond.notify_all()

    def adopt(self, dependent: "NativeHandle") -> None:
        """Close ``dependent`` before this object is destroyed.

        Call it inside :meth:`use`, so that this object cannot be closed
        between creating the dependent and registering it.
        """
        with self._cond:
            self._dependents.add(dependent)
        dependent._parent = self

    def close(self) -> None:
        """Destroy the object once no call uses it; later calls raise RuntimeError."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            while self._calls:
                self._cond.wait()
            dependents = list(self._dependents)
            self._dependents.clear()
        for dependent in dependents:
            dependent.close()
        self._destroy(self._handle)
        self._handle = None
        parent, self._parent = self._parent, None
        if parent is not None:
            with parent._cond:
                parent._dependents.discard(self)


def fsencode_or_none(path: Optional[PathLike]) -> Optional[bytes]:
    """Encode a path for the C ABI (UTF-8 on Windows, as ggml_fopen expects).

    Native calls that open such paths must run inside
    :func:`nemo_speech._paths.utf8_file_paths`; see there.
    """
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
