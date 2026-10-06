"""High-level standalone speaker diarization ("who spoke when").

For speaker tags on transcript words, pass ``diarization_model_path`` to
:class:`nemo_speech.Recognizer` instead.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from ._common import NativeHandle, PathLike, as_mono_f32, fsencode_or_none, status_checker
from ._paths import utf8_file_paths
from .capi import asr as _asr
from .capi import diar as C

__all__ = [
    "DiarizationResult",
    "DiarizationStream",
    "Diarizer",
    "SegmentationConfig",
    "SpeakerSegment",
]

_check = status_checker(_asr.nemo_speech_asr_last_error)


@dataclass(frozen=True)
class SpeakerSegment:
    start: float  # seconds
    end: float  # seconds
    speaker: int  # 1-based, matching Word.speaker_tag


@dataclass(frozen=True)
class SegmentationConfig:
    """Segment postprocessing; ``None`` keeps the library default.

    The defaults are NeMo's callhome-tuned values; clean read speech usually
    wants a lower onset and larger pads.
    """

    onset: Optional[float] = None
    offset: Optional[float] = None
    pad_onset: Optional[float] = None
    pad_offset: Optional[float] = None
    min_gap: Optional[float] = None
    min_duration: Optional[float] = None

    def _struct(self) -> C.nemo_speech_diar_segmentation_config:
        # Zero keeps the library default for every field.
        return C.nemo_speech_diar_segmentation_config(
            onset=self.onset or 0.0,
            offset=self.offset or 0.0,
            pad_onset_sec=self.pad_onset or 0.0,
            pad_offset_sec=self.pad_offset or 0.0,
            min_gap_sec=self.min_gap or 0.0,
            min_duration_sec=self.min_duration or 0.0,
        )


@dataclass(frozen=True)
class DiarizationResult:
    segments: List[SpeakerSegment]
    # (frames, num_speakers) speaker probabilities, starting at frame_offset.
    frame_probs: np.ndarray
    frame_offset: int
    seconds_per_frame: float


class _Job:
    """Accessors shared by streams and finished offline jobs."""

    def __init__(self, diarizer: "Diarizer", native: NativeHandle) -> None:
        # The model must outlive the job; it also closes the job before
        # destroying itself (NativeHandle.adopt).
        self._diarizer = diarizer
        self._native = native

    @property
    def frame_count(self) -> int:
        """Labeled frames so far (see :attr:`Diarizer.seconds_per_frame`)."""
        with self._native.use() as handle:
            return C.nemo_speech_diar_frame_count(handle)

    def frame_probs(self) -> np.ndarray:
        """Retained per-frame probabilities, shape ``(frames, num_speakers)``.

        Covers frames ``[frame_probs_start, frame_count)``; long streams drop
        the oldest raw probabilities after converting them into segments.
        """
        speakers = self._diarizer._num_speakers
        with self._native.use() as handle:
            start = C.nemo_speech_diar_frame_probs_start(handle)
            frames = C.nemo_speech_diar_frame_count(handle) - start
            out = np.zeros((max(frames, 0), speakers), dtype=np.float32)
            if out.size:
                _check(
                    C.nemo_speech_diar_frame_probs(
                        handle, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size
                    )
                )
        return out

    @property
    def frame_probs_start(self) -> int:
        with self._native.use() as handle:
            return C.nemo_speech_diar_frame_probs_start(handle)

    def segments(self, config: Optional[SegmentationConfig] = None) -> List[SpeakerSegment]:
        """Speaker segments sorted by start time."""
        cfg = ctypes.byref(config._struct()) if config else None
        count = ctypes.c_size_t()
        with self._native.use() as handle:
            _check(C.nemo_speech_diar_segments(handle, cfg, None, 0, ctypes.byref(count)))
            if count.value == 0:
                return []
            buffer = (C.nemo_speech_diar_segment * count.value)()
            _check(
                C.nemo_speech_diar_segments(handle, cfg, buffer, count.value, ctypes.byref(count))
            )
        return [
            SpeakerSegment(start=s.start_time, end=s.end_time, speaker=s.speaker)
            for s in buffer[: count.value]
        ]

    def close(self) -> None:
        self._native.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DiarizationStream(_Job):
    """Live diarization; push audio, then :meth:`finish`. Single-threaded."""

    def __init__(self, diarizer: "Diarizer", native: NativeHandle) -> None:
        super().__init__(diarizer, native)
        self._sample_rate: Optional[int] = None

    def push(self, audio: "np.typing.ArrayLike", sample_rate: int = 0) -> None:
        """Buffer audio; diarization advances in whole chunks."""
        if self._sample_rate is None:
            self._sample_rate = sample_rate
        elif sample_rate != self._sample_rate:
            raise ValueError("the sample rate cannot change within a stream")
        samples = as_mono_f32(audio)
        with self._native.use() as handle:
            if samples.size:
                _check(
                    C.nemo_speech_diar_stream_push_f32(
                        handle,
                        samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        samples.size,
                        sample_rate,
                    )
                )

    def finish(self) -> None:
        """No more audio: label the remaining tail."""
        with self._native.use() as handle:
            _check(C.nemo_speech_diar_stream_finish(handle))

    def __enter__(self) -> "DiarizationStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Diarizer:
    """A loaded Sortformer diarization model.

    Offline :meth:`diarize` calls and separate streams can run on different
    threads; :meth:`close` waits for the calls running on other threads.

    Args:
        model_path: Sortformer GGUF.
        gpu: GPU device index, ``-1`` for CPU; ``None`` uses device 0.
        preset: Geometry preset ("streaming", "offline", "v3-streaming",
            "v3-offline"); ``None`` selects the model's low-latency default.
        chunk_frames, right_context_frames, left_context_frames, fifo_frames,
        spkcache_frames, update_period_frames: Geometry overrides in 80 ms
            frames applied on top of the preset; ``None`` keeps the preset.
    """

    def __init__(
        self,
        model_path: PathLike,
        *,
        gpu: Optional[int] = None,
        preset: Optional[str] = None,
        chunk_frames: Optional[int] = None,
        right_context_frames: Optional[int] = None,
        left_context_frames: Optional[int] = None,
        fifo_frames: Optional[int] = None,
        spkcache_frames: Optional[int] = None,
        update_period_frames: Optional[int] = None,
    ) -> None:
        self._native: Optional[NativeHandle] = None
        cfg = C.nemo_speech_diar_model_config(
            model_path=fsencode_or_none(model_path),
            gpu=0 if gpu is None else gpu,
            preset=preset.encode() if preset else None,
            chunk_frames=chunk_frames or 0,
            right_context_frames=right_context_frames or 0,
            # 0 is a valid explicit value here; < 0 keeps the preset.
            left_context_frames=-1 if left_context_frames is None else left_context_frames,
            fifo_frames=fifo_frames or 0,
            spkcache_frames=spkcache_frames or 0,
            update_period_frames=update_period_frames or 0,
        )
        handle = C.nemo_speech_diar_model_p()
        # Some components open their files with narrow C runtime APIs.
        with utf8_file_paths():
            status = C.nemo_speech_diar_create(ctypes.byref(cfg), ctypes.byref(handle))
        _check(status)
        self._native = NativeHandle(handle, C.nemo_speech_diar_destroy, "Diarizer")
        # Fixed by the model.
        self._num_speakers = C.nemo_speech_diar_num_speakers(handle)
        self._seconds_per_frame = C.nemo_speech_diar_seconds_per_frame(handle)

    @classmethod
    def from_pretrained(cls, name: Optional[str] = None, **kwargs: Any) -> "Diarizer":
        """Load a model from the catalog (downloading it if needed)."""
        from . import models

        return cls(models.download(name or models.default("diarization"))["diarization"], **kwargs)

    def _use(self) -> Any:
        if self._native is None:
            raise RuntimeError("Diarizer is closed")
        return self._native.use()

    @property
    def num_speakers(self) -> int:
        """Speaker capacity of the model (4 for V2, 8 for V3)."""
        with self._use():
            return self._num_speakers

    @property
    def seconds_per_frame(self) -> float:
        """Native output cadence (0.08 s for V2, 0.01 s for V3)."""
        with self._use():
            return self._seconds_per_frame

    def diarize(
        self,
        audio: "np.typing.ArrayLike",
        sample_rate: int = 0,
        *,
        segmentation: Optional[SegmentationConfig] = None,
    ) -> DiarizationResult:
        """One stateless full-attention pass over the whole recording.

        Bounded by the encoder's positional limit; use :meth:`stream` for
        long recordings.
        """
        samples = as_mono_f32(audio)
        if samples.size == 0:
            raise ValueError("audio is empty")
        with self._use() as handle:
            job_handle = C.nemo_speech_diar_stream_p()
            _check(
                C.nemo_speech_diar_offline_f32(
                    handle,
                    samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    samples.size,
                    sample_rate,
                    ctypes.byref(job_handle),
                )
            )
            native = NativeHandle(job_handle, C.nemo_speech_diar_stream_close, "diarization job")
            self._native.adopt(native)
        job = _Job(self, native)
        try:
            return DiarizationResult(
                segments=job.segments(segmentation),
                frame_probs=job.frame_probs(),
                frame_offset=job.frame_probs_start,
                seconds_per_frame=self._seconds_per_frame,
            )
        finally:
            job.close()

    def diarize_file(self, path: PathLike, **options: Any) -> DiarizationResult:
        from .audio import load_wav

        samples, rate = load_wav(path)
        return self.diarize(samples, rate, **options)

    def stream(self) -> DiarizationStream:
        with self._use() as handle:
            stream = C.nemo_speech_diar_stream_p()
            _check(C.nemo_speech_diar_stream_open(handle, ctypes.byref(stream)))
            native = NativeHandle(stream, C.nemo_speech_diar_stream_close, "DiarizationStream")
            self._native.adopt(native)
        return DiarizationStream(self, native)

    def close(self) -> None:
        """Release the model, closing any streams still open on it.

        Waits for calls running on other threads; later calls raise
        RuntimeError.
        """
        if self._native is not None:
            self._native.close()

    def __enter__(self) -> "Diarizer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
