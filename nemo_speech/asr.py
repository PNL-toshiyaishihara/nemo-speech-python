"""High-level speech recognition API over :mod:`nemo_speech.capi.asr`."""

from __future__ import annotations

import ctypes
import weakref
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ._common import PathLike, as_mono_f32, decode, fsencode_or_none, status_checker
from ._paths import utf8_file_paths
from .capi import asr as C

__all__ = [
    "Alternative",
    "RecognitionResult",
    "RecognitionStream",
    "Recognizer",
    "SpeechContext",
    "Word",
    "version",
]


_check = status_checker(C.nemo_speech_asr_last_error)


def version() -> str:
    """Version string of the loaded NeMo-Speech.cpp ASR library."""
    return decode(C.nemo_speech_asr_version())


# ---- Results ----


@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int
    confidence: float
    speaker_tag: int  # 1-based speaker id; 0 = untagged

    @property
    def start(self) -> float:
        """Start time in seconds."""
        return self.start_ms / 1000.0

    @property
    def end(self) -> float:
        """End time in seconds."""
        return self.end_ms / 1000.0


@dataclass(frozen=True)
class Alternative:
    transcript: str
    confidence: float
    words: Tuple[Word, ...]
    language_codes: Tuple[str, ...]


@dataclass(frozen=True)
class RecognitionResult:
    alternatives: Tuple[Alternative, ...]
    is_final: bool
    audio_processed: float  # seconds of audio consumed so far
    channel_tag: int

    @property
    def text(self) -> str:
        """Transcript of the best alternative ("" when there is none)."""
        return self.alternatives[0].transcript if self.alternatives else ""

    @property
    def words(self) -> Tuple[Word, ...]:
        """Words of the best alternative."""
        return self.alternatives[0].words if self.alternatives else ()


def _take_result(handle: ctypes.c_void_p) -> RecognitionResult:
    """Copy a result handle into Python objects and destroy it."""
    try:
        alternatives = []
        for a in range(C.nemo_speech_asr_result_alternative_count(handle)):
            words = tuple(
                Word(
                    text=decode(C.nemo_speech_asr_result_word_text(handle, a, i)),
                    start_ms=C.nemo_speech_asr_result_word_start_time(handle, a, i),
                    end_ms=C.nemo_speech_asr_result_word_end_time(handle, a, i),
                    confidence=C.nemo_speech_asr_result_word_confidence(handle, a, i),
                    speaker_tag=C.nemo_speech_asr_result_word_speaker_tag(handle, a, i),
                )
                for i in range(C.nemo_speech_asr_result_word_count(handle, a))
            )
            languages = tuple(
                decode(C.nemo_speech_asr_result_language_code(handle, a, i))
                for i in range(C.nemo_speech_asr_result_language_count(handle, a))
            )
            alternatives.append(
                Alternative(
                    transcript=decode(C.nemo_speech_asr_result_transcript(handle, a)),
                    confidence=C.nemo_speech_asr_result_confidence(handle, a),
                    words=words,
                    language_codes=languages,
                )
            )
        return RecognitionResult(
            alternatives=tuple(alternatives),
            is_final=bool(C.nemo_speech_asr_result_is_final(handle)),
            audio_processed=C.nemo_speech_asr_result_audio_processed(handle),
            channel_tag=C.nemo_speech_asr_result_channel_tag(handle),
        )
    finally:
        C.nemo_speech_asr_result_destroy(handle)


# ---- Request options ----


@dataclass(frozen=True)
class SpeechContext:
    """Phrases to boost (word boosting / context biasing)."""

    phrases: Sequence[str]
    boost: float


class _RequestOptions:
    """Owns a recognition_options struct and every buffer it points to."""

    def __init__(
        self,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        punctuation: bool = False,
        verbatim: bool = False,
        profanity_filter: bool = False,
        interim_results: bool = False,
        eou_silence_ms: Optional[int] = None,
        speech_contexts: Sequence[SpeechContext] = (),
        max_alternatives: int = 1,
        speaker_diarization: bool = False,
        request_id: Optional[str] = None,
    ) -> None:
        o = C.nemo_speech_asr_recognition_options_default()
        o.request_id = request_id.encode() if request_id else None
        o.language_code = language.encode() if language else None
        o.interim_results = interim_results
        o.enable_word_time_offsets = word_timestamps
        o.enable_automatic_punctuation = punctuation
        o.verbatim_transcripts = verbatim
        o.profanity_filter = profanity_filter
        if eou_silence_ms is not None:
            o.stop_history_eou_ms = eou_silence_ms
        o.max_alternatives = max_alternatives
        o.enable_speaker_diarization = speaker_diarization

        self._keepalive: List[Any] = []
        if speech_contexts:
            contexts = (C.nemo_speech_asr_speech_context * len(speech_contexts))()
            for dst, src in zip(contexts, speech_contexts):
                phrases = (ctypes.c_char_p * len(src.phrases))(
                    *(p.encode("utf-8") for p in src.phrases)
                )
                self._keepalive.append(phrases)
                dst.size = ctypes.sizeof(dst)
                dst.phrases = ctypes.cast(phrases, ctypes.POINTER(ctypes.c_char_p))
                dst.phrase_count = len(src.phrases)
                dst.boost = src.boost
            self._keepalive.append(contexts)
            o.speech_contexts = ctypes.cast(
                contexts, ctypes.POINTER(C.nemo_speech_asr_speech_context)
            )
            o.speech_context_count = len(speech_contexts)
        self.struct = o

    def pointer(self) -> Any:
        return ctypes.byref(self.struct)


def _samples_pointer(samples: np.ndarray) -> Any:
    return samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ---- Recognizer ----


class Recognizer:
    """A loaded ASR model.

    One recognizer can serve concurrent :meth:`transcribe` calls and streams
    from multiple threads (ctypes releases the GIL during native calls).

    Args:
        model_path: ASR GGUF model.
        gpu: GPU device index, ``-1`` for CPU, ``None`` for the library
            default (device 0 when the build has a GPU backend).
        model_name: Optional model name; derived from the model when omitted.
        vad_model_path: Silero VAD model enabling VAD features.
        vad_masking: Mask non-speech audio using the VAD.
        endpointing: Enable end-of-utterance detection for streaming.
        endpointing_vad_based: Use the VAD instead of the model for endpoints.
        eou_silence_ms: Default end-of-utterance silence threshold.
        pnc_model_path: Punctuation-and-capitalization model.
        itn_model_dir: Inverse text normalization grammars.
        profanity_list_path: Word list for ``profanity_filter``.
        diarization_model_path: Sortformer model enabling ``speaker_diarization``.
        batching: Combine compatible work from concurrent calls into batches.
        max_batch_size: Upper bound for ``batching``; library default if None.
        streaming: Advanced: raw ``nemo_speech_asr_streaming_config``. Every
            field of it is applied, so fill all of them.
        decoder: Advanced: raw ``nemo_speech_asr_decoder_config``.
    """

    def __init__(
        self,
        model_path: PathLike,
        *,
        gpu: Optional[int] = None,
        model_name: Optional[str] = None,
        vad_model_path: Optional[PathLike] = None,
        vad_masking: bool = False,
        endpointing: bool = False,
        endpointing_vad_based: bool = False,
        eou_silence_ms: Optional[int] = None,
        pnc_model_path: Optional[PathLike] = None,
        itn_model_dir: Optional[PathLike] = None,
        profanity_list_path: Optional[PathLike] = None,
        diarization_model_path: Optional[PathLike] = None,
        batching: bool = False,
        max_batch_size: Optional[int] = None,
        streaming: Optional[C.nemo_speech_asr_streaming_config] = None,
        decoder: Optional[C.nemo_speech_asr_decoder_config] = None,
    ) -> None:
        self._handle: Optional[ctypes.c_void_p] = None
        self._streams: "weakref.WeakSet[RecognitionStream]" = weakref.WeakSet()

        cfg = C.nemo_speech_asr_recognizer_config()
        model = C.nemo_speech_asr_model_config(
            path=fsencode_or_none(model_path),
            name=model_name.encode() if model_name else None,
        )
        cfg.model = ctypes.pointer(model)
        if gpu is not None:
            cfg.backend = ctypes.pointer(C.nemo_speech_asr_backend_config(gpu=gpu))
        if vad_model_path is not None:
            cfg.vad = ctypes.pointer(
                C.nemo_speech_asr_vad_config(
                    model_path=fsencode_or_none(vad_model_path),
                    enable_masking=vad_masking,
                )
            )
        if endpointing:
            cfg.endpointing = ctypes.pointer(
                C.nemo_speech_asr_endpointing_config(
                    enable=True,
                    vad_based=endpointing_vad_based,
                    stop_history_eou_ms=eou_silence_ms or 0,  # <= 0 = default
                )
            )
        if pnc_model_path or itn_model_dir or profanity_list_path:
            cfg.postproc = ctypes.pointer(
                C.nemo_speech_asr_postproc_config(
                    profanity_list_path=fsencode_or_none(profanity_list_path),
                    itn_model_dir=fsencode_or_none(itn_model_dir),
                    pnc_model_path=fsencode_or_none(pnc_model_path),
                )
            )
        if diarization_model_path is not None:
            cfg.diar = ctypes.pointer(
                C.nemo_speech_asr_diar_config(
                    model_path=fsencode_or_none(diarization_model_path),
                    left_context_frames=-1,  # 0 is a valid value; < 0 = default
                )
            )
        if batching:
            cfg.batching = ctypes.pointer(
                C.nemo_speech_asr_batching_config(
                    enable=True,
                    max_batch_size=max_batch_size or 0,  # <= 0 = default
                    max_queue_delay_us=-1,  # 0 is a valid value; < 0 = default
                    ingress_cohort_delay_us=-1,
                )
            )
        if streaming is not None:
            cfg.streaming = ctypes.pointer(streaming)
        if decoder is not None:
            cfg.decoder = ctypes.pointer(decoder)

        handle = C.nemo_speech_asr_recognizer_p()
        # Some components open their files with narrow C runtime APIs.
        with utf8_file_paths():
            status = C.nemo_speech_asr_create(ctypes.byref(cfg), ctypes.byref(handle))
        _check(status)
        self._handle = handle

    @classmethod
    def from_pretrained(cls, name: Optional[str] = None, **kwargs: Any) -> "Recognizer":
        """Load a model from the catalog (downloading it if needed).

        ``name`` is a repository or alias from :func:`nemo_speech.models.list_models`;
        the catalog's default ASR model is used when omitted.
        """
        from . import models

        return cls(models.download(name or models.default("asr"))["asr"], **kwargs)

    # -- lifetime --

    def close(self) -> None:
        """Release the model, closing any streams still open on it."""
        for stream in list(self._streams):
            stream.close()
        if self._handle:
            C.nemo_speech_asr_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "Recognizer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_handle(self) -> ctypes.c_void_p:
        if not self._handle:
            raise RuntimeError("Recognizer is closed")
        return self._handle

    # -- recognition --

    def transcribe(
        self,
        audio: "np.typing.ArrayLike",
        sample_rate: int = 0,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        punctuation: bool = False,
        verbatim: bool = False,
        profanity_filter: bool = False,
        speech_contexts: Sequence[SpeechContext] = (),
        max_alternatives: int = 1,
        speaker_diarization: bool = False,
        request_id: Optional[str] = None,
    ) -> RecognitionResult:
        """Recognize a whole utterance.

        Args:
            audio: Mono samples (float in [-1, 1] or integer PCM).
            sample_rate: 8-96 kHz input is resampled internally; 0 means the
                samples are already at the model rate.
        """
        samples = as_mono_f32(audio)
        if samples.size == 0:
            raise ValueError("audio is empty")
        options = _RequestOptions(
            language=language,
            word_timestamps=word_timestamps,
            punctuation=punctuation,
            verbatim=verbatim,
            profanity_filter=profanity_filter,
            speech_contexts=speech_contexts,
            max_alternatives=max_alternatives,
            speaker_diarization=speaker_diarization,
            request_id=request_id,
        )
        result = C.nemo_speech_asr_result_p()
        _check(
            C.nemo_speech_asr_recognize_f32(
                self._require_handle(),
                options.pointer(),
                _samples_pointer(samples),
                samples.size,
                sample_rate,
                ctypes.byref(result),
            )
        )
        return _take_result(result)

    def transcribe_file(self, path: PathLike, **options: Any) -> RecognitionResult:
        """Recognize an integer PCM WAV file (see :func:`nemo_speech.audio.load_wav`)."""
        from .audio import load_wav

        samples, rate = load_wav(path)
        return self.transcribe(samples, rate, **options)

    def stream(
        self,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        punctuation: bool = False,
        verbatim: bool = False,
        profanity_filter: bool = False,
        interim_results: bool = True,
        eou_silence_ms: Optional[int] = None,
        speech_contexts: Sequence[SpeechContext] = (),
        speaker_diarization: bool = False,
        request_id: Optional[str] = None,
    ) -> "RecognitionStream":
        """Start a streaming recognition. Drive it from a single thread."""
        options = _RequestOptions(
            language=language,
            word_timestamps=word_timestamps,
            punctuation=punctuation,
            verbatim=verbatim,
            profanity_filter=profanity_filter,
            interim_results=interim_results,
            eou_silence_ms=eou_silence_ms,
            speech_contexts=speech_contexts,
            speaker_diarization=speaker_diarization,
            request_id=request_id,
        )
        handle = C.nemo_speech_asr_stream_p()
        _check(
            C.nemo_speech_asr_streaming_recognize(
                self._require_handle(), options.pointer(), ctypes.byref(handle)
            )
        )
        stream = RecognitionStream(self, handle)
        self._streams.add(stream)
        return stream


class RecognitionStream:
    """A streaming recognition created by :meth:`Recognizer.stream`.

    :meth:`push` buffers audio and returns every result that became available;
    :meth:`finish` flushes the tail and returns the remaining results,
    including the end-of-stream final.
    """

    def __init__(self, recognizer: Recognizer, handle: ctypes.c_void_p) -> None:
        # The recognizer must outlive the stream.
        self._recognizer = recognizer
        self._handle: Optional[ctypes.c_void_p] = handle
        self._sample_rate: Optional[int] = None

    def _require_handle(self) -> ctypes.c_void_p:
        if not self._handle:
            raise RuntimeError("RecognitionStream is closed")
        return self._handle

    def _drain(self) -> List[RecognitionResult]:
        results = []
        handle = self._require_handle()
        while True:
            result = C.nemo_speech_asr_result_p()
            _check(C.nemo_speech_asr_stream_next(handle, ctypes.byref(result)))
            if not result:
                return results
            results.append(_take_result(result))

    def push(self, audio: "np.typing.ArrayLike", sample_rate: int = 0) -> List[RecognitionResult]:
        """Add audio and return the results it produced (possibly none).

        The sample rate must not change within a stream.
        """
        if self._sample_rate is None:
            self._sample_rate = sample_rate
        elif sample_rate != self._sample_rate:
            raise ValueError("the sample rate cannot change within a stream")
        samples = as_mono_f32(audio)
        if samples.size:
            _check(
                C.nemo_speech_asr_stream_push_f32(
                    self._require_handle(), _samples_pointer(samples), samples.size, sample_rate
                )
            )
        return self._drain()

    def force_endpoint(self) -> List[RecognitionResult]:
        """End the current utterance now and return the resulting results."""
        _check(C.nemo_speech_asr_stream_force_endpoint(self._require_handle()))
        return self._drain()

    def finish(self) -> List[RecognitionResult]:
        """Signal end of audio and return the remaining results."""
        _check(C.nemo_speech_asr_stream_finish(self._require_handle()))
        return self._drain()

    def close(self) -> None:
        if self._handle:
            C.nemo_speech_asr_stream_close(self._handle)
            self._handle = None

    def __enter__(self) -> "RecognitionStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
