"""High-level text-to-speech (MagpieTTS + NanoCodec)."""

from __future__ import annotations

import ctypes
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from ._common import PathLike, decode, fsencode_or_none, status_checker
from ._paths import utf8_file_paths
from .capi import tts as C

__all__ = ["SynthesisResult", "Synthesizer", "version"]

_check = status_checker(C.nemo_speech_tts_last_error)

# Called with each PCM16 chunk; returning False cancels synthesis.
AudioCallback = Callable[[np.ndarray], Optional[bool]]


def version() -> str:
    """Version string of the loaded NeMo-Speech.cpp TTS library."""
    return decode(C.nemo_speech_tts_version())


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray  # int16 mono PCM
    sample_rate: int
    stats: Dict[str, float] = field(repr=False)
    cancelled: bool = False

    @property
    def duration(self) -> float:
        return self.audio.size / self.sample_rate if self.sample_rate else 0.0

    def to_float(self) -> np.ndarray:
        """Samples as float32 in [-1, 1)."""
        return self.audio.astype(np.float32) / 32768.0

    def save(self, path: PathLike) -> None:
        """Write a 16-bit PCM WAV file."""
        from .audio import save_wav

        save_wav(path, self.audio, self.sample_rate)


def _stats_dict(stats: C.nemo_speech_tts_synthesis_stats) -> Dict[str, float]:
    return {name: getattr(stats, name) for name, _ in stats._fields_ if name != "size"}


class Synthesizer:
    """A loaded MagpieTTS model with its NanoCodec decoder.

    Calls on one synthesizer are serialized; create several synthesizers for
    parallel synthesis.

    Args:
        model_path: MagpieTTS GGUF.
        codec_path: NanoCodec decoder GGUF.
        tokenizer_dir: Extracted tokenizer assets; required for text input
            (see :mod:`nemo_speech.models`).
        text_normalizer_dir: Optional written-to-spoken grammar root.
        language: Default language code (``None`` = en-US).
        voice: Default speaker name or numeric index.
        threads: CPU threads (``None`` = min(8, hardware threads)).
        seed: Default sampling seed.
        runtime: Advanced: raw ``nemo_speech_tts_runtime_config``; start from
            ``capi.tts.nemo_speech_tts_runtime_config_default()``.
    """

    def __init__(
        self,
        model_path: PathLike,
        codec_path: PathLike,
        *,
        tokenizer_dir: Optional[PathLike] = None,
        text_normalizer_dir: Optional[PathLike] = None,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        threads: Optional[int] = None,
        seed: Optional[int] = None,
        runtime: Optional[C.nemo_speech_tts_runtime_config] = None,
    ) -> None:
        self._handle: Optional[ctypes.c_void_p] = None
        self._lock = threading.Lock()

        model = C.nemo_speech_tts_model_config(
            magpie_model=fsencode_or_none(model_path),
            codec_model=fsencode_or_none(codec_path),
            tokenizer_model_dir=fsencode_or_none(tokenizer_dir),
            text_normalizer_model_dir=fsencode_or_none(text_normalizer_dir),
        )
        if runtime is None:
            runtime = C.nemo_speech_tts_runtime_config_default()
        if threads is not None:
            runtime.threads = threads
        if seed is not None:
            runtime.seed = seed
        cfg = C.nemo_speech_tts_synthesizer_config(
            model=ctypes.pointer(model),
            runtime=ctypes.pointer(runtime),
            default_language_code=language.encode() if language else None,
            default_voice_name=voice.encode() if voice else None,
        )
        handle = C.nemo_speech_tts_synthesizer_p()
        # Some components open their files with narrow C runtime APIs.
        with utf8_file_paths():
            status = C.nemo_speech_tts_create(ctypes.byref(cfg), ctypes.byref(handle))
        _check(status)
        self._handle = handle
        # Fixed by the model; read once so the properties never wait for a
        # synthesis holding the lock (e.g. while iterating stream()).
        self._sample_rate = C.nemo_speech_tts_sample_rate(handle)
        self._speakers = [
            decode(C.nemo_speech_tts_speaker_name(handle, i))
            for i in range(C.nemo_speech_tts_speaker_count(handle))
        ]

    @classmethod
    def from_pretrained(cls, name: Optional[str] = None, **kwargs: Any) -> "Synthesizer":
        """Load a model and its codec/tokenizer from the catalog (downloading if needed)."""
        from . import models

        paths = models.download(name or models.default("tts"), companions=True)
        kwargs.setdefault("tokenizer_dir", paths.get("tokenizer"))
        return cls(paths["tts"], paths["codec"], **kwargs)

    def _require_handle(self) -> ctypes.c_void_p:
        if not self._handle:
            raise RuntimeError("Synthesizer is closed")
        return self._handle

    @property
    def sample_rate(self) -> int:
        """Native output rate of the codec."""
        self._require_handle()
        return self._sample_rate

    @property
    def speakers(self) -> List[str]:
        """Speaker names, in index order."""
        self._require_handle()
        return list(self._speakers)

    # -- synthesis --

    def _options(
        self,
        language: Optional[str],
        voice: Optional[str],
        speaker: Optional[int],
        seed: Optional[int],
        steps: Optional[int],
        top_k: Optional[int],
        temperature: Optional[float],
        cfg_scale: Optional[float],
        sample_rate: Optional[int],
        request_id: Optional[str],
    ) -> C.nemo_speech_tts_synthesis_options:
        o = C.nemo_speech_tts_synthesis_options_default()
        o.request_id = request_id.encode() if request_id else None
        o.language_code = language.encode() if language else None
        o.voice_name = voice.encode() if voice else None
        if speaker is not None:
            o.speaker = speaker
        if seed is not None:
            o.seed = seed
        if steps is not None:
            o.steps = steps
        if top_k is not None:
            o.top_k = top_k
        if temperature is not None:
            o.temperature = temperature
            o.override_temperature = True
        if cfg_scale is not None:
            o.cfg_scale = cfg_scale
            o.override_cfg_scale = True
        if sample_rate is not None:
            o.output_sample_rate = sample_rate
        return o

    def _run(
        self,
        invoke: Callable[[Any, Any, Any], int],
        options: C.nemo_speech_tts_synthesis_options,
        on_audio: Optional[AudioCallback],
    ) -> SynthesisResult:
        """Call ``invoke(handle, callback, stats)`` under the lock."""
        chunks: List[np.ndarray] = []
        errors: List[BaseException] = []
        cancelled = False

        def callback(pcm: Any, n_bytes: int, _user: Any) -> bool:
            nonlocal cancelled
            try:
                chunk = np.frombuffer(ctypes.string_at(pcm, n_bytes), dtype="<i2")
                chunks.append(chunk)
                if on_audio is not None and on_audio(chunk) is False:
                    cancelled = True
                    return False
                return True
            except BaseException as e:  # surfaced after the native call returns
                errors.append(e)
                return False

        c_callback = C.nemo_speech_tts_pcm_callback(callback)
        stats = C.nemo_speech_tts_synthesis_stats_default()
        # The handle is read under the lock: close() takes it too, so it cannot
        # destroy the synthesizer between the check and the call.
        with self._lock:
            status = invoke(self._require_handle(), c_callback, ctypes.byref(stats))
        # A cancelled synthesis returns before filling the stats, but the
        # chunks it delivered were already resampled to the requested rate.
        rate = stats.sample_rate or options.output_sample_rate or self._sample_rate
        if errors:
            raise errors[0]
        # Upstream's streaming codec path reports a callback cancellation as a
        # RUNTIME failure rather than CANCELLED, so any non-OK status after a
        # cancellation we requested counts as that cancellation.
        if not cancelled:
            _check(status)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return SynthesisResult(audio.astype(np.int16), rate, _stats_dict(stats), cancelled)

    def synthesize(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speaker: Optional[int] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        sample_rate: Optional[int] = None,
        on_audio: Optional[AudioCallback] = None,
        request_id: Optional[str] = None,
    ) -> SynthesisResult:
        """Synthesize ``text``; requires ``tokenizer_dir``.

        ``on_audio`` receives each int16 chunk as it is produced; returning
        ``False`` from it cancels synthesis (the result is then partial and
        ``cancelled`` is set). ``sample_rate`` resamples the output
        (8000..native rate).
        """
        options = self._options(
            language, voice, speaker, seed, steps, top_k, temperature, cfg_scale,
            sample_rate, request_id,
        )
        encoded = text.encode("utf-8")
        return self._run(
            lambda handle, cb, stats: C.nemo_speech_tts_synthesize_text(
                handle, ctypes.byref(options), encoded, cb, None, stats
            ),
            options,
            on_audio,
        )

    def synthesize_tokens(
        self,
        tokens: Sequence[int],
        *,
        on_audio: Optional[AudioCallback] = None,
        **options: Any,
    ) -> SynthesisResult:
        """Synthesize pre-tokenized Magpie text tokens as a single chunk."""
        opts = self._options(
            options.get("language"), options.get("voice"), options.get("speaker"),
            options.get("seed"), options.get("steps"), options.get("top_k"),
            options.get("temperature"), options.get("cfg_scale"),
            options.get("sample_rate"), options.get("request_id"),
        )
        array = (ctypes.c_int32 * len(tokens))(*tokens)
        return self._run(
            lambda handle, cb, stats: C.nemo_speech_tts_synthesize_tokens(
                handle, ctypes.byref(opts), array, len(tokens), cb, None, stats
            ),
            opts,
            on_audio,
        )

    def stream(self, text: str, **options: Any) -> Iterator[np.ndarray]:
        """Yield int16 chunks while synthesis runs on a worker thread.

        Stopping the iteration early cancels the synthesis.
        """
        done = object()
        chunks: "queue.Queue[Any]" = queue.Queue()
        stop = threading.Event()
        failure: List[BaseException] = []

        def on_audio(chunk: np.ndarray) -> bool:
            if stop.is_set():
                return False
            chunks.put(chunk)
            return True

        def worker() -> None:
            try:
                self.synthesize(text, on_audio=on_audio, **options)
            except BaseException as e:
                failure.append(e)
            finally:
                chunks.put(done)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                item = chunks.get()
                if item is done:
                    break
                yield item
            if failure:
                raise failure[0]
        finally:
            stop.set()
            thread.join()

    # -- lifetime --

    def close(self) -> None:
        """Release the model, waiting for a synthesis running on another thread.

        Do not call it from an ``on_audio`` callback: the synthesis waits for
        the callback, so the two would wait for each other.
        """
        with self._lock:
            if self._handle:
                C.nemo_speech_tts_destroy(self._handle)
                self._handle = None

    def __enter__(self) -> "Synthesizer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
