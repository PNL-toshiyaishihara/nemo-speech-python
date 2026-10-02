"""High-level text translation (Riva-Translate through llama.cpp)."""

from __future__ import annotations

import ctypes
from typing import Any, List, Optional, Sequence, Union, overload

from ._common import PathLike, decode, fsencode_or_none, status_checker
from .capi import nmt as C

__all__ = ["Translator", "version"]

_check = status_checker(C.nemo_speech_nmt_last_error)


def version() -> str:
    """Version string of the loaded NeMo-Speech.cpp NMT library."""
    return decode(C.nemo_speech_nmt_version())


class Translator:
    """A loaded translation model. Safe to call from multiple threads.

    Args:
        model_path: Riva-Translate GGUF (see the upstream docs/nmt/models.md).
        gpu: GPU device index, ``-1`` for CPU, ``None`` for the library default.
        n_ctx: Decode context length in tokens; library default if None.
        max_new_tokens: Generation cap per input text; library default if None.
        contexts: Concurrent decode contexts; library default if None.
    """

    def __init__(
        self,
        model_path: PathLike,
        *,
        gpu: Optional[int] = None,
        n_ctx: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        contexts: Optional[int] = None,
    ) -> None:
        self._handle: Optional[ctypes.c_void_p] = None
        cfg = C.nemo_speech_nmt_translator_config()
        model = C.nemo_speech_nmt_model_config(
            path=fsencode_or_none(model_path), n_ctx=n_ctx or 0
        )
        cfg.model = ctypes.pointer(model)
        if gpu is not None:
            cfg.backend = ctypes.pointer(C.nemo_speech_nmt_backend_config(gpu=gpu))
        if max_new_tokens:
            cfg.generation = ctypes.pointer(
                C.nemo_speech_nmt_generation_config(max_new_tokens=max_new_tokens)
            )
        if contexts:
            cfg.pool = ctypes.pointer(C.nemo_speech_nmt_pool_config(contexts=contexts))
        handle = C.nemo_speech_nmt_translator_p()
        _check(C.nemo_speech_nmt_create(ctypes.byref(cfg), ctypes.byref(handle)))
        self._handle = handle

    @overload
    def translate(self, texts: str, source_language: str, target_language: str) -> str: ...

    @overload
    def translate(
        self, texts: Sequence[str], source_language: str, target_language: str
    ) -> List[str]: ...

    def translate(
        self,
        texts: Union[str, Sequence[str]],
        source_language: str,
        target_language: str,
    ) -> Union[str, List[str]]:
        """Translate one text or a batch.

        Languages are two-letter codes ("en", "ja") or the model's pair tag
        ("en-ja"). Unsupported pairs raise :class:`NemoSpeechError`.
        """
        if not self._handle:
            raise RuntimeError("Translator is closed")
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return []
        array = (ctypes.c_char_p * len(items))(*(t.encode("utf-8") for t in items))
        result = C.nemo_speech_nmt_result_p()
        _check(
            C.nemo_speech_nmt_translate(
                self._handle,
                array,
                len(items),
                source_language.encode(),
                target_language.encode(),
                ctypes.byref(result),
            )
        )
        try:
            out = [
                decode(C.nemo_speech_nmt_result_text(result, i))
                for i in range(C.nemo_speech_nmt_result_count(result))
            ]
        finally:
            C.nemo_speech_nmt_result_destroy(result)
        return out[0] if single else out

    def close(self) -> None:
        if self._handle:
            C.nemo_speech_nmt_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "Translator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
