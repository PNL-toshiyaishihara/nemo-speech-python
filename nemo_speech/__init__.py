"""Python bindings for NeMo-Speech.cpp.

High-level API::

    from nemo_speech import Recognizer, Synthesizer

    with Recognizer.from_pretrained("nemotron-3.5", gpu=-1) as asr:
        print(asr.transcribe_file("speech.wav").text)

    with Synthesizer.from_pretrained("magpie") as tts:
        tts.synthesize("Hello world.").save("hello.wav")

Components load lazily, so a build without (say) NMT still imports. The
ctypes mirror of the C ABI is available as :mod:`nemo_speech.capi`.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0rc2"

from ._common import NemoSpeechError
from .audio import load_wav, save_wav

# Public name -> (module, attribute). Importing a module loads its library.
_LAZY = {
    "Alternative": ("asr", "Alternative"),
    "RecognitionResult": ("asr", "RecognitionResult"),
    "RecognitionStream": ("asr", "RecognitionStream"),
    "Recognizer": ("asr", "Recognizer"),
    "SpeechContext": ("asr", "SpeechContext"),
    "Word": ("asr", "Word"),
    "asr_version": ("asr", "version"),
    "DiarizationResult": ("diar", "DiarizationResult"),
    "DiarizationStream": ("diar", "DiarizationStream"),
    "Diarizer": ("diar", "Diarizer"),
    "SegmentationConfig": ("diar", "SegmentationConfig"),
    "SpeakerSegment": ("diar", "SpeakerSegment"),
    "Translator": ("nmt", "Translator"),
    "nmt_version": ("nmt", "version"),
    "SynthesisResult": ("tts", "SynthesisResult"),
    "Synthesizer": ("tts", "Synthesizer"),
    "tts_version": ("tts", "version"),
}

__all__ = ["NemoSpeechError", "build_info", "load_wav", "save_wav", *_LAZY]


def build_info() -> dict:
    """How this installation was built.

    Returns the package version, the variant (``default``, ``vulkan``,
    ``cu128``, ...), the enabled GPU backends and components, CUDA toolkit
    and architectures for CUDA builds, and the commit of every vendored
    source (NeMo-Speech.cpp, llama.cpp, SentencePiece). Include it in bug
    reports.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parent / "_build_info.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} is missing; build_info() needs an installed package, not the source tree"
        ) from None


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except FileNotFoundError as e:
        raise ImportError(
            f"nemo_speech.{name} is unavailable: this build does not include the "
            f"{module_name} library ({e})"
        ) from e
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY))
