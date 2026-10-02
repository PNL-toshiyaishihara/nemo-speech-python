"""Shared fixtures.

Model-backed tests never download: each model comes from an environment
variable or an already-populated cache (``nemo_speech.models``, shared with
the nemo-speech CLI), and the test is skipped otherwise.

    NEMO_SPEECH_TEST_ASR_MODEL    ASR GGUF
    NEMO_SPEECH_TEST_DIAR_MODEL   Sortformer GGUF
    NEMO_SPEECH_TEST_NMT_MODEL    Riva-Translate GGUF (not in the catalog)
    NEMO_SPEECH_TEST_GPU          device index for every model (default -1 = CPU)
"""

import os
import pathlib
from typing import Dict, Optional

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / "vendor" / "NeMo-Speech.cpp"
JFK_WAV = UPSTREAM_DIR / "test_files" / "asr" / "wav" / "test" / "jfk.wav"
LOCAL_MODELS = REPO_ROOT / "models"


def test_gpu() -> int:
    return int(os.environ.get("NEMO_SPEECH_TEST_GPU", "-1"))


test_gpu.__test__ = False  # not a test function


def _cached(name: str) -> Optional[Dict[str, pathlib.Path]]:
    """Paths of a catalog model when every artifact is already cached."""
    from nemo_speech import models

    model = models.find(name)
    directory = models.cache_root().joinpath(*model["repo"].split("/"), model["revision"])
    paths = {}
    for artifact in model["artifacts"]:
        path = directory / (artifact.get("filename") if artifact["type"] == "file" else artifact["directory"])
        if not path.exists():
            return None
        paths[artifact["role"]] = path
    return paths


def _model(env: str, *candidates: Optional[pathlib.Path]) -> pathlib.Path:
    if os.environ.get(env):
        path = pathlib.Path(os.environ[env])
        if not path.exists():
            pytest.fail(f"{env} points at a missing file: {path}")
        return path
    for path in candidates:
        if path is not None and path.exists():
            return path
    pytest.skip(f"model not available (set {env})")


@pytest.fixture(scope="session")
def upstream_dir() -> pathlib.Path:
    return UPSTREAM_DIR


@pytest.fixture(scope="session")
def jfk_wav() -> pathlib.Path:
    if not JFK_WAV.exists():
        pytest.skip(f"test audio not found: {JFK_WAV}")
    return JFK_WAV


@pytest.fixture(scope="session")
def asr_model_path() -> pathlib.Path:
    cached = _cached("nemotron-3.5") or {}
    return _model(
        "NEMO_SPEECH_TEST_ASR_MODEL",
        LOCAL_MODELS / "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        cached.get("asr"),
    )


@pytest.fixture(scope="session")
def recognizer(asr_model_path):
    from nemo_speech import Recognizer

    with Recognizer(asr_model_path, gpu=test_gpu()) as r:
        yield r


@pytest.fixture(scope="session")
def diar_model_path() -> pathlib.Path:
    cached = _cached("nemotron-diar") or {}
    return _model("NEMO_SPEECH_TEST_DIAR_MODEL", cached.get("diarization"))


@pytest.fixture(scope="session")
def diarizer(diar_model_path):
    from nemo_speech import Diarizer

    with Diarizer(diar_model_path, gpu=test_gpu()) as d:
        yield d


@pytest.fixture(scope="session")
def tts_paths() -> Dict[str, pathlib.Path]:
    tts = _cached("magpie")
    codec = _cached("nano-codec")
    if not tts or not codec:
        pytest.skip("MagpieTTS not cached (run nemo_speech.models.download('magpie', companions=True))")
    return {**tts, **codec}


@pytest.fixture(scope="session")
def synthesizer(tts_paths):
    from nemo_speech import Synthesizer

    with Synthesizer(tts_paths["tts"], tts_paths["codec"], tokenizer_dir=tts_paths["tokenizer"]) as s:
        yield s


@pytest.fixture(scope="session")
def nmt_model_path() -> pathlib.Path:
    return _model(
        "NEMO_SPEECH_TEST_NMT_MODEL", LOCAL_MODELS / "Riva-Translate-4B-Instruct-v2-Q4_K_M.gguf"
    )


@pytest.fixture(scope="session")
def translator(nmt_model_path):
    from nemo_speech import Translator

    with Translator(nmt_model_path, gpu=test_gpu()) as t:
        yield t
