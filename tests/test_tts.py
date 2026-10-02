"""MagpieTTS synthesis; skipped unless the model, codec and tokenizer are cached."""

import re

import numpy as np
import pytest

from nemo_speech import load_wav

TEXT = "Hello from the Python bindings."


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())


def test_properties(synthesizer):
    assert synthesizer.sample_rate == 22050
    assert synthesizer.speakers and all(synthesizer.speakers)


def test_synthesize(synthesizer):
    chunks = []
    result = synthesizer.synthesize(TEXT, on_audio=chunks.append)
    assert not result.cancelled
    assert result.audio.dtype == np.int16
    assert result.sample_rate == synthesizer.sample_rate
    assert 0.5 < result.duration < 10
    assert np.abs(result.audio).max() > 1000  # not silence
    assert sum(c.size for c in chunks) == result.audio.size
    assert result.stats["sample_rate"] == result.sample_rate


def test_round_trip_through_asr(synthesizer, recognizer):
    result = synthesizer.synthesize(TEXT, seed=1)
    text = recognizer.transcribe(result.audio, result.sample_rate).text
    assert "python bindings" in _normalized(text)


def test_output_sample_rate(synthesizer):
    result = synthesizer.synthesize(TEXT, sample_rate=16000)
    assert result.sample_rate == 16000


def test_callback_cancels(synthesizer):
    result = synthesizer.synthesize(
        "This sentence is long enough to produce several chunks of audio before it ends.",
        on_audio=lambda chunk: False,
    )
    assert result.cancelled
    assert result.duration < 2


def test_callback_exception_propagates(synthesizer):
    def boom(chunk):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        synthesizer.synthesize(TEXT, on_audio=boom)


def test_stream_and_early_stop(synthesizer):
    full = sum(c.size for c in synthesizer.stream(TEXT))
    assert full > 0
    for i, _ in enumerate(synthesizer.stream(TEXT * 3)):
        if i == 0:
            break
    # The synthesizer is still usable after an abandoned stream.
    assert synthesizer.synthesize(TEXT).audio.size > 0


def test_save(synthesizer, tmp_path):
    result = synthesizer.synthesize(TEXT)
    path = tmp_path / "out.wav"
    result.save(path)
    samples, rate = load_wav(path)
    assert rate == result.sample_rate
    assert samples.size == result.audio.size
