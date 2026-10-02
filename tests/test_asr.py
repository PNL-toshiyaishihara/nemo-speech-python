"""End-to-end recognition tests; skipped when no ASR model is available."""

import re
import threading

import numpy as np
import pytest

from nemo_speech import NemoSpeechError, Recognizer, load_wav

JFK_PHRASE = "ask not what your country can do for you"


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z' ]", " ", text.lower()).split())


def test_missing_model_raises():
    with pytest.raises(NemoSpeechError) as info:
        Recognizer("does-not-exist.gguf", gpu=-1)
    assert info.value.message


def test_transcribe_file(recognizer, jfk_wav):
    result = recognizer.transcribe_file(jfk_wav)
    assert JFK_PHRASE in _normalized(result.text)
    assert result.is_final


def test_word_timestamps(recognizer, jfk_wav):
    result = recognizer.transcribe_file(jfk_wav, word_timestamps=True)
    words = result.words
    assert words
    starts = [w.start_ms for w in words]
    assert starts == sorted(starts)
    assert all(w.end_ms >= w.start_ms for w in words)
    assert words[-1].end <= 11.5  # the clip is 11 s long


def test_int16_and_float_input_agree(recognizer, jfk_wav):
    samples, rate = load_wav(jfk_wav)
    pcm16 = np.round(samples * 32767).astype(np.int16)
    assert recognizer.transcribe(pcm16, rate).text == recognizer.transcribe(samples, rate).text


def test_streaming(recognizer, jfk_wav):
    samples, rate = load_wav(jfk_wav)
    chunk = rate // 2
    results = []
    with recognizer.stream() as stream:
        for start in range(0, samples.size, chunk):
            results += stream.push(samples[start : start + chunk], rate)
        results += stream.finish()
    final_text = " ".join(r.text for r in results if r.is_final)
    assert JFK_PHRASE in _normalized(final_text)


def test_concurrent_transcribe(recognizer, jfk_wav):
    samples, rate = load_wav(jfk_wav)
    expected = recognizer.transcribe(samples, rate).text
    texts = [None, None]

    def worker(i):
        texts[i] = recognizer.transcribe(samples, rate).text

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert texts == [expected, expected]


def test_closed_recognizer_rejects_calls(asr_model_path):
    r = Recognizer(asr_model_path, gpu=-1)
    stream = r.stream()
    r.close()  # also closes the open stream
    with pytest.raises(RuntimeError):
        stream.push(np.zeros(1600, dtype=np.float32))
    with pytest.raises(RuntimeError):
        r.transcribe(np.zeros(1600, dtype=np.float32))
