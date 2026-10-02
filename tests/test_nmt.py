"""Riva-Translate NMT; skipped when no model is available (it is not in the catalog)."""

import pytest

from nemo_speech import NemoSpeechError, Translator


def test_missing_model_raises():
    with pytest.raises(NemoSpeechError):
        Translator("does-not-exist.gguf", gpu=-1)


def test_single_text(translator):
    out = translator.translate("Ich liebe Programmieren.", "de", "en")
    assert isinstance(out, str)
    assert "love" in out.lower()


def test_batch_keeps_order(translator):
    out = translator.translate(["Good morning.", "Thank you very much."], "en", "de")
    assert isinstance(out, list) and len(out) == 2
    assert "morgen" in out[0].lower()
    assert "dank" in out[1].lower()  # "Danke ..." or "Vielen Dank."


def test_empty_batch(translator):
    assert translator.translate([], "en", "de") == []


def test_unsupported_language_raises(translator):
    with pytest.raises(NemoSpeechError):
        translator.translate("Hello.", "en", "xx")
