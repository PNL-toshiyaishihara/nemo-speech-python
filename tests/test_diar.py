"""Standalone diarization; skipped when no Sortformer model is available."""

import numpy as np
import pytest

from nemo_speech import NemoSpeechError, SegmentationConfig, load_wav
from nemo_speech import Diarizer


def test_missing_model_raises():
    with pytest.raises(NemoSpeechError):
        Diarizer("does-not-exist.gguf", gpu=-1)


def test_model_properties(diarizer):
    assert diarizer.num_speakers in (4, 8)
    assert 0 < diarizer.seconds_per_frame <= 0.08 + 1e-6


def test_offline_single_speaker(diarizer, jfk_wav):
    samples, rate = load_wav(jfk_wav)
    result = diarizer.diarize(samples, rate)
    assert result.segments
    assert {s.speaker for s in result.segments} == {1}
    assert all(s.end > s.start for s in result.segments)
    starts = [s.start for s in result.segments]
    assert starts == sorted(starts)
    assert result.frame_probs.shape[1] == diarizer.num_speakers
    duration = result.frame_probs.shape[0] * result.seconds_per_frame
    assert abs(duration - samples.size / rate) < 0.5


def test_stream_matches_offline(diarizer, jfk_wav):
    samples, rate = load_wav(jfk_wav)
    offline = diarizer.diarize(samples, rate).segments
    with diarizer.stream() as stream:
        for i in range(0, samples.size, rate // 2):
            stream.push(samples[i : i + rate // 2], rate)
        stream.finish()
        segments = stream.segments()
        probs = stream.frame_probs()
    assert len(segments) == len(offline)
    for a, b in zip(segments, offline):
        assert a.speaker == b.speaker
        assert abs(a.start - b.start) < 0.25 and abs(a.end - b.end) < 0.25
    assert probs.shape[1] == diarizer.num_speakers


def test_segmentation_config_changes_segments(diarizer, jfk_wav):
    samples, rate = load_wav(jfk_wav)
    merged = diarizer.diarize(samples, rate, segmentation=SegmentationConfig(min_gap=2.0))
    default = diarizer.diarize(samples, rate)
    assert len(merged.segments) <= len(default.segments)


def test_closed_diarizer_closes_streams(diar_model_path):
    d = Diarizer(diar_model_path, gpu=-1)
    stream = d.stream()
    d.close()
    with pytest.raises(RuntimeError):
        stream.push(np.zeros(1600, dtype=np.float32))
