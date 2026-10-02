"""Model catalog and archive handling (no network)."""

import io
import pathlib
import tarfile

import pytest

from nemo_speech import models


def test_find_by_alias_and_repo():
    assert models.find("nemotron-3.5")["repo"] == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert models.find("NVIDIA/Nemotron-3-Diarization")["repo"] == "nvidia/Nemotron-3-Diarization"
    with pytest.raises(KeyError):
        models.find("no-such-model")


def test_defaults_resolve():
    for role in ("asr", "diarization", "tts", "codec"):
        models.find(models.default(role))


def test_cache_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(tmp_path))
    assert models.cache_root() == tmp_path


def test_base_url_requires_https(monkeypatch):
    monkeypatch.setenv("NEMO_SPEECH_HF_BASE_URL", "http://example.com")
    with pytest.raises(ValueError):
        models._base_url()
    monkeypatch.setenv("NEMO_SPEECH_HF_BASE_URL", "http://127.0.0.1:8080/")
    assert models._base_url() == "http://127.0.0.1:8080"


def _tar(entries) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_extract_segment_stops_before_member(tmp_path):
    data = _tar([("./a.txt", b"alpha"), ("./model_weights.ckpt", b"x" * 1024), ("./b.txt", b"b")])
    models._extract_segment(data, tmp_path, "model_weights.ckpt")
    assert (tmp_path / "a.txt").read_bytes() == b"alpha"
    assert not (tmp_path / "model_weights.ckpt").exists()
    assert not (tmp_path / "b.txt").exists()


def test_extract_segment_requires_stop_member(tmp_path):
    with pytest.raises(RuntimeError, match="stop member"):
        models._extract_segment(_tar([("a.txt", b"a")]), tmp_path, "missing.ckpt")


def test_extract_segment_rejects_unsafe_paths(tmp_path):
    with pytest.raises(RuntimeError, match="unsafe"):
        models._extract_segment(_tar([("../evil.txt", b"x")]), tmp_path, None)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_cached_file_is_verified(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert models._valid_file(path, 3, sha)
    assert not models._valid_file(path, 3, "0" * 64)
    assert not models._valid_file(path, 4, None)
    assert not models._valid_file(pathlib.Path(tmp_path / "missing"), 3, None)
