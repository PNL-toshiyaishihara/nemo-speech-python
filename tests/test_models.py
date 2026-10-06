"""Model catalog, archive handling and downloads (no network beyond loopback)."""

import hashlib
import http.server
import io
import os
import pathlib
import tarfile
import threading
import urllib.error

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


@pytest.mark.parametrize("name", ["../evil.txt", "..\\evil.txt", "a\\..\\..\\evil.txt", "C:evil.txt"])
def test_extract_segment_rejects_unsafe_paths(tmp_path, name):
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(RuntimeError, match="unsafe"):
        models._extract_segment(_tar([(name, b"x")]), target, None)
    assert not (tmp_path / "evil.txt").exists()
    assert not any(target.iterdir())


def test_cached_file_is_verified(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert models._valid_file(path, 3, sha)
    assert not models._valid_file(path, 3, "0" * 64)
    assert not models._valid_file(path, 4, None)
    assert not models._valid_file(pathlib.Path(tmp_path / "missing"), 3, None)


# ---- Downloads against a loopback server ----

PAYLOAD = bytes(range(256)) * 64  # 16 KiB


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        server = self.server
        server.requests.append(self.headers.get("Range"))
        if server.failures:
            server.failures -= 1
            self.send_error(503)
            return
        if server.status != 200:
            self.send_error(server.status)
            return
        body, status = server.payload, 200
        requested = self.headers.get("Range")
        if requested:
            start, _, end = requested[len("bytes="):].partition("-")
            end = int(end) if end else len(body) - 1
            body, status = body[int(start) : end + 1], 206
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch, tmp_path):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.payload, httpd.requests, httpd.failures, httpd.status = PAYLOAD, [], 0, 200
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NEMO_SPEECH_HF_BASE_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(models, "_RETRY_DELAY", 0.0)
    yield httpd
    httpd.shutdown()
    httpd.server_close()


MODEL = {"repo": "org/name", "revision": "rev1", "license": "test", "license_url": "about:blank"}
ARTIFACT = {
    "type": "file",
    "role": "asr",
    "filename": "model.gguf",
    "size": len(PAYLOAD),
    "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
}


def _destination():
    return models._model_dir(MODEL) / ARTIFACT["filename"]


def test_download_verifies_and_marks(server):
    path = models._materialize(MODEL, ARTIFACT, progress=False)
    assert path == _destination() and path.read_bytes() == PAYLOAD
    marker = path.with_name("model.gguf.verified").read_text(encoding="utf-8").splitlines()
    assert marker[0] == f"sha256={ARTIFACT['sha256']}" and marker[1] == f"size={len(PAYLOAD)}"
    assert marker[2].startswith("mtime=")
    assert not path.with_name("model.gguf.partial").exists()
    assert not path.with_name("model.gguf.partial.revision").exists()


def test_marker_skips_rehash_until_the_file_changes(server, monkeypatch):
    path = models._materialize(MODEL, ARTIFACT, progress=False)
    hashed = []
    real_sha256 = models._sha256
    monkeypatch.setattr(models, "_sha256", lambda p: hashed.append(p) or real_sha256(p))
    assert models._materialize(MODEL, ARTIFACT, progress=False) == path
    assert hashed == [] and server.requests == [None]
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # touched: hash again
    models._materialize(MODEL, ARTIFACT, progress=False)
    assert hashed == [path] and server.requests == [None]


def test_partial_download_resumes_after_a_transient_failure(server):
    destination = _destination()
    destination.parent.mkdir(parents=True)
    destination.with_name("model.gguf.partial").write_bytes(PAYLOAD[:5000])
    destination.with_name("model.gguf.partial.revision").write_text("rev1\n", encoding="utf-8")
    server.failures = 1  # 503 first, then success
    path = models._materialize(MODEL, ARTIFACT, progress=False)
    assert path.read_bytes() == PAYLOAD
    assert server.requests == ["bytes=5000-", "bytes=5000-"]


def test_partial_download_of_another_revision_restarts(server):
    destination = _destination()
    destination.parent.mkdir(parents=True)
    destination.with_name("model.gguf.partial").write_bytes(b"x" * 5000)
    destination.with_name("model.gguf.partial.revision").write_text("rev0\n", encoding="utf-8")
    assert models._materialize(MODEL, ARTIFACT, progress=False).read_bytes() == PAYLOAD
    assert server.requests == [None]


def test_client_errors_are_not_retried(server):
    server.status = 404
    with pytest.raises(urllib.error.HTTPError):
        models._materialize(MODEL, ARTIFACT, progress=False)
    assert server.requests == [None]


def test_server_errors_give_up_after_the_last_attempt(server):
    server.failures = models._ATTEMPTS
    with pytest.raises(urllib.error.HTTPError):
        models._materialize(MODEL, ARTIFACT, progress=False)
    assert len(server.requests) == models._ATTEMPTS


def test_artifact_lock_is_exclusive(tmp_path):
    destination = tmp_path / "model.gguf"
    acquired = threading.Event()

    def contender():
        with models._artifact_lock(destination):
            acquired.set()

    with models._artifact_lock(destination):
        thread = threading.Thread(target=contender)
        thread.start()
        assert not acquired.wait(0.5)
    assert acquired.wait(10)
    thread.join()
    assert (tmp_path / "model.gguf.lock").exists()
