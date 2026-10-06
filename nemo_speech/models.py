"""Model catalog and downloads, compatible with the ``nemo-speech`` CLI.

The catalog is upstream's ``models/index.json`` (bundled as
``model-index.json``). Files go to the same cache as the CLI, so models
pulled by either are shared:

- ``NEMO_SPEECH_MODEL_DIR`` if set, else
- Windows: ``%LOCALAPPDATA%\\NeMoSpeech\\models``
- macOS: ``~/Library/Caches/NeMoSpeech/models``
- Linux: ``$XDG_CACHE_HOME/nemo-speech/models`` or ``~/.cache/nemo-speech/models``

laid out as ``<root>/<org>/<name>/<revision>/<file or directory>``.
``NEMO_SPEECH_HF_BASE_URL`` overrides the Hugging Face endpoint (HTTPS only,
HTTP for loopback hosts).

Downloads behave like the CLI's, so both can share the cache safely: each
artifact is guarded by the same ``<artifact>.lock`` file, an interrupted
download resumes from ``<file>.partial`` when its ``.revision`` matches,
stalled connections time out and transient failures are retried, and a
verified file gets the CLI's ``<file>.verified`` marker so later loads skip
the SHA-256 pass.

Ported to Python from NeMo-Speech.cpp app/model_store.cpp (Copyright (c) 2026
NVIDIA CORPORATION & AFFILIATES, Apache-2.0). Downloaded models are governed by
their own licenses (see each catalog entry's ``license_url``).
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import io
import json
import os
import pathlib
import shutil
import stat
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, TypeVar

__all__ = ["cache_root", "default", "download", "find", "index", "list_models"]

_INDEX_PATH = pathlib.Path(__file__).resolve().parent / "model-index.json"
_CHUNK = 1 << 20
# Like the CLI's curl options (--connect-timeout 20, --speed-time 30,
# --retry 3): give up on a connection that stalls this long, and try a
# transient failure this many times in all, waiting longer each time.
_TIMEOUT = 30.0
_ATTEMPTS = 4
_RETRY_DELAY = 1.0

_T = TypeVar("_T")


def index() -> Dict[str, Any]:
    """The parsed model catalog."""
    with open(_INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_models() -> List[Dict[str, Any]]:
    return list(index()["models"])


def default(role: str) -> str:
    """Repository of the catalog's default model for ``role`` (e.g. "asr")."""
    defaults = index().get("defaults", {})
    if role not in defaults:
        raise KeyError(f"no default model for role {role!r}; known: {sorted(defaults)}")
    return defaults[role]


def find(name: str) -> Dict[str, Any]:
    """Look up a model by repository ("nvidia/...") or alias (case-insensitive)."""
    key = name.lower()
    for model in index()["models"]:
        if model["repo"].lower() == key or key in (a.lower() for a in model.get("aliases", [])):
            return model
    raise KeyError(f"unknown model {name!r}")


def cache_root() -> pathlib.Path:
    override = os.environ.get("NEMO_SPEECH_MODEL_DIR")
    if override:
        return pathlib.Path(override)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return pathlib.Path(local) / "NeMoSpeech" / "models"
    elif sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Caches" / "NeMoSpeech" / "models"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return pathlib.Path(xdg) / "nemo-speech" / "models"
        return pathlib.Path.home() / ".cache" / "nemo-speech" / "models"
    raise RuntimeError("cannot determine the model cache directory; set NEMO_SPEECH_MODEL_DIR")


def _base_url() -> str:
    url = os.environ.get("NEMO_SPEECH_HF_BASE_URL") or "https://huggingface.co"
    url = url.rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    loopback = parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
    if parsed.scheme != "https" and not loopback:
        raise ValueError(
            "NEMO_SPEECH_HF_BASE_URL must use HTTPS (HTTP is allowed only for loopback tests)"
        )
    return url


def _model_dir(model: Dict[str, Any]) -> pathlib.Path:
    org, name = model["repo"].split("/", 1)
    return cache_root() / org / name / model["revision"]


def _revision(model: Dict[str, Any], artifact: Dict[str, Any]) -> str:
    return artifact.get("revision") or model["revision"]


def _url(model: Dict[str, Any], artifact: Dict[str, Any]) -> str:
    revision = _revision(model, artifact)
    return f"{_base_url()}/{model['repo']}/resolve/{revision}/{artifact['filename']}?download=true"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _valid_file(path: pathlib.Path, size: int, sha256: Optional[str]) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    return sha256 is None or _sha256(path) == sha256


def _sidecar(path: pathlib.Path, suffix: str) -> pathlib.Path:
    return path.with_name(path.name + suffix)


def _write_atomically(path: pathlib.Path, text: str) -> None:
    temporary = _sidecar(path, ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


# ---- Verification markers, shared with the CLI ----


def _file_time_ns(st: os.stat_result) -> int:
    """``st_mtime`` as the CLI records it: std::filesystem::file_time_type in ns."""
    if sys.platform == "win32":
        return st.st_mtime_ns + 11_644_473_600 * 10**9  # MSVC: since 1601-01-01
    if sys.platform == "darwin":
        return st.st_mtime_ns  # libc++: the Unix epoch
    return st.st_mtime_ns - 6_437_664_000 * 10**9  # libstdc++: since 2174-01-01


def _marker_text(artifact: Dict[str, Any], st: os.stat_result) -> str:
    return f"sha256={artifact['sha256']}\nsize={st.st_size}\nmtime={_file_time_ns(st)}\n"


def _write_marker(path: pathlib.Path, artifact: Dict[str, Any]) -> None:
    """Record that ``path`` (unchanged since it was hashed) matches the artifact."""
    if artifact.get("sha256"):
        with contextlib.suppress(OSError):
            _write_atomically(_sidecar(path, ".verified"), _marker_text(artifact, path.stat()))


def _verified_file(path: pathlib.Path, artifact: Dict[str, Any]) -> bool:
    """Whether a cached file matches the artifact, hashing only when its marker is stale.

    A mismatch between this marker format and the CLI's (for example a
    different file-time epoch) only costs another SHA-256 pass.
    """
    try:
        before = path.stat()
    except OSError:
        return False
    if not stat.S_ISREG(before.st_mode) or before.st_size != artifact["size"]:
        return False
    if not artifact.get("sha256"):
        return True
    with contextlib.suppress(OSError):
        if _sidecar(path, ".verified").read_text(encoding="utf-8") == _marker_text(artifact, before):
            return True
    if _sha256(path) != artifact["sha256"]:
        return False
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns):
        _write_marker(path, artifact)
    return True


# ---- Locking and retries ----


@contextlib.contextmanager
def _artifact_lock(destination: pathlib.Path) -> Iterator[None]:
    """Hold ``<destination>.lock`` exclusively, the CLI's lock for the same artifact.

    Windows: the file is opened without sharing, so other openers get a
    sharing violation until it is closed. Elsewhere: ``flock``.
    """
    path = _sidecar(destination, ".lock")
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        create.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        invalid = ctypes.c_void_p(-1).value
        generic_read_write, open_always, normal, sharing_violation = 0xC0000000, 4, 0x80, 32
        while True:
            handle = create(str(path), generic_read_write, 0, None, open_always, normal, None)
            if handle != invalid:
                break
            error = ctypes.get_last_error()
            if error != sharing_violation:
                raise ctypes.WinError(error)
            time.sleep(0.1)
        try:
            yield
        finally:
            kernel32.CloseHandle(handle)
    else:
        import fcntl

        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # releases the lock


def _retrying(action: Callable[[], _T]) -> _T:
    """Run ``action``, retrying the network failures that may be transient."""
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return action()
        except urllib.error.HTTPError as e:
            if (e.code < 500 and e.code != 429) or attempt == _ATTEMPTS:
                raise
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError):
            if attempt == _ATTEMPTS:
                raise
        time.sleep(_RETRY_DELAY * attempt)
    raise AssertionError("unreachable")


class _Progress:
    def __init__(self, label: str, total: int, enabled: bool) -> None:
        self.label, self.total, self.enabled, self.done, self._shown = label, total, enabled, 0, -1

    def update(self, n: int) -> None:
        self.done += n
        if not self.enabled or not self.total:
            return
        percent = self.done * 100 // self.total
        if percent != self._shown:
            self._shown = percent
            sys.stderr.write(f"\r[model] {self.label}: {percent:3d}%")
            if self.done >= self.total:
                sys.stderr.write("\n")
            sys.stderr.flush()


def _fetch(url: str, out: BinaryIO, progress: _Progress, byte_range: Optional[str] = None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "nemo-speech-python"})
    if byte_range:
        request.add_header("Range", f"bytes={byte_range}")
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        if byte_range and response.status != 206:
            raise RuntimeError(f"server ignored the byte range request for {url}")
        for block in iter(lambda: response.read(_CHUNK), b""):
            out.write(block)
            progress.update(len(block))


def _safe_member_path(destination: pathlib.Path, name: str) -> pathlib.Path:
    relative = pathlib.PurePosixPath(name)
    # Backslashes and drive letters would be path syntax on Windows.
    if not name or "\\" in name or ":" in name or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe path in tokenizer artifact: {name!r}")
    return destination.joinpath(*relative.parts)


def _extract_segment(data: bytes, destination: pathlib.Path, stop_before: Optional[str]) -> None:
    """Extract regular files from one TAR segment of a ranged archive."""
    reached_stop = False
    with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as archive:
        for member in archive:
            if stop_before and pathlib.PurePosixPath(member.name).name == stop_before:
                reached_stop = True
                break
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                assert source is not None
                with open(target, "wb") as f:
                    shutil.copyfileobj(source, f)
            else:
                raise RuntimeError(f"unsupported TAR entry in tokenizer artifact: {member.name!r}")
    if stop_before and not reached_stop:
        raise RuntimeError("tokenizer archive range did not reach the expected stop member")


def _announce(model: Dict[str, Any], artifact: Dict[str, Any], progress: bool) -> None:
    if progress:
        sys.stderr.write(
            f"[model] downloading {model['repo']}@{_revision(model, artifact)[:12]} "
            f"({artifact['role']}, {artifact['size'] / 1048576:.1f} MiB)\n"
            f"[model] license: {model.get('license', 'see the model card')}"
            f" - {model.get('license_url', '')}\n"
        )


def _download_file(
    model: Dict[str, Any], artifact: Dict[str, Any], destination: pathlib.Path, progress: bool
) -> None:
    size = artifact["size"]
    revision = _revision(model, artifact)
    partial = _sidecar(destination, ".partial")
    revision_file = _sidecar(partial, ".revision")
    # Continue a partial download (ours or the CLI's) only for the same revision.
    try:
        resumable = (
            revision_file.read_text(encoding="utf-8") == revision + "\n"
            and partial.stat().st_size <= size
        )
    except OSError:
        resumable = False
    if not resumable:
        partial.unlink(missing_ok=True)
        _write_atomically(revision_file, revision + "\n")

    bar = _Progress(artifact["filename"], size, progress)

    def attempt() -> None:
        offset = partial.stat().st_size if partial.exists() else 0
        bar.done = offset
        if offset < size:
            with open(partial, "ab") as out:
                _fetch(_url(model, artifact), out, bar, f"{offset}-" if offset else None)
        if partial.stat().st_size < size:
            # The connection closed early without an error; resume.
            raise ConnectionError(f"download of {artifact['filename']} ended early")

    _retrying(attempt)
    if not _valid_file(partial, size, artifact.get("sha256")):
        partial.unlink()
        revision_file.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {artifact['filename']} failed verification")
    os.replace(partial, destination)
    revision_file.unlink(missing_ok=True)
    _write_marker(destination, artifact)


def _download_tar_ranges(
    model: Dict[str, Any], artifact: Dict[str, Any], destination: pathlib.Path, progress: bool
) -> None:
    bar = _Progress(artifact["directory"], artifact["size"], progress)
    segments = []
    for r in artifact["ranges"]:
        length = r["end"] - r["start"] + 1

        def attempt(r: Dict[str, Any] = r, done: int = bar.done) -> bytes:
            bar.done = done
            buffer = io.BytesIO()
            _fetch(_url(model, artifact), buffer, bar, f"{r['start']}-{r['end']}")
            if buffer.tell() != length:
                raise ConnectionError("truncated tokenizer archive range")
            return buffer.getvalue()

        segments.append((_retrying(attempt), r.get("stop_before")))
    extracting = _sidecar(destination, ".extracting")
    shutil.rmtree(extracting, ignore_errors=True)
    extracting.mkdir()
    try:
        for data, stop_before in segments:
            _extract_segment(data, extracting, stop_before)
        for m in artifact["members"]:
            if not _valid_file(extracting / m["name"], m["size"], m.get("sha256")):
                raise RuntimeError(f"tokenizer artifact member failed verification: {m['name']}")
    except BaseException:
        shutil.rmtree(extracting, ignore_errors=True)
        raise
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(extracting, destination)


def _materialize(model: Dict[str, Any], artifact: Dict[str, Any], progress: bool) -> pathlib.Path:
    directory = _model_dir(model)
    directory.mkdir(parents=True, exist_ok=True)
    kind = artifact["type"]

    if kind == "file":
        destination = directory / artifact["filename"]
        with _artifact_lock(destination):
            if not _verified_file(destination, artifact):
                _announce(model, artifact, progress)
                _download_file(model, artifact, destination, progress)
        return destination

    if kind == "tar-ranges":
        destination = directory / artifact["directory"]
        with _artifact_lock(destination):
            if not destination.is_dir() or not all(
                _valid_file(destination / m["name"], m["size"], m.get("sha256"))
                for m in artifact["members"]
            ):
                _announce(model, artifact, progress)
                _download_tar_ranges(model, artifact, destination, progress)
        return destination

    raise NotImplementedError(f"artifact type {kind!r} is not supported")


def download(
    name: str, *, companions: bool = False, progress: bool = True
) -> Dict[str, pathlib.Path]:
    """Ensure every artifact of ``name`` is cached; return ``{role: path}``.

    Cached files are verified (size and SHA-256) and reused. With
    ``companions=True`` the model's companions (e.g. the TTS codec) are
    downloaded too and merged into the result.
    """
    model = find(name)
    paths: Dict[str, pathlib.Path] = {}
    if companions:
        for companion in model.get("companions", []):
            paths.update(download(companion, progress=progress))
    for artifact in model["artifacts"]:
        paths[artifact["role"]] = _materialize(model, artifact, progress)
    return paths
